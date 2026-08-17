"""Group-wise, position-randomized LLM-as-a-Judge evaluation."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .prompts import JudgePromptPack


SCORE_FIELDS = (
    "coherence",
    "identification",
    "comforting",
    "suggestion",
    "information",
)

JUDGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_label": {"type": "string"},
                    "scores": {
                        "type": "object",
                        "properties": {
                            field: {"type": "integer"} for field in SCORE_FIELDS
                        },
                        "required": list(SCORE_FIELDS),
                        "additionalProperties": False,
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["candidate_label", "scores", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["evaluations"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class JudgeConfig:
    """OpenAI judge and reproducibility settings."""

    model: str = "gpt-5.6"
    shuffle_seed: int = 3407
    max_output_tokens: int = 1600
    max_retries: int = 3


def load_inference_outputs(path: str | Path) -> dict[int, str]:
    """Load ``{id, output}`` records from a JSON array/object or JSONL file."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]

    outputs: dict[int, str] = {}
    for record in records:
        if not isinstance(record, dict) or "id" not in record or "output" not in record:
            raise ValueError(f"Invalid inference record in {path}: {record!r}")
        sample_id = record["id"]
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise ValueError(f"Inference id must be an integer in {path}: {sample_id!r}")
        if sample_id in outputs:
            raise ValueError(f"Duplicate inference id {sample_id} in {path}")
        outputs[sample_id] = str(record["output"])
    return outputs


def find_model_output(inference_dir: str | Path, model_name: str) -> Path:
    """Resolve either ``model.jsonl`` or ``model.json``."""
    inference_dir = Path(inference_dir)
    matches = [
        path
        for path in (
            inference_dir / f"{model_name}.jsonl",
            inference_dir / f"{model_name}.json",
        )
        if path.exists()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No inference file for '{model_name}' in {inference_dir}; "
            f"expected {model_name}.jsonl or {model_name}.json"
        )
    if len(matches) > 1:
        raise ValueError(f"Ambiguous inference files for '{model_name}': {matches}")
    return matches[0]


class OpenAIGroupJudge:
    """Evaluate an anonymous group through the OpenAI Responses API."""

    def __init__(self, client=None, config: JudgeConfig | None = None):
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.config = config or JudgeConfig()

    def evaluate(self, context: str, candidates: Mapping[str, str]) -> dict[str, Any]:
        if not candidates:
            raise ValueError("At least one candidate is required")
        messages = JudgePromptPack.group(context, dict(candidates))
        response = self.client.responses.create(
            model=self.config.model,
            input=messages,
            max_output_tokens=self.config.max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "esc_group_judgment",
                    "strict": True,
                    "schema": JUDGMENT_SCHEMA,
                }
            },
        )
        judgment = json.loads(response.output_text)
        self._validate(judgment, set(candidates))
        return judgment

    @staticmethod
    def _validate(judgment: Mapping[str, Any], expected_labels: set[str]) -> None:
        evaluations = judgment.get("evaluations", [])
        labels = [evaluation.get("candidate_label") for evaluation in evaluations]
        if len(labels) != len(set(labels)) or set(labels) != expected_labels:
            raise ValueError(
                f"Judge returned labels {labels}; expected each of {sorted(expected_labels)} once"
            )
        for evaluation in evaluations:
            for field in SCORE_FIELDS:
                score = evaluation["scores"][field]
                if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
                    raise ValueError(
                        f"Invalid {field} score for {evaluation['candidate_label']}: {score}"
                    )


def _sample_rng(seed: int, group_name: str, sample_id: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{group_name}:{sample_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _completed_ids(
    path: Path,
    *,
    group_name: str,
    model_names: Sequence[str],
    model: str,
    seed: int,
) -> set[int]:
    """Validate a result checkpoint and return its completed sample ids."""
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        candidate_order = record.get("candidate_order", [])
        recorded_models = {
            item.get("model") for item in candidate_order if isinstance(item, dict)
        }
        if (
            record.get("group") != group_name
            or record.get("judge_model") != model
            or record.get("shuffle_seed") != seed
            or recorded_models != set(model_names)
        ):
            raise ValueError(
                f"Existing {path} uses a different group, candidate set, judge "
                "model, or shuffle seed. "
                "Choose another results directory or remove the old result intentionally."
            )
        sample_id = record.get("id")
        if isinstance(sample_id, bool) or not isinstance(sample_id, int):
            raise ValueError(f"Invalid sample id in existing judge result: {sample_id!r}")
        if sample_id in completed:
            raise ValueError(f"Duplicate sample id {sample_id} in {path}")
        completed.add(sample_id)
    return completed


def run_evaluation_group(
    contexts: Sequence[str],
    group_name: str,
    model_names: Sequence[str],
    judge: OpenAIGroupJudge,
    *,
    inference_dir: str | Path = "inference",
    results_dir: str | Path = "inference/judge",
    resume: bool = True,
    show_progress: bool = True,
) -> Path:
    """Judge all shared sample ids for one experimental group."""
    if len(model_names) < 2 or len(set(model_names)) != len(model_names):
        raise ValueError("An evaluation group requires at least two distinct models")

    outputs = {
        name: load_inference_outputs(find_model_output(inference_dir, name))
        for name in model_names
    }
    reference_ids = set(next(iter(outputs.values())))
    for name, model_outputs in outputs.items():
        if set(model_outputs) != reference_ids:
            raise ValueError(
                f"Model '{name}' does not contain exactly the same ids as the other group members"
            )
    invalid_ids = sorted(sample_id for sample_id in reference_ids if not 0 <= sample_id < len(contexts))
    if invalid_ids:
        raise IndexError(f"Inference ids outside the test context range: {invalid_ids[:5]}")

    results_path = Path(results_dir) / f"{group_name}.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        _completed_ids(
            results_path,
            group_name=group_name,
            model_names=model_names,
            model=judge.config.model,
            seed=judge.config.shuffle_seed,
        )
        if resume
        else set()
    )
    mode = "a" if resume else "w"
    unexpected_completed = sorted(completed - reference_ids)
    if unexpected_completed:
        raise ValueError(
            f"Existing {results_path} contains ids absent from current inference "
            f"outputs: {unexpected_completed[:10]}"
        )
    pending_ids = sorted(reference_ids - completed)

    sample_ids: Any = pending_ids
    if show_progress:
        try:
            from tqdm.auto import tqdm

            sample_ids = tqdm(
                sample_ids,
                desc=f"Judging {group_name}",
                initial=len(completed),
                total=len(reference_ids),
            )
        except ImportError:
            pass

    with results_path.open(mode, encoding="utf-8", newline="\n") as output_file:
        for sample_id in sample_ids:
            shuffled_models = list(model_names)
            _sample_rng(judge.config.shuffle_seed, group_name, sample_id).shuffle(shuffled_models)
            labels = [chr(ord("A") + index) for index in range(len(shuffled_models))]
            if len(labels) > 26:
                raise ValueError("At most 26 candidates are supported in one group")
            label_to_model = dict(zip(labels, shuffled_models))
            candidates = {
                label: outputs[model_name][sample_id]
                for label, model_name in label_to_model.items()
            }

            for attempt in range(judge.config.max_retries):
                try:
                    judgment = judge.evaluate(contexts[sample_id], candidates)
                    break
                except Exception:
                    if attempt + 1 == judge.config.max_retries:
                        raise
                    time.sleep(2**attempt)

            evaluations_by_label = {
                item["candidate_label"]: item for item in judgment["evaluations"]
            }
            record = {
                "id": sample_id,
                "group": group_name,
                "judge_model": judge.config.model,
                "shuffle_seed": judge.config.shuffle_seed,
                "candidate_order": [
                    {"candidate_label": label, "model": label_to_model[label]}
                    for label in labels
                ],
                "evaluations": [
                    {
                        "model": label_to_model[label],
                        **evaluations_by_label[label],
                    }
                    for label in labels
                ],
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()

    summarize_group_results(results_path)
    print(
        f"Judge group {group_name}: resumed {len(completed)}, "
        f"evaluated this run {len(pending_ids)}, total {len(reference_ids)}."
    )
    return results_path


def summarize_group_results(results_path: str | Path) -> Path:
    """Write per-model mean criterion scores next to a group JSONL result."""
    results_path = Path(results_path)
    totals: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        for evaluation in record["evaluations"]:
            for field in SCORE_FIELDS:
                totals[evaluation["model"]][field].append(evaluation["scores"][field])

    summary = {
        "group": records[0]["group"] if records else results_path.stem,
        "judge_model": records[0]["judge_model"] if records else None,
        "num_samples": len(records),
        "mean_scores": {
            model: {
                field: round(sum(values) / len(values), 4)
                for field, values in fields.items()
            }
            for model, fields in totals.items()
        },
    }
    summary_path = results_path.with_name(f"{results_path.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def run_evaluation_groups(
    contexts: Sequence[str],
    groups: Mapping[str, Sequence[str]],
    judge: OpenAIGroupJudge,
    **kwargs,
) -> dict[str, Path]:
    """Run each research-question group as an independent judging experiment."""
    return {
        group_name: run_evaluation_group(
            contexts,
            group_name,
            model_names,
            judge,
            **kwargs,
        )
        for group_name, model_names in groups.items()
    }
