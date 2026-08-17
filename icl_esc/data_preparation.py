"""ESConv loading and preprocessing utilities.

The functions in this module deliberately do not load a dataset at import time. This
makes them usable from notebooks, scripts, and tests without hidden network access.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

def extract_context(dialog_turns: Sequence[Mapping[str, Any]]) -> str:
    """Render ESConv turns as the text context expected by the Llama model."""
    lines = []
    for turn in dialog_turns:
        speaker = "seeker" if turn["speaker"] == "usr" else "supporter"
        lines.append(f"{speaker}: {turn['text']}")
    return "\n".join(lines)


def load_esconv_dataset():
    """Load the ESConv dataset from Hugging Face."""
    from datasets import load_dataset

    return load_dataset("thu-coai/esconv")


def _valid_test_cut_indices(dialog: Sequence[Mapping[str, Any]]) -> list[int]:
    """Return seeker turns from rounds 5-23 that are followed by a supporter."""
    return [
        index
        for index in range(4, min(len(dialog) - 1, 23))
        if dialog[index]["speaker"] == "usr"
        and dialog[index + 1]["speaker"] == "sys"
    ]


def make_test_batch_processor(*, seed: int = 3407):
    """Create a deterministic processor suitable for ``Dataset.map``."""
    rng = random.Random(seed)

    def process_test_batch(examples: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
        contexts: list[str] = []
        blender_contexts: list[str] = []

        for raw_json in examples["text"]:
            dialog = json.loads(raw_json).get("dialog", [])
            valid_indices = _valid_test_cut_indices(dialog)
            if not valid_indices:
                continue

            cut_index = rng.choice(valid_indices)
            context_turns = dialog[: cut_index + 1]
            contexts.append(extract_context(context_turns))
            utterances = [turn["text"] for turn in context_turns]
            blender_contexts.append("  ".join(f" {utterance}" for utterance in utterances))

        return {"context": contexts, "blender_context": blender_contexts}

    return process_test_batch


def prepare_test_dataset(dataset, *, split: str = "test", seed: int = 3407):
    """Create standard and BlenderBot contexts from an ESConv split."""
    return dataset[split].map(
        make_test_batch_processor(seed=seed),
        batched=True,
        remove_columns=["text"],
    )


def make_train_batch_formatter(tokenizer, *, context_window: int = 4):
    """Create the training formatter used by Hugging Face ``Dataset.map``."""
    if context_window <= 0:
        raise ValueError("context_window must be greater than zero")

    def format_train_batch(examples: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
        texts: list[str] = []
        for raw_json in examples["text"]:
            dialog = json.loads(raw_json)["dialog"]
            for index, turn in enumerate(dialog):
                if turn["speaker"] != "sys":
                    continue

                messages = [
                    {
                        "role": "user",
                        "content": extract_context(dialog[max(0, index - context_window) : index]),
                    },
                    {
                        "role": "assistant",
                        "content": f"[{turn['strategy']}] {turn['text']}",
                    },
                ]
                texts.append(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )
        return {"text": texts}

    return format_train_batch


def prepare_train_dataset(
    dataset,
    tokenizer,
    *,
    split: str = "train",
    context_window: int = 4,
):
    """Expand each dialogue into one SFT example per supporter response."""
    return dataset[split].map(
        make_train_batch_formatter(tokenizer, context_window=context_window),
        batched=True,
    )

# Export ESConv train dataset for dot-skill
def export_context_split(
    dataset,
    output_path: str | Path,
    *,
    split: str = "train",
) -> dict[str, int | str]:
    """Export every dialogue in a split as numbered, human-readable text.

    The resulting TXT is suitable as direct source material for dot-skill. Dialogue
    boundaries are explicit so unrelated ESConv conversations are not interpreted as
    one continuous conversation.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dialogue_count = 0
    turn_count = 0

    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(f"# ESConv {split} contexts\n\n")
        for dialogue_count, example in enumerate(dataset[split], start=1):
            raw_text = example["text"]
            data = json.loads(raw_text) if isinstance(raw_text, str) else raw_text
            dialog = data.get("dialog", [])
            turn_count += len(dialog)

            output_file.write(f"## Dialogue {dialogue_count:06d}\n\n")
            output_file.write(extract_context(dialog))
            output_file.write("\n\n---\n\n")

    return {
        "output_path": str(path.resolve()),
        "dialogues": dialogue_count,
        "turns": turn_count,
        "bytes": path.stat().st_size,
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export local ESConv context artifacts for dot-skill."
    )
    parser.add_argument(
        "--output",
        default="data/esconv_train_contexts.txt",
        help="Destination TXT path (default: data/esconv_train_contexts.txt)",
    )
    parser.add_argument("--split", default="train", help="Dataset split to export")
    args = parser.parse_args()

    context_output = Path(args.output)

    if context_output.exists():
        print(f"Skill context output already exists; skipping: {context_output.resolve()}")
    else:
        dataset = load_esconv_dataset()
        summary = export_context_split(dataset, context_output, split=args.split)
        size_mib = summary["bytes"] / (1024 * 1024)
        print(f"Exported {summary['dialogues']} dialogues / {summary['turns']} turns")
        print(f"Output: {summary['output_path']}")
        print(f"Size: {summary['bytes']:,} bytes ({size_mib:.2f} MiB)")


if __name__ == "__main__":
    main()
