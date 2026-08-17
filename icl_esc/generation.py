"""Model inference wrappers used by the notebook."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .prompts import PromptPack


def _extract_cot_response(output: str, method_name: str) -> str:
    """Return the last non-empty line from a free-text CoT generation."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{method_name} returned an empty output")
    return lines[-1]


@dataclass(frozen=True)
class RefineCoTStep:
    """Parsed response, explicit reasoning, and raw output from one refinement."""

    response: str
    reasoning: str
    raw_output: str


@dataclass(frozen=True)
class RefineCoTResult:
    """Optional trace of the supplied ESC-CoT response and all refinements."""

    response: str
    initial_response: str
    refinements: tuple[RefineCoTStep, ...]


def _parse_refinement_output(output: str) -> RefineCoTStep:
    """Parse the JSON emitted by one Refine-CoT iteration."""
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        output.strip(),
        flags=re.IGNORECASE,
    )
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Refine-CoT output does not contain a JSON object")
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError("Refine-CoT returned invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Refine-CoT JSON output must be an object")

    fields = {str(key).casefold(): item for key, item in value.items()}
    response = fields.get("response")
    reasoning = fields.get("reasoning")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Refine-CoT JSON output requires a non-empty Response")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("Refine-CoT JSON output requires a non-empty Reasoning")
    return RefineCoTStep(
        response=response.strip(),
        reasoning=reasoning.strip(),
        raw_output=output.strip(),
    )


@dataclass
class LlamaGenerator:
    """Bind a model and tokenizer once, then expose named prompt strategies."""

    model: object
    tokenizer: object
    device: str = "cuda"

    def _generate(
        self,
        messages,
        *,
        max_new_tokens: int = 800,
        temperature: float | None = 0.7,
        top_p: float | None = 0.9,
        return_dict: bool = False,
        do_sample: bool = True,
    ) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=return_dict,
        ).to(self.device)

        generation_args = {
            "max_new_tokens": max_new_tokens,
            "use_cache": True,
            "do_sample": do_sample,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample and temperature is not None:
            generation_args["temperature"] = temperature
        if do_sample and top_p is not None:
            generation_args["top_p"] = top_p

        if return_dict:
            outputs = self.model.generate(**inputs, **generation_args)
            prompt_length = inputs["input_ids"].shape[1]
        else:
            outputs = self.model.generate(input_ids=inputs, **generation_args)
            prompt_length = inputs.shape[1]
        return self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    def generate_messages(self, messages, **generation_args) -> str:
        """Public low-level entry point for orchestrators such as E-MAS."""
        return self._generate(messages, **generation_args)

    def vanilla(self, context: str) -> str:
        return self._generate(PromptPack.vanilla(context))

    def ecot(self, context: str, *, with_cot: bool = False):
        full_response = self._generate(PromptPack.ecot(context))
        response = _extract_cot_response(full_response, "ECoT")
        return (response, full_response) if with_cot else response

    def fine_tuned(self, context: str) -> str:
        return self._generate(
            PromptPack.fine_tuned(context),
            return_dict=True,
        )

    def judge(self, context: str, response: str) -> str:
        return self._generate(
            PromptPack.judge(context, response),
            temperature=None,
            top_p=None,
            do_sample=False,
        )

    def esccot(self, context: str, *, with_cot: bool = False):
        full_response = self._generate(PromptPack.esccot(context))
        response = _extract_cot_response(full_response, "ESC-CoT")
        return (response, full_response) if with_cot else response

    def skds(self, context: str) -> str:
        return self._generate(PromptPack.skds(context)).strip()

    def refine(
        self,
        context: str,
        esccot_response: str,
        *,
        steps: int = 5,
        return_trace: bool = False,
    ) -> str | RefineCoTResult:
        """Iteratively refine a previously generated ESC-CoT response."""
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if not isinstance(esccot_response, str) or not esccot_response.strip():
            raise ValueError("esccot_response must be a non-empty string")

        initial_response = esccot_response.strip()
        current_response = initial_response
        refinements = []

        for _ in range(steps):
            raw_output = self._generate(
                PromptPack.refine(context, current_response),
                max_new_tokens=400,
            ).strip()
            refinement = _parse_refinement_output(raw_output)
            current_response = refinement.response
            refinements.append(refinement)

        if return_trace:
            return RefineCoTResult(
                response=current_response,
                initial_response=initial_response,
                refinements=tuple(refinements),
            )
        return current_response


@dataclass
class BlenderGenerator:
    model: object
    tokenizer: object
    max_input_tokens: int = 128

    def __call__(self, context: str) -> str:
        """Generate from the pre-formatted BlenderBot context string."""
        import torch

        input_sequence = context + self.tokenizer.eos_token
        token_ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(input_sequence)
        )[-self.max_input_tokens :]
        try:
            device = next(self.model.parameters()).device
        except (AttributeError, StopIteration):
            device = torch.device("cpu")
        input_ids = torch.LongTensor([token_ids]).to(device)

        # BlenderBot is a regular Transformers model. Its inference path is
        # model.eval() plus torch.inference_mode(), not Unsloth's Llama helper.
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                num_beams=1,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                # Disable cache to avoid Transformers-version incompatibilities.
                use_cache=False,
            )
        response = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return " ".join(response.strip().split())
