"""A compact E-MAS orchestration pipeline for emotional-support responses.

E-MAS uses one language model as seven logical agents. Three independent analysis
roles inspect the same dialogue, three identically instructed strategy agents produce
candidate responses, and one integration agent refines the final response. All agents
share one sampling configuration. There is no debate, voting, AutoGen dependency, or
agent-to-agent chat loop.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from .prompts import EMASPromptPack


@dataclass(frozen=True)
class EMASConfig:
    """Inference budget and sampling settings for E-MAS."""

    strategy_agent_count: int = 3
    analysis_max_new_tokens: int = 400
    candidate_max_new_tokens: int = 400
    integration_max_new_tokens: int = 400
    temperature: float = 0.7
    top_p: float = 0.9
    # soft limit on the number of words in the final response, aligned with ECoT
    max_response_words: int = 30

    def __post_init__(self) -> None:
        if self.strategy_agent_count != 3:
            raise ValueError("E-MAS requires exactly three strategy agents")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.max_response_words < 1:
            raise ValueError("max_response_words must be at least 1")


@dataclass(frozen=True)
class EMASResult:
    """Final response plus inspectable intermediate agent outputs."""

    response: str
    analyses: dict[str, str]
    candidates: list[str]
    integration: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _json_object(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _extract_response(integration: str) -> str:
    parsed = _json_object(integration)
    if parsed and isinstance(parsed.get("response"), str):
        return parsed["response"].strip()

    response = integration.strip().strip("`")
    response = re.sub(r"^\s*(?:final\s+)?response\s*:\s*", "", response, flags=re.IGNORECASE)
    response = re.sub(r"^\[[^\]]+\]\s*", "", response)
    return response.strip()


@dataclass
class EMASGenerator:
    """Coordinate E-MAS agents through an existing ``LlamaGenerator`` backend."""

    backend: Any
    config: EMASConfig = EMASConfig()

    def _run_agent(self, messages, max_new_tokens: int) -> str:
        return self.backend.generate_messages(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
        )

    def generate(self, context: str, *, return_trace: bool = False):
        if not context or not context.strip():
            raise ValueError("context must not be empty")

        # These agents are logically independent: each sees only the original context.
        analyses = {
            name: self._run_agent(
                EMASPromptPack.analysis(name, context),
                self.config.analysis_max_new_tokens,
            )
            for name in EMASPromptPack.ANALYSIS_ROLES
        }

        shared_candidate_messages = EMASPromptPack.strategy(
            context,
            analyses,
            self.config.max_response_words,
        )
        candidates = [
            self._run_agent(
                shared_candidate_messages,
                self.config.candidate_max_new_tokens,
            )
            for _ in range(self.config.strategy_agent_count)
        ]

        integration = self._run_agent(
            EMASPromptPack.integration(
                context,
                analyses,
                candidates,
                self.config.max_response_words,
            ),
            self.config.integration_max_new_tokens,
        )
        result = EMASResult(
            response=_extract_response(integration),
            analyses=analyses,
            candidates=candidates,
            integration=integration,
        )
        return result if return_trace else result.response

    def __call__(self, context: str) -> str:
        return self.generate(context)
