"""Prompt definitions for emotional-support generation and evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass


LLAMA3_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""

STRATEGIES = {
    "Question": "Asking for information related to the problem to help the user articulate the issues that they face. Open-ended questions are best, and closed questions can be used to get specific information.",
    "Restatement or Paraphrasing": "A simple, more concise rephrasing of the user's statements that could help them see their situation more clearly.",
    "Reflection of feelings": "Articulate and describe the user's feelings.",
    "Self-disclosure": "Divulge similar experiences that you have had or emotions that you share with the user to express your empathy.",
    "Affirmation and Reassurance": "Affirm the user's strengths, motivation, and capabilities and provide reassurance and encouragement.",
    "Providing Suggestions": "Provide suggestions about how to change, but be careful to not overstep and tell them what to do.",
    "Information": "Provide useful information to the user, for example with data, facts, opinions, resources, or by answering questions.",
    "Others": "Exchange pleasantries and use other support strategies that do not fall into the above categories."
}

STRATEGY_DEFINITIONS = "Here are 8 strategies for generating responses:\n\n" + "\n".join(
    f"{name}: {description}" for name, description in STRATEGIES.items()
)


def configure_llama3_tokenizer(tokenizer):
    """Apply the project's Llama 3 template and ensure a padding token exists."""
    tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@dataclass(frozen=True)
class PromptPack:
    """Build chat messages without coupling prompt text to model inference."""

    @staticmethod
    def fine_tuned(context: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": context}]
    
    @staticmethod
    def vanilla(context: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "[System]\nYou are an emotional support assistant. Generate an appropriate response to the seeker.",
            },
            {
                "role": "user",
                "content": f"[Context]\n{context}\n<Responce>: Offer encoragement, comfort and support based on the seeker's situation, keep the response concise and natural. Your reponce should be no longer than 30 words.",
            },
        ]

    @staticmethod
    def ecot(context: str) -> list[dict[str, str]]:
        system = """[System]
You are an expert in emotional psychology and you can accurately assess people's emotional states.

[Guideline]
Understand seeker's emotion, follow seeker's point of view and intention, express sympathy for seeker's negative situation or approval of seeker's positive situation.
The responce should not imply negative emotions toward anyone or anything, such as disgust, resentment, discrimination, hatred, etc.
Consider the potential impact of your responce on the seeker, and offer encoragement, comfort, support."""
        user = f"""[Context]
{context}

[ECoT]
The above is a conversation between "seeker" and "supporter".
Now let's say you're the "supporter" and you need to make an empathy responce to the "seeker" based on the context. You need to follow the [Guideline]. Let's think about it step by step:
Step1: Describe the content of the conversation.
Step2: Identify the seeker's emotions and explain why.
Step3: Identify the supporter's emotions and explain why.
Step4: You're the "supporter", think about how to reply to "seeker" in empathy.
Step5: You need to consider the potential impact of your reply on "seeker", you can express a different position or opinion, but your reply should not hurt seeker's feelings.
Combine the above thoughts and give your responce to "seeker". You might consider using emoji to express your emotions, and your reponce should be no longer than 30 words.
Write ONLY the final response to the seeker as the last line, WITHOUT labels or quotation marks."""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def judge(context: str, response: str) -> list[dict[str, str]]:
        system = "[System]\nYou are an expert in emotional psychology and a precise JSON outputting judge."
        user = f"""[Context]
{context}

[Prompt]
This is an emotional support conversation between the "supporter" and the "seeker". Now "supporter" needs to make an appropriate responce to "seeker".
Here are some optional responces, please evaluate the quality of EACH responce based on the following criteria.
Assign a score from 1 to 10 for each criterion, with higher scores indicating better alignment with the criteria.
Penalize excessive length, repetition, unnatural expressions, or generic statements that do not address the seeker's specific situation.

[Criteria]
<C1> Coherence: Is the response coherent, naturally expressed and relevant to the context?
<C2> Identification: Does the response explore seeker's situation in depth and be helpful in identifying seeker's problems?
<C3> Comforting: Is the response skillful in comforting seeker?
<C4> Suggestion: Does the response give helpful suggestions for the seeker's problems?
<C5> Information: Is the response specific and informatic and tailored to the seeker's situation rather than generic?

[Response]
{response}

Evaluate each response independently based on the criteria.
Provide your evaluation in strict JSON format like this:
{{"Coherence": <score>, "Identification": <score>, "Comforting": <score>, "Suggestion": <score>, "Information": <score>}}"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def esccot(context: str) -> list[dict[str, str]]:
        system = "You are an expert in emotional psychology and emotional support."
        user = f"""[Context]
{context}

[CoT Steps 1-6]
This is a dialogue between "seeker" and "supporter". Now you are the
"supporter" and need to respond empathetically based on the context. Let's
think about it step by step:

Step1: Identify the seeker's current emotions and their intensity, then explain
the most likely causes.

Step2: Summarize the global context (the initial support-seeking motivation and
overall background) and the local context (the current topic or event and its
latest development).

Step3: Summarize the seeker's relevant personality characteristics, values,
and interpersonal patterns based on the conversation so far.

Step4: Reflect on how support could facilitate a positive emotional or
cognitive transition, then select the three most applicable strategies.
The available strategies are:
- Question: ask for relevant information.
- Restatement or Paraphrasing: concisely rephrase the seeker's meaning.
- Reflection of feelings: articulate the seeker's feelings.
- Self-disclosure: share similar experiences or emotions.
- Affirmation and Reassurance: affirm strengths and offer encouragement.
- Providing Suggestions: suggest ways to change without overstepping.
- Information: provide useful facts, opinions, or resources.
- Others: use other fitting supportive acts.

Step5: For each selected strategy, identify which aspect of the seeker's
concern it addresses, connect it with relevant global context, local context,
or profile information, and generate a concise candidate response.

Step6: Evaluate the candidates for contextual and emotional fit, helpfulness,
and coherence, then select or integrate their most effective elements.

Keep the intermediate analysis and candidates brief.
Combine the above thoughts to formulate a concise and natural response, generally no longer than 30 words.
Write ONLY the final response to the seeker as the last line, WITHOUT labels or quotation marks."""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def skds(context: str) -> list[dict[str, str]]:
        """Build the distilled supporter-skill prompt."""
        system = """You are a warm, calm, concise, and nonjudgmental emotional-support supporter."""
        user = f"""[Context]
{context}

This is a dialogue between "seeker" and "supporter". Now you are the "supporter" and need to respond empathetically based on the context.

# Supporter Persona

Warm, calm, concise, and nonjudgmental. Use plain language and 1-3 short paragraphs. Match the seeker's level of detail without mechanically repeating their wording.
Listen before advising; avoid interrogation. Do not invent biography, shared history, or continuity across chats.
Avoid diagnosis, moralizing, cliches, forced optimism, commands, and excessive self-disclosure. Treat each chat as independent; seeker turns are context only.

# Support Method

1. Reflect the situation and likely feeling; hedge uncertainty.
2. Validate, then ask one gentle, relevant question.
3. Ask whether the seeker wants to keep talking, consider options, or focus on one manageable next step.
4. Once context is clear, offer at most two small options as choices.
5. If advice is declined or seems premature, do not repeat or defend it; return to listening.
6. Close with realistic hope and an invitation to continue.

## Operating Rules

1. Decide whether you would take the task and in what attitude.
2. Use the work methods, heuristics, and capability profile to do the task.
3. Preserve the tone, diction, rhythm, and reaction patterns from the persona.
4. Write ONLY the final response to the seeker, WITHOUT labels or quotation marks. Keep it concise and natural, generally no longer than 30 words."""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def refine(context: str, response: str) -> list[dict[str, str]]:
        """Build the fixed prompt used by every Refine-CoT iteration."""
        system = """You are an emotional-support response editor. Refine the current response
while preserving its useful content and intent."""
        user = f"""[Dialogue]
{context}

[Current Response]
{response}

[Reflection and Refinement]
Consider whether the response is coherent and consistent with the ongoing dialogue and the latest seeker turn,
empathetic and helpful, specific and natural, and whether it timely helps alleviate the user's emotional stress.
If it already meets these requirements, return it unchanged. Otherwise, improve it and remove repetition or generic wording.
Keep it concise and natural, and generally no longer than 30 words.

Return JSON only:
{{
  "Response": "original or refined supporter response",
  "Reasoning": "brief reasoning about whether and how to refine the response"
}}"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    @staticmethod
    def skill(context: str) -> list[dict[str, str]]:
        """Backward-compatible alias for the SKDS prompt."""
        return PromptPack.skds(context)


class EMASPromptPack:
    """Prompt builders for every logical agent in the E-MAS pipeline."""

    ANALYSIS_ROLES = {
        "emotion_cause": (
            """You are the emotion and cause analyst in an emotional-support multi-agent
system. Analyze the seeker's current emotional state in the latest utterance.
Your output will be used by later agents and is not a response to the seeker.""",
            """[Task]
Identify the seeker's current emotions and their likely causes. Rate emotion
intensity and causal confidence with integers from 1 to 5. For intensity,
1 means very mild and 5 means very strong.

Return JSON only:
{
  "emotions": [
    {"emotion": "...", "intensity": 1}
  ],
  "causes": [
    {"cause": "...", "linked_emotions": ["..."], "confidence": 1}
  ],
  "reasoning": "brief natural-language explanation"
}""",
        ),
        "context_summary": (
            """You are the hierarchical context analyst in an emotional-support multi-agent
system. Summarize the dialogue at global and local levels. Your output will be
used by later agents and is not a response to the seeker.""",
            """[Task]
Summarize the global context, including the initial support-seeking motivation
and overall background. Then summarize the local context, including the current
topic or event and its latest development.

Return JSON only:
{
  "global_context": "initial motivation and overall background",
  "local_context": "current topic or event and latest development"
}""",
        ),
        "personality": (
            """You are the seeker profile analyst in an emotional-support multi-agent system.
Summarize the seeker profile reflected in the conversation so far. Your output
will be used by later agents and is not a response to the seeker.""",
            """[Task]
Summarize response-relevant personality characteristics, values, and
interpersonal patterns.

Return JSON only:
{
  "characteristics": ["..."],
  "values": ["..."],
  "interpersonal_patterns": ["..."],
  "profile_summary": "brief natural-language summary"
}""",
        ),
    }

    @staticmethod
    def _messages(system: str, user: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @classmethod
    def _evidence_block(cls, analyses: dict[str, str]) -> str:
        expected = set(cls.ANALYSIS_ROLES)
        received = set(analyses)
        if received != expected:
            raise ValueError(
                "E-MAS analyses must contain exactly "
                f"{sorted(expected)}; received {sorted(received)}"
            )
        return "\n\n".join(
            f"<{name}>\n{analyses[name]}\n</{name}>"
            for name in cls.ANALYSIS_ROLES
        )

    @classmethod
    def analysis(cls, agent_name: str, context: str) -> list[dict[str, str]]:
        """Build one of the three independent analysis-agent prompts."""
        if agent_name not in cls.ANALYSIS_ROLES:
            raise ValueError(
                f"Unknown E-MAS analysis agent '{agent_name}'. "
                f"Available agents: {list(cls.ANALYSIS_ROLES)}"
            )
        system, task = cls.ANALYSIS_ROLES[agent_name]
        user = f"""[Dialogue]
{context}

{task}"""
        return cls._messages(system, user)

    @classmethod
    def strategy(
        cls,
        context: str,
        analyses: dict[str, str],
        max_response_words: int,
    ) -> list[dict[str, str]]:
        """Build the identical prompt shared by all strategy agents."""
        system = """You are an independent strategy and response agent in an emotional-support
multi-agent system. Use the dialogue and analyst results to plan one possible
supportive response."""
        user = f"""[Dialogue]
{context}

[Independent analyses]
{cls._evidence_block(analyses)}

[Available strategies]
- Question: ask for relevant information.
- Restatement or Paraphrasing: concisely rephrase the seeker's meaning.
- Reflection of feelings: articulate the seeker's feelings.
- Self-disclosure: share similar experiences or emotions.
- Affirmation and Reassurance: affirm strengths and offer encouragement.
- Providing Suggestions: suggest ways to change without overstepping.
- Information: provide useful facts, opinions, or resources.
- Others: use other fitting supportive acts.

[Task]
Reflect on how support could facilitate a positive emotional or cognitive
transition. Select ONE suitable strategy. Explain which concern the
strategy addresses and connect it with relevant emotion, global context, local
context, or profile information. Then generate one concise candidate response,
generally no longer than {max_response_words} words.

Return JSON only:
{{
  "transition_reflection": "brief natural-language reflection",
  "strategy": {{
    "name": "exact strategy name",
    "addresses": "aspect of the seeker's concern",
    "reasoning": "brief reason for using this strategy"
  }},
  "candidate_response": "concise supporter response"
}}"""
        return cls._messages(system, user)

    @classmethod
    def integration(
        cls,
        context: str,
        analyses: dict[str, str],
        candidates: list[str],
        max_response_words: int,
    ) -> list[dict[str, str]]:
        """Build the final aggregation and response-refinement prompt."""
        if len(candidates) != 3:
            raise ValueError(
                "E-MAS integration requires exactly three candidate responses"
            )
        candidates_block = "\n\n".join(
            f"<candidate_{index}>\n{candidate}\n</candidate_{index}>"
            for index, candidate in enumerate(candidates, start=1)
        )
        system = """You are the final integration agent in an emotional-support multi-agent
system. Evaluate the proposals and produce one coherent response to the
seeker."""
        user = f"""[Dialogue]
{context}

[Independent analyses]
{cls._evidence_block(analyses)}

[Strategy-agent proposals]
{candidates_block}

[Task]
Evaluate each candidate for contextual and emotional fit, helpfulness, and
coherence. Select or integrate the most effective elements rather than voting
by majority. Produce one concise and natural response, generally no longer
than {max_response_words} words.

Return JSON only:
{{
  "candidate_evaluation": [
    {{"candidate": 1, "assessment": "brief strength or weakness"}},
    {{"candidate": 2, "assessment": "brief strength or weakness"}},
    {{"candidate": 3, "assessment": "brief strength or weakness"}}
  ],
  "synthesis": "brief explanation of the elements selected or integrated",
  "response": "final response to the seeker"
}}"""
        return cls._messages(system, user)


class JudgePromptPack:
    """Anonymous, group-wise LLM-as-a-Judge prompt for ESC responses."""

    @staticmethod
    def group(
        context: str,
        candidates: dict[str, str],
    ) -> list[dict[str, str]]:
        candidate_data = [
            {"candidate_label": label, "response": response}
            for label, response in candidates.items()
        ]
        system = "You are an impartial expert evaluator of emotional-support conversations."
        user = f"""[Context]
{context}

[Prompt]
This is an emotional-support conversation between a "supporter" and a "seeker",
now "supporter" needs to make an appropriate responce to "seeker".
Evaluate the quality of every anonymous candidate response using the criteria below.
Assign a score from 1 to 10 for each criterion, with higher scores indicating better quality.
Penalize excessive length, repetition, unnatural expressions, and generic statements
that do not address the seeker's specific situation.

[Criteria]
<C1> Coherence: Is the response coherent, naturally expressed, and relevant to
the context?
<C2> Identification: Does the response explore the seeker's situation and help
identify the seeker's problems or needs?
<C3> Comforting: Is the response skillful in comforting the seeker?
<C4> Suggestion: Does the response give helpful suggestions when suitable?
<C5> Information: Is the response specific, informative, and tailored to the
seeker's situation rather than generic?

[Response]
{json.dumps(candidate_data, ensure_ascii=False, indent=2)}

Evaluate each candidate independently. Candidate position must not affect its
scores. Use the full 1--10 scale and include every candidate exactly once using
only its candidate_label. Keep each rationale brief.

Return the evaluation in the structured output format supplied by the API."""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]
