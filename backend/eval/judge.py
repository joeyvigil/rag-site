"""LLM-as-judge: grade an answer for correctness and faithfulness.

Known biases of LLM judges (worth knowing for interviews):
- Self-preference: models rate their own family's outputs higher. Use a judge
  from a different provider when you can (JUDGE_BASE_URL / JUDGE_MODEL).
- Verbosity bias: longer answers tend to get higher scores.
- Leniency or inconsistency: spot-check a sample of judgments by hand.
We reduce noise with a strict rubric, discrete labels instead of 1-10
scores, temperature 0, and JSON output.
"""

import json
import os
import re

from openai import AsyncOpenAI

from app.config import settings

JUDGE_PROMPT = """You are grading an AI assistant's answer to a question about a company's internal documents.

You are given:
- EARLIER CONVERSATION (optional): previous turns, for follow-up questions
- QUESTION: what the user asked
- REFERENCE: the correct answer, written by a human. "NOT IN DOCUMENTS" means the documents do not contain the answer.
- EXCERPTS: the document excerpts the assistant was shown
- ANSWER: the assistant's answer

Grade three things:

1. correctness (compare ANSWER to REFERENCE only):
   The REFERENCE is the ground truth. The EXCERPTS may be incomplete or wrong (retrieval can fail), so do NOT use them to judge correctness. If the answer says the information isn't available but the REFERENCE has an answer, that is "incorrect", even if the excerpts indeed lack it.
   Key facts are the facts needed to answer the QUESTION. Extra context in the reference (e.g. in parentheses) is optional.
   - "correct": contains all the key facts, and nothing that contradicts the reference. Extra correct detail is fine.
   - "partial": some key facts are right but others are missing or wrong.
   - "incorrect": wrong, missing the key facts, or refuses when the reference has an answer.
   If REFERENCE is "NOT IN DOCUMENTS": "correct" only if the answer says the information isn't available (and doesn't invent one); otherwise "incorrect".

2. refused: true if the answer says it could not find (some or all of) the requested information.

3. faithful (compare ANSWER to EXCERPTS, ignoring REFERENCE): true if every factual claim in the answer is supported by the excerpts (or restates the earlier conversation). Simple arithmetic on numbers from the excerpts counts as supported. A refusal is faithful. List any unsupported claims.

Respond with JSON only:
{"correctness": "correct" | "partial" | "incorrect", "refused": true | false, "faithful": true | false, "unsupported_claims": ["..."], "explanation": "one short sentence"}"""

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


class Judge:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.getenv("JUDGE_MODEL") or settings.llm_model
        self.client = AsyncOpenAI(
            base_url=os.getenv("JUDGE_BASE_URL") or settings.llm_base_url,
            api_key=os.getenv("JUDGE_API_KEY") or settings.effective_llm_api_key,
            max_retries=1,
            timeout=60,
        )

    async def grade(
        self, question: str, reference: str, excerpts: list[str], answer: str, *,
        answerable: bool, history: list[dict] | None = None,
    ) -> dict:
        excerpt_text = "\n\n".join(f"[{i}] {e}" for i, e in enumerate(excerpts, start=1)) or "(none)"
        convo = ""
        if history:  # follow-up questions only make sense with the earlier turns
            convo = "EARLIER CONVERSATION:\n" + "\n".join(f"{m['role']}: {m['content']}" for m in history) + "\n\n"
        user = (
            f"{convo}QUESTION:\n{question}\n\nREFERENCE:\n{reference}\n\n"
            f"EXCERPTS:\n{excerpt_text}\n\nANSWER:\n{answer}"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        verdict = _parse_json(resp.choices[0].message.content or "")
        correctness = str(verdict.get("correctness", "incorrect")).lower()
        if correctness not in CORRECTNESS_SCORE:
            correctness = "incorrect"
        refused = bool(verdict.get("refused", False))

        # Deterministic guard: never trust the judge for what code can check.
        # In the first baseline run the judge graded refusals on answerable
        # questions as "correct" (it compared to the excerpts, not the reference).
        overridden = False
        if answerable and refused and correctness == "correct":
            correctness, overridden = "incorrect", True

        return {
            "correctness": correctness,
            "correctness_score": CORRECTNESS_SCORE[correctness],
            "judge_overridden": overridden,
            "refused": refused,
            "faithful": bool(verdict.get("faithful", False)),
            "unsupported_claims": list(verdict.get("unsupported_claims") or []),
            "explanation": str(verdict.get("explanation", "")),
            "judge_model": self.model,
            "judge_usage": {
                "prompt_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
        }


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)  # tolerate ```json fences or chatter
        if match:
            return json.loads(match.group(0))
        raise
