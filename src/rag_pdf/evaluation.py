"""Small, transparent evaluation metrics for reference-answer comparisons."""

from __future__ import annotations

import re
from collections import Counter


def token_f1(reference: str, answer: str) -> float:
    """Return lexical token F1 against a reference answer, from zero to one."""

    expected = re.findall(r"\w+", reference.casefold())
    answer_without_citations = re.sub(r"\[[^\]]*\]", " ", answer)
    actual = re.findall(r"\w+", answer_without_citations.casefold())
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = sum((Counter(expected) & Counter(actual)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


_ABSTENTION_PHRASES = (
    "i don't know",
    "i do not know",
    "i cannot find",
    "i could not find",
    "not found in the selected documents",
    "not provided in the documents",
    "not available in the documents",
    "the documents do not contain",
    "no relevant information",
    "there is no information",
)


def is_abstention(answer: str) -> bool:
    """Detect common explicit statements that the answer is unknown or absent."""

    normalized = " ".join(answer.casefold().split())
    return any(phrase in normalized for phrase in _ABSTENTION_PHRASES)


def score_answer(
    answer: str,
    *,
    expected_answer: str = "",
    should_abstain: bool = False,
    f1_threshold: float = 0.7,
) -> dict[str, object]:
    """Score answerability behavior and lexical overlap with a reference answer."""

    abstained = is_abstention(answer)
    if should_abstain:
        return {
            "abstained": abstained,
            "token_f1": None,
            "outcome_correct": abstained,
            "outcome_label": "Correct abstention" if abstained else "Unsupported answer",
        }

    score = token_f1(expected_answer, answer)
    passed = not abstained and score >= f1_threshold
    return {
        "abstained": abstained,
        "token_f1": score,
        "outcome_correct": passed,
        "outcome_label": (
            "Answer passed F1 threshold"
            if passed
            else "Incorrect abstention"
            if abstained
            else "Below F1 threshold"
        ),
    }
