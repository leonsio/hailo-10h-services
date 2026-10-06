"""Conservative slot repair, independently implemented; never rewrite action words."""

import re

from rapidfuzz.fuzz import ratio


def slot_rankings(text, values, *, threshold=90.0, margin=8.0):
    """Rank whole spans against actual catalogue slots, retaining ambiguous hits.

    Scores measure character similarity, not calibrated probabilities. Threshold
    and runner-up margin control when a correction can be used without inference.

    Args:
        text (str): Text to parse, normalize, match or render.
        values (Iterable[str]): Actual catalogue names and areas eligible for slot repair.
        threshold (float): Minimum fuzzy score or detection confidence for acceptance.
        margin (float): Required distance between the best and runner-up fuzzy score.

    Returns:
        list[dict[str, Any]]: Span repairs with scores, runner-up candidates and ambiguity flags.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    words = list(re.finditer(r"\w+", text, re.UNICODE))
    values = sorted(set(value for value in values if value))
    repairs = []
    for count in sorted({len(value.split()) for value in values}):
        for index in range(len(words) - count + 1):
            start, end = words[index].start(), words[index + count - 1].end()
            fragment = text[start:end]
            if len(fragment) < 5:
                continue
            candidates = sorted(
                [
                    (ratio(fragment.casefold(), value.casefold()), value)
                    for value in values
                    if len(value.split()) == count
                ],
                key=lambda item: (-item[0], item[1]),
            )
            if not candidates or candidates[0][0] == 100 or candidates[0][0] < threshold:
                continue
            second = candidates[1][0] if len(candidates) > 1 else 0
            score, value = candidates[0]
            repairs.append(
                {
                    "input": fragment,
                    "resolved": value,
                    "score": score,
                    "runner_up": second,
                    "text": text[:start] + value + text[end:],
                    "candidates": [
                        {"value": v, "score": s}
                        for s, v in candidates
                        if candidates[0][0] - s < max(margin, 0.000001)
                    ],
                    "ambiguous": candidates[0][0] == second or candidates[0][0] - second < margin,
                }
            )
    return repairs


def slot_repairs(text, values, *, threshold=90.0, margin=8.0):
    """Repair only one uniquely ranked catalogue span; defer ties to inference.

    Args:
        text (str): Text to parse, normalize, match or render.
        values (Iterable[str]): Actual catalogue names and areas eligible for slot repair.
        threshold (float): Minimum fuzzy score or detection confidence for acceptance.
        margin (float): Required distance between the best and runner-up fuzzy score.

    Returns:
        list[dict[str, Any]]: Single safe repair, or an empty list when absent/ambiguous.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    repairs = slot_rankings(text, values, threshold=threshold, margin=margin)
    return repairs if len(repairs) == 1 and not repairs[0]["ambiguous"] else []
