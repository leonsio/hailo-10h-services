"""Conservative slot repair, independently implemented; never rewrite action words."""

import re

from rapidfuzz.fuzz import ratio


def slot_repairs(text, values, *, threshold=90.0, margin=8.0):
    """Return at most one uniquely resolved spelling correction for a known slot.

    Matching is whole-span, uses actual request catalogue values and requires a
    runner-up margin. The caller must re-parse the entire result with HassIL.
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
                [(ratio(fragment.casefold(), value.casefold()), value) for value in values
                 if len(value.split()) == count],
                key=lambda item: (-item[0], item[1]),
            )
            if not candidates or candidates[0][0] == 100 or candidates[0][0] < threshold:
                continue
            second = candidates[1][0] if len(candidates) > 1 else 0
            if candidates[0][0] - second < margin:
                continue
            score, value = candidates[0]
            repairs.append({"input": fragment, "resolved": value, "score": score,
                            "runner_up": second, "text": text[:start] + value + text[end:]})
    # Multiple plausible replacements are deliberately deferred to the LLM.
    return repairs if len(repairs) == 1 else []
