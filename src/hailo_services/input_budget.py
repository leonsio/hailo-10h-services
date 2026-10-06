"""Trim whole old turns, retaining system messages and the active tool round."""


class InputBudgetError(ValueError):
    """Report that required input cannot fit without truncating the active turn."""

    def __init__(self, tokens, limit, requested, context, output):
        """Initialize InputBudgetError configuration and owned dependencies.

        Args:
            tokens (int): Conservative measured input-token count including safety reserves.
            limit (int): Upper bound for bytes, input tokens or retained enum values.
            requested (int): Requested maximum input-token count.
            context (Any): Native context manager or total model context-token capacity.
            output (Any): Native detector output or reserved output-token count.

        Returns:
            None: Creates the object without running inference.

        Notes:
            No application-specific exceptions are raised for valid inputs.
        """
        self.tokens = tokens
        self.limit = limit
        super().__init__(
            f"System prompt, tools and the current user/tool turn require {tokens} input tokens; "
            f"the effective input limit is {limit} (max_input_tokens={requested}, "
            f"context={context}, max_tokens={output}). Reduce the system prompt/exposed "
            "entities/tool schemas or the current request. These required contents were not truncated."
        )


def history_candidates(messages):
    """Yield progressively shorter histories without orphaning tool results.

    The caller validates the original tool history before using these candidates.
    Everything from the last user message onward is the active turn, including
    all tool calls/results and further tool iterations. Never delete any of it.
    System messages keep their original position relative to retained messages.
    Without a user boundary, retain everything rather than guessing dependencies.

    Args:
        messages (list[dict[str, Any]]): Ordered OpenAI or native conversation messages.

    Yields:
        Any: Result as described by the operation.

    Notes:
        No application-specific exceptions are raised for valid inputs.
    """
    yield messages
    for index, message in enumerate(messages):
        if message["role"] != "user":
            continue
        if any(item["role"] != "system" for item in messages[:index]):
            yield [
                item
                for position, item in enumerate(messages)
                if position >= index or item["role"] == "system"
            ]
