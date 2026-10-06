"""Shared speech language selection helpers independent of a model runtime."""


def language_matches(expected, actual):
    """Compare language codes with optional regional specificity.

    Args:
        expected (str): Requested or configured language.
        actual (str): Language declared by the voice metadata.

    Returns:
        bool: Whether language and any requested region match.
    """
    expected, actual = expected.replace("-", "_").lower(), actual.replace("-", "_").lower()
    return expected == actual or ("_" not in expected and expected == actual.split("_")[0])
