# Python documentation and IDE support

Use PEP 257 docstrings with Google-style sections. Keep the first sentence
specific: describe what the operation does and the condition under which it
returns a fallback. For typed signatures, the type need not be repeated in the
parameter description. Dynamic native APIs should name the actual expected
contract instead of guessing a vendor class that may differ by SDK version.

```python
def json_object(value: object) -> dict[str, Any] | None:
    """Parse a JSON object without raising for invalid JSON.

    Args:
        value: A dictionary or JSON-encoded string.

    Returns:
        Original or decoded dictionary; None for invalid JSON or other types.

    Notes:
        Invalid input is represented by None rather than an exception.
    """
```

For functions that can fail, document the actionable condition for each relevant
exception, including errors propagated from the backend:

```python
def target_model(settings: Settings, request: ChatRequest) -> str:
    """Select the configured HA-Assist text or image backend.

    Args:
        settings: Enabled backends and virtual-model target configuration.
        request: Validated chat request; image parts require a VLM target.

    Returns:
        Enabled backend model identifier.

    Raises:
        ValueError: HA-Assist is disabled or the selected target is incompatible.
    """
```

Use `Yields` for generators/context managers. Document side effects such as
in-place schema changes, emitted chunks, metrics updates, model residency and
resource cleanup. Units belong in parameter descriptions: bytes, seconds,
milliseconds, tokens and image dimensions are not interchangeable.
`self` and `cls` are implicit; `*args` and `**kwargs` must describe forwarding or
named values. Examples are useful for complex contracts, not required on every
small predicate. Class docs use `Attributes` for settings, request fields and
value objects; constructor docs describe injected dependencies.

In PyCharm, select **Google** as the Python docstring format. In VS Code use the
project Python environment with Pylance, install the package editable and open
its repository root. Hover, signature help and go-to-definition then show source
docstrings and the explicit chat/schema types. No generated documentation
package or Hailo hardware is needed for source browsing.

```bash
python -m pip install -e '.[test]'
python -m ruff check .
python -m ruff format --check src tests
python -m pytest -q
node --test tests/test_web.cjs
```

Ruff enforces production docstring presence, layout and parameter documentation.
The Python tests also check that all production definitions include a return or
yield contract. Existing native inference tests use controlled fake bindings;
real accelerator execution must additionally be checked on a Hailo host.
