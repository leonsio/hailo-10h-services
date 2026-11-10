"""Prompt-based structured generation with validation before publishing content."""

import json

from fastapi import HTTPException
from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for
from referencing import Registry
from referencing.exceptions import Unresolvable

from hailo_services.config import FRIGATE_ASSIST_MODEL, HA_ASSIST_MODEL


def schema_validator(schema):
    """Build a validator without permitting network or filesystem references.

    Args:
        schema (dict): Requested JSON Schema.

    Returns:
        Validator: Locally resolving JSON Schema validator.

    Raises:
        ValueError: Invalid schemas or external references are unsupported.
    """

    def check(node):
        """Reject external references in nested schema contents.

        Args:
            node (object): Schema subtree.

        Returns:
            None: Validates the subtree in place.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                if key in {"$ref", "$dynamicRef", "$recursiveRef"} and (
                    not isinstance(value, str) or not value.startswith("#")
                ):
                    raise ValueError("Only local JSON Schema references are supported")
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(schema)
    try:
        validator = validator_for(schema)
        validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError("Invalid JSON Schema") from exc
    return validator(schema, registry=Registry())


def prepare_structured_request(request):
    """Add output instructions to a request before backend token budgeting.

    Args:
        request (ChatRequest): Validated chat request.

    Returns:
        ChatRequest: Request carrying the structured output instructions.

    Raises:
        HTTPException: A virtual assistant was selected for structured generation.
    """
    fmt = request.response_format
    if fmt is None or fmt.type == "text":
        return request
    if request.model in {HA_ASSIST_MODEL, FRIGATE_ASSIST_MODEL}:
        raise HTTPException(
            400, "Structured output requires a direct chat model, e.g. gemma-4-E2B-it"
        )
    instruction = (
        "Return only valid JSON, without markdown fences or explanation. "
        "Follow the requested output format for your final answer. "
        "If tools are needed, call them first and return JSON after receiving their results. "
    )
    if fmt.type == "json_object":
        instruction += "The answer must be a JSON object."
    else:
        instruction += "The answer must match this JSON Schema: " + json.dumps(
            fmt.json_schema.schema_, ensure_ascii=False
        )
        if fmt.json_schema.description:
            instruction += "\nOutput description: " + fmt.json_schema.description
    request.messages = [
        {"role": "system", "content": instruction},
        *request.messages,
    ]
    return request


def validate_output(request, result):
    """Validate final JSON while preserving intermediate tool calls.

    Args:
        request (ChatRequest): Request specifying the desired response format.
        result (str | dict): Backend assistant output.

    Returns:
        str | dict: Validated output, or an intermediate tool call.

    Raises:
        HTTPException: Generated content is invalid JSON or violates the schema.
    """
    fmt = request.response_format
    if fmt is None or fmt.type == "text" or (isinstance(result, dict) and result.get("tool_calls")):
        return result
    content = result.get("content") if isinstance(result, dict) else result
    try:

        def invalid_constant(value):
            """Reject constants excluded by the JSON standard.

            Args:
                value (str): Non-finite numeric constant.

            Returns:
                None: Always raises ValueError.
            """
            raise ValueError("Non-finite JSON number")

        value = json.loads(content, parse_constant=invalid_constant)
        if fmt.type == "json_object":
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
        else:
            schema_validator(fmt.json_schema.schema_).validate(value)
    except (TypeError, ValueError, ValidationError, Unresolvable) as exc:
        raise HTTPException(
            502,
            detail={
                "code": "structured_output_invalid",
                "message": "Model output is not valid JSON matching the requested response format",
            },
        ) from exc
    return result
