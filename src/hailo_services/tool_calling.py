"""OpenAI/LiteRT tool translation. Actions are executed by the client, never here."""

import json
import re
import uuid

from jsonschema import Draft202012Validator, SchemaError, ValidationError


def arguments_object(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise ValueError("Tool arguments must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be a JSON object")
    return value


def validate_tools(tools):
    names = set()
    for tool in tools:
        function = tool.get("function")
        if tool.get("type") != "function" or not isinstance(function, dict):
            raise ValueError("Only function tools are supported")
        name = function.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            raise ValueError("Invalid function name")
        if name in names:
            raise ValueError("Duplicate function name")
        names.add(name)
        schema = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(schema, dict):
            raise ValueError("Function parameters must be a JSON schema object")
        # Never retrieve remote schemas supplied by a caller/model.
        def check_refs(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"$ref", "$dynamicRef"} and (not isinstance(value, str) or not value.startswith("#")):
                        raise ValueError("Only local JSON schema references are supported")
                    check_refs(value)
            elif isinstance(node, list):
                for value in node:
                    check_refs(value)
        check_refs(schema)
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError("Invalid function parameter schema") from exc


def validate_history_calls(calls):
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be a list")
    ids = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("Invalid historical tool call")
        identifier, function = call.get("id"), call.get("function")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("Tool calls require unique non-empty ids")
        ids.add(identifier)
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("Tool calls require a function name")
        arguments_object(function.get("arguments"))


def selected_tools(request):
    tools = request.tools or []
    if request.tool_choice == "none":
        return []
    if request.tool_choice == "required" and not tools:
        raise ValueError("tool_choice=required requires tools")
    if isinstance(request.tool_choice, dict):
        name = request.tool_choice["function"]["name"]
        tools = [tool for tool in tools if tool["function"]["name"] == name]
        if not tools:
            raise ValueError("tool_choice names an unavailable function")
    return tools


def native_tools(litert_lm, tools):
    class ClientTool(litert_lm.Tool):
        def __init__(self, description):
            self.description = description

        def get_tool_description(self):
            return self.description

        def execute(self, param):
            raise RuntimeError("Home Assistant tools must be executed by Home Assistant")

    return [ClientTool(tool) for tool in tools]


def native_messages(messages):
    """Translate history and match every tool result to its pending call ID."""
    converted, pending = [], {}
    for message in messages:
        role, content = message["role"], message.get("content")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("Gemma accepts text only")
            content = "\n".join(part["text"] for part in content)
        if role == "tool":
            identifier = message["tool_call_id"]
            if identifier not in pending:
                raise ValueError("Tool result does not match a pending tool_call_id")
            part = {"type": "tool_response", "name": pending.pop(identifier), "response": content}
            if converted and converted[-1]["role"] == "tool":
                converted[-1]["content"].append(part)
            else:
                converted.append({"role": "tool", "content": [part]})
            continue
        if pending:
            raise ValueError("Missing results for previous tool calls")
        value = {"role": role, "content": content or ""}
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call["function"]
                pending[call["id"]] = function["name"]
                calls.append({"id": call["id"], "type": "function", "function": {
                    "name": function["name"], "arguments": arguments_object(function["arguments"]),
                }})
            value["tool_calls"] = calls
        converted.append(value)
    if pending:
        raise ValueError("Missing results for previous tool calls")
    if not converted or converted[-1]["role"] not in {"user", "tool"}:
        raise ValueError("The final Gemma message must be from the user or a tool")
    return converted


def response_message(response, request, text):
    """Validate generated function names and arguments before returning actions."""
    if hasattr(response, "to_json"):
        response = response.to_json()
    calls = response.get("tool_calls", []) if isinstance(response, dict) else []
    tools = {tool["function"]["name"]: tool["function"] for tool in selected_tools(request)}
    if not isinstance(calls, list):
        raise ValueError("Model returned invalid tool_calls")
    if not request.parallel_tool_calls and len(calls) > 1:
        raise ValueError("Model returned parallel calls when disabled")
    normalized = []
    for call in calls:
        function = call.get("function", {}) if isinstance(call, dict) else {}
        if not isinstance(function, dict):
            raise ValueError("Model returned an invalid function call")
        name = function.get("name")
        if not isinstance(name, str) or name not in tools:
            raise ValueError("Model requested an unavailable function")
        arguments = arguments_object(function.get("arguments"))
        try:
            Draft202012Validator(tools[name].get("parameters", {})).validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"Model returned invalid arguments for {name}: {exc.message}") from exc
        normalized.append({"id": "call_" + uuid.uuid4().hex, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
        }})
    if not normalized:
        if request.tool_choice == "required" or isinstance(request.tool_choice, dict):
            raise ValueError("Model did not return the required tool call")
        return text
    return {"role": "assistant", "content": text or None, "tool_calls": normalized}


def has_tool_context(request):
    return bool(request.tools) or request.tool_choice is not None or any(
        message.get("role") == "tool" or message.get("tool_calls") for message in request.messages
    )
