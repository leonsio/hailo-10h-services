"""OpenAI/LiteRT tool translation. Actions are executed by the client, never here."""

import json
import logging
import re
import uuid

from jsonschema import Draft202012Validator, SchemaError, ValidationError

_LOG = logging.getLogger(__name__)


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


def _normalized_text(value):
    text = str(value).casefold().replace("ß", "ss")
    text = text.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _latest_user_text(messages):
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                part.get("text", "") for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return ""
    return ""


def _relevant_entities(messages):
    entities = []
    for message in messages:
        if message.get("role") != "system" or not isinstance(message.get("content"), str):
            continue
        blocks = re.findall(
            r"^- names:\s*(.+?)\n\s+domain:\s*(.+?)(?:\n\s+areas:\s*(.+?))?(?=\n- names:|\n\n|\Z)",
            message["content"],
            flags=re.MULTILINE | re.DOTALL,
        )
        for name, domain, area in blocks:
            entities.append({
                "name": name.strip(),
                "domain": domain.strip(),
                "area": area.strip() if area else "",
            })
    return entities


def _prefer_area_target(arguments, schema, messages):
    """Convert an accidental single-device choice into an area/domain target.

    This is intentionally narrow: the model must already have selected a name
    and a domain, the user must mention an area, the selected name must not be
    explicitly present in the user text, and at least two relevant entities of
    that domain must exist in the mentioned area.
    """
    if not isinstance(arguments, dict) or not isinstance(arguments.get("name"), str):
        return arguments
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    if "area" not in properties or "domain" not in properties:
        return arguments
    domains = arguments.get("domain")
    if isinstance(domains, str):
        domains = [domains]
    if not isinstance(domains, list) or len(domains) != 1 or not isinstance(domains[0], str):
        return arguments
    user_text = _normalized_text(_latest_user_text(messages))
    chosen_name = _normalized_text(arguments["name"])
    if chosen_name and chosen_name in user_text:
        return arguments
    domain = _normalized_text(domains[0])
    entities = _relevant_entities(messages)
    area_groups = {}
    for entity in entities:
        if _normalized_text(entity["domain"]) != domain or not entity["area"]:
            continue
        area_groups.setdefault(entity["area"], []).append(entity)
    matching = [
        (area, members) for area, members in area_groups.items()
        if _normalized_text(area) in user_text and len(members) >= 2
    ]
    if len(matching) != 1:
        return arguments
    area, members = matching[0]
    if not any(_normalized_text(item["name"]) == chosen_name for item in members):
        return arguments
    repaired = dict(arguments)
    repaired.pop("name", None)
    repaired["area"] = area
    return repaired

def _expand_name_list_arguments(arguments, schema, parallel_tool_calls):
    """Repair Gemma's common HA multi-target shape without inventing arguments.

    Home Assistant intent schemas use name: string for one target, while Gemma
    may emit name: [..] when the user asks to control several devices. A
    singleton list can be normalized directly. Multiple names are represented
    as independent tool calls only when the client explicitly allows parallel
    calls. All expanded argument objects are still validated against the
    caller-provided JSON schema afterwards.
    """
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    name_schema = properties.get("name", {}) if isinstance(properties, dict) else {}
    names = arguments.get("name") if isinstance(arguments, dict) else None
    if not (
        isinstance(names, list)
        and isinstance(name_schema, dict)
        and name_schema.get("type") == "string"
    ):
        return [arguments]
    if not names or not all(isinstance(value, str) for value in names):
        return [arguments]
    if len(names) > 1 and not parallel_tool_calls:
        raise ValueError(
            "Model returned multiple device names while parallel_tool_calls is disabled"
        )
    return [{**arguments, "name": value} for value in names]


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
        schema = tools[name].get("parameters", {})
        original_arguments = dict(arguments)
        arguments = _prefer_area_target(arguments, schema, request.messages)
        if arguments != original_arguments:
            _LOG.debug(
                "event=tool_argument_normalization request_id=%s tool=%s "
                "kind=area_target before=%s after=%s",
                getattr(request, "_request_id", "-"),
                name,
                json.dumps(original_arguments, ensure_ascii=False, separators=(",", ":")),
                json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
            )
        expanded_arguments = _expand_name_list_arguments(
            arguments, schema, request.parallel_tool_calls
        )
        if len(expanded_arguments) > 1:
            _LOG.debug(
                "event=tool_argument_normalization request_id=%s tool=%s "
                "kind=expand_name_list count=%d before=%s after=%s",
                getattr(request, "_request_id", "-"),
                name,
                len(expanded_arguments),
                json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                json.dumps(expanded_arguments, ensure_ascii=False, separators=(",", ":")),
            )
        for expanded in expanded_arguments:
            try:
                Draft202012Validator(schema).validate(expanded)
            except ValidationError as exc:
                raise ValueError(
                    f"Model returned invalid arguments for {name}: {exc.message}"
                ) from exc
            normalized.append({
                "id": "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(expanded, ensure_ascii=False),
                },
            })
    if not normalized:
        if request.tool_choice == "required" or isinstance(request.tool_choice, dict):
            raise ValueError("Model did not return the required tool call")
        return text
    return {"role": "assistant", "content": text or None, "tool_calls": normalized}


def has_tool_context(request):
    return bool(request.tools) or request.tool_choice is not None or any(
        message.get("role") == "tool" or message.get("tool_calls") for message in request.messages
    )
