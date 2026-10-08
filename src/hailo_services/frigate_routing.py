"""Conservative Frigate tool planning; execution always remains with Frigate."""

import copy
import json
import logging
import re
from datetime import datetime, timedelta

from .frigate_prompt import TOOL_HINTS, camera_catalogue, has_images, server_time, text_content
from .tool_calling import arguments_object, response_message, selected_tools

_LOG = logging.getLogger(__name__)

_ABSENCE = re.compile(
    r"(?:was (?:ist )?passiert(?:e)?[, ]+(?:während|waehrend|als) ich weg war|"
    r"was (?:ist )?passiert[, ]+(?:während|waehrend) meiner abwesenheit|"
    r"what happened (?:while i was away|during my absence))",
    re.I,
)
_LIVE = re.compile(
    r"(?:was (?:ist|passiert)|what (?:is|do you see)|show|zeige|beschreibe|describe).*(?:jetzt|aktuell|gerade|live|now|visible).*|(?:zeige|show).*(?:kamera|camera).*",
    re.I,
)
_WRITES = {"set_camera_state", "start_camera_watch", "stop_camera_watch", "create_export"}
_EVENT_QUERY = re.compile(
    r"(?:(?:zeige|zeig|show)(?: mir| me)?(?: die| alle| the| all)? )?"
    r"(?:ereignisse|events|aktivitäten|activity|erkennungen|detections|historie|history) (.+)",
    re.I,
)
_TIME_REPLY = re.compile(
    r"(?:(?:von|seit|ab|from|since) )?"
    r"(?:heute morgen|this morning|heute|today|gestern|yesterday|"
    r"(?:heute |today )?\d{1,2}(?::\d{2})?(?: uhr)?)"
    r"(?: bis jetzt| until now)?",
    re.I,
)


def last_seen_name(question):
    """Recognize narrow named-entity last-sighting questions without inferring identity.

    Args:
        question: Latest user text, without terminal punctuation.

    Returns:
        str | None: Literal name to search as sub_label, otherwise None.
    """
    match = re.fullmatch(
        r"(?:wann wurde (.+?) zuletzt (?:gesehen|erkannt)|when was (.+?) last seen|"
        r"wann hast du zuletzt (.+?) gesehen|wann ist (.+?) zuletzt gesehen worden|"
        r"when did you last see (.+?))",
        question,
        re.I,
    )
    if match:
        name = next(group for group in match.groups() if group)
        name = re.sub(r"^(?:die person|the person) ", "", name, flags=re.I)
    else:
        match = re.fullmatch(
            r"ich meine (?:die )?person [\"“](.+?)[\"”][, ]+wann wurde (?:er|sie|es) zuletzt (?:gesehen|erkannt)",
            question,
            re.I,
        )
        if not match:
            return None
        name = match[1]
    name = name.strip().strip('"“”')
    if not re.fullmatch(r"[\w][\w .'-]{0,79}", name) or re.search(
        r"\b(person|personen|people|jemand|someone|auto|autos|car|cars|hund|dog|katze|cat|"
        r"paket|package|vogel|bird|mit|with|wearing|in|der|die|das|ein|eine|a|the)\b",
        name,
        re.I,
    ):
        return None
    return name


def event_interval(request):
    """Resolve narrow event/time forms using Frigate's supplied local clock.

    A time-only follow-up inherits only the preceding event query through time clarifications;
    assistant suggestions never supply dates or filters. Morning has no silently
    invented start hour.

    Args:
        request: Original Frigate request with conversation and server clock.

    Returns:
        tuple: Recognition flag, local ISO interval or None, clarification or None.
    """
    users = [
        text_content(m).strip().rstrip("?.!") for m in request.messages if m.get("role") == "user"
    ]
    match = _EVENT_QUERY.fullmatch(users[-1])
    if match:
        phrase = match[1].casefold()
    elif len(users) > 1 and _TIME_REPLY.fullmatch(users[-1]):
        previous = next((q for q in reversed(users[:-1]) if not _TIME_REPLY.fullmatch(q)), "")
        if not _EVENT_QUERY.fullmatch(previous):
            return False, None, None
        phrase = users[-1].casefold()
    else:
        return False, None, None
    german = not re.search(r"\b(show|today|yesterday|morning|since|from)\b", users[-1], re.I)
    phrase = re.sub(r"^(?:von|seit|ab|from|since|of) ", "", phrase)
    phrase = re.sub(r"(?: bis jetzt| until now)$", "", phrase)
    duration = re.fullmatch(
        r"(?:(?:der |in den )?letzten?|(?:in )?(?:the )?last) "
        r"(\d+|eine[rn]?|one|an?)?\s*(stunden?|hours?|minuten?|minutes?)",
        phrase,
    )
    since = re.fullmatch(r"(?:heute |today )?(\d{1,2})(?::(\d{2}))?(?: uhr)?", phrase)
    if (
        not duration
        and not since
        and phrase not in {"heute", "today", "gestern", "yesterday", "heute morgen", "this morning"}
    ):
        return False, None, None
    if phrase in {"heute morgen", "this morning"}:
        return (
            True,
            None,
            (
                "Ab welcher Uhrzeit meinst du mit heute Morgen? Zum Beispiel: ab 06:00 Uhr bis jetzt."
                if german
                else "What start time do you mean by this morning? For example: from 06:00 until now."
            ),
        )
    now = _local_datetime(server_time(request.messages))
    if now is None:
        return (
            True,
            None,
            (
                "Frigate hat keine gültige lokale Serverzeit mitgeschickt. Bitte gib Anfang und Ende mit Datum und Uhrzeit an."
                if german
                else "Frigate supplied no valid local server time. Please provide explicit start and end dates and times."
            ),
        )
    before = now
    if duration:
        number = duration[1] or "1"
        amount = int(number) if number.isdigit() else 1
        minutes = amount * (60 if duration[2].startswith(("stund", "hour")) else 1)
        if minutes <= 0:
            return (
                True,
                None,
                (
                    "Bitte gib einen positiven Zeitraum an."
                    if german
                    else "Please provide a positive time period."
                ),
            )
        try:
            after = now - timedelta(minutes=minutes)
        except OverflowError:
            return True, None, "Please provide a time period within the supported calendar."
    elif since:
        try:
            after = now.replace(hour=int(since[1]), minute=int(since[2] or 0), second=0)
        except ValueError:
            after = now
        if after >= now:
            return (
                True,
                None,
                (
                    "Bitte gib eine gültige Uhrzeit vor der mitgeschickten Serverzeit an."
                    if german
                    else "Please provide a valid start time before the supplied server time."
                ),
            )
    else:
        after = now.replace(hour=0, minute=0, second=0)
        if phrase in {"gestern", "yesterday"}:
            before = after
            after -= timedelta(days=1)
        if after == before:
            return (
                True,
                None,
                "Für diesen Zeitraum liegt noch keine vergangene Zeit vor."
                if german
                else "This period has no elapsed time yet.",
            )
    return (
        True,
        {
            "after": after.isoformat(timespec="seconds"),
            "before": before.isoformat(timespec="seconds"),
        },
        None,
    )


def latest_question(request):
    """Get the last user question for a text-only request.

    Args:
        request: Original request.

    Returns:
        str: User text stripped of terminal sentence punctuation.
    """
    return next(
        (
            text_content(m).strip().rstrip("?.!")
            for m in reversed(request.messages)
            if m.get("role") == "user"
        ),
        "",
    )


def resolved_context(request):
    """Resolve unambiguous calendar and camera facts without replacing the user's task.

    Args:
        request: Frigate request with authoritative local clock and camera catalogue.

    Returns:
        dict: Explicit facts for the compact model prompt; absent facts are not guessed.
    """
    question = latest_question(request)
    context = {}
    days = set(re.findall(r"\b(gestern|yesterday|heute|today)\b", question, re.I))
    days = {"yesterday" if day.casefold() in {"gestern", "yesterday"} else "today" for day in days}
    now = _local_datetime(server_time(request.messages))
    if (
        now
        and len(days) == 1
        and not re.search(
            r"\b(morgen|morning|abend|evening|nachmittag|afternoon)\b|\d{1,2}:\d{2}|\d{1,2}\s*(?:uhr|am|pm)\b",
            question,
            re.I,
        )
    ):
        midnight = now.replace(hour=0, minute=0, second=0)
        day = next(iter(days))
        start, end = (
            (midnight - timedelta(days=1), midnight) if day == "yesterday" else (midnight, now)
        )
        context["local_time_window"] = {
            "after": start.isoformat(timespec="seconds"),
            "before": end.isoformat(timespec="seconds"),
            "meaning": "entire previous day, ending at next midnight"
            if day == "yesterday"
            else "today until supplied server time",
        }
    cameras = camera_catalogue(request.messages)
    matches = matched_cameras(question, cameras)
    if len(matches) == 1:
        context["camera_id"] = matches[0]
    duration = re.search(
        r"\b(?:letzten?|last) (\d+) (stunden?|hours?|minuten?|minutes?)\b", question, re.I
    )
    if now and duration:
        minutes = int(duration[1]) * (
            60 if duration[2].casefold().startswith(("stund", "hour")) else 1
        )
        if minutes > 0:
            try:
                start = now - timedelta(minutes=minutes)
            except OverflowError:
                pass
            else:
                context["local_time_window"] = {
                    "after": start.isoformat(timespec="seconds"),
                    "before": now.isoformat(timespec="seconds"),
                    "meaning": "requested elapsed interval ending at supplied server time",
                }
    return context


def tool_results(request):
    """Read matched results only from the active user turn.

    Args:
        request: Original text request.

    Returns:
        dict: Tool names mapped to decoded client results.
    """
    start = max(i for i, m in enumerate(request.messages) if m.get("role") == "user")
    calls, results = {}, {}
    for message in request.messages[start:]:
        for call in message.get("tool_calls", []):
            calls[call["id"]] = call["function"]["name"]
        if message.get("role") == "tool":
            name = calls.get(message.get("tool_call_id"))
            if name:
                try:
                    results[name] = json.loads(message.get("content", ""))
                except (ValueError, TypeError):
                    results[name] = {"error": "Tool returned non-JSON data"}
    return results


def _local_datetime(value):
    """Parse supplied local times without converting or assuming a timezone.

    Args:
        value: Timestamp text from Frigate.

    Returns:
        datetime | None: Naive local datetime, or None for unsupported/ambiguous values.
    """
    if not isinstance(value, str):
        return None
    for fmt in (
        "%Y-%m-%d at %I:%M:%S %p",
        "%Y-%m-%d %I:%M:%S %p",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def recap_interval(data, request, settings):
    """Resolve a known absence interval using supplied profile activation times.

    Args:
        data: get_profile_status result.
        request: Original Frigate request carrying server-local time.
        settings: Service settings defining explicitly recognized away profiles.

    Returns:
        dict | None: Local ISO after/before values, or None when evidence is insufficient.
    """
    if not isinstance(data, dict) or data.get("error"):
        return None
    now = _local_datetime(server_time(request.messages))
    activated = data.get("last_activated", {})
    aliases = {
        p.strip().casefold() for p in settings.frigate_assist_away_profiles.split(",") if p.strip()
    }
    if not now or not isinstance(activated, dict):
        return None
    candidates = [
        _local_datetime(ts) for name, ts in activated.items() if name.casefold() in aliases
    ]
    candidates = [ts for ts in candidates if ts and ts < now]
    if not candidates:
        return None
    after = max(candidates)
    active = data.get("active_profile")
    if not isinstance(active, str):
        return None
    if active.casefold() in aliases:
        before = now
    else:
        # A named non-away profile provides an end only if activated after departure.
        before = _local_datetime(activated.get(active))
        if not before or not after < before <= now:
            return None
    return {
        "after": after.isoformat(timespec="seconds"),
        "before": before.isoformat(timespec="seconds"),
    }


def matched_cameras(text, cameras):
    """Match exact friendly names or IDs without fuzzy guessing.

    Args:
        text: User question.
        cameras: Supplied ID/name catalogue.

    Returns:
        list[str]: Unambiguous mentioned camera IDs.
    """
    hits = []
    for identifier, friendly in cameras.items():
        if any(
            re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I)
            for value in {identifier, friendly}
        ):
            hits.append(identifier)
    return hits


def _call(request, name, arguments):
    """Build a schema-validated OpenAI tool call without model inference.

    Args:
        request: Request with selected tools.
        name: Tool name.
        arguments: Proposed argument mapping.

    Returns:
        dict: Validated assistant tool call message.
    """
    return response_message(
        {"tool_calls": [{"function": {"name": name, "arguments": arguments}}]}, request, ""
    )


def last_sighting_query(question, cameras):
    """Parse a narrow last-sighting intent, retaining explicit camera and image requests.

    Args:
        question: Current user question without trailing punctuation.
        cameras: Supplied camera ID/friendly-name mapping.

    Returns:
        tuple | None: Search arguments and image-request flag, or None for unsupported wording.
    """
    appearance = re.fullmatch(
        r"welche farbe hatte die kleidung von (.+?) als (?:er|sie|die person) zuletzt gesehen wurde|"
        r"what was (.+?) wearing when (?:they|he|she) (?:was|were) last seen",
        question,
        re.I,
    )
    if appearance:
        name = last_seen_name("When was " + (appearance[1] or appearance[2]) + " last seen")
        if name:
            return {"sub_label": name}, True
    image = re.search(
        r"[?.!]\s*(?:gib|zeige|zeig) mir (?:das )?letzte (?:bild|foto)(?: dazu)?$|"
        r"[?.!]\s*show me (?:the )?last (?:image|picture)(?: too)?$",
        question,
        re.I,
    )
    core = question[: image.start()] if image else question
    camera = re.search(r" (?:an|bei|in|at|on) (?:kamera|camera) (.+)$", core, re.I)
    args = {}
    if camera:
        matches = [
            key
            for key, value in cameras.items()
            if camera[1].casefold() in {key.casefold(), value.casefold()}
        ]
        if len(matches) != 1:
            return None
        args["camera"] = matches[0]
        core = core[: camera.start()]
    name = last_seen_name(core)
    if name:
        return {**args, "sub_label": name}, bool(image)
    generic = re.fullmatch(
        r"(?:wann wurde zuletzt (?:eine?|die|der|das) (.+?) (?:vor|an) (?:dem |der )?(.+?) (?:erkannt|gesehen)|"
        r"when was (?:a|the) (.+?) last (?:seen|detected) (?:at|in front of) (.+))",
        core,
        re.I,
    )
    if generic:
        label, target = (generic[1], generic[2]) if generic[1] else (generic[3], generic[4])
        classes = {
            "person": "person",
            "auto": "car",
            "car": "car",
            "hund": "dog",
            "dog": "dog",
            "katze": "cat",
            "cat": "cat",
            "paket": "package",
            "package": "package",
            "fahrrad": "bicycle",
            "bicycle": "bicycle",
        }
        matches = [
            key
            for key, value in cameras.items()
            if target.casefold() in {key.casefold(), value.casefold()}
        ]
        if label.casefold() in classes and len(matches) == 1:
            return {"label": classes[label.casefold()], "camera": matches[0]}, bool(image)
    return None


def request_question(request, cameras):
    """Resolve an exact camera reply only against the immediately preceding user intent.

    Args:
        request: Incoming Frigate request.
        cameras: Supplied camera catalogue.

    Returns:
        tuple: Effective question and optional exact camera ID.
    """
    question = latest_question(request)
    matches = [
        key
        for key, value in cameras.items()
        if question.casefold() in {key.casefold(), value.casefold()}
    ]
    if len(matches) == 1:
        users = [
            text_content(m).strip().rstrip("?.!")
            for m in request.messages
            if m.get("role") == "user"
        ]
        if len(users) > 1 and (
            _LIVE.fullmatch(users[-2]) or last_sighting_query(users[-2], cameras)
        ):
            return users[-2], matches[0]
    return question, None


def event_image_call(request, tools, event_id, german):
    """Fetch an identified event frame only through an explicitly provided image tool.

    Args:
        request: Incoming request used to validate tools.
        tools: Selected read tool declarations.
        event_id: Exact supplied event identifier.
        german: Whether to use a German clarification.

    Returns:
        tuple: Tool selection, direct result, and route reason.
    """
    selected = [t for t in tools if t["function"]["name"] == "get_event_image"]
    if selected and "event_id" in selected[0]["function"].get("parameters", {}).get(
        "properties", {}
    ):
        return selected, _call(request, "get_event_image", {"event_id": event_id}), "event_image"
    return (
        [],
        (
            "Für dieses Ereignis wurde kein Werkzeug zum Abrufen des Bildes bereitgestellt. "
            "Öffne das Ereignisbild in Frigate oder sende es zur Bildanalyse mit."
            if german
            else "No event-image retrieval tool was supplied. Open the event image in Frigate or attach it for image analysis."
        ),
        "event_image_unavailable",
    )


def last_sighting_result(request, query, results, tools, cameras, german):
    """Summarize a verified one-result last-sighting search without generative inference.

    Args:
        request: Incoming request with active tool calls.
        query: Parsed search constraints and image flag.
        results: Active matched tool results.
        tools: Read tool declarations.
        cameras: Camera names supplied by Frigate.
        german: Answer-language flag.

    Returns:
        tuple | None: Deterministic response when evidence is sufficient, otherwise None.
    """
    args, image = query
    calls = [
        c
        for m in request.messages
        for c in m.get("tool_calls", [])
        if c["function"]["name"] == "search_objects"
    ]
    if not calls:
        return None
    actual = arguments_object(calls[-1]["function"]["arguments"])
    if actual != {**args, "limit": 1} or set(results) != {"search_objects"}:
        return None
    events = results["search_objects"]
    if not isinstance(events, list) or len(events) > 1:
        return None
    if not events:
        return (
            [],
            (
                "Für diese Suchfilter wurde keine Sichtung zurückgegeben."
                if german
                else "No sighting was returned for these search filters."
            ),
            "last_seen_empty",
        )
    event = events[0]
    if not isinstance(event, dict) or event.get("error") or event.get("_omitted"):
        return None
    if any(event.get(key) != value for key, value in args.items()):
        return None
    start, end = event.get("start_time_local"), event.get("end_time_local")
    if not _local_datetime(start) or (end is not None and not _local_datetime(end)):
        return None
    if end is not None and _local_datetime(end) < _local_datetime(start):
        return None
    camera = event.get("camera")
    if camera not in cameras:
        return None
    who = event.get("sub_label") or event.get("label")
    if not isinstance(who, str):
        return None
    if image and isinstance(event.get("id"), str) and event["id"]:
        return event_image_call(request, tools, event["id"], german)
    period = start + (" – " + end if end and end != start else "")
    return (
        [],
        (
            f"Letzte zurückgegebene Sichtung: {who}, Kamera {cameras[camera]}, {period}."
            if german
            else f"Latest returned sighting: {who}, camera {cameras[camera]}, {period}."
        ),
        "last_seen_summary",
    )


def plan(request, settings, images):
    """Select tools and return deterministic read calls or conservative clarifications.

    Args:
        request: Original Frigate request.
        settings: Service settings.
        images: Whether the incoming request includes images.

    Returns:
        tuple: Selected declarations, direct response/call or None, and route reason.

    Raises:
        ValueError: A forced tool violates the supported conservative action policy.
    """
    tools = selected_tools(request)
    if images:
        if request.tool_choice == "required" or isinstance(request.tool_choice, dict):
            return tools, None, "image_tools"
        question = latest_question(request)
        if question.startswith("Here is the current live image"):
            questions = [
                text_content(m)
                for m in request.messages
                if m.get("role") == "user"
                and not text_content(m).startswith("Here is the current live image")
            ]
            question = questions[-1] if questions else question
        if re.search(
            r"\b(notify|watch|benachrichtige|überwache|enable|disable|schalte)\b", question, re.I
        ):
            return [t for t in tools if t["function"]["name"] in _WRITES], None, "image_action"
        if re.search(r"\b(search|find|suche|finde|similar|ähnlich|aehnlich)\b", question, re.I):
            return (
                [
                    t
                    for t in tools
                    if t["function"]["name"] in {"search_objects", "find_similar_objects"}
                ],
                None,
                "image_search",
            )
        return [], None, "image_observation"
    names = {t["function"]["name"] for t in tools}
    cameras = camera_catalogue(request.messages)
    question, reply_camera = request_question(request, cameras)
    german = not re.search(r"\b(what|when|show|describe|stop|turn|camera status)\b", question, re.I)
    results = tool_results(request)
    targets = [reply_camera] if reply_camera else matched_cameras(question, cameras)
    query = last_sighting_query(question, cameras)
    if query and reply_camera:
        query = ({**query[0], "camera": reply_camera}, query[1])
    if query and "search_objects" in results:
        direct = last_sighting_result(request, query, results, tools, cameras, german)
        if direct:
            return direct
    anchor = re.match(r"\[attached_event:([^]\s]+)\]\s*(.*)", question, re.S)
    if anchor and re.search(
        r"\b(kleidung|farbe|bild|foto|clothing|wearing|color|colour|image|picture)\b",
        anchor[2],
        re.I,
    ):
        if "get_event_image" not in results:
            return event_image_call(request, tools, anchor[1], german)
        return (
            [],
            (
                "Das Werkzeug hat kein Bild zur Analyse mitgeschickt."
                if german
                else "The tool did not attach an image for analysis."
            ),
            "event_image_missing",
        )
    absent = bool(_ABSENCE.fullmatch(question))
    recap = results.get("get_recap")
    if (
        isinstance(recap, dict)
        and set(results).issubset({"get_recap", "get_profile_status"})
        and set(recap).issubset({"events", "message"})
        and recap.get("events") == []
        and recap.get("message") in (None, "No activity was found during this time period.")
    ):
        # Only the canonical successful empty shape is conclusive. Error/partial
        # fields must never be converted into a claim of no activity.
        return (
            [],
            (
                "Für den abgefragten Zeitraum wurden keine Aktivitäten zurückgegeben."
                if german
                else "No activity was returned for the requested time period."
            ),
            "empty_recap",
        )
    # Exact actions bypass generation. Other supplied tools remain model-accessible.
    if question.casefold() in {
        "stop watching",
        "stop camera watch",
        "stoppe die überwachung",
        "beende die überwachung",
    }:
        if "stop_camera_watch" in names and not results:
            return (
                [t for t in tools if t["function"]["name"] == "stop_camera_watch"],
                _call(request, "stop_camera_watch", {}),
                "stop_watch",
            )
    state = re.fullmatch(
        r"(?:schalte|turn) (?:die |the )?(erkennung|detection|aufnahme|recording) (?:für|fuer|for) (?:kamera |camera )?(.+?) (ein|aus|on|off)",
        question,
        re.I,
    )
    if state and "set_camera_state" in names and not results:
        selected = matched_cameras(state[2], cameras)
        if len(selected) == 1 and state[2].casefold() in {
            selected[0].casefold(),
            cameras[selected[0]].casefold(),
        }:
            args = {
                "camera": selected[0],
                "feature": "detect"
                if state[1].casefold() in {"erkennung", "detection"}
                else "record",
                "value": "ON" if state[3].casefold() in {"ein", "on"} else "OFF",
            }
            return (
                [t for t in tools if t["function"]["name"] == "set_camera_state"],
                _call(request, "set_camera_state", args),
                "camera_state",
            )
    if any(name in _WRITES for name in results):
        # Let Gemma explain the actual returned success/error; never repeat the action.
        return tools, None, "action_result"
    if absent:
        if "get_recap" in results:
            return tools, None, "recap_summary"
        if "get_profile_status" in results:
            interval = recap_interval(results["get_profile_status"], request, settings)
            if interval and "get_recap" in names:
                return (
                    [t for t in tools if t["function"]["name"] == "get_recap"],
                    _call(request, "get_recap", interval),
                    "absence_interval",
                )
            clarification = (
                "Für die Abwesenheit lässt sich kein eindeutiger Zeitraum aus den Profilen bestimmen. Von wann bis wann soll ich nachsehen?"
                if german
                else "The profiles do not establish an unambiguous absence interval. What start and end time should I use?"
            )
            return tools, clarification, "absence_ambiguous"
        if "get_profile_status" in names:
            return (
                [t for t in tools if t["function"]["name"] == "get_profile_status"],
                _call(request, "get_profile_status", {}),
                "absence_profile",
            )
    # Technical health is not equivalent to a live image/detection status.
    live_reply = question.casefold() in {
        "kamerabild",
        "das aktuelle kamerabild",
        "das kamerabild",
        "das aktuelle livebild",
        "the current camera image",
        "the live image",
    }
    if live_reply and "get_live_context" in names and not results:
        if len(cameras) == 1:
            return (
                [t for t in tools if t["function"]["name"] == "get_live_context"],
                _call(request, "get_live_context", {"camera": next(iter(cameras))}),
                "live_image_reply",
            )
        return (
            tools,
            (
                "Welche Kamera meinst du? Bitte nenne den Kameranamen."
                if german
                else "Which camera do you mean? Please provide its name."
            ),
            "camera_clarification",
        )
    if (
        re.fullmatch(
            r"(?:wie ist der aktuelle status meiner kameras|(?:wie ist der )?(?:kamera|kameras)[ -]?status|what is (?:the )?(?:current )?status of my cameras|camera status)",
            question,
            re.I,
        )
        and not results
    ):
        return (
            tools,
            (
                "Meinst du das aktuelle Kamerabild und die Erkennungen oder den technischen Betriebsstatus? Für den technischen Status wurde kein Werkzeug bereitgestellt."
                if german
                else "Do you mean the live image and detections or technical camera health? No technical-health tool was supplied."
            ),
            "status_clarification",
        )
    historical = bool(
        re.search(
            r"\b(letzten?|gestern|heute|last|yesterday|today|ereignisse|events|history)\b",
            question,
            re.I,
        )
    )
    live = (
        not historical
        and bool(
            _LIVE.fullmatch(question)
            or re.fullmatch(
                r"(?:zeige|show|beschreibe|describe).*\b(?:bild|image|foto|picture)\b.*",
                question,
                re.I,
            )
        )
        and bool(
            targets
            or re.search(
                r"\b(kamera|kameras|camera|cameras|live|livebild|liveimage)\b", question, re.I
            )
        )
    )
    if live and "get_live_context" in names and not results:
        if len(targets) == 1:
            return (
                [t for t in tools if t["function"]["name"] == "get_live_context"],
                _call(request, "get_live_context", {"camera": targets[0]}),
                "live_camera",
            )
        if not targets:
            # An unnamed live request can use the sole supplied camera; a named
            # unknown target must still be clarified rather than silently replaced.
            if len(cameras) == 1 and re.fullmatch(
                r"(?:zeige|show)(?: mir| me)? (?:(?:das|ein|the|a) )?(?:aktuelle |current )?live[ -]?(?:bild|image)(?: an)?",
                question,
                re.I,
            ):
                return (
                    [t for t in tools if t["function"]["name"] == "get_live_context"],
                    _call(request, "get_live_context", {"camera": next(iter(cameras))}),
                    "live_camera",
                )
            return (
                tools,
                "Welche Kamera meinst du? Bitte verwende einen der angegebenen Kameranamen."
                if german
                else "Which camera do you mean? Please use one of the supplied camera names.",
                "camera_clarification",
            )
    if "get_profile_status" in results and "get_recap" not in results:
        return [t for t in tools if t["function"]["name"] == "get_recap"], None, "profile_reasoning"
    if "search_objects" in results and re.search(r"\b(ähnlich|aehnlich|similar)\b", question, re.I):
        return (
            [t for t in tools if t["function"]["name"] == "find_similar_objects"],
            None,
            "similarity_anchor",
        )
    if results:
        # Reading live context without a frame cannot describe unseen objects.
        return tools, None, "tool_summary"
    if query and "search_objects" in names:
        selected = [t for t in tools if t["function"]["name"] == "search_objects"]
        args = dict(query[0])
        if "limit" in selected[0]["function"].get("parameters", {}).get("properties", {}):
            args["limit"] = 1
        return (
            selected,
            _call(request, "search_objects", args),
            "named_last_seen" if "sub_label" in args else "class_last_seen",
        )
    recognized, interval, clarification = event_interval(request)
    if recognized and names.intersection({"get_recap", "search_objects"}):
        if clarification:
            return tools, clarification, "event_time_clarification"
        name = "get_recap" if "get_recap" in names else "search_objects"
        return (
            [t for t in tools if t["function"]["name"] == name],
            _call(request, name, interval),
            "event_time_interval",
        )
    if request.tool_choice == "required" or isinstance(request.tool_choice, dict):
        return tools, None, "explicit_tool_choice"
    relevant = set()
    if re.search(
        r"\b(benachrichtige|notify|überwache|watch)\b|pass(?:e)? auf|(?:sag|sage|gib) mir bescheid",
        question,
        re.I,
    ):
        relevant.update({"start_camera_watch", "stop_camera_watch", "get_live_context"})
    if re.search(r"\b(schalte|turn|deaktiviere|aktiviere|enable|disable)\b", question, re.I):
        relevant.add("set_camera_state")
    if absent:
        relevant.update({"get_profile_status", "get_recap"})
    if live:
        relevant.add("get_live_context")
    if re.search(r"\b(ähnlich|aehnlich|similar|attached_event)\b", question, re.I):
        relevant.update({"find_similar_objects", "search_objects"})
    if re.search(
        r"\b(gestern|heute|yesterday|today|detections|erkennungen|autos|cars|personen|people)\b",
        question,
        re.I,
    ):
        relevant.add("search_objects")
    if relevant:
        # Unknown/new tools cannot be classified safely and must remain available.
        selected = [
            t
            for t in tools
            if t["function"]["name"] in relevant or t["function"]["name"] not in TOOL_HINTS
        ]
        if selected:
            tools = selected
    return tools, None, "text_reasoning"


def validate_result(result, prepared, original):
    """Validate generative calls against selected schemas and the supplied camera catalogue.

    Args:
        result: Backend output.
        prepared: Compact request with allowed tools.
        original: Incoming Frigate request.

    Returns:
        object: Validated text or assistant call message.

    Raises:
        ValueError: Generated camera IDs or tool names are unavailable.
    """
    if (
        has_images(prepared)
        and isinstance(result, str)
        and (not result.strip() or "\ufffd" in result)
    ):
        chat = bool(original.tools) or any(
            m.get("role") == "system" and "helpful assistant for Frigate" in text_content(m)
            for m in original.messages
        )
        reason = "replacement_character" if "\ufffd" in result else "empty_output"
        action = "chat_fallback" if chat else "rejected"
        original._metrics["frigate_vision_quality"] = {
            "status": "unusable",
            "reason": reason,
            "action": action,
        }
        _LOG.warning(
            "event=frigate_vision_quality request_id=%s status=unusable reason=%s action=%s",
            original._request_id,
            reason,
            action,
        )
        if not chat:
            # A description may have a JSON contract; never substitute prose that
            # Frigate could store as a successful description/structured result.
            raise ValueError(
                "Frigate-Assist received empty or damaged VLM output; description rejected"
            )
        system = text_content(prepared.messages[0])
        if "Antworte auf Deutsch" in system:
            return "Die Bildbeschreibung konnte nicht zuverlässig erstellt werden. Bitte prüfe das Kamerabild direkt in Frigate."
        if "Answer in Russian" in system:
            return "Не удалось получить надёжное описание изображения. Проверьте изображение камеры в Frigate."
        return "A reliable image description could not be produced. Please check the camera image directly in Frigate."
    if not isinstance(result, dict):
        return result
    # Normalize exact catalogue names before schema validation, including enum IDs.
    # Do not mutate backend output or infer a camera that was not supplied.
    result = copy.deepcopy(result)
    cameras = camera_catalogue(original.messages)
    for call in result.get("tool_calls", []):
        function = call.get("function", {})
        args = arguments_object(function.get("arguments", {}))
        for key in ("camera", "cameras"):
            value = args.get(key)
            if (
                not isinstance(value, (str, list))
                or value == "all"
                or (function.get("name") == "set_camera_state" and value == "*")
            ):
                continue
            identifiers = value if isinstance(value, list) else value.split(",")
            normalized = []
            for identifier in identifiers:
                matches = {
                    cam
                    for cam, friendly in cameras.items()
                    if isinstance(identifier, str)
                    and identifier.casefold() in {cam.casefold(), friendly.casefold()}
                }
                if len(matches) != 1:
                    raise ValueError("Frigate-Assist rejected an unknown or ambiguous camera ID")
                normalized.append(next(iter(matches)))
            args[key] = normalized if isinstance(value, list) else ",".join(normalized)
        function["arguments"] = args
    result = response_message(result, prepared, result.get("content") or "")
    if not isinstance(result, dict):
        return result
    for call in result.get("tool_calls", []):
        name = call["function"]["name"]
        args = arguments_object(call["function"]["arguments"])
        if name in {"search_objects", "find_similar_objects", "get_recap"}:
            times = {}
            for key in ("after", "before"):
                if key not in args:
                    continue
                value = args[key]
                parsed = _local_datetime(value)
                if (
                    not isinstance(value, str)
                    or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value)
                    or parsed is None
                ):
                    raise ValueError(
                        "Frigate-Assist requires valid local ISO timestamps for historical tools"
                    )
                times[key] = parsed
            if "after" in times and "before" in times and times["after"] >= times["before"]:
                raise ValueError(
                    "Frigate-Assist rejected an empty or reversed historical time interval"
                )
            now = _local_datetime(server_time(original.messages))
            if now and any(value > now for value in times.values()):
                raise ValueError(
                    "Frigate-Assist rejected a future timestamp in a historical tool call"
                )
        for key in ("camera", "cameras"):
            if key not in args:
                continue
            value = args[key]
            identifiers = (
                value
                if isinstance(value, list)
                else value.split(",")
                if isinstance(value, str)
                else []
            )
            if name == "get_recap" and value == "all":
                continue
            if name == "set_camera_state" and value == "*":
                continue
            if not identifiers or any(identifier not in cameras for identifier in identifiers):
                raise ValueError("Frigate-Assist rejected an unknown or ambiguous camera ID")
    return result
