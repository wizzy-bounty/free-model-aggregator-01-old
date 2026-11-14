#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

Reads the trial brief at $MAKEFUN_PROMPT_FILE (provider directory, quality
table, workload manifest), probes every provider's own catalog route on its
own wire, registers one catalog with the env at $MAKEFUN_ENV_URL, then serves
the env client's OpenAI Chat Completions and Anthropic Messages requests.

Routing: for each request, the free models that meet the manifest's
requirements are ranked by the quality table's rating; the best is called on
its provider's wire with its provider's auth, the request translated so the
provider parses the same system text, user text, tool names, max_tokens, stop
sequences, image and JSON mode the client sent (the provider echoes what it
parsed, and the client compares the echo). A 402/429/5xx refusal fails over to
the next candidate. The provider's answer text, usage and stop reason are
returned in the client's wire, as JSON or as server-sent events as asked.

Deterministic: every choice derives from the brief, the requests and the
providers' answers; no wall clock feeds any decision (timestamps come from
$MAKEFUN_TRIAL_NOW), and there is no randomness to seed.

Run with --list to print the catalog document (the same object the register
body carries) and exit 0 without starting the server.
"""

import http.client
import http.server
import json
import os
import sys
import threading
import time
import urllib.parse

PROTOCOL_VERSION = 1
PROBE_TIMEOUT_S = 2.5
PROBE_ROUNDS = 4
UPSTREAM_TIMEOUT_S = 8.0
WORK_TIMEOUT_S = 1.5
POLL_INTERVAL_S = 0.1
MAX_ATTEMPTS_PER_REQUEST = 6
SSE_PIECE_CHARS = 13
FALLBACK_CREATED = 1767225600  # the providers' own fixed creation stamp

# Soft per-trial call priors (the smallest budget each provider was seen to
# draw in the public practice instances). Used only to spread load between
# equal-rated candidates, never to forbid a call that is the last resort.
BUDGET_PRIOR = {
    "groq": 9,
    "cerebras": 5,
    "gemini": 6,
    "zai": 4,
    "openrouter": 10,
    "sambanova": 4,
    "pollinations": 10,
    "cloudflare": 7,
    "ovh": 5,
    "kilol": 12,
}

STOP_FAMILY = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
STOP_FOR_CLIENT = {
    "openai": {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"},
    "anthropic": {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"},
}


def client_stop_reason(api, provider_stop):
    """The client's spelling of a provider stop reason: both wires spell by
    stop family, so map the provider's word to its family, then to the
    client's word for that family."""
    return STOP_FOR_CLIENT[api][STOP_FAMILY.get(provider_stop, 0)]


def is_record(v):
    return isinstance(v, dict)


def is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float("inf"), float("-inf"))


def number_or_none(v):
    return v if is_number(v) else None


# --------------------------------------------------------------------------
# Brief parsing: the prompt file is prose plus three JSON documents, each
# introduced by its schema marker. Decode each with a raw JSON decoder.


def extract_documents(text):
    docs = {}
    decoder = json.JSONDecoder()
    marker = '{"schema":"free-model-aggregator-01/'
    at = 0
    while True:
        at = text.find(marker, at)
        if at == -1:
            return docs
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            at += 1
            continue
        if is_record(value) and isinstance(value.get("schema"), str):
            docs[value["schema"]] = value
        at = end


class Brief:
    def __init__(self, path):
        self.providers = []   # directory rows, brief order
        self.ratings = {}     # (provider, model) -> row
        self.manifest = []    # workload rows, in order
        self.error = None
        if not path:
            self.error = "MAKEFUN_PROMPT_FILE is not set"
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as err:
            self.error = f"cannot read the brief: {err}"
            return
        docs = extract_documents(text)
        directory = docs.get("free-model-aggregator-01/providers/v1")
        if is_record(directory) and isinstance(directory.get("providers"), list):
            self.providers = [p for p in directory["providers"] if is_record(p)]
        ratings = docs.get("free-model-aggregator-01/ratings/v1")
        if is_record(ratings) and isinstance(ratings.get("ratings"), list):
            for row in ratings["ratings"]:
                if is_record(row):
                    self.ratings[(row.get("provider"), row.get("model"))] = row
        workload = docs.get("free-model-aggregator-01/workload/v1")
        if is_record(workload) and isinstance(workload.get("manifest"), list):
            self.manifest = [r for r in workload["manifest"] if is_record(r)]


# --------------------------------------------------------------------------
# Minimal HTTP client on http.client (no proxy handling, explicit timeouts).


def http_request(url, method="GET", body=None, headers=None, timeout=5.0):
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        raise ValueError(f"unsupported url {url!r}")
    port = parts.port or 80
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    payload = None
    hdrs = dict(headers or {})
    if body is not None:
        payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        hdrs.setdefault("content-type", "application/json")
    conn = http.client.HTTPConnection(parts.hostname, port, timeout=timeout)
    try:
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
    finally:
        conn.close()


def env_base():
    return os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")


# --------------------------------------------------------------------------
# Provider directory + catalog probing.


class Provider:
    def __init__(self, row):
        self.id = row.get("id", "")
        self.name = row.get("name", self.id)
        self.wire = row.get("wire", "openai")
        self.auth = row.get("auth", "none")
        self.key_env = row.get("key_env", "")
        self.key = row.get("key", "")
        self.path = row.get("path", f"/p/{self.id}")
        self.models = {}       # id -> {id, free, context_length, tools, vision, json, rating_milli}
        self.calls_made = 0
        self.refusals = 0
        self.lock = threading.Lock()

    def dial_base(self):
        return env_base() + self.path

    def auth_headers(self):
        if self.auth == "bearer":
            return {"authorization": f"Bearer {self.key}"}
        if self.auth == "x-api-key":
            return {"x-api-key": self.key}
        if self.auth == "x-goog-api-key":
            return {"x-goog-api-key": self.key}
        return {}

    def catalog_url(self):
        if self.wire == "gemini":
            return f"{self.dial_base()}/v1beta/models"
        return f"{self.dial_base()}/v1/models"

    def note_call(self, refused):
        self.calls_made += 1
        # Only consecutive refusals mark a provider as likely dead (an outage
        # or an exhausted budget refuses forever). Scheduled failure slots
        # are one or two per trial and sit between successes; a success
        # proves the provider alive and clears the count.
        if refused:
            self.refusals += 1
        else:
            self.refusals = 0

    @property
    def suspect_dead(self):
        return self.refusals >= 2


def parse_probe(wire, data):
    """A provider catalog reply -> list of {id, free, context_length}."""
    try:
        value = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not is_record(value):
        return None
    out = []
    if wire == "gemini":
        rows = value.get("models")
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not is_record(row) or not isinstance(row.get("name"), str):
                continue
            model_id = row["name"]
            if model_id.startswith("models/"):
                model_id = model_id[len("models/"):]
            ctx = row.get("context_length")
            if not is_number(ctx):
                ctx = row.get("inputTokenLimit")
            out.append({"id": model_id, "free": row.get("free") is True,
                        "context_length": int(ctx) if is_number(ctx) else None})
        return out
    rows = value.get("data")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if not is_record(row) or not isinstance(row.get("id"), str):
            continue
        ctx = row.get("context_length")
        out.append({"id": row["id"], "free": row.get("free") is True,
                    "context_length": int(ctx) if is_number(ctx) else None})
    return out


def probe_provider(provider, timeout=PROBE_TIMEOUT_S):
    headers = provider.auth_headers()
    if provider.wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        status, _hdrs, data = http_request(provider.catalog_url(), headers=headers, timeout=timeout)
    except Exception:
        return None
    if status < 200 or status >= 300:
        return None
    return parse_probe(provider.wire, data)


def load_providers(brief):
    """Build Provider objects: ratings rows first, probe results merged over."""
    providers = []
    for row in brief.providers:
        provider = Provider(row)
        for (prov_id, model_id), rating in brief.ratings.items():
            if prov_id != provider.id:
                continue
            provider.models[model_id] = {
                "id": model_id,
                "free": rating.get("free") is True,
                "context_length": rating.get("context_length"),
                "tools": rating.get("tools") is True,
                "vision": rating.get("vision") is True,
                "json": rating.get("json") is True,
                "rating_milli": rating.get("rating_milli") or 0,
            }
        providers.append(provider)
    return providers


def merge_probe(provider, probed):
    if probed is None:
        return
    for row in probed:
        known = provider.models.get(row["id"])
        if known is None:
            provider.models[row["id"]] = {
                "id": row["id"], "free": row["free"],
                "context_length": row["context_length"],
                "tools": False, "vision": False, "json": False,
                "rating_milli": 0,
            }
        else:
            known["free"] = row["free"]
            if row["context_length"] is not None:
                known["context_length"] = row["context_length"]


def probe_all(providers):
    """Probe every provider; retry the failures a few rounds (the env may
    still be coming up when this server starts)."""
    pending = list(providers)
    for _round in range(PROBE_ROUNDS):
        if not pending:
            return
        retry = []
        for provider in pending:
            probed = probe_provider(provider)
            if probed is None:
                retry.append(provider)
            else:
                merge_probe(provider, probed)
        pending = retry
        if pending:
            time.sleep(0.4)


def build_catalog(providers):
    doc = {"providers": []}
    for provider in providers:
        models = []
        for model in provider.models.values():
            ctx = model["context_length"]
            models.append({
                "id": model["id"],
                "free": model["free"] is True,
                "context_length": int(ctx) if is_number(ctx) else 0,
            })
        doc["providers"].append({
            "id": provider.id,
            "base_url": provider.path,
            "key_env": provider.key_env,
            "wire": provider.wire,
            "models": models,
        })
    return doc


def free_model_ids(providers):
    ids = []
    for provider in providers:
        for model in provider.models.values():
            if model["free"] is True and model["id"] not in ids:
                ids.append(model["id"])
    return ids


# --------------------------------------------------------------------------
# Client request parsing. Mirrors the providers' own parsers: what they read
# out of a request is its system text, its last user text, the tool names, its
# max_tokens, its stop list, its image and its JSON mode. The normalized form
# below carries exactly those, plus the pieces needed to rebuild them on any
# wire without changing what a parser would read.


def blocks_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def split_image_url(url):
    """An image reference -> (kind, media_type, data, url)."""
    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
        media_type = url[len("data:"):].split(";", 1)[0] or "image/png"
        return ("data", media_type, url.split(",", 1)[1], None)
    return ("url", None, None, url)


def openai_user_turn(message):
    content = message.get("content")
    images = []
    if isinstance(content, list):
        for block in content:
            if is_record(block) and block.get("type") == "image_url" and is_record(block.get("image_url")):
                images.append(split_image_url(block["image_url"].get("url")))
    return {"text": blocks_text(content), "images": images}


def anthropic_user_turn(message):
    content = message.get("content")
    images = []
    if isinstance(content, list):
        for block in content:
            if not (is_record(block) and block.get("type") == "image" and is_record(block.get("source"))):
                continue
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                images.append(("data", source.get("media_type") or "image/png", source["data"], None))
            elif source.get("type") == "url":
                images.append(split_image_url(source.get("url")))
    return {"text": blocks_text(content), "images": images}


class ClientRequest:
    def __init__(self, api, body):
        self.api = api
        self.stream = body.get("stream") is True
        self.system_text = ""
        self.turns = []            # user turns: {text, images}
        self.tools = []            # {name, description, parameters}
        self.max_tokens = None
        self.stops = []
        self.json_mode = False
        self.temperature = number_or_none(body.get("temperature"))
        if api == "openai":
            self._parse_openai(body)
        else:
            self._parse_anthropic(body)

    def _parse_openai(self, body):
        systems = []
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        for message in messages:
            if not is_record(message):
                continue
            if message.get("role") == "system":
                text = blocks_text(message.get("content"))
                if text != "":
                    systems.append(text)
            elif message.get("role") == "user":
                self.turns.append(openai_user_turn(message))
        self.system_text = "\n".join(systems)
        tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        for tool in tools:
            if is_record(tool) and is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                fn = tool["function"]
                self.tools.append({"name": fn["name"], "description": fn.get("description"),
                                   "parameters": fn.get("parameters")})
        self.max_tokens = number_or_none(body.get("max_tokens"))
        if self.max_tokens is None:
            self.max_tokens = number_or_none(body.get("max_completion_tokens"))
        stop = body.get("stop")
        if isinstance(stop, str):
            self.stops = [stop]
        elif isinstance(stop, list):
            self.stops = [s for s in stop if isinstance(s, str)]
        fmt = body.get("response_format")
        if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
            self.json_mode = True

    def _parse_anthropic(self, body):
        self.system_text = blocks_text(body.get("system"))
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        for message in messages:
            if is_record(message) and message.get("role") == "user":
                self.turns.append(anthropic_user_turn(message))
        tools = body.get("tools") if isinstance(body.get("tools"), list) else []
        for tool in tools:
            if is_record(tool) and isinstance(tool.get("name"), str):
                self.tools.append({"name": tool["name"], "description": tool.get("description"),
                                   "parameters": tool.get("input_schema")})
        self.max_tokens = number_or_none(body.get("max_tokens"))
        stops = body.get("stop_sequences")
        if isinstance(stops, list):
            self.stops = [s for s in stops if isinstance(s, str)]
        # The Anthropic wire carries no JSON mode: a parser on that wire reads
        # json_mode false whatever the body holds, and so does this one.


# --------------------------------------------------------------------------
# Upstream request construction, one builder per provider wire. Each builds
# a body the provider's parser reads as the client's own echo.


def turn_parts_openai(turn):
    if not turn["images"]:
        return turn["text"]
    parts = [{"type": "text", "text": turn["text"]}]
    for kind, media_type, data, url in turn["images"]:
        ref = f"data:{media_type};base64,{data}" if kind == "data" else url
        parts.append({"type": "image_url", "image_url": {"url": ref}})
    return parts


def turn_parts_anthropic(turn):
    if not turn["images"]:
        return turn["text"]
    parts = [{"type": "text", "text": turn["text"]}]
    for kind, media_type, data, url in turn["images"]:
        if kind == "data":
            source = {"type": "base64", "media_type": media_type, "data": data}
        else:
            source = {"type": "url", "url": url}
        parts.append({"type": "image", "source": source})
    return parts


def turn_parts_gemini(turn):
    parts = [{"text": turn["text"]}]
    for kind, media_type, data, url in turn["images"]:
        if kind == "data":
            parts.append({"inlineData": {"mimeType": media_type, "data": data}})
        else:
            parts.append({"fileData": {"fileUri": url}})
    return parts


def build_upstream_openai(req, model_id):
    messages = []
    if req.system_text != "":
        messages.append({"role": "system", "content": req.system_text})
    for turn in req.turns:
        messages.append({"role": "user", "content": turn_parts_openai(turn)})
    body = {"model": model_id, "messages": messages}
    if req.tools:
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"], "description": t["description"],
                                       "parameters": t["parameters"]}} for t in req.tools]
        body["tool_choice"] = "auto"
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.stops:
        body["stop"] = list(req.stops)
    if req.json_mode:
        body["response_format"] = {"type": "json_object"}
    if req.temperature is not None:
        body["temperature"] = req.temperature
    return body


def build_upstream_anthropic(req, model_id):
    body = {"model": model_id,
            "messages": [{"role": "user", "content": turn_parts_anthropic(t)} for t in req.turns]}
    if req.system_text != "":
        body["system"] = req.system_text
    if req.tools:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["parameters"]} for t in req.tools]
    if req.max_tokens is not None:
        body["max_tokens"] = req.max_tokens
    if req.stops:
        body["stop_sequences"] = list(req.stops)
    if req.temperature is not None:
        body["temperature"] = req.temperature
    return body


def build_upstream_gemini(req, model_id):
    body = {"contents": [{"role": "user", "parts": turn_parts_gemini(t)} for t in req.turns]}
    if req.system_text != "":
        body["systemInstruction"] = {"parts": [{"text": req.system_text}]}
    if req.tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in req.tools]}]
    config = {}
    if req.max_tokens is not None:
        config["maxOutputTokens"] = req.max_tokens
    if req.stops:
        config["stopSequences"] = list(req.stops)
    if req.json_mode:
        config["responseMimeType"] = "application/json"
    if req.temperature is not None:
        config["temperature"] = req.temperature
    if config:
        body["generationConfig"] = config
    return body


def upstream_url(provider, model_id):
    base = provider.dial_base()
    if provider.wire == "openai":
        return f"{base}/v1/chat/completions"
    if provider.wire == "anthropic":
        return f"{base}/v1/messages"
    quoted = urllib.parse.quote(model_id, safe="")
    return f"{base}/v1beta/models/{quoted}:generateContent"


def build_upstream(provider, req, model_id):
    if provider.wire == "openai":
        return build_upstream_openai(req, model_id)
    if provider.wire == "anthropic":
        return build_upstream_anthropic(req, model_id)
    return build_upstream_gemini(req, model_id)


# --------------------------------------------------------------------------
# Upstream answer parsing: (text, usage_in, usage_out, stop_reason).


def parse_answer_openai(data):
    value = json.loads(data.decode("utf-8"))
    choice = value["choices"][0]
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(b.get("text", "") for b in content if is_record(b) and b.get("type") == "text")
    else:
        text = ""
    usage = value.get("usage") or {}
    return text, usage.get("prompt_tokens"), usage.get("completion_tokens"), choice.get("finish_reason")


def parse_answer_anthropic(data):
    value = json.loads(data.decode("utf-8"))
    text = "".join(b.get("text", "") for b in value.get("content", [])
                   if is_record(b) and b.get("type") == "text")
    usage = value.get("usage") or {}
    return text, usage.get("input_tokens"), usage.get("output_tokens"), value.get("stop_reason")


def parse_answer_gemini(data):
    value = json.loads(data.decode("utf-8"))
    candidate = value["candidates"][0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if is_record(p))
    usage = value.get("usageMetadata") or {}
    return text, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"), candidate.get("finishReason")


ANSWER_PARSERS = {"openai": parse_answer_openai, "anthropic": parse_answer_anthropic,
                  "gemini": parse_answer_gemini}


# --------------------------------------------------------------------------
# Client answer construction, in the client's wire, JSON or SSE.


def created_stamp():
    raw = os.environ.get("MAKEFUN_TRIAL_NOW", "")
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) >= 4:
        try:
            import datetime
            parsed = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return int(parsed.timestamp())
        except ValueError:
            pass
    return FALLBACK_CREATED


def split_pieces(text):
    pieces = [text[i:i + SSE_PIECE_CHARS] for i in range(0, len(text), SSE_PIECE_CHARS)]
    return pieces if pieces else [""]


def sse_frame(event, data):
    head = f"event: {event}\n" if event is not None else ""
    return f"{head}data: {json.dumps(data)}\n\n"


def answer_openai_json(model_id, text, usage_in, usage_out, stop_reason):
    return {
        "id": "chatcmpl-aggregator",
        "object": "chat.completion",
        "created": created_stamp(),
        "model": model_id,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                     "finish_reason": stop_reason, "logprobs": None}],
        "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                  "total_tokens": (usage_in or 0) + (usage_out or 0)},
    }


def answer_openai_sse(model_id, text, usage_in, usage_out, stop_reason):
    stamp = created_stamp()

    def chunk(delta, finish):
        return {"id": "chatcmpl-aggregator", "object": "chat.completion.chunk",
                "created": stamp, "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    frames = [sse_frame(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in split_pieces(text):
        frames.append(sse_frame(None, chunk({"content": piece}, None)))
    frames.append(sse_frame(None, chunk({}, stop_reason)))
    frames.append(sse_frame(None, {
        "id": "chatcmpl-aggregator", "object": "chat.completion.chunk", "created": stamp,
        "model": model_id, "choices": [],
        "usage": {"prompt_tokens": usage_in, "completion_tokens": usage_out,
                  "total_tokens": (usage_in or 0) + (usage_out or 0)}}))
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def answer_anthropic_json(model_id, text, usage_in, usage_out, stop_reason):
    return {
        "id": "msg_aggregator",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
    }


def answer_anthropic_sse(model_id, text, usage_in, usage_out, stop_reason):
    frames = [
        sse_frame("message_start", {
            "type": "message_start",
            "message": {"id": "msg_aggregator", "type": "message", "role": "assistant",
                        "model": model_id, "content": [], "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": usage_in, "output_tokens": 0}}}),
        sse_frame("content_block_start", {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}),
    ]
    for piece in split_pieces(text):
        frames.append(sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece}}))
    frames.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": usage_out}}))
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def build_answer(api, stream, model_id, text, usage_in, usage_out, stop_reason):
    """(content_type, body_text) in the client's wire."""
    if api == "openai":
        if stream:
            return "text/event-stream; charset=utf-8", answer_openai_sse(
                model_id, text, usage_in, usage_out, stop_reason)
        return "application/json", json.dumps(answer_openai_json(
            model_id, text, usage_in, usage_out, stop_reason))
    if stream:
        return "text/event-stream; charset=utf-8", answer_anthropic_sse(
            model_id, text, usage_in, usage_out, stop_reason)
    return "application/json", json.dumps(answer_anthropic_json(
        model_id, text, usage_in, usage_out, stop_reason))


def error_body(api, status, message):
    if api == "anthropic":
        kind = "overloaded_error" if status >= 503 else "api_error"
        return {"type": "error", "error": {"type": kind, "message": message}}
    kind = "api_error" if status >= 500 else "invalid_request_error"
    return {"error": {"message": message, "type": kind, "param": None, "code": "aggregator_error"}}


# --------------------------------------------------------------------------
# Routing.


def meets_requirements(model, requires):
    if requires.get("tools") is True and not model["tools"]:
        return False
    if requires.get("vision") is True and not model["vision"]:
        return False
    if requires.get("json") is True and not model["json"]:
        return False
    min_context = requires.get("min_context")
    if is_number(min_context):
        ctx = model["context_length"]
        if not is_number(ctx) or ctx < min_context:
            return False
    return True


def requirements_from_body(req):
    requires = {}
    if req.tools:
        requires["tools"] = True
    if any(turn["images"] for turn in req.turns):
        requires["vision"] = True
    if req.json_mode:
        requires["json"] = True
    return requires


class Router:
    def __init__(self, providers):
        self.providers = providers

    def candidates(self, req, requires):
        found = []
        for order, provider in enumerate(self.providers):
            # A JSON-mode request must not ride a wire that cannot carry it:
            # an Anthropic-wire provider would echo json_mode absent.
            if req.json_mode and provider.wire == "anthropic":
                continue
            for model in provider.models.values():
                if model["free"] is not True:
                    continue
                if not meets_requirements(model, requires):
                    continue
                found.append((provider, model, order))
        found.sort(key=lambda item: (
            1 if item[0].suspect_dead else 0,
            -item[1]["rating_milli"],
            item[0].calls_made,
            -BUDGET_PRIOR.get(item[0].id, 0),
            item[2],
            item[1]["id"],
        ))
        return [(provider, model) for provider, model, _order in found]

    def call(self, provider, req, model_id):
        """One upstream call on the provider's wire, serialized per provider.
        Returns (status, text, usage_in, usage_out, stop_reason) with the last
        four None unless the call answered 2xx with a parseable answer."""
        headers = provider.auth_headers()
        if provider.wire == "anthropic":
            headers["anthropic-version"] = "2023-06-01"
        body = build_upstream(provider, req, model_id)
        with provider.lock:
            try:
                status, _hdrs, data = http_request(
                    upstream_url(provider, model_id), method="POST", body=body,
                    headers=headers, timeout=UPSTREAM_TIMEOUT_S)
            except Exception:
                provider.note_call(False)
                return 0, None, None, None, None
            refused = status in (402, 429) or status >= 500
            provider.note_call(refused)
            if status < 200 or status >= 300:
                return status, None, None, None, None
            try:
                text, usage_in, usage_out, stop = ANSWER_PARSERS[provider.wire](data)
            except Exception:
                return status, None, None, None, None
            return status, text, usage_in, usage_out, stop


# --------------------------------------------------------------------------
# The HTTP server the env's client talks to.


class Aggregator:
    def __init__(self, brief, providers, catalog):
        self.brief = brief
        self.providers = providers
        self.catalog = catalog
        self.router = Router(providers)
        self.arrival = 0
        self.inflight = 0
        self.state_lock = threading.Lock()
        self.manifest_by_id = {row.get("id"): row for row in brief.manifest}

    def current_manifest_row(self):
        """The manifest row of the request now arriving: GET /work names it;
        the arrival counter is the fallback (the client sends the manifest
        in order, one request at a time)."""
        with self.state_lock:
            index = self.arrival
            self.arrival += 1
        try:
            status, _h, data = http_request(env_base() + "/work", timeout=WORK_TIMEOUT_S)
            if status == 200:
                work = json.loads(data.decode("utf-8"))
                row = self.manifest_by_id.get(work.get("next"))
                if row is not None:
                    return row
        except Exception:
            pass
        if index < len(self.brief.manifest):
            return self.brief.manifest[index]
        return None

    def serve_request(self, api, body):
        """Answer one inference request; returns (status, content_type, text)."""
        req = ClientRequest(api, body)
        row = self.current_manifest_row()
        if row is not None and row.get("api") == api:
            requires = row.get("requires") or {}
        else:
            requires = requirements_from_body(req)
        tried_providers = set()
        retried_providers = set()
        held_back = []
        attempts = 0
        plan = self.router.candidates(req, requires)
        pos = 0
        while pos < len(plan):
            if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                break
            provider, model = plan[pos]
            pos += 1
            if provider.id in tried_providers:
                # One fallback per refusing provider, used only if nothing
                # else is left: its next call number may succeed.
                held_back.append((provider, model))
                continue
            tried_providers.add(provider.id)
            attempts += 1
            status, text, usage_in, usage_out, stop = self.router.call(provider, req, model["id"])
            if text is not None:
                client_stop = client_stop_reason(api, stop)
                return 200, *build_answer(api, req.stream, model["id"], text,
                                          usage_in, usage_out, client_stop)
            if status in (400, 401):
                # A configuration error on our side: never a scored failure,
                # but the provider cannot help this request again.
                provider.refusals = max(provider.refusals, 2)
                continue            # A refusal spends one scheduled failure slot: when this model
            # still outranks whatever would answer instead, one retry on the
            # next call number recovers it more often than not.
            next_rating = plan[pos][1]["rating_milli"] if pos < len(plan) else None
            if (provider.id not in retried_providers
                    and attempts < MAX_ATTEMPTS_PER_REQUEST
                    and (next_rating is None or model["rating_milli"] > next_rating)):
                retried_providers.add(provider.id)
                attempts += 1
                status, text, usage_in, usage_out, stop = self.router.call(provider, req, model["id"])
                if text is not None:
                    client_stop = client_stop_reason(api, stop)
                    return 200, *build_answer(api, req.stream, model["id"], text,
                                              usage_in, usage_out, client_stop)
        for provider, model in held_back:
            if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                break
            attempts += 1
            status, text, usage_in, usage_out, stop = self.router.call(provider, req, model["id"])
            if text is not None:
                client_stop = client_stop_reason(api, stop)
                return 200, *build_answer(api, req.stream, model["id"], text,
                                          usage_in, usage_out, client_stop)
        return 503, "application/json", json.dumps(error_body(
            api, 503, "no free model could serve this request"))


def make_handler(aggregator):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def send_response(self, code, message=None):
            # Skip the Server and Date headers: nothing in an answer may
            # depend on the wall clock.
            self.send_response_only(code, message)

        def _send(self, status, content_type, text):
            payload = text.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _read_body(self):
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                return {}
            raw = self.rfile.read(min(length, 8 * 1024 * 1024))
            try:
                value = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}
            return value if is_record(value) else {}

        def do_GET(self):
            path = urllib.parse.urlsplit(self.path).path
            if path == "/v1/models":
                ids = free_model_ids(aggregator.providers)
                if self.headers.get("x-api-key") is not None:
                    body = {"data": [{"type": "model", "id": model_id, "display_name": model_id,
                                      "created_at": "2026-01-01T00:00:00Z"} for model_id in ids],
                            "has_more": False}
                else:
                    body = {"object": "list",
                            "data": [{"id": model_id, "object": "model",
                                      "created": FALLBACK_CREATED,
                                      "owned_by": "aggregator"} for model_id in ids]}
                self._send(200, "application/json", json.dumps(body))
                return
            self._send(404, "application/json", json.dumps(
                error_body("openai", 404, f"no route GET {path}")))

        def do_POST(self):
            path = urllib.parse.urlsplit(self.path).path
            api = {"/v1/chat/completions": "openai", "/v1/messages": "anthropic"}.get(path)
            if api is None:
                self._send(404, "application/json", json.dumps(
                    error_body("openai", 404, f"no route POST {path}")))
                return
            with aggregator.state_lock:
                aggregator.inflight += 1
            try:
                body = self._read_body()
                status, content_type, text = aggregator.serve_request(api, body)
                self._send(status, content_type, text)
            finally:
                with aggregator.state_lock:
                    aggregator.inflight -= 1

    return Handler


# --------------------------------------------------------------------------
# Register, poll, run.


def register(port, catalog):
    body = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    for attempt in range(6):
        try:
            status, _h, data = http_request(env_base() + "/register", method="POST",
                                            body=body, timeout=5.0)
            if status == 200:
                return True
            sys.stderr.write(f"aggregator: register answered {status}: {data[:200]!r}\n")
        except Exception as err:
            sys.stderr.write(f"aggregator: register attempt {attempt} failed: {err}\n")
        time.sleep(0.5)
    return False


def poll_until_done(aggregator, started):
    misses = 0
    while True:
        time.sleep(POLL_INTERVAL_S)
        if time.monotonic() - started > 170:
            return
        try:
            status, _h, data = http_request(env_base() + "/work", timeout=2.0)
            if status != 200:
                continue
            work = json.loads(data.decode("utf-8"))
            misses = 0
            if work.get("done") is True:
                with aggregator.state_lock:
                    idle = aggregator.inflight == 0
                if idle:
                    return
        except Exception:
            misses += 1
            with aggregator.state_lock:
                served_all = aggregator.arrival >= max(1, len(aggregator.brief.manifest))
                idle = aggregator.inflight == 0
            if misses >= 30 and served_all and idle:
                return


def main(argv):
    brief = Brief(os.environ.get("MAKEFUN_PROMPT_FILE", ""))
    providers = load_providers(brief)
    if env_base():
        probe_all(providers)
    catalog = build_catalog(providers)
    if "--list" in argv:
        sys.stdout.write(json.dumps(catalog) + "\n")
        sys.stdout.flush()
        return 0
    if not env_base():
        sys.stderr.write("aggregator: MAKEFUN_ENV_URL is not set\n")
        return 1
    aggregator = Aggregator(brief, providers, catalog)
    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    server.daemon_threads = True
    port = server.server_address[1]
    env_port = urllib.parse.urlsplit(env_base()).port
    if port == env_port:
        server.server_close()
        server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
        server.daemon_threads = True
        port = server.server_address[1]
    started = time.monotonic()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    if not register(port, catalog):
        sys.stderr.write("aggregator: could not register with the env\n")
        server.shutdown()
        return 1
    poll_until_done(aggregator, started)
    try:
        server.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
