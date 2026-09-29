#!/usr/bin/env python3
"""Native EXL3/DFlash2 HTTP adapter. SSE buffers a whole response, not tokens."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import queue
import re
import signal
import stat
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar, override

from jinja2 import TemplateError
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

if TYPE_CHECKING:
    import torch
    from exllamav3.generator.persist import PrefixStore
    from jsonschema.protocols import Validator

type JSON = str | int | float | bool | list[JSON] | dict[str, JSON] | None
MAX_BODY = 32 * 1024 * 1024
CONTEXT = 262144
CACHE_TOKENS = 270336
MODEL_NAME = "qwen3.8-27b"
FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
MARKUP = re.compile(r"</?(?:tool_call|function|parameter)(?:[=>]|\Z)")
SCHEMA_ANNOTATIONS = {
    "$schema",
    "$defs",
    "definitions",
    "$comment",
    "title",
    "description",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
}
SCHEMA_MAPS = {
    "$defs",
    "definitions",
    "properties",
    "patternProperties",
    "dependentSchemas",
}
SCHEMA_SINGLE = {
    "additionalProperties",
    "unevaluatedProperties",
    "items",
    "contains",
    "unevaluatedItems",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
}
SCHEMA_ARRAYS = {"allOf", "anyOf", "oneOf", "prefixItems"}
FORMAT_CHECKER = FormatChecker()
# Persistent prefix cache (engine generator/persist.py). Saved on SIGTERM and after 30 s idle, at most every 5 min.
PERSIST_ENV = "QWEN_PREFIX_PERSIST"
PERSIST_IDLE_SECONDS = 30
PERSIST_INTERVAL_SECONDS = 300
# Stop budget from SIGTERM to exit, inside the guardian's 45 s and Compose's 60 s grace.
STOP_BUDGET_SECONDS = 40
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
LAUNCH_GATE = "/maintenance-control/launch-gate.py"
ENGINE_MANIFEST = "/opt/qwen/exl3-patches.json"
MODEL_MANIFEST = "/model-preparation/exl3-manifest.json"
BINDING_ENV_PREFIXES = ("EXL3_", "QWEN_", "CUDA_", "PYTORCH_", "TORCH_", "NVIDIA_")


class APIError(Exception):
    """An HTTP-safe failure with an explicit status and machine-readable code."""

    def __init__(
        self, message: str, status: int = 400, code: str = "invalid_request"
    ) -> None:
        """Keep public error text separate from private native exceptions."""
        super().__init__(message)
        self.status = status
        self.code = code


def object_value(value: JSON, label: str) -> dict[str, JSON]:
    """Require an object without coercion.

    Returns:
        The validated object.

    Raises:
        APIError: The supplied value is not an object.
    """
    if not isinstance(value, dict):
        raise APIError(f"{label} must be an object")
    return value


def string_value(value: JSON, label: str) -> str:
    """Require a string without coercion.

    Returns:
        The validated string.

    Raises:
        APIError: The supplied value is not a string.
    """
    if not isinstance(value, str):
        raise APIError(f"{label} must be a string")
    return value


def boolean_value(value: JSON, label: str) -> bool:
    """Require a JSON boolean, excluding numeric stand-ins.

    Returns:
        The validated boolean.

    Raises:
        APIError: The supplied value is not a boolean.
    """
    if type(value) is not bool:
        raise APIError(f"{label} must be a boolean")
    return value


def positive_integer(value: JSON, label: str) -> int:
    """Require a positive integer, excluding booleans.

    Returns:
        The validated integer.

    Raises:
        APIError: The supplied value is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise APIError(f"{label} must be a positive integer")
    return value


def only_fields(value: dict[str, JSON], allowed: set[str], label: str) -> None:
    """Reject fields whose semantics this adapter does not implement.

    Raises:
        APIError: An unsupported field is present.
    """
    unknown = value.keys() - allowed
    if unknown:
        raise APIError(f"Unsupported {label} field: {sorted(unknown)[0]}")


def json_value(value: object) -> JSON:
    """Rebuild decoded data as finite, string-keyed JSON values.

    Returns:
        The validated JSON value.

    Raises:
        ValueError: A non-JSON value or nonfinite number is present.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, JSON] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON object keys must be strings")
            result[key] = json_value(item)
        return result
    raise ValueError("Expected finite JSON values")


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build an object without silently overwriting duplicate JSON keys.

    Returns:
        The decoded key-value mapping.

    Raises:
        ValueError: A key appears more than once.
    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def load_json(text: str) -> JSON:
    """Decode JSON with duplicate-key and nonfinite-number rejection.

    Returns:
        The validated JSON value.
    """
    return json_value(json.loads(text, object_pairs_hook=unique_object))


def parameter_name(value: str) -> None:
    """Require a name representable in the model's parameter tag.

    Raises:
        APIError: The name is empty or contains structural delimiters.
    """
    if not value or any(char in value for char in "<>\r\n"):
        raise APIError(
            "Parameter names must be nonempty and cannot contain markup delimiters"
        )


def check_schema_nodes(schema: JSON, root: dict[str, JSON]) -> None:
    """Reject unsupported schema features without fetching client-supplied URIs.

    Raises:
        APIError: A dialect, reference, format, or keyword is unsupported.
    """
    if isinstance(schema, bool):
        return
    node = object_value(schema, "JSON Schema")
    allowed = set(Draft202012Validator.VALIDATORS) | SCHEMA_ANNOTATIONS
    only_fields(node, allowed, "JSON Schema")
    if (
        "$schema" in node
        and node["$schema"] != "https://json-schema.org/draft/2020-12/schema"
    ):
        raise APIError("Only JSON Schema draft 2020-12 is supported")
    # No resource identifiers, remote references, or dynamic scope: all references
    # must be JSON pointers into this one request's parameters object.
    if "$dynamicRef" in node:
        raise APIError("$dynamicRef is unsupported; use local $ref JSON pointers")
    if "$ref" in node:
        ref = string_value(node["$ref"], "$ref")
        if ref != "#" and not ref.startswith("#/"):
            raise APIError("Only local JSON-pointer $ref values are supported")
        target: JSON = root
        if ref != "#":
            for part in ref[2:].split("/"):
                key = part.replace("~1", "/").replace("~0", "~")
                if isinstance(target, dict) and key in target:
                    target = target[key]
                elif (
                    isinstance(target, list)
                    and key.isascii()
                    and key.isdecimal()
                    and (key == "0" or not key.startswith("0"))
                    and int(key) < len(target)
                ):
                    target = target[int(key)]
                else:
                    raise APIError("Unresolvable local JSON Schema $ref")
        if not isinstance(target, (dict, bool)):
            raise APIError("JSON Schema $ref must resolve to a schema")
    if "format" in node:
        fmt = string_value(node["format"], "format")
        if fmt not in FORMAT_CHECKER.checkers:
            raise APIError(f"Unsupported JSON Schema format: {fmt}")
    for keyword in SCHEMA_MAPS & node.keys():
        children = object_value(node[keyword], keyword)
        for name, child in children.items():
            if keyword == "properties":
                parameter_name(name)
            check_schema_nodes(child, root)
    for keyword in SCHEMA_SINGLE & node.keys():
        check_schema_nodes(node[keyword], root)
    for keyword in SCHEMA_ARRAYS & node.keys():
        children = node[keyword]
        if not isinstance(children, list):
            raise APIError(f"{keyword} must be an array")
        for child in children:
            check_schema_nodes(child, root)


@dataclass
class Tool:
    """A declared function and the validator governing its returned arguments."""

    name: str
    wire: dict[str, JSON]
    schema: dict[str, JSON]
    validator: Validator

    def parameter(self, name: str, raw: str) -> JSON:
        """Decode one parameter using its schema and the template's raw strings.

        Returns:
            A schema-compatible JSON value, preferring valid raw strings.

        Raises:
            APIError: Generated JSON is malformed or violates its schema.
            RuntimeError: Validated schema state is internally inconsistent.
        """
        properties = self.schema.get("properties", {})
        patterns = self.schema.get("patternProperties", {})
        if not isinstance(properties, dict) or not isinstance(patterns, dict):
            raise RuntimeError("Validated schema has invalid properties")
        schemas: list[JSON] = []
        if name in properties:
            schemas.append(properties[name])
        for pattern, schema in patterns.items():
            if re.search(pattern, name):
                schemas.append(schema)
        if not schemas:
            schemas.append(self.schema.get("additionalProperties", True))
        validator = self.validator.evolve(schema={"allOf": schemas})
        # The actual template renders strings verbatim (not JSON quoted). Prefer
        # that interpretation for string/nonstring unions, preserving e.g. "001".
        if validator.is_valid(raw):
            return raw
        try:
            value = load_json(raw)
        except (ValueError, RecursionError) as exc:
            raise APIError(
                f"Model emitted invalid JSON for {self.name}.{name}",
                502,
                "invalid_tool_arguments",
            ) from exc
        if not validator.is_valid(value):
            raise APIError(
                f"Model argument violates schema: {self.name}.{name}",
                502,
                "invalid_tool_arguments",
            )
        return value


def parse_tools(value: JSON) -> dict[str, Tool]:
    """Validate function declarations and construct local-only schema validators.

    Returns:
        Functions indexed by their unique declared names.

    Raises:
        APIError: A tool definition or schema is invalid or unsupported.
    """
    if not isinstance(value, list):
        raise APIError("tools must be an array")
    tools: dict[str, Tool] = {}
    for item in value:
        wire = object_value(item, "tool")
        only_fields(wire, {"type", "function"}, "tool")
        if wire.get("type") != "function":
            raise APIError("Only function tools are supported")
        function = object_value(wire.get("function"), "tool.function")
        only_fields(
            function, {"name", "description", "parameters", "strict"}, "tool.function"
        )
        name = string_value(function.get("name"), "tool.function.name")
        if not FUNCTION_NAME.fullmatch(name) or name in tools:
            raise APIError(
                "Tool names must be unique, 1..64 ASCII letters/digits/underscores/hyphens"
            )
        if "description" in function:
            string_value(function["description"], "tool.function.description")
        if "strict" in function:
            # Strictness is a response postcondition, not constrained decoding:
            # every returned call is schema-valid, or the request fails with 502.
            boolean_value(function["strict"], "tool.function.strict")
        schema = object_value(
            function.get(
                "parameters",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            "tool.function.parameters",
        )
        # A direct object root makes the XML parameter-to-JSON type mapping
        # unambiguous. Nested schemas have the full admitted draft vocabulary.
        if schema.get("type") != "object":
            raise APIError("Tool parameters must declare type: object")
        if set(schema) & {
            "$ref",
            "allOf",
            "anyOf",
            "oneOf",
            "if",
            "then",
            "else",
            "dependentSchemas",
        }:
            raise APIError(
                "Tool parameter root composition is unsupported; put schemas in properties"
            )
        try:
            Draft202012Validator.check_schema(schema)
            check_schema_nodes(schema, schema)
            registry = Registry().with_resource(
                "urn:exl3:parameters", DRAFT202012.create_resource(schema)
            )
            validator = Draft202012Validator(
                schema, registry=registry, format_checker=FORMAT_CHECKER
            )
        except (SchemaError, re.error, RecursionError) as exc:
            raise APIError("Invalid tool JSON Schema") from exc
        tools[name] = Tool(name, wire, schema, validator)
    return tools


def text_content(value: JSON, label: str, nullable: bool = False) -> JSON:
    """Validate text-only content, preserving the caller's representation.

    Returns:
        Text, text parts, or an explicitly permitted null value.

    Raises:
        APIError: Content is malformed or requires unsupported modalities.
    """
    if isinstance(value, str) or (nullable and value is None):
        return value
    if isinstance(value, list):
        for part in value:
            part = object_value(part, label)
            only_fields(part, {"type", "text"}, "text content")
            if part.get("type") != "text":
                raise APIError("Only text content parts are supported")
            string_value(part.get("text"), "content.text")
        return value
    raise APIError(f"{label} must be text or an array of text parts")


def parse_messages(value: JSON) -> list[dict[str, JSON]]:
    """Normalize tool arguments and associate each tool result with its call ID.

    Returns:
        Template-ready messages with results in assistant call order.

    Raises:
        APIError: Message structure, tool history, or arguments are invalid.
    """
    if not isinstance(value, list) or not value:
        raise APIError("messages must be a nonempty array")
    messages: list[dict[str, JSON]] = []
    seen_ids: set[str] = set()
    pending: dict[str, str] = {}
    results: dict[str, dict[str, JSON]] = {}
    user_found = False
    for index, item in enumerate(value):
        message = object_value(item, "message")
        role = message.get("role")
        if role == "tool":
            only_fields(
                message, {"role", "content", "tool_call_id", "name"}, "tool message"
            )
            call_id = string_value(message.get("tool_call_id"), "tool_call_id")
            if call_id not in pending or call_id in results:
                raise APIError(
                    "Tool result must match one unresolved assistant tool_call_id"
                )
            if "name" in message and message["name"] != pending[call_id]:
                raise APIError("Tool result name does not match its tool_call_id")
            results[call_id] = {
                **message,
                "name": pending[call_id],
                "content": text_content(message.get("content"), "tool content"),
            }
            if len(results) == len(pending):
                # Qwen's actual template omits IDs and names; order the responses
                # by the prior call IDs rather than associating results by arrival.
                messages.extend(results[call_id] for call_id in pending)
                pending = {}
                results = {}
            continue
        if pending:
            raise APIError(
                "All assistant tool calls need contiguous tool results before the next message"
            )
        if role not in ("system", "user", "assistant"):
            raise APIError("Supported message roles: system, user, assistant, tool")
        allowed = {"role", "content"}
        if role == "assistant":
            allowed |= {"tool_calls", "reasoning_content"}
        only_fields(message, allowed, "message")
        if role == "system" and index != 0:
            raise APIError("System message must be first")
        if role == "user":
            user_found = True
        normalized = dict(message)
        normalized["content"] = text_content(
            message.get("content"), "message.content", role == "assistant"
        )
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            if reasoning is not None:
                string_value(reasoning, "reasoning_content")
            if "tool_calls" in message:
                calls = message["tool_calls"]
                if not isinstance(calls, list) or not calls:
                    raise APIError("assistant.tool_calls must be a nonempty array")
                normalized_calls: list[JSON] = []
                for call in calls:
                    call = object_value(call, "assistant tool call")
                    only_fields(call, {"id", "type", "function"}, "assistant tool call")
                    call_id = string_value(call.get("id"), "assistant tool call id")
                    if not call_id or call_id in seen_ids:
                        raise APIError(
                            "Assistant tool call IDs must be nonempty and unique"
                        )
                    if call.get("type") != "function":
                        raise APIError("Only function tool calls are supported")
                    function = object_value(
                        call.get("function"), "assistant tool call function"
                    )
                    only_fields(
                        function, {"name", "arguments"}, "assistant tool call function"
                    )
                    name = string_value(
                        function.get("name"), "assistant tool function name"
                    )
                    if not FUNCTION_NAME.fullmatch(name):
                        raise APIError("Invalid assistant tool function name")
                    raw = string_value(
                        function.get("arguments"), "assistant tool arguments"
                    )
                    try:
                        arguments = object_value(
                            load_json(raw), "assistant tool arguments JSON"
                        )
                    except (ValueError, RecursionError) as exc:
                        raise APIError(
                            "Assistant tool arguments must be a valid JSON object string"
                        ) from exc
                    for key in arguments:
                        parameter_name(key)
                    normalized_calls.append({
                        **call,
                        "function": {"name": name, "arguments": arguments},
                    })
                    seen_ids.add(call_id)
                    pending[call_id] = name
                normalized["tool_calls"] = normalized_calls
            elif normalized["content"] is None:
                raise APIError("Assistant content may be null only with tool_calls")
        messages.append(normalized)
    if pending:
        raise APIError("All assistant tool calls need tool results before generation")
    if not user_found:
        raise APIError("At least one user message is required")
    return messages


@dataclass
class Options:
    """Validated generation budget and response transport options."""

    max_tokens: int
    stream: bool
    include_usage: bool
    skip_special_tokens: bool


# Sampling controls a greedy client may send; each is admitted only at the value that
# leaves argmax decoding unchanged, never silently ignored at any other value.
GREEDY_IDENTITY: dict[str, int] = {
    "top_p": 1,
    "min_p": 0,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "repetition_penalty": 1,
}


def parse_options(body: dict[str, JSON], model_name: str, chat: bool) -> Options:
    """Validate model selection, greedy generation, limits, and transport options.

    Returns:
        Options shared by rendering and native generation.

    Raises:
        APIError: An option is invalid, conflicting, or unsupported.
    """
    model = body.get("model", model_name)
    if not isinstance(model, str) or model != model_name:
        raise APIError(f"model must be {model_name}")
    if "max_tokens" in body and "max_completion_tokens" in body:
        raise APIError("Specify only one of max_tokens and max_completion_tokens")
    limit = body.get(
        "max_completion_tokens", body.get("max_tokens", 4096 if chat else 256)
    )
    max_tokens = positive_integer(limit, "output token limit")
    temperature = body.get("temperature", 0)
    if type(temperature) not in (int, float) or temperature != 0:
        raise APIError("temperature must be numeric zero (greedy)")
    for name, identity in GREEDY_IDENTITY.items():
        if name in body and (
            type(body[name]) not in (int, float) or body[name] != identity
        ):
            raise APIError(f"{name} must be numeric {identity} (greedy identity)")
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        raise APIError("Only n=1 is supported")
    stream = boolean_value(body.get("stream", False), "stream")
    skip = boolean_value(body.get("skip_special_tokens", True), "skip_special_tokens")
    include_usage = False
    if "stream_options" in body:
        options = object_value(body["stream_options"], "stream_options")
        only_fields(options, {"include_usage"}, "stream_options")
        include_usage = boolean_value(
            options.get("include_usage", False), "stream_options.include_usage"
        )
        if not stream:
            raise APIError("stream_options requires stream=true")
    if not chat and stream:
        raise APIError("Streaming raw completions is unsupported")
    return Options(max_tokens, stream, include_usage, skip)


@dataclass
class Chat:
    """Validated history, rendering inputs, and output tool-choice postconditions."""

    messages: list[dict[str, JSON]]
    template_kwargs: dict[str, JSON]
    tools: dict[str, Tool]
    choice: str
    parallel: bool
    thinking: bool


def parse_chat(body: dict[str, JSON]) -> Chat:
    """Prepare the actual model template inputs and explicit tool-choice policy.

    Returns:
        A complete chat request ready for rendering.

    Raises:
        APIError: Tool choice or template options are invalid.
        RuntimeError: Validated system content is internally inconsistent.
    """
    tools = parse_tools(body.get("tools", []))
    choice_value = body.get("tool_choice", "auto" if tools else "none")
    if isinstance(choice_value, str) and choice_value in ("auto", "none", "required"):
        choice = choice_value
    elif isinstance(choice_value, dict):
        only_fields(choice_value, {"type", "function"}, "tool_choice")
        if choice_value.get("type") != "function":
            raise APIError("Named tool_choice must have type=function")
        function = object_value(choice_value.get("function"), "tool_choice.function")
        only_fields(function, {"name"}, "tool_choice.function")
        name = string_value(function.get("name"), "tool_choice.function.name")
        if name not in tools:
            raise APIError("Named tool_choice must name a supplied tool")
        choice = "named:" + name
    else:
        raise APIError(
            "tool_choice must be auto, none, required, or a named function object"
        )
    if choice != "none" and not tools:
        raise APIError("This tool_choice requires nonempty tools")
    parallel = boolean_value(
        body.get("parallel_tool_calls", True), "parallel_tool_calls"
    )
    messages = parse_messages(body.get("messages"))
    kwargs = object_value(body.get("chat_template_kwargs", {}), "chat_template_kwargs")
    only_fields(
        kwargs,
        {"enable_thinking", "preserve_thinking", "reasoning_effort"},
        "chat_template_kwargs",
    )
    thinking = boolean_value(kwargs.get("enable_thinking", True), "enable_thinking")
    if "preserve_thinking" in kwargs:
        boolean_value(kwargs["preserve_thinking"], "preserve_thinking")
    effort = kwargs.get("reasoning_effort", "xhigh")
    if not isinstance(effort, str) or effort not in ("xhigh", "medium", "low"):
        raise APIError("reasoning_effort must be xhigh, medium, or low")
    if not thinking and "reasoning_effort" in kwargs:
        raise APIError("reasoning_effort requires enable_thinking=true")
    template_kwargs = dict(kwargs)
    if "reasoning_effort" in body:
        requested_effort = body["reasoning_effort"]
        if not isinstance(requested_effort, str) or requested_effort not in (
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ):
            raise APIError(
                "reasoning_effort must be none, minimal, low, medium, high, xhigh, or max"
            )
        requested_thinking = requested_effort != "none"
        if "enable_thinking" in kwargs and thinking != requested_thinking:
            raise APIError("reasoning_effort conflicts with enable_thinking")
        if requested_effort == "minimal":
            normalized_effort = "low"
        elif requested_effort in ("high", "max"):
            normalized_effort = "xhigh"
        else:
            normalized_effort = requested_effort
        if "reasoning_effort" in kwargs and (
            not requested_thinking or effort != normalized_effort
        ):
            raise APIError(
                "reasoning_effort conflicts with chat_template_kwargs.reasoning_effort"
            )
        thinking = requested_thinking
        template_kwargs["enable_thinking"] = thinking
        if thinking:
            template_kwargs["reasoning_effort"] = normalized_effort
    instructions: list[str] = []
    if choice == "none" and ("tool_choice" in body or tools):
        instructions.append(
            "Do not call functions. Answer without function-call markup."
        )
    elif choice == "required":
        instructions.append(
            "You must call at least one of the provided functions in this response."
        )
    elif choice.startswith("named:"):
        instructions.append(
            f"You must call only the function {choice[6:]} in this response."
        )
    if not parallel and tools and choice != "none":
        instructions.append("Do not call more than one function in this response.")
    if instructions:
        policy = "\n\nTool choice for this response: " + " ".join(instructions)
        if messages[0]["role"] == "system":
            content = messages[0]["content"]
            if isinstance(content, str):
                messages[0] = {**messages[0], "content": content + policy}
            elif isinstance(content, list):
                messages[0] = {
                    **messages[0],
                    "content": [*content, {"type": "text", "text": policy}],
                }
            else:
                raise RuntimeError("Validated system message lacks text")
        else:
            messages.insert(0, {"role": "system", "content": policy.lstrip()})
    template_kwargs["tools"] = (
        [] if choice == "none" else [tool.wire for tool in tools.values()]
    )
    return Chat(messages, template_kwargs, tools, choice, parallel, thinking)


def split_reasoning(text: str, thinking: bool) -> tuple[str | None, str]:
    """Keep reasoning separate, including an unfinished thinking-only response.

    Returns:
        Optional reasoning text and the remaining assistant content.
    """
    if thinking:
        reasoning, separator, content = text.partition("</think>")
        if reasoning.startswith("<think>"):
            reasoning = reasoning[len("<think>") :].lstrip("\n")
        return reasoning, content.lstrip("\n ") if separator else ""
    if text.startswith("<think>"):
        reasoning, separator, content = text[len("<think>") :].partition("</think>")
        return reasoning.lstrip("\n"), content.lstrip("\n ") if separator else ""
    return None, text


def parse_tool_output(
    content: str, chat: Chat, finish: str
) -> tuple[str, list[dict[str, JSON]]]:
    """Decode complete Qwen calls, withholding all calls on a length-ended turn.

    Returns:
        Preserved leading content and validated OpenAI function calls.

    Raises:
        APIError: Model markup, arguments, or tool-choice postconditions fail.
    """
    # A budget-ended response may include one complete call followed by a partial
    # second call. Dispatch neither: the entire model turn must be complete.
    if finish == "length":
        return content, []
    start = content.find("<tool_call>")
    if start < 0:
        if MARKUP.search(content):
            raise APIError(
                "Model emitted malformed tool markup", 502, "invalid_tool_call"
            )
        if chat.choice == "required" or chat.choice.startswith("named:"):
            raise APIError(
                "Model did not satisfy required tool_choice",
                502,
                "tool_choice_not_satisfied",
            )
        return content, []
    if chat.choice == "none":
        raise APIError(
            "Model emitted a tool call while tool_choice=none",
            502,
            "tool_choice_not_satisfied",
        )
    if MARKUP.search(content[:start]):
        raise APIError(
            "Model emitted malformed markup before a tool call",
            502,
            "invalid_tool_call",
        )
    calls: list[dict[str, JSON]] = []
    cursor = start

    def consume(token: str) -> None:
        nonlocal cursor
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if not content.startswith(token, cursor):
            raise APIError(
                "Model emitted incomplete or malformed tool markup",
                502,
                "invalid_tool_call",
            )
        cursor += len(token)

    def tag_name(prefix: str) -> str:
        nonlocal cursor
        consume(prefix)
        end = content.find(">", cursor)
        if end < 0:
            raise APIError(
                "Model emitted an incomplete tool tag", 502, "invalid_tool_call"
            )
        name = content[cursor:end]
        cursor = end + 1
        return name

    while cursor < len(content):
        consume("<tool_call>")
        name = tag_name("<function=")
        if name not in chat.tools:
            raise APIError(
                "Model called an undeclared function", 502, "invalid_tool_call"
            )
        if chat.choice.startswith("named:") and name != chat.choice[6:]:
            raise APIError(
                "Model did not satisfy named tool_choice",
                502,
                "tool_choice_not_satisfied",
            )
        arguments: dict[str, JSON] = {}
        while True:
            while cursor < len(content) and content[cursor].isspace():
                cursor += 1
            if content.startswith("</function>", cursor):
                break
            key = tag_name("<parameter=")
            if not key or any(char in key for char in "<>\r\n") or key in arguments:
                raise APIError(
                    "Model emitted invalid or duplicate parameter names",
                    502,
                    "invalid_tool_arguments",
                )
            end = content.find("</parameter>", cursor)
            if end < 0:
                raise APIError(
                    "Model emitted an incomplete parameter",
                    502,
                    "invalid_tool_arguments",
                )
            raw = content[cursor:end]
            if MARKUP.search(raw):
                raise APIError(
                    "Model emitted ambiguous nested tool markup",
                    502,
                    "invalid_tool_arguments",
                )
            if raw.startswith("\r\n"):
                raw = raw[2:]
            elif raw.startswith("\n"):
                raw = raw[1:]
            if raw.endswith("\r\n"):
                raw = raw[:-2]
            elif raw.endswith("\n"):
                raw = raw[:-1]
            try:
                arguments[key] = chat.tools[name].parameter(key, raw)
            except (Unresolvable, RecursionError) as exc:
                raise APIError(
                    "Tool schema reference cannot be evaluated", 400, "invalid_schema"
                ) from exc
            cursor = end + len("</parameter>")
        consume("</function>")
        consume("</tool_call>")
        try:
            valid = chat.tools[name].validator.is_valid(arguments)
        except (Unresolvable, RecursionError) as exc:
            raise APIError(
                "Tool schema reference cannot be evaluated", 400, "invalid_schema"
            ) from exc
        if not valid:
            raise APIError(
                f"Model arguments violate the schema for {name}",
                502,
                "invalid_tool_arguments",
            )
        calls.append({
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False),
            },
        })
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1
        if cursor < len(content) and not content.startswith("<tool_call>", cursor):
            raise APIError(
                "Model emitted text after a tool call", 502, "invalid_tool_call"
            )
    if not chat.parallel and len(calls) > 1:
        raise APIError(
            "Model violated parallel_tool_calls=false", 502, "tool_choice_not_satisfied"
        )
    return content[:start], calls


def speculative_counts(
    final: dict[str, object], window: int, completion_tokens: int
) -> tuple[int, int]:
    """Derive verify rounds and their committed tokens from native draft counters.

    Every verify round resolves each of its fixed `window` drafted positions exactly
    once (accepted, rejected, or unresolved after EOS) and commits one target token
    beyond its accepted drafts: the corrective, bonus, or terminal token.

    Returns:
        Verify rounds and the tokens those rounds committed.

    Raises:
        RuntimeError: The native draft counters are absent or inconsistent.
    """
    accepted = final.get("accepted_draft_tokens")
    rejected = final.get("rejected_draft_tokens")
    if type(accepted) is not int or type(rejected) is not int:
        raise RuntimeError("Native terminal event lacks draft accounting")
    if accepted < 0 or rejected < 0:
        raise RuntimeError("Native draft accounting is negative")
    rounds, remainder = divmod(accepted + rejected, window)
    if remainder:
        raise RuntimeError("Native draft accounting is not whole verify rounds")
    committed = accepted + rounds
    if committed > completion_tokens:
        raise RuntimeError("Native draft accounting exceeds the completion")
    return rounds, committed


@dataclass
class Result:
    """A complete native generation and its engine-reported token accounting."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    spec_rounds: int
    spec_committed: int

    def usage(self) -> dict[str, JSON]:
        """Expose native token counts without inferring counts from decoded text.

        Returns:
            OpenAI-compatible prompt, completion, and total token counts, plus the
            request's speculative verify rounds and the tokens they committed.
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "exl3_spec": {
                "rounds": self.spec_rounds,
                "committed": self.spec_committed,
            },
        }


@dataclass
class Pending:
    """One FIFO generation request and its cross-thread completion state."""

    input_ids: torch.Tensor
    options: Options
    identifier: str = field(default_factory=lambda: uuid.uuid4().hex)
    event: threading.Event = field(default_factory=threading.Event)
    result: Result | None = None
    error: Exception | None = None


class ShuttingDown(Exception):
    """The job was cancelled, or never started, because the server is stopping."""


def env_flag(name: str, default: bool) -> bool:
    """Read a strict 0/1 environment switch.

    Raises:
        ValueError: The variable holds anything but 0 or 1.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1")
    return value == "1"


def env_seconds(name: str, default: int) -> int:
    """Read a strict whole-second duration (test hook for the save schedule).

    Raises:
        ValueError: The variable is not an integer in 0..86400.
    """
    value = os.environ.get(name)
    if value is None:
        return default
    if not value.isdigit() or int(value) > 86400:
        raise ValueError(f"{name} must be whole seconds in 0..86400")
    return int(value)


def launch_image() -> str | None:
    """Identify the running image without trusting request data.

    The guardian's launch gate is the container entrypoint and receives the verified image
    ID as --candidate-image; docker-init (PID 1) keeps that argv. QWEN_IMAGE_ID serves
    launches without the gate and must agree with the gate when both exist.

    Returns:
        The full image ID, or None when it is unknown or contradictory.
    """
    gate = None
    try:
        with open("/proc/1/cmdline", "rb") as source:
            argv = [a.decode() for a in source.read().split(b"\0")]
    except (OSError, UnicodeDecodeError):
        argv = []
    if LAUNCH_GATE in argv and "--candidate-image" in argv:
        at = argv.index("--candidate-image") + 1
        gate = argv[at] if at < len(argv) and IMAGE_ID.fullmatch(argv[at]) else None
        if gate is None:
            return None
    env = os.environ.get("QWEN_IMAGE_ID") or None
    if env is not None and IMAGE_ID.fullmatch(env) is None:
        return None
    if gate is not None and env is not None and gate != env:
        return None
    return gate or env


def file_digest(path: str) -> str | None:
    """Hash one file, or None if it does not exist."""
    try:
        with open(path, "rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except FileNotFoundError:
        return None


def prefix_binding(args: argparse.Namespace, torch_: ModuleType) -> dict[str, JSON] | None:
    """Everything outside the cache geometry that persisted K/V and recurrent bytes depend on.

    Returns:
        The JSON binding, or None when the image cannot be identified.
    """
    image = launch_image()
    if image is None:
        return None
    try:
        with open("/proc/driver/nvidia/version", encoding="utf-8") as source:
            driver = source.readline().strip()
    except OSError:
        driver = None
    return {
        "image": image,
        "files": {
            path: file_digest(path)
            for path in (
                ENGINE_MANIFEST,
                MODEL_MANIFEST,
                os.path.abspath(__file__),
                os.path.join(args.target, "config.json"),
                os.path.join(args.draft, "config.json"),
            )
        },
        "args": {
            "target": args.target,
            "draft": args.draft,
            "model_name": args.model_name,
            "max_model_len": args.max_model_len,
            "cache_tokens": args.cache_tokens,
            "cq": args.cq,
        },
        "runtime": {
            "torch": str(torch_.__version__),
            "cuda": str(torch_.version.cuda),
            "gpu": str(torch_.cuda.get_device_name(0)),
            "capability": list(torch_.cuda.get_device_capability(0)),
            "driver": driver,
        },
        "env": {
            name: value
            for name, value in sorted(os.environ.items())
            if name.startswith(BINDING_ENV_PREFIXES)
            and not name.startswith(PERSIST_ENV)
            and name != "QWEN_IMAGE_ID"
        },
    }


class Server:
    """Own the fixed native EXL3/DFlash2 stack and its single generation worker."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Load the approved cache geometry before exposing the worker."""
        import torch
        from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
        from exllamav3.cache import CacheLayer_quant
        from exllamav3.generator.sampler.presets import ArgmaxSampler

        self.torch = torch
        self.job_type = Job
        self.sampler_type = ArgmaxSampler
        self.model_name = args.model_name
        self.max_model_len = args.max_model_len
        target_config = Config.from_directory(args.target)
        draft_config = Config.from_directory(args.draft)
        draft_model = Model.from_config(draft_config)
        max_history = draft_model.caps.get("default_draft_size", 4)
        model = Model.from_config(target_config)
        print(
            f"[serve] loading native EXL3 target, cache={args.cache_tokens}, cq={args.cq}",
            flush=True,
        )
        start = time.monotonic()
        cache = Cache(
            model,
            max_num_tokens=args.cache_tokens,
            layer_type=CacheLayer_quant,
            k_bits=args.cq,
            v_bits=args.cq,
            max_history=max_history,
            max_batch_size=1,
        )
        model.load(progressbar=False)
        self.tokenizer = Tokenizer.from_config(target_config)
        draft_cache = Cache(
            draft_model,
            max_num_tokens=args.cache_tokens,
            layer_type=CacheLayer_quant,
            k_bits=args.cq,
            v_bits=args.cq,
            max_batch_size=1,
        )
        draft_model.load(progressbar=False)
        self.gen = Generator(
            model,
            cache,
            self.tokenizer,
            draft_model=draft_model,
            draft_cache=draft_cache,
        )
        # Fixed (non-dynamic) verify window; usage accounting divides by it.
        self.draft_window: int = self.gen.num_draft_tokens
        if type(self.draft_window) is not int or self.draft_window != max_history:
            raise RuntimeError("Generator draft window differs from the cache history")
        # The candidate engine commits greedy verify rounds through its admitted acceptance
        # decision and counts them; the baseline image must carry the unpatched engine.
        variant = os.environ.get("QWEN_EXL3_VARIANT")
        if variant not in ("baseline", "candidate"):
            raise RuntimeError("QWEN_EXL3_VARIANT must name the baked image variant")
        self.verified_acceptance = variant == "candidate"
        if self.verified_acceptance != hasattr(self.gen, "greedy_verify_rounds"):
            raise RuntimeError("Installed engine does not match the image variant")
        # Dynamic tree verify (exl3 0006): the candidate engine parses EXL3_TREE and the test hook
        # EXL3_TREE_FORCE_CHAIN strictly (0|1) and counts tree rounds; the baseline engine has no
        # tree, so both must be unset or 0 there.
        if self.verified_acceptance:
            if not hasattr(self.gen, "tree_verify_rounds"):
                raise RuntimeError("Installed engine lacks the tree verify (exl3 0006)")
            self.tree_rounds_expected = bool(self.gen.tree and not self.gen.tree_force_chain)
        else:
            for name in ("EXL3_TREE", "EXL3_TREE_FORCE_CHAIN"):
                if os.environ.get(name) not in (None, "0"):
                    raise RuntimeError(f"{name} requires the candidate engine")
            self.tree_rounds_expected = False
        self.stop_ids = list(model.config.eos_token_id_list or [])
        if (
            self.tokenizer.eos_token_id is not None
            and self.tokenizer.eos_token_id not in self.stop_ids
        ):
            self.stop_ids.append(self.tokenizer.eos_token_id)
        self.tokenizer_lock = threading.Lock()
        # None is the stop sentinel queued by begin_shutdown.
        self.queue: queue.Queue[Pending | None] = queue.Queue()
        self.failure: Exception | None = None
        self.stopping = threading.Event()
        self.stopped = threading.Event()
        self.stop_deadline = math.inf
        self.persist_debug = env_flag("QWEN_PREFIX_PERSIST_DEBUG", False)
        self.persist_idle = env_seconds("QWEN_PREFIX_PERSIST_IDLE_SECONDS", PERSIST_IDLE_SECONDS)
        self.persist_interval = env_seconds(
            "QWEN_PREFIX_PERSIST_INTERVAL_SECONDS", PERSIST_INTERVAL_SECONDS
        )
        self.persist_dirty = False
        self.last_job_end = -math.inf
        self.last_save = -math.inf
        self.persist = self.open_prefix_cache(args) if args.prefix_cache else None
        threading.Thread(target=self._worker, daemon=True).start()
        free, total = torch.cuda.mem_get_info()
        print(
            f"[serve] READY in {time.monotonic() - start:.0f}s, VRAM={(total - free) / 1e9:.2f} GB",
            flush=True,
        )

    def check_context(self, ids: torch.Tensor, output_tokens: int = 0) -> None:
        """Enforce one sequence and the complete input-plus-output context budget.

        Raises:
            APIError: Shape, input length, or the native context limit is invalid.
        """
        if ids.ndim not in (1, 2) or (ids.ndim == 2 and ids.shape[0] != 1):
            raise APIError("Exactly one input sequence is required")
        length = ids.shape[-1]
        if length <= 0 or length + output_tokens > self.max_model_len:
            raise APIError(
                "Input plus output budget exceeds native 262144 context",
                code="context_length_exceeded",
            )

    def render_chat(self, chat: Chat) -> torch.Tensor:
        """Render through the target tokenizer's actual Hugging Face template.

        Returns:
            The rendered token-ID tensor.

        Raises:
            APIError: The model template cannot render the supplied messages.
            RuntimeError: The tokenizer does not return its documented tensor type.
        """
        with self.tokenizer_lock:
            try:
                ids: object = self.tokenizer.hf_chat_template(
                    chat.messages, add_generation_prompt=True, **chat.template_kwargs
                )
            except (ValueError, TypeError, TemplateError) as exc:
                raise APIError(
                    "Messages could not be rendered by the model chat template"
                ) from exc
        if not isinstance(ids, self.torch.Tensor):
            raise RuntimeError("Native chat template did not return a token-ID tensor")
        self.check_context(ids)
        return ids

    def submit(self, ids: torch.Tensor, options: Options) -> Result:
        """Enqueue exactly one native job and wait for its terminal result.

        Returns:
            Complete decoded text and native token accounting.

        Raises:
            APIError: Generation times out or the native worker fails.
            RuntimeError: The worker signals completion without a result.
        """
        self.check_context(ids, options.max_tokens)
        if self.stopping.is_set():
            raise APIError("Server is shutting down", 503, "server_shutting_down")
        pending = Pending(ids, options)
        self.queue.put(pending)
        if not pending.event.wait(timeout=7200):
            raise APIError(
                "Generation timed out after 7200 seconds", 504, "generation_timeout"
            )
        if isinstance(pending.error, ShuttingDown):
            raise APIError("Server is shutting down", 503, "server_shutting_down")
        if pending.error is not None:
            raise APIError(
                "Native generation failed; inspect server logs",
                503,
                "generation_failed",
            ) from pending.error
        if pending.result is None:
            raise RuntimeError("Native worker completed without a result")
        return pending.result

    def open_prefix_cache(self, args: argparse.Namespace) -> PrefixStore | None:
        """Open and restore the persistent prefix cache; any problem means running without it.

        Returns:
            The engine's PrefixStore, or None when persistence is off or unusable.
        """
        if not env_flag(PERSIST_ENV, True):
            print(f"[persist] disabled by {PERSIST_ENV}=0", flush=True)
            return None
        from exllamav3.generator.persist import PersistError, PrefixStore

        binding = prefix_binding(args, self.torch)
        if binding is None:
            print(
                "[persist] image identity unknown (launch gate --candidate-image or "
                "QWEN_IMAGE_ID); persistence disabled",
                flush=True,
            )
            return None
        try:
            store = PrefixStore(
                args.prefix_cache, binding, self.gen, log=lambda m: print(m, flush=True)
            )
        except (PersistError, OSError) as exc:
            print(f"[persist] disabled: {exc}", flush=True)
            return None
        print(f"[persist] binding {store.key}", flush=True)
        # Test hooks: a permuted physical placement, and a re-read of every restored page
        permute = os.environ.get("QWEN_PREFIX_PERSIST_DEBUG_PERMUTE")
        if permute is not None and not permute.isdigit():
            raise ValueError("QWEN_PREFIX_PERSIST_DEBUG_PERMUTE must be a non-negative integer")
        stats = store.restore(permute_seed=None if permute is None else int(permute))
        if self.persist_debug and stats.get("restored"):
            checked, mismatched = store.verify()
            print(f"[persist] verify checked={checked} mismatched={mismatched}", flush=True)
            if mismatched:
                store.enabled = False
        return store if store.enabled else None

    def begin_shutdown(self, deadline: float) -> None:
        """Stop taking jobs, cancel the running one, then save and let the worker exit."""
        self.stop_deadline = deadline
        self.stopping.set()
        self.queue.put(None)

    def wait_stopped(self) -> None:
        """Wait for the worker's final save, bounded by the stop budget."""
        _ = self.stopped.wait(max(0.0, self.stop_deadline - time.monotonic()) + 1)

    def _save_due(self) -> float | None:
        """Seconds until the next idle save, or None if none is pending."""
        store = self.persist
        if store is None or not store.enabled or not self.persist_dirty or self.failure:
            return None
        due = max(
            self.last_job_end + self.persist_idle, self.last_save + self.persist_interval
        )
        return max(0.0, due - time.monotonic())

    def _save(self, final: bool) -> None:
        """Capture the prefix cache on this (generator) thread and write it.

        Idle saves write in the background; the final save writes here, bounded by
        the stop deadline. Persistence errors disable persistence, never a request.
        """
        store = self.persist
        if store is None or not store.enabled or self.failure is not None:
            return
        if final:
            if not store.wait(max(0.0, self.stop_deadline - time.monotonic())):
                store.abort()
                print("[persist] final save skipped: earlier save still running", flush=True)
                return
            if not self.persist_dirty:
                return
        self.last_save = time.monotonic()
        try:
            capture = store.capture(verify=self.persist_debug)
        except Exception as exc:
            store.enabled = False
            print(f"[persist] capture failed ({type(exc).__name__}); persistence disabled", flush=True)
            return
        if capture is None:
            # Writer still busy: retry shortly instead of a whole interval later
            self.last_save -= max(0, self.persist_interval - 1)
            return
        self.persist_dirty = False
        if final:
            _ = store.save(capture, background=False, deadline=self.stop_deadline)
        else:
            _ = store.save(capture)

    def _next_pending(self) -> Pending | None:
        """The next job to run, running idle saves while waiting; None once stopping."""
        while True:
            try:
                pending = self.queue.get(timeout=self._save_due())
            except queue.Empty:
                self._save(final=False)
                continue
            if pending is not None and not self.stopping.is_set():
                return pending
            if pending is not None:
                pending.error = ShuttingDown()
                pending.event.set()
            self.queue.task_done()
            if pending is None:
                # Fail whatever raced in behind the stop sentinel
                while True:
                    try:
                        late = self.queue.get_nowait()
                    except queue.Empty:
                        return None
                    if late is not None:
                        late.error = ShuttingDown()
                        late.event.set()
                    self.queue.task_done()

    def _worker(self) -> None:
        while True:
            pending = self._next_pending()
            if pending is None:
                break
            try:
                if self.failure is not None:
                    raise RuntimeError(
                        "Generator requires restart after a native failure"
                    ) from self.failure
                ids = pending.input_ids
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                job = self.job_type(
                    input_ids=ids,
                    max_new_tokens=pending.options.max_tokens,
                    sampler=self.sampler_type(),
                    stop_conditions=self.stop_ids,
                    decode_special_tokens=not pending.options.skip_special_tokens,
                    identifier=pending.identifier,
                )
                verified_before = (
                    self.gen.greedy_verify_rounds if self.verified_acceptance else 0
                )
                tree_before = (
                    self.gen.tree_verify_rounds if self.verified_acceptance else 0
                )
                self.gen.enqueue(job)
                final = None
                while self.gen.num_remaining_jobs():
                    if self.stopping.is_set():
                        self.gen.cancel(job)
                        raise ShuttingDown()
                    for event in self.gen.iterate():
                        if event.get("identifier") == pending.identifier and event.get(
                            "eos"
                        ):
                            final = event
                if final is None:
                    raise RuntimeError(
                        "Native generation ended without a terminal event"
                    )
                # Terminal `text` is only the final delta. Never substitute it for
                # the full completion, and never re-decode a speculative token list.
                text = final.get("full_completion")
                prompt_tokens = final.get("prompt_tokens")
                completion_tokens = final.get("new_tokens")
                if (
                    not isinstance(text, str)
                    or type(prompt_tokens) is not int
                    or type(completion_tokens) is not int
                ):
                    raise RuntimeError(
                        "Native terminal event lacks complete text or usage"
                    )
                spec_rounds, spec_committed = speculative_counts(
                    final, self.draft_window, completion_tokens
                )
                if (
                    self.verified_acceptance
                    and self.gen.greedy_verify_rounds - verified_before != spec_rounds
                ):
                    raise RuntimeError(
                        "A verify round bypassed the admitted acceptance decision"
                    )
                # With the dynamic tree on (and not forced to the chain) every greedy verify
                # round of this single-sequence argmax job is a tree round; otherwise none is
                tree_rounds = (
                    self.gen.tree_verify_rounds - tree_before if self.verified_acceptance else 0
                )
                if tree_rounds != (spec_rounds if self.tree_rounds_expected else 0):
                    raise RuntimeError(
                        "Tree verify rounds disagree with the EXL3_TREE configuration"
                    )
                if self.persist_debug:
                    print(
                        f"[persist] job prompt_tokens={prompt_tokens} "
                        f"cached_pages={final.get('cached_pages')} "
                        f"cached_tokens={final.get('cached_tokens')}",
                        flush=True,
                    )
                pending.result = Result(
                    text,
                    prompt_tokens,
                    completion_tokens,
                    "length" if final.get("eos_reason") == "max_new_tokens" else "stop",
                    spec_rounds,
                    spec_committed,
                )
            except ShuttingDown as exc:
                pending.error = exc
            except Exception as exc:
                self.failure = exc
                pending.error = exc
                print(
                    f"[serve] native worker failure: {type(exc).__name__}", flush=True
                )
            finally:
                pending.event.set()
                self.queue.task_done()
                self.persist_dirty = True
                self.last_job_end = time.monotonic()
        self._save(final=True)
        self.stopped.set()


def load_authorization(path: str = "/app/api_key.txt") -> str:
    """Read a bounded private regular key file without following symlinks.

    Returns:
        The exact expected ASCII Bearer authorization value.

    Raises:
        ValueError: Key permissions, file type, or bytes violate the boundary.
    """
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, "rb") as key_file:
        info = os.fstat(key_file.fileno())
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) not in (
            0o400,
            0o600,
        ):
            raise ValueError("API key must be a private regular file")
        key = key_file.read(4098).removesuffix(b"\n")
    if not 1 <= len(key) <= 4096 or not all(33 <= byte <= 126 for byte in key):
        raise ValueError("Invalid API key bytes")
    return "Bearer " + key.decode("ascii")


COMMON_FIELDS = {
    "model",
    "max_tokens",
    "temperature",
    "stream",
    "stream_options",
    "skip_special_tokens",
    "n",
} | set(GREEDY_IDENTITY)
CHAT_FIELDS = COMMON_FIELDS | {
    "messages",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "chat_template_kwargs",
    "reasoning_effort",
    "max_completion_tokens",
}


class Handler(BaseHTTPRequestHandler):
    """Expose authenticated OpenAI endpoints without executing arbitrary tools."""

    protocol_version = "HTTP/1.1"
    engine: ClassVar[Server]
    authorization: ClassVar[str]

    @override
    def log_message(self, format: str, *args: object) -> None:
        # Do not echo request targets, credentials, bodies, or model text.
        print(f"[http] {self.address_string()} request complete", flush=True)

    def authorized(self) -> bool:
        """Check exactly one ASCII authorization header in constant time.

        Returns:
            Whether the supplied header matches the private server key.
        """
        values = self.headers.get_all("Authorization", [])
        if len(values) != 1:
            return False
        supplied: object = values[0]
        if not isinstance(supplied, str) or not supplied.isascii():
            return False
        return hmac.compare_digest(supplied, self.authorization)

    def require_auth(self) -> bool:
        """Close unauthorized requests before consuming their bodies.

        Returns:
            Whether request handling may continue.
        """
        if self.authorized():
            return True
        self.close_connection = True
        self.send_error_json(APIError("Unauthorized", 401, "invalid_api_key"))
        return False

    def send_json(self, code: int, payload: dict[str, JSON]) -> None:
        """Serialize a finite JSON response before committing its HTTP status."""
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, error: APIError) -> None:
        """Emit an OpenAI-shaped error without exposing native exception details."""
        error_type = (
            "authentication_error"
            if error.status == 401
            else "invalid_request_error"
            if error.status < 500
            else "server_error"
        )
        self.send_json(
            error.status,
            {
                "error": {
                    "message": str(error),
                    "type": error_type,
                    "param": None,
                    "code": error.code,
                }
            },
        )

    def body_length(self) -> int:
        """Validate unambiguous, bounded HTTP request framing.

        Returns:
            The positive body length admitted by the framing boundary.

        Raises:
            APIError: Framing is ambiguous, absent, invalid, or too large.
        """
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") is not None or len(lengths) != 1:
            raise APIError("Expected one Content-Length and no Transfer-Encoding")
        length = lengths[0]
        if not length.isascii() or not length.isdecimal() or len(length) > 10:
            raise APIError("Invalid Content-Length")
        size = int(length)
        if size > MAX_BODY:
            raise APIError("Request body exceeds 32 MiB", 413, "body_too_large")
        if size <= 0:
            raise APIError("Request body must be nonempty")
        return size

    @override
    def handle_expect_100(self) -> bool:
        if not self.require_auth():
            return False
        try:
            self.body_length()
        except APIError as exc:
            self.close_connection = True
            self.send_error_json(exc)
            return False
        return super().handle_expect_100()

    def read_body(self) -> dict[str, JSON]:
        """Read one framed UTF-8 JSON object, closing invalid request connections.

        Returns:
            The validated request object.

        Raises:
            APIError: The body is incomplete or is not a valid JSON object.
        """
        try:
            size = self.body_length()
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise APIError("Incomplete request body")
            return object_value(load_json(raw.decode("utf-8")), "Request body")
        except (UnicodeError, ValueError, RecursionError) as exc:
            self.close_connection = True
            raise APIError(
                "Request body must be valid UTF-8 JSON without duplicate keys or nonfinite numbers"
            ) from exc
        except APIError:
            self.close_connection = True
            raise

    def do_GET(self) -> None:
        """Serve authenticated health and model metadata without generation."""
        if not self.require_auth():
            return
        if self.headers.get("Transfer-Encoding") is not None or self.headers.get_all(
            "Content-Length", []
        ) not in ([], ["0"]):
            self.close_connection = True
            self.send_error_json(APIError("GET requests must not have a body"))
            return
        if self.path in ("/health", "/v1/health"):
            if self.engine.failure is not None:
                self.send_error_json(
                    APIError(
                        "Native generator requires restart", 503, "generation_failed"
                    )
                )
            else:
                self.send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self.send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.engine.model_name,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "local",
                            "max_model_len": self.engine.max_model_len,
                        }
                    ],
                },
            )
        else:
            self.send_error_json(APIError("Not found", 404, "not_found"))

    def do_POST(self) -> None:
        """Dispatch authenticated requests and translate failures into JSON errors."""
        if not self.require_auth():
            return
        try:
            if self.path not in (
                "/v1/chat/completions",
                "/v1/chat/completions/render",
                "/v1/completions",
            ):
                self.close_connection = True
                raise APIError("Not found", 404, "not_found")
            body = self.read_body()
            if self.path == "/v1/completions":
                self.completions(body)
            else:
                self.chat(body, self.path.endswith("/render"))
        except APIError as exc:
            self.send_error_json(exc)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            print(f"[serve] request failure: {type(exc).__name__}", flush=True)
            self.send_error_json(
                APIError("Internal server error", 500, "internal_error")
            )

    def chat(self, body: dict[str, JSON], render: bool) -> None:
        """Render or generate a chat response, including buffered tool-call SSE.

        Raises:
            APIError: Rendering was requested with streaming enabled.
        """
        only_fields(body, CHAT_FIELDS, "chat request")
        options = parse_options(body, self.engine.model_name, True)
        chat = parse_chat(body)
        if render and options.stream:
            raise APIError("Rendering does not support stream=true")
        ids = self.engine.render_chat(chat)
        self.engine.check_context(
            ids,
            options.max_tokens
            if not render or "max_tokens" in body or "max_completion_tokens" in body
            else 0,
        )
        if render:
            self.send_json(200, {"token_ids": ids.flatten().tolist()})
            return
        result = self.engine.submit(ids, options)
        reasoning, content = split_reasoning(result.text, chat.thinking)
        content, calls = parse_tool_output(content, chat, result.finish_reason)
        finish = "tool_calls" if calls else result.finish_reason
        message: dict[str, JSON] = {
            "role": "assistant",
            "content": content or (None if calls else ""),
            "reasoning_content": reasoning,
        }
        if calls:
            message["tool_calls"] = list(calls)
        common: dict[str, JSON] = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "created": int(time.time()),
            "model": self.engine.model_name,
        }
        usage = result.usage()
        if not options.stream:
            self.send_json(
                200,
                {
                    **common,
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
                    ],
                    "usage": usage,
                },
            )
            return
        frames: list[bytes] = []

        def frame(delta: dict[str, JSON], reason: str | None = None) -> None:
            chunk: dict[str, JSON] = {
                **common,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
            }
            if options.include_usage:
                chunk["usage"] = None
            frames.append(
                (
                    "data: "
                    + json.dumps(chunk, ensure_ascii=False, allow_nan=False)
                    + "\n\n"
                ).encode("utf-8")
            )

        frame({"role": "assistant", "content": ""})
        if reasoning is not None:
            frame({"reasoning_content": reasoning})
        if content:
            frame({"content": content})
        for index, call in enumerate(calls):
            frame({"tool_calls": [{"index": index, **call}]})
        frame({}, finish)
        if options.include_usage:
            chunk = {
                **common,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": usage,
            }
            frames.append(
                (
                    "data: "
                    + json.dumps(chunk, ensure_ascii=False, allow_nan=False)
                    + "\n\n"
                ).encode("utf-8")
            )
        frames.append(b"data: [DONE]\n\n")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(sum(len(item) for item in frames)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-EXL3-Transport", "buffered-sse")
        self.end_headers()
        for item in frames:
            self.wfile.write(item)
        self.wfile.flush()

    def completions(self, body: dict[str, JSON]) -> None:
        """Generate one raw text completion with native context and usage checks.

        Raises:
            APIError: The prompt is not a supported string or token-ID sequence.
        """
        only_fields(body, COMMON_FIELDS | {"prompt"}, "completion request")
        options = parse_options(body, self.engine.model_name, False)
        prompt = body.get("prompt")
        with self.engine.tokenizer_lock:
            if isinstance(prompt, str):
                ids = self.engine.tokenizer.encode(prompt)
            elif isinstance(prompt, list) and prompt:
                tokens = (
                    prompt[0]
                    if len(prompt) == 1 and isinstance(prompt[0], list)
                    else prompt
                )
                if not tokens or any(
                    type(token) is not int
                    or token < 0
                    or token >= self.engine.tokenizer.actual_vocab_size
                    for token in tokens
                ):
                    raise APIError(
                        "prompt token IDs must be nonempty integers within the tokenizer vocabulary"
                    )
                ids = self.engine.torch.tensor([tokens], dtype=self.engine.torch.long)
            else:
                raise APIError(
                    "prompt must be a string, a token-ID list, or one nested token-ID list"
                )
        result = self.engine.submit(ids, options)
        self.send_json(
            200,
            {
                "id": "cmpl-" + uuid.uuid4().hex[:24],
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.engine.model_name,
                "choices": [
                    {
                        "index": 0,
                        "text": result.text,
                        "finish_reason": result.finish_reason,
                    }
                ],
                "usage": result.usage(),
            },
        )


def main() -> None:
    """Start the authenticated adapter only with the fixed native model geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=CONTEXT)
    parser.add_argument("--cache-tokens", type=int, default=CACHE_TOKENS)
    parser.add_argument("--cq", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8889)
    parser.add_argument(
        "--prefix-cache",
        default=None,
        help="private directory for the persistent prefix cache (off when omitted)",
    )
    args = parser.parse_args()
    if (
        args.max_model_len != CONTEXT
        or args.cache_tokens != CACHE_TOKENS
        or args.cq != 3
    ):
        parser.error(
            "This native EXL3 stack requires context=262144, cache-tokens=270336, cq=3"
        )
    if args.model_name != MODEL_NAME:
        parser.error("The served model alias must be qwen3.8-27b")
    os.umask(0o077)
    Handler.authorization = load_authorization()
    Handler.engine = Server(args)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    stopping = threading.Event()

    def on_sigterm(_signum: int, _frame: object) -> None:
        # Docker stop: finish at the next generator step, save the prefix cache, exit 0.
        if stopping.is_set():
            return
        stopping.set()
        Handler.engine.begin_shutdown(time.monotonic() + STOP_BUDGET_SECONDS)
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    _ = signal.signal(signal.SIGTERM, on_sigterm)
    print(
        f"[serve] listening on http://{args.host}:{args.port}; SSE transport is buffered",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    if stopping.is_set():
        Handler.engine.wait_stopped()
        print("[serve] stopped", flush=True)


if __name__ == "__main__":
    main()
