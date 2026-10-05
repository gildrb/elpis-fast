#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Native EXL3/DFlash2 HTTP adapter. SSE buffers a whole response, not tokens."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import logging
import math
import os
import pathlib
import queue
import re
import select
import signal
import stat
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, ClassVar, Protocol, override

import regex
import torch
from exllamav3 import Cache, Config, Job, Model, Tokenizer
from exllamav3 import Generator as NativeGenerator
from exllamav3.cache import CacheLayer_quant
from exllamav3.generator.sampler.presets import ArgmaxSampler
from jinja2 import TemplateError
from jsonschema import Draft202012Validator, FormatChecker, validators
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

# The persistent prefix cache module comes from patch 9501b, which only the
# candidate-ext engine carries; other engines must run with persistence off.
PERSIST_MODULE = "exllamav3.generator.persist"
try:
    from exllamav3.generator.persist import PersistError, PrefixStore
except ModuleNotFoundError as missing:
    if missing.name != PERSIST_MODULE:
        raise
    PERSIST_INSTALLED = False
else:
    PERSIST_INSTALLED = True

if TYPE_CHECKING:
    import socket
    from collections.abc import Buffer, Generator, Iterable, Iterator
    from types import TracebackType

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
# Keywords applying subschemas to the whole tool-argument object, and the keywords
# by which a subschema would declare a parameter's value schema.
ROOT_COMPOSITION = {
    "allOf",
    "anyOf",
    "oneOf",
    "not",
    "if",
    "then",
    "else",
    "dependentSchemas",
}
PARAMETER_SCHEMAS = {
    "$ref",
    "$dynamicRef",
    "properties",
    "patternProperties",
    "additionalProperties",
    "unevaluatedProperties",
}
FORMAT_CHECKER = FormatChecker()
# Total time for client-schema regex matches while one model response is checked.
# Python's backtracking `re` has no time bound; `regex` stops at the deadline.
PATTERN_SECONDS = 2.0
PATTERN_DEADLINE: ContextVar[float] = ContextVar("pattern_deadline")
# Client read limits: request line and headers (this includes keep-alive idle time),
# then the whole body. Generation and response writes have no socket timeout.
HEADER_SECONDS = 60
BODY_SECONDS = 300
# Persistent prefix cache (engine generator/persist.py). Saved on SIGTERM and after
# 30 s idle, at most every 5 min.
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
        msg = f"{label} must be an object"
        raise APIError(msg)
    return value


def string_value(value: JSON, label: str) -> str:
    """Require a string without coercion.

    Returns:
        The validated string.

    Raises:
        APIError: The supplied value is not a string.

    """
    if not isinstance(value, str):
        msg = f"{label} must be a string"
        raise APIError(msg)
    return value


def boolean_value(value: JSON, label: str) -> bool:
    """Require a JSON boolean, excluding numeric stand-ins.

    Returns:
        The validated boolean.

    Raises:
        APIError: The supplied value is not a boolean.

    """
    if type(value) is not bool:
        msg = f"{label} must be a boolean"
        raise APIError(msg)
    return value


def positive_integer(value: JSON, label: str) -> int:
    """Require a positive integer, excluding booleans.

    Returns:
        The validated integer.

    Raises:
        APIError: The supplied value is not a positive integer.

    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"{label} must be a positive integer"
        raise APIError(msg)
    return value


def only_fields(value: dict[str, JSON], allowed: set[str], label: str) -> None:
    """Reject fields whose semantics this adapter does not implement.

    Raises:
        APIError: An unsupported field is present.

    """
    unknown = value.keys() - allowed
    if unknown:
        msg = f"Unsupported {label} field: {min(unknown)}"
        raise APIError(msg)


class InvariantTypeError(TypeError, RuntimeError):
    """Validated state or a native result has an impossible type."""


class JSONTypeError(TypeError, ValueError):
    """A decoded value has a non-JSON type; still a ValueError for JSON callers."""


def json_value(value: object) -> JSON:
    """Rebuild decoded data as finite, string-keyed JSON values.

    Returns:
        The validated JSON value.

    Raises:
        JSONTypeError: An object key is not a string.
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
                msg = "JSON object keys must be strings"
                raise JSONTypeError(msg)
            result[key] = json_value(item)
        return result
    msg = "Expected finite JSON values"
    raise ValueError(msg)


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
            msg = "Duplicate JSON object key"
            raise ValueError(msg)
        result[key] = value
    return result


PYTHON_LITERALS: dict[str, tuple[JSON]] = {
    "True": (True,),
    "False": (False,),
    "None": (None,),
}


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
        msg = "Parameter names must be nonempty and cannot contain markup delimiters"
        raise APIError(msg)


def check_schema_nodes(schema: JSON, root: dict[str, JSON]) -> None:
    """Reject unsupported schema features without fetching client-supplied URIs.

    Raises:
        APIError: A dialect, reference, format, or keyword is unsupported.

    """
    if isinstance(schema, bool):
        return
    node = object_value(schema, "JSON Schema")
    # jsonschema checks then/else inside its "if" validator, not as VALIDATORS keys.
    allowed = (
        set(Draft202012Validator.VALIDATORS) | SCHEMA_ANNOTATIONS | {"then", "else"}
    )
    only_fields(node, allowed, "JSON Schema")
    if (
        "$schema" in node
        and node["$schema"] != "https://json-schema.org/draft/2020-12/schema"
    ):
        msg = "Only JSON Schema draft 2020-12 is supported"
        raise APIError(msg)
    # No resource identifiers, remote references, or dynamic scope: all references
    # must be JSON pointers into this one request's parameters object.
    if "$dynamicRef" in node:
        msg = "$dynamicRef is unsupported; use local $ref JSON pointers"
        raise APIError(msg)
    if "$ref" in node:
        check_local_ref(string_value(node["$ref"], "$ref"), root)
    if "format" in node:
        fmt = string_value(node["format"], "format")
        if fmt not in FORMAT_CHECKER.checkers:
            msg = f"Unsupported JSON Schema format: {fmt}"
            raise APIError(msg)
    check_schema_children(node, root)


def pointer_index(key: str, target: list[JSON]) -> int | None:
    """Return the in-range array index a JSON-pointer token names, if any.

    Returns:
        The index, or None when the token is not a canonical in-range index.

    """
    if not (key.isascii() and key.isdecimal()):
        return None
    if key != "0" and key.startswith("0"):
        return None
    index = int(key)
    return index if index < len(target) else None


def check_local_ref(ref: str, root: dict[str, JSON]) -> None:
    """Require a `$ref` to be a local JSON pointer resolving to a schema.

    Raises:
        APIError: The reference is nonlocal, unresolvable, or not a schema.

    """
    if ref != "#" and not ref.startswith("#/"):
        msg = "Only local JSON-pointer $ref values are supported"
        raise APIError(msg)
    target: JSON = root
    if ref != "#":
        for part in ref[2:].split("/"):
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(target, dict) and key in target:
                target = target[key]
                continue
            index = pointer_index(key, target) if isinstance(target, list) else None
            if not isinstance(target, list) or index is None:
                msg = "Unresolvable local JSON Schema $ref"
                raise APIError(msg)
            target = target[index]
    if not isinstance(target, (dict, bool)):
        msg = "JSON Schema $ref must resolve to a schema"
        raise APIError(msg)


def check_schema_children(node: dict[str, JSON], root: dict[str, JSON]) -> None:
    """Check every subschema of one schema object.

    Raises:
        APIError: A subschema container has the wrong JSON type.

    """
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
            msg = f"{keyword} must be an array"
            raise APIError(msg)
        for child in children:
            check_schema_nodes(child, root)


def check_root_composition(node: dict[str, JSON]) -> None:
    """Admit whole-object composition only when it leaves parameter schemas at the root.

    Each XML parameter decodes against the root's properties, patternProperties and
    additionalProperties alone, so a branch that declares parameter schemas would make
    that mapping ambiguous. Branches that only constrain the object (e.g. which
    parameters are required) cannot; the whole-argument validation enforces them.

    Raises:
        APIError: A composed subschema declares or references parameter schemas.

    """
    for keyword in ROOT_COMPOSITION & node.keys():
        value = node[keyword]
        children: list[JSON]
        if keyword == "dependentSchemas":
            children = list(object_value(value, keyword).values())
        elif isinstance(value, list):
            children = value
        else:
            children = [value]
        for item in children:
            if isinstance(item, bool):
                continue
            child = object_value(item, keyword)
            if PARAMETER_SCHEMAS & child.keys():
                msg = (
                    "Tool parameter root composition may only constrain the object; "
                    "put parameter schemas in the root properties"
                )
                raise APIError(msg)
            check_root_composition(child)


@contextmanager
def pattern_budget() -> Generator[None]:
    """Allow PATTERN_SECONDS of client-schema regex matching in this context.

    Yields:
        Control while the deadline applies.

    """
    token = PATTERN_DEADLINE.set(time.monotonic() + PATTERN_SECONDS)
    try:
        yield
    finally:
        PATTERN_DEADLINE.reset(token)


def pattern_search(pattern: str, text: str) -> bool:
    """Search text with a client-schema regex inside the current pattern budget.

    Returns:
        Whether the pattern matches anywhere in the text.

    Raises:
        APIError: The budget is spent, or `regex` cannot compile the pattern.

    """
    timeout = APIError(
        f"Tool schema patterns exceeded the {PATTERN_SECONDS:g} s evaluation budget",
        400,
        "pattern_timeout",
    )
    remaining = PATTERN_DEADLINE.get() - time.monotonic()
    if remaining <= 0:
        raise timeout
    try:
        # concurrent=True releases the GIL, so a long match does not stop generation.
        match = regex.search(pattern, text, timeout=remaining, concurrent=True)
    except TimeoutError as exc:
        raise timeout from exc
    except regex.error as exc:
        msg = "Tool schema pattern cannot be evaluated"
        raise APIError(msg, 400, "invalid_schema") from exc
    return match is not None


# jsonschema 4.26 matches these keywords with `re`. These replacements use the
# bounded search. unevaluatedProperties also matches patternProperties with `re`;
# check_unevaluated_patterns does not admit that pair.
def bounded_pattern(
    _validator: Validator, pattern: str, instance: JSON, _schema: dict[str, JSON]
) -> Iterator[ValidationError]:
    """Apply `pattern` to strings with the bounded search.

    Yields:
        A validation error when the string does not match.

    """
    if isinstance(instance, str) and not pattern_search(pattern, instance):
        yield ValidationError(f"{instance!r} does not match {pattern!r}")


class Descending(Protocol):
    """The jsonschema validator method the bounded keyword callbacks use."""

    def descend(
        self,
        instance: JSON,
        schema: JSON,
        path: str | None = None,
    ) -> Iterable[object]:
        """Validate a child instance against a subschema."""
        ...


def checked_errors(errors: Iterable[object]) -> Iterator[ValidationError]:
    """Pass through jsonschema's untyped `descend` errors, checking their type.

    Yields:
        Each validation error.

    Raises:
        InvariantTypeError: `descend` yielded something other than an error.

    """
    for error in errors:
        if not isinstance(error, ValidationError):
            msg = "jsonschema descend yielded a non-ValidationError"
            raise InvariantTypeError(msg)
        yield error


def bounded_pattern_properties(
    validator: Descending,
    patterns: dict[str, JSON],
    instance: JSON,
    _schema: dict[str, JSON],
) -> Iterator[ValidationError]:
    """Apply `patternProperties` subschemas to matching keys with the bounded search.

    Yields:
        Validation errors of the matching values.

    """
    if not isinstance(instance, dict):
        return
    for pattern, subschema in patterns.items():
        for key, value in instance.items():
            if pattern_search(pattern, key):
                yield from checked_errors(validator.descend(value, subschema, path=key))


def bounded_additional_properties(
    validator: Descending,
    additional: JSON,
    instance: JSON,
    schema: dict[str, JSON],
) -> Iterator[ValidationError]:
    """Apply `additionalProperties` to keys outside properties and patternProperties.

    Yields:
        Validation errors of the additional values, or one error if they are banned.

    """
    if not isinstance(instance, dict):
        return
    properties = object_value(schema.get("properties", {}), "properties")
    patterns = object_value(schema.get("patternProperties", {}), "patternProperties")
    extras = [
        key
        for key in instance
        if key not in properties
        and not any(pattern_search(pattern, key) for pattern in patterns)
    ]
    if isinstance(additional, dict):
        for key in extras:
            yield from checked_errors(
                validator.descend(instance[key], additional, path=key)
            )
    elif additional is False and extras:
        yield ValidationError(f"Additional properties are not allowed: {extras!r}")


SchemaValidator = validators.extend(
    Draft202012Validator,
    {
        "pattern": bounded_pattern,
        "patternProperties": bounded_pattern_properties,
        "additionalProperties": bounded_additional_properties,
    },
)


def contains_key(value: JSON, key: str) -> bool:
    """Check if any object in a JSON tree, annotation values included, has a key.

    Returns:
        Whether the key occurs at any depth.

    """
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if key in item:
                return True
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return False


def check_unevaluated_patterns(schema: dict[str, JSON]) -> None:
    """Reject unevaluatedProperties with patternProperties in one tool schema.

    jsonschema matches patternProperties for unevaluatedProperties with `re`, which
    has no time bound. References resolve only into this object or the fixed draft
    metaschemas, so a scan of all keys, annotation values included, finds every
    client schema that validation can reach.

    Raises:
        APIError: The schema contains both keywords.

    """
    if contains_key(schema, "unevaluatedProperties") and contains_key(
        schema, "patternProperties"
    ):
        msg = "unevaluatedProperties cannot be combined with patternProperties"
        raise APIError(msg)


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
            InvariantTypeError: Validated schema state is internally inconsistent.

        """
        properties = self.schema.get("properties", {})
        patterns = self.schema.get("patternProperties", {})
        if not isinstance(properties, dict) or not isinstance(patterns, dict):
            msg = "Validated schema has invalid properties"
            raise InvariantTypeError(msg)
        schemas: list[JSON] = []
        if name in properties:
            schemas.append(properties[name])
        for pattern, schema in patterns.items():
            if pattern_search(pattern, name):
                schemas.append(schema)
        if not schemas:
            schemas.append(self.schema.get("additionalProperties", True))
        validator = self.validator.evolve(schema={"allOf": schemas})
        # The actual template renders strings verbatim (not JSON quoted). Prefer
        # that interpretation for string/nonstring unions, preserving e.g. "001".
        if validator.is_valid(raw):
            return raw
        # Qwen's template prints Python values; the model may echo that casing.
        # Accept it only where the schema accepts the resulting JSON value.
        literal = PYTHON_LITERALS.get(raw.strip())
        if literal is not None and validator.is_valid(literal[0]):
            return literal[0]
        try:
            value = load_json(raw)
        except (ValueError, RecursionError) as exc:
            msg = f"Model emitted invalid JSON for {self.name}.{name}"
            raise APIError(
                msg,
                502,
                "invalid_tool_arguments",
            ) from exc
        if not validator.is_valid(value):
            msg = f"Model argument violates schema: {self.name}.{name}"
            raise APIError(
                msg,
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
        msg = "tools must be an array"
        raise APIError(msg)
    tools: dict[str, Tool] = {}
    for item in value:
        wire = object_value(item, "tool")
        only_fields(wire, {"type", "function"}, "tool")
        if wire.get("type") != "function":
            msg = "Only function tools are supported"
            raise APIError(msg)
        function = object_value(wire.get("function"), "tool.function")
        only_fields(
            function, {"name", "description", "parameters", "strict"}, "tool.function"
        )
        name = string_value(function.get("name"), "tool.function.name")
        if not FUNCTION_NAME.fullmatch(name) or name in tools:
            msg = (
                "Tool names must be unique, 1..64 ASCII letters/digits/"
                "underscores/hyphens"
            )
            raise APIError(msg)
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
            msg = "Tool parameters must declare type: object"
            raise APIError(msg)
        if "$ref" in schema:
            msg = "Tool parameters root $ref is unsupported; inline the object"
            raise APIError(msg)
        check_unevaluated_patterns(schema)
        try:
            Draft202012Validator.check_schema(schema)
            check_schema_nodes(schema, schema)
            check_root_composition(schema)
            registry = Registry().with_resource(
                "urn:exl3:parameters", DRAFT202012.create_resource(schema)
            )
            validator = SchemaValidator(
                schema, registry=registry, format_checker=FORMAT_CHECKER
            )
        except (SchemaError, re.error, RecursionError) as exc:
            msg = "Invalid tool JSON Schema"
            raise APIError(msg) from exc
        tools[name] = Tool(name, wire, schema, validator)
    return tools


def text_content(value: JSON, label: str, *, nullable: bool = False) -> JSON:
    """Validate text-only content, preserving the caller's representation.

    Returns:
        Text, text parts, or an explicitly permitted null value.

    Raises:
        APIError: Content is malformed or requires unsupported modalities.

    """
    if isinstance(value, str) or (nullable and value is None):
        return value
    if isinstance(value, list):
        for item in value:
            part = object_value(item, label)
            only_fields(part, {"type", "text"}, "text content")
            if part.get("type") != "text":
                msg = "Only text content parts are supported"
                raise APIError(msg)
            string_value(part.get("text"), "content.text")
        return value
    msg = f"{label} must be text or an array of text parts"
    raise APIError(msg)


@dataclass
class History:
    """Messages normalized so far and the tool-call bookkeeping between them."""

    messages: list[dict[str, JSON]] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    pending: dict[str, str] = field(default_factory=dict)
    results: dict[str, dict[str, JSON]] = field(default_factory=dict)
    user_found: bool = False

    def add_tool_result(self, message: dict[str, JSON]) -> None:
        """Record one tool result, flushing them in call order once all arrived.

        Raises:
            APIError: The result does not match an unresolved call.

        """
        only_fields(
            message, {"role", "content", "tool_call_id", "name"}, "tool message"
        )
        call_id = string_value(message.get("tool_call_id"), "tool_call_id")
        if call_id not in self.pending or call_id in self.results:
            msg = "Tool result must match one unresolved assistant tool_call_id"
            raise APIError(msg)
        if "name" in message and message["name"] != self.pending[call_id]:
            msg = "Tool result name does not match its tool_call_id"
            raise APIError(msg)
        self.results[call_id] = {
            **message,
            "name": self.pending[call_id],
            "content": text_content(message.get("content"), "tool content"),
        }
        if len(self.results) == len(self.pending):
            # Qwen's actual template omits IDs and names; order the responses
            # by the prior call IDs rather than associating results by arrival.
            self.messages.extend(self.results[pending] for pending in self.pending)
            self.pending = {}
            self.results = {}

    def add_message(self, index: int, message: dict[str, JSON]) -> None:
        """Validate and append one system, user, or assistant message.

        Raises:
            APIError: The message is out of place or malformed.

        """
        role = message.get("role")
        if self.pending:
            msg = (
                "All assistant tool calls need contiguous tool results before the "
                "next message"
            )
            raise APIError(msg)
        if role not in {"system", "user", "assistant"}:
            msg = "Supported message roles: system, user, assistant, tool"
            raise APIError(msg)
        allowed = {"role", "content"}
        if role == "assistant":
            allowed |= {"tool_calls", "reasoning_content"}
        only_fields(message, allowed, "message")
        if role == "system" and index != 0:
            msg = "System message must be first"
            raise APIError(msg)
        if role == "user":
            self.user_found = True
        normalized = dict(message)
        normalized["content"] = text_content(
            message.get("content"), "message.content", nullable=role == "assistant"
        )
        if role == "assistant":
            self.add_assistant_fields(message, normalized)
        self.messages.append(normalized)

    def add_assistant_fields(
        self, message: dict[str, JSON], normalized: dict[str, JSON]
    ) -> None:
        """Validate assistant reasoning and tool calls into the normalized message.

        Raises:
            APIError: The reasoning or tool calls are malformed.

        """
        reasoning = message.get("reasoning_content")
        if reasoning is not None:
            string_value(reasoning, "reasoning_content")
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not calls:
                msg = "assistant.tool_calls must be a nonempty array"
                raise APIError(msg)
            normalized_calls: list[JSON] = []
            for item in calls:
                call_id, name, call = parse_assistant_call(item, self.seen_ids)
                normalized_calls.append(call)
                self.seen_ids.add(call_id)
                self.pending[call_id] = name
            normalized["tool_calls"] = normalized_calls
        elif normalized["content"] is None:
            msg = "Assistant content may be null only with tool_calls"
            raise APIError(msg)


def parse_assistant_call(item: JSON, seen_ids: set[str]) -> tuple[str, str, JSON]:
    """Validate one historical assistant tool call and decode its arguments.

    Returns:
        The call ID, the function name, and the normalized call.

    Raises:
        APIError: The call is malformed, duplicated, or has invalid arguments.

    """
    call = object_value(item, "assistant tool call")
    only_fields(call, {"id", "type", "function"}, "assistant tool call")
    call_id = string_value(call.get("id"), "assistant tool call id")
    if not call_id or call_id in seen_ids:
        msg = "Assistant tool call IDs must be nonempty and unique"
        raise APIError(msg)
    if call.get("type") != "function":
        msg = "Only function tool calls are supported"
        raise APIError(msg)
    function = object_value(call.get("function"), "assistant tool call function")
    only_fields(function, {"name", "arguments"}, "assistant tool call function")
    name = string_value(function.get("name"), "assistant tool function name")
    if not FUNCTION_NAME.fullmatch(name):
        msg = "Invalid assistant tool function name"
        raise APIError(msg)
    raw = string_value(function.get("arguments"), "assistant tool arguments")
    try:
        arguments = object_value(load_json(raw), "assistant tool arguments JSON")
    except (ValueError, RecursionError) as exc:
        msg = "Assistant tool arguments must be a valid JSON object string"
        raise APIError(msg) from exc
    for key in arguments:
        parameter_name(key)
    normalized: JSON = {**call, "function": {"name": name, "arguments": arguments}}
    return call_id, name, normalized


def parse_messages(value: JSON) -> list[dict[str, JSON]]:
    """Normalize tool arguments and associate each tool result with its call ID.

    Returns:
        Template-ready messages with results in assistant call order.

    Raises:
        APIError: Message structure, tool history, or arguments are invalid.

    """
    if not isinstance(value, list) or not value:
        msg = "messages must be a nonempty array"
        raise APIError(msg)
    history = History()
    for index, item in enumerate(value):
        message = object_value(item, "message")
        if message.get("role") == "tool":
            history.add_tool_result(message)
        else:
            history.add_message(index, message)
    if history.pending:
        msg = "All assistant tool calls need tool results before generation"
        raise APIError(msg)
    if not history.user_found:
        msg = "At least one user message is required"
        raise APIError(msg)
    return history.messages


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


def parse_options(body: dict[str, JSON], model_name: str, *, chat: bool) -> Options:
    """Validate model selection, greedy generation, limits, and transport options.

    Returns:
        Options shared by rendering and native generation.

    Raises:
        APIError: An option is invalid, conflicting, or unsupported.

    """
    model = body.get("model", model_name)
    if not isinstance(model, str) or model != model_name:
        msg = f"model must be {model_name}"
        raise APIError(msg)
    if "max_tokens" in body and "max_completion_tokens" in body:
        msg = "Specify only one of max_tokens and max_completion_tokens"
        raise APIError(msg)
    limit = body.get(
        "max_completion_tokens", body.get("max_tokens", 4096 if chat else 256)
    )
    max_tokens = positive_integer(limit, "output token limit")
    temperature = body.get("temperature", 0)
    if type(temperature) not in {int, float} or temperature != 0:
        msg = "temperature must be numeric zero (greedy)"
        raise APIError(msg)
    for name, identity in GREEDY_IDENTITY.items():
        if name in body and (
            type(body[name]) not in {int, float} or body[name] != identity
        ):
            msg = f"{name} must be numeric {identity} (greedy identity)"
            raise APIError(msg)
    if "n" in body and (type(body["n"]) is not int or body["n"] != 1):
        msg = "Only n=1 is supported"
        raise APIError(msg)
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
            msg = "stream_options requires stream=true"
            raise APIError(msg)
    if not chat and stream:
        msg = "Streaming raw completions is unsupported"
        raise APIError(msg)
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


REQUEST_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def parse_tool_choice(body: dict[str, JSON], tools: dict[str, Tool]) -> str:
    """Validate tool_choice against the supplied tools.

    Returns:
        "auto", "none", "required", or "named:" followed by a tool name.

    Raises:
        APIError: The tool choice is malformed or names no supplied tool.

    """
    choice_value = body.get("tool_choice", "auto" if tools else "none")
    if isinstance(choice_value, str) and choice_value in {"auto", "none", "required"}:
        choice = choice_value
    elif isinstance(choice_value, dict):
        only_fields(choice_value, {"type", "function"}, "tool_choice")
        if choice_value.get("type") != "function":
            msg = "Named tool_choice must have type=function"
            raise APIError(msg)
        function = object_value(choice_value.get("function"), "tool_choice.function")
        only_fields(function, {"name"}, "tool_choice.function")
        name = string_value(function.get("name"), "tool_choice.function.name")
        if name not in tools:
            msg = "Named tool_choice must name a supplied tool"
            raise APIError(msg)
        choice = "named:" + name
    else:
        msg = "tool_choice must be auto, none, required, or a named function object"
        raise APIError(msg)
    if choice != "none" and not tools:
        msg = "This tool_choice requires nonempty tools"
        raise APIError(msg)
    return choice


def normalize_effort(requested_effort: str) -> str:
    """Map a request-level reasoning effort onto the template's effort levels.

    Returns:
        The template effort level.

    """
    if requested_effort == "minimal":
        return "low"
    if requested_effort in {"high", "max"}:
        return "xhigh"
    return requested_effort


def parse_template_kwargs(body: dict[str, JSON]) -> tuple[dict[str, JSON], bool]:
    """Validate chat_template_kwargs and the request-level reasoning_effort.

    Returns:
        The template keyword arguments and whether thinking is enabled.

    Raises:
        APIError: A template option is invalid or conflicts with another.

    """
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
    if not isinstance(effort, str) or effort not in {"xhigh", "medium", "low"}:
        msg = "reasoning_effort must be xhigh, medium, or low"
        raise APIError(msg)
    if not thinking and "reasoning_effort" in kwargs:
        msg = "reasoning_effort requires enable_thinking=true"
        raise APIError(msg)
    template_kwargs = dict(kwargs)
    if "reasoning_effort" not in body:
        return template_kwargs, thinking
    requested_effort = body["reasoning_effort"]
    if not isinstance(requested_effort, str) or requested_effort not in REQUEST_EFFORTS:
        msg = "reasoning_effort must be none, minimal, low, medium, high, xhigh, or max"
        raise APIError(msg)
    requested_thinking = requested_effort != "none"
    if "enable_thinking" in kwargs and thinking != requested_thinking:
        msg = "reasoning_effort conflicts with enable_thinking"
        raise APIError(msg)
    normalized_effort = normalize_effort(requested_effort)
    if "reasoning_effort" in kwargs and (
        not requested_thinking or effort != normalized_effort
    ):
        msg = "reasoning_effort conflicts with chat_template_kwargs.reasoning_effort"
        raise APIError(msg)
    template_kwargs["enable_thinking"] = requested_thinking
    if requested_thinking:
        template_kwargs["reasoning_effort"] = normalized_effort
    return template_kwargs, requested_thinking


def choice_instructions(
    choice: str, *, explicit: bool, tools: dict[str, Tool], parallel: bool
) -> list[str]:
    """Phrase the tool-choice policy as system-prompt instructions.

    Returns:
        Instruction sentences, empty when no policy applies.

    """
    instructions: list[str] = []
    if choice == "none" and (explicit or tools):
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
    return instructions


def add_system_policy(messages: list[dict[str, JSON]], policy: str) -> None:
    """Append the policy to the system message, inserting one when absent.

    Raises:
        InvariantTypeError: Validated system content is internally inconsistent.

    """
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": policy.lstrip()})
        return
    content = messages[0]["content"]
    if isinstance(content, str):
        messages[0] = {**messages[0], "content": content + policy}
    elif isinstance(content, list):
        messages[0] = {
            **messages[0],
            "content": [*content, {"type": "text", "text": policy}],
        }
    else:
        msg = "Validated system message lacks text"
        raise InvariantTypeError(msg)


def parse_chat(body: dict[str, JSON]) -> Chat:
    """Prepare the actual model template inputs and explicit tool-choice policy.

    Returns:
        A complete chat request ready for rendering.

    """
    tools = parse_tools(body.get("tools", []))
    choice = parse_tool_choice(body, tools)
    parallel = boolean_value(
        body.get("parallel_tool_calls", True), "parallel_tool_calls"
    )
    messages = parse_messages(body.get("messages"))
    template_kwargs, thinking = parse_template_kwargs(body)
    instructions = choice_instructions(
        choice, explicit="tool_choice" in body, tools=tools, parallel=parallel
    )
    if instructions:
        add_system_policy(
            messages, "\n\nTool choice for this response: " + " ".join(instructions)
        )
    template_kwargs["tools"] = (
        [] if choice == "none" else [tool.wire for tool in tools.values()]
    )
    return Chat(messages, template_kwargs, tools, choice, parallel, thinking)


def split_reasoning(text: str, *, thinking: bool) -> tuple[str | None, str]:
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


def strip_parameter_newlines(raw: str) -> str:
    """Remove the one newline the template writes around each parameter value.

    Returns:
        The raw value without its framing newlines.

    """
    if raw.startswith("\r\n"):
        raw = raw[2:]
    elif raw.startswith("\n"):
        raw = raw[1:]
    if raw.endswith("\r\n"):
        return raw[:-2]
    if raw.endswith("\n"):
        return raw[:-1]
    return raw


@dataclass
class ToolMarkup:
    """A cursor over the tool-call section of one model response."""

    content: str
    chat: Chat
    cursor: int

    def skip_space(self) -> None:
        """Advance past whitespace."""
        while self.cursor < len(self.content) and self.content[self.cursor].isspace():
            self.cursor += 1

    def consume(self, token: str) -> None:
        """Advance past optional whitespace and one required token.

        Raises:
            APIError: The token is absent.

        """
        self.skip_space()
        if not self.content.startswith(token, self.cursor):
            msg = "Model emitted incomplete or malformed tool markup"
            raise APIError(msg, 502, "invalid_tool_call")
        self.cursor += len(token)

    def tag_name(self, prefix: str) -> str:
        """Read the name of a `<prefix...>` tag.

        Returns:
            The text between the prefix and the closing bracket.

        Raises:
            APIError: The tag is not closed.

        """
        self.consume(prefix)
        end = self.content.find(">", self.cursor)
        if end < 0:
            msg = "Model emitted an incomplete tool tag"
            raise APIError(msg, 502, "invalid_tool_call")
        name = self.content[self.cursor : end]
        self.cursor = end + 1
        return name

    def parameter(self, name: str, arguments: dict[str, JSON]) -> None:
        """Decode one `<parameter=...>` element into the arguments.

        Raises:
            APIError: The parameter is malformed, duplicated, or invalid.

        """
        key = self.tag_name("<parameter=")
        if not key or any(char in key for char in "<>\r\n") or key in arguments:
            msg = "Model emitted invalid or duplicate parameter names"
            raise APIError(msg, 502, "invalid_tool_arguments")
        end = self.content.find("</parameter>", self.cursor)
        if end < 0:
            msg = "Model emitted an incomplete parameter"
            raise APIError(msg, 502, "invalid_tool_arguments")
        raw = self.content[self.cursor : end]
        if MARKUP.search(raw):
            msg = "Model emitted ambiguous nested tool markup"
            raise APIError(msg, 502, "invalid_tool_arguments")
        raw = strip_parameter_newlines(raw)
        try:
            arguments[key] = self.chat.tools[name].parameter(key, raw)
        except (Unresolvable, RecursionError) as exc:
            msg = "Tool schema reference cannot be evaluated"
            raise APIError(msg, 400, "invalid_schema") from exc
        self.cursor = end + len("</parameter>")

    def call(self) -> dict[str, JSON]:
        """Decode and validate one complete `<tool_call>` element.

        Returns:
            The OpenAI function call.

        Raises:
            APIError: The call is malformed, undeclared, or violates its schema.

        """
        self.consume("<tool_call>")
        name = self.tag_name("<function=")
        if name not in self.chat.tools:
            msg = "Model called an undeclared function"
            raise APIError(msg, 502, "invalid_tool_call")
        if self.chat.choice.startswith("named:") and name != self.chat.choice[6:]:
            msg = "Model did not satisfy named tool_choice"
            raise APIError(msg, 502, "tool_choice_not_satisfied")
        arguments: dict[str, JSON] = {}
        while True:
            self.skip_space()
            if self.content.startswith("</function>", self.cursor):
                break
            self.parameter(name, arguments)
        self.consume("</function>")
        self.consume("</tool_call>")
        try:
            valid = self.chat.tools[name].validator.is_valid(arguments)
        except (Unresolvable, RecursionError) as exc:
            msg = "Tool schema reference cannot be evaluated"
            raise APIError(msg, 400, "invalid_schema") from exc
        if not valid:
            msg = f"Model arguments violate the schema for {name}"
            raise APIError(msg, 502, "invalid_tool_arguments")
        return {
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False, allow_nan=False),
            },
        }


def check_no_tool_call(content: str, chat: Chat) -> None:
    """Check a response without `<tool_call>` against markup and tool_choice.

    Raises:
        APIError: Stray markup is present or a tool call was required.

    """
    if MARKUP.search(content):
        msg = "Model emitted malformed tool markup"
        raise APIError(msg, 502, "invalid_tool_call")
    if chat.choice == "required" or chat.choice.startswith("named:"):
        msg = "Model did not satisfy required tool_choice"
        raise APIError(msg, 502, "tool_choice_not_satisfied")


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
        check_no_tool_call(content, chat)
        return content, []
    if chat.choice == "none":
        msg = "Model emitted a tool call while tool_choice=none"
        raise APIError(msg, 502, "tool_choice_not_satisfied")
    if MARKUP.search(content[:start]):
        msg = "Model emitted malformed markup before a tool call"
        raise APIError(msg, 502, "invalid_tool_call")
    calls: list[dict[str, JSON]] = []
    markup = ToolMarkup(content, chat, start)
    while markup.cursor < len(content):
        calls.append(markup.call())
        markup.skip_space()
        if markup.cursor < len(content) and not content.startswith(
            "<tool_call>", markup.cursor
        ):
            msg = "Model emitted text after a tool call"
            raise APIError(msg, 502, "invalid_tool_call")
    if not chat.parallel and len(calls) > 1:
        msg = "Model violated parallel_tool_calls=false"
        raise APIError(msg, 502, "tool_choice_not_satisfied")
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
        msg = "Native terminal event lacks draft accounting"
        raise RuntimeError(msg)
    if accepted < 0 or rejected < 0:
        msg = "Native draft accounting is negative"
        raise RuntimeError(msg)
    rounds, remainder = divmod(accepted + rejected, window)
    if remainder:
        msg = "Native draft accounting is not whole verify rounds"
        raise RuntimeError(msg)
    committed = accepted + rounds
    if committed > completion_tokens:
        msg = "Native draft accounting exceeds the completion"
        raise RuntimeError(msg)
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


class ShutdownError(Exception):
    """The job was cancelled, or never started, because the server is stopping."""


def log_line(line: object) -> None:
    """Write one line to stdout and flush it for the container log."""
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


class RedactedTracebackFormatter(logging.Formatter):
    """Format tracebacks as frames and exception types, without exception messages.

    Messages can carry request bodies or model text, which the log must not hold.
    """

    @override
    def formatException(
        self,
        ei: tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None],
    ) -> str:
        chain: list[str] = []
        seen: set[int] = set()
        error = ei[1]
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            frames = "".join(traceback.format_tb(error.__traceback__))
            chain.append(
                f"Traceback (most recent call last):\n{frames}"
                f"{type(error).__module__}.{type(error).__qualname__}"
            )
            error = error.__cause__ or (
                None if error.__suppress_context__ else error.__context__
            )
        return "\n\nThe above exception led to:\n\n".join(reversed(chain))


class LogLineHandler(logging.Handler):
    """Emit log records through `log_line`, like every other server log line."""

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            log_line(self.format(record))
        except (OSError, ValueError):
            self.handleError(record)


def traceback_logger() -> logging.Logger:
    """Build the logger for last-resort handlers: message line plus redacted frames.

    Returns:
        The module logger, writing only through `log_line`.

    """
    handler = LogLineHandler()
    handler.setFormatter(RedactedTracebackFormatter("%(message)s"))
    logger = logging.getLogger(__name__)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


LOGGER = traceback_logger()


MAX_ENV_SECONDS = 86400


def env_flag(name: str, *, default: bool) -> bool:
    """Read a strict 0/1 environment switch.

    Returns:
        The switch value, or the default when unset.

    Raises:
        ValueError: The variable holds anything but 0 or 1.

    """
    value = os.environ.get(name)
    if value is None:
        return default
    if value not in {"0", "1"}:
        msg = f"{name} must be 0 or 1"
        raise ValueError(msg)
    return value == "1"


def env_seconds(name: str, default: int) -> int:
    """Read a strict whole-second duration (test hook for the save schedule).

    Returns:
        The duration, or the default when unset.

    Raises:
        ValueError: The variable is not an integer in 0..86400.

    """
    value = os.environ.get(name)
    if value is None:
        return default
    if not value.isdigit() or int(value) > MAX_ENV_SECONDS:
        msg = f"{name} must be whole seconds in 0..86400"
        raise ValueError(msg)
    return int(value)


def launch_image() -> str | None:
    """Identify the running image without trusting request data.

    The guardian's launch gate is the container entrypoint and receives the verified
    image ID as --candidate-image; docker-init (PID 1) keeps that argv. QWEN_IMAGE_ID
    serves launches without the gate and must agree with the gate when both exist.

    Returns:
        The full image ID, or None when it is unknown or contradictory.

    """
    gate = None
    argv: list[str]
    try:
        argv = [
            a.decode()
            for a in pathlib.Path("/proc/1/cmdline").read_bytes().split(b"\0")
        ]
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
    """Hash one file.

    Returns:
        The SHA-256 hex digest, or None if the file does not exist.

    """
    try:
        with pathlib.Path(path).open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    except FileNotFoundError:
        return None


def prefix_binding(args: argparse.Namespace) -> dict[str, JSON] | None:
    """Bind persisted K/V and recurrent bytes to everything outside cache geometry.

    Returns:
        The JSON binding, or None when the image cannot be identified.

    """
    image = launch_image()
    if image is None:
        return None
    try:
        with pathlib.Path("/proc/driver/nvidia/version").open(
            encoding="utf-8"
        ) as source:
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
                str(pathlib.Path(__file__).absolute()),
                str(pathlib.Path(args.target) / "config.json"),
                str(pathlib.Path(args.draft) / "config.json"),
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
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "gpu": str(torch.cuda.get_device_name(0)),
            "capability": list(torch.cuda.get_device_capability(0)),
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


BATCHED_NDIM = 2


def call_untyped(function: object) -> object:
    """Call a native method whose static type is wrong or unknown.

    Returns:
        The call's result, for the caller to narrow.

    Raises:
        InvariantTypeError: The object is not callable.

    """
    if not callable(function):
        msg = "Native generator iterate is not callable"
        raise InvariantTypeError(msg)
    return function()


class Server:
    """Own the fixed native EXL3/DFlash2 stack and its single generation worker."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Load the approved cache geometry before exposing the worker."""
        start = self._load_engine(args)
        self._check_variant()
        self.tokenizer_lock = threading.Lock()
        # None is the stop sentinel queued by begin_shutdown.
        self.queue: queue.Queue[Pending | None] = queue.Queue()
        self.failure: Exception | None = None
        self.stopping = threading.Event()
        self.stopped = threading.Event()
        self.stop_deadline = math.inf
        self.persist_debug = env_flag("QWEN_PREFIX_PERSIST_DEBUG", default=False)
        self.persist_idle = env_seconds(
            "QWEN_PREFIX_PERSIST_IDLE_SECONDS", PERSIST_IDLE_SECONDS
        )
        self.persist_interval = env_seconds(
            "QWEN_PREFIX_PERSIST_INTERVAL_SECONDS", PERSIST_INTERVAL_SECONDS
        )
        self.persist_dirty = False
        self.last_job_end = -math.inf
        self.last_save = -math.inf
        self.persist = self.open_prefix_cache(args) if args.prefix_cache else None
        threading.Thread(target=self._worker, daemon=True).start()
        free, total = torch.cuda.mem_get_info()
        log_line(
            f"[serve] READY in {time.monotonic() - start:.0f}s, "
            f"VRAM={(total - free) / 1e9:.2f} GB"
        )

    def _load_engine(self, args: argparse.Namespace) -> float:
        """Load target, draft, caches, tokenizer and generator.

        Returns:
            The monotonic time at which loading began, for the READY line.

        Raises:
            RuntimeError: The generator's draft window differs from the cache.

        """
        self.model_name = args.model_name
        self.max_model_len = args.max_model_len
        target_config = Config.from_directory(args.target)
        draft_config = Config.from_directory(args.draft)
        draft_model = Model.from_config(draft_config)
        max_history = draft_model.caps.get("default_draft_size", 4)
        model = Model.from_config(target_config)
        log_line(
            f"[serve] loading native EXL3 target, cache={args.cache_tokens}, "
            f"cq={args.cq}"
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
        self.gen = NativeGenerator(
            model,
            cache,
            self.tokenizer,
            draft_model=draft_model,
            draft_cache=draft_cache,
        )
        # Fixed (non-dynamic) verify window; usage accounting divides by it.
        self.draft_window: int = self.gen.num_draft_tokens
        if type(self.draft_window) is not int or self.draft_window != max_history:
            msg = "Generator draft window differs from the cache history"
            raise RuntimeError(msg)
        self.stop_ids = list(model.config.eos_token_id_list or [])
        if (
            self.tokenizer.eos_token_id is not None
            and self.tokenizer.eos_token_id not in self.stop_ids
        ):
            self.stop_ids.append(self.tokenizer.eos_token_id)
        return start

    def _check_variant(self) -> None:
        """Match the installed engine to the baked image variant.

        Raises:
            RuntimeError: The engine, variant, or tree settings disagree.

        """
        # The candidate engine commits greedy verify rounds through its admitted
        # acceptance decision and counts them; the baseline image must carry the
        # unpatched engine.
        variant = os.environ.get("QWEN_EXL3_VARIANT")
        if variant not in {"baseline", "candidate"}:
            msg = "QWEN_EXL3_VARIANT must name the baked image variant"
            raise RuntimeError(msg)
        self.verified_acceptance = variant == "candidate"
        if self.verified_acceptance != hasattr(self.gen, "greedy_verify_rounds"):
            msg = "Installed engine does not match the image variant"
            raise RuntimeError(msg)
        # Dynamic tree verify (exl3 0006): the candidate engine parses EXL3_TREE and
        # the test hook EXL3_TREE_FORCE_CHAIN strictly (0|1) and counts tree rounds;
        # the baseline engine has no tree, so both must be unset or 0 there.
        if self.verified_acceptance:
            if not hasattr(self.gen, "tree_verify_rounds"):
                msg = "Installed engine lacks the tree verify (exl3 0006)"
                raise RuntimeError(msg)
            self.tree_rounds_expected = bool(
                self.gen.tree and not self.gen.tree_force_chain
            )
        else:
            for name in ("EXL3_TREE", "EXL3_TREE_FORCE_CHAIN"):
                if os.environ.get(name) not in {None, "0"}:
                    msg = f"{name} requires the candidate engine"
                    raise RuntimeError(msg)
            self.tree_rounds_expected = False

    def check_context(self, ids: torch.Tensor, output_tokens: int = 0) -> None:
        """Enforce one sequence and the complete input-plus-output context budget.

        Raises:
            APIError: Shape, input length, or the native context limit is invalid.

        """
        if ids.ndim not in {1, BATCHED_NDIM} or (
            ids.ndim == BATCHED_NDIM and ids.shape[0] != 1
        ):
            msg = "Exactly one input sequence is required"
            raise APIError(msg)
        length = ids.shape[-1]
        if length <= 0 or length + output_tokens > self.max_model_len:
            msg = "Input plus output budget exceeds native 262144 context"
            raise APIError(
                msg,
                code="context_length_exceeded",
            )

    def render_chat(self, chat: Chat) -> torch.Tensor:
        """Render through the target tokenizer's actual Hugging Face template.

        Returns:
            The rendered token-ID tensor.

        Raises:
            APIError: The model template cannot render the supplied messages.
            InvariantTypeError: The tokenizer does not return its tensor type.

        """
        with self.tokenizer_lock:
            try:
                ids: object = self.tokenizer.hf_chat_template(
                    chat.messages, add_generation_prompt=True, **chat.template_kwargs
                )
            except (ValueError, TypeError, TemplateError) as exc:
                msg = "Messages could not be rendered by the model chat template"
                raise APIError(msg) from exc
        if not isinstance(ids, torch.Tensor):
            msg = "Native chat template did not return a token-ID tensor"
            raise InvariantTypeError(msg)
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
            msg = "Server is shutting down"
            raise APIError(msg, 503, "server_shutting_down")
        pending = Pending(ids, options)
        self.queue.put(pending)
        if not pending.event.wait(timeout=7200):
            msg = "Generation timed out after 7200 seconds"
            raise APIError(msg, 504, "generation_timeout")
        if isinstance(pending.error, ShutdownError):
            msg = "Server is shutting down"
            raise APIError(msg, 503, "server_shutting_down")
        if pending.error is not None:
            msg = "Native generation failed; inspect server logs"
            raise APIError(
                msg,
                503,
                "generation_failed",
            ) from pending.error
        if pending.result is None:
            msg = "Native worker completed without a result"
            raise RuntimeError(msg)
        return pending.result

    def open_prefix_cache(self, args: argparse.Namespace) -> PrefixStore | None:
        """Open and restore the persistent prefix cache, or run without it.

        Any persistence problem means running without the cache.

        Returns:
            The engine's PrefixStore, or None when persistence is off or unusable.

        Raises:
            ModuleNotFoundError: The engine lacks the persist module (patch 9501b).
            ValueError: The permutation test hook is not a non-negative integer.

        """
        if not env_flag(PERSIST_ENV, default=True):
            log_line(f"[persist] disabled by {PERSIST_ENV}=0")
            return None
        if not PERSIST_INSTALLED:
            msg = f"No module named {PERSIST_MODULE!r}"
            raise ModuleNotFoundError(msg, name=PERSIST_MODULE)

        binding = prefix_binding(args)
        if binding is None:
            log_line(
                "[persist] image identity unknown (launch gate --candidate-image or "
                "QWEN_IMAGE_ID); persistence disabled"
            )
            return None
        try:
            store = PrefixStore(args.prefix_cache, binding, self.gen, log=log_line)
        except (PersistError, OSError) as exc:
            log_line(f"[persist] disabled: {exc}")
            return None
        log_line(f"[persist] binding {store.key}")
        # Test hooks: a permuted physical placement, and a re-read of every restored
        # page
        permute = os.environ.get("QWEN_PREFIX_PERSIST_DEBUG_PERMUTE")
        if permute is not None and not permute.isdigit():
            msg = "QWEN_PREFIX_PERSIST_DEBUG_PERMUTE must be a non-negative integer"
            raise ValueError(msg)
        stats = store.restore(permute_seed=None if permute is None else int(permute))
        if self.persist_debug and stats.get("restored"):
            checked, mismatched = store.verify()
            log_line(f"[persist] verify checked={checked} mismatched={mismatched}")
            if mismatched:
                store.enabled = False
        return store if store.enabled else None

    def begin_shutdown(self, deadline: float) -> None:
        """Stop taking jobs and cancel the running one; the worker saves and exits."""
        self.stop_deadline = deadline
        self.stopping.set()
        self.queue.put(None)

    def wait_stopped(self) -> None:
        """Wait for the worker's final save, bounded by the stop budget."""
        _ = self.stopped.wait(max(0.0, self.stop_deadline - time.monotonic()) + 1)

    def _save_due(self) -> float | None:
        """Compute the delay until the next idle save.

        Returns:
            Seconds until the next idle save, or None if none is pending.

        """
        store = self.persist
        if store is None or not store.enabled or not self.persist_dirty or self.failure:
            return None
        due = max(
            self.last_job_end + self.persist_idle,
            self.last_save + self.persist_interval,
        )
        return max(0.0, due - time.monotonic())

    def _save(self, *, final: bool) -> None:
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
                log_line("[persist] final save skipped: earlier save still running")
                return
            if not self.persist_dirty:
                return
        self.last_save = time.monotonic()
        try:
            capture = store.capture(verify=self.persist_debug)
        except Exception as exc:
            # Last resort: any capture failure disables persistence, not the server.
            store.enabled = False
            LOGGER.exception(
                "[persist] capture failed (%s); persistence disabled",
                type(exc).__name__,
            )
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

    def _fail_late_jobs(self) -> None:
        """Fail whatever raced in behind the stop sentinel."""
        while True:
            try:
                late = self.queue.get_nowait()
            except queue.Empty:
                return
            if late is not None:
                late.error = ShutdownError()
                late.event.set()
            self.queue.task_done()

    def _next_pending(self) -> Pending | None:
        """Wait for the next job, running idle saves meanwhile.

        Returns:
            The next job to run, or None once stopping.

        """
        while True:
            try:
                pending = self.queue.get(timeout=self._save_due())
            except queue.Empty:
                self._save(final=False)
                continue
            if pending is not None and not self.stopping.is_set():
                return pending
            if pending is not None:
                pending.error = ShutdownError()
                pending.event.set()
            self.queue.task_done()
            if pending is None:
                self._fail_late_jobs()
                return None

    def _step(self) -> list[dict[object, object]]:
        """Run one generator iteration.

        Returns:
            The native events of this iteration.

        Raises:
            InvariantTypeError: The generator does not return a list of event dicts.

        """
        # `iterate` is wrapped by @torch.inference_mode, which type checkers see as
        # the decorator object rather than the method; narrow it at this boundary.
        events = call_untyped(self.gen.iterate)
        if not isinstance(events, list):
            msg = "Native generator iterate did not return a list"
            raise InvariantTypeError(msg)
        checked: list[dict[object, object]] = []
        for event in events:
            if not isinstance(event, dict):
                msg = "Native generator event is not a dict"
                raise InvariantTypeError(msg)
            checked.append(event)
        return checked

    def _generate(self, pending: Pending) -> dict[str, object]:
        """Run one native job to its terminal event.

        Returns:
            The job's terminal event.

        Raises:
            RuntimeError: The generator needs a restart or ended without an event.
            ShutdownError: The server began stopping during the job.

        """
        if self.failure is not None:
            msg = "Generator requires restart after a native failure"
            raise RuntimeError(msg) from self.failure
        ids = pending.input_ids
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        job = Job(
            input_ids=ids,
            max_new_tokens=pending.options.max_tokens,
            sampler=ArgmaxSampler(),
            stop_conditions=self.stop_ids,
            decode_special_tokens=not pending.options.skip_special_tokens,
            identifier=pending.identifier,
        )
        self.gen.enqueue(job)
        final = None
        while self.gen.num_remaining_jobs():
            if self.stopping.is_set():
                self.gen.cancel(job)
                raise ShutdownError
            for event in self._step():
                if event.get("identifier") == pending.identifier and event.get("eos"):
                    final = {
                        key: value
                        for key, value in event.items()
                        if isinstance(key, str)
                    }
        if final is None:
            msg = "Native generation ended without a terminal event"
            raise RuntimeError(msg)
        return final

    def _run_job(self, pending: Pending) -> Result:
        """Generate one job and check its native accounting.

        Returns:
            The complete result.

        Raises:
            RuntimeError: Native text, usage, or verify accounting is inconsistent.

        """
        verified_before = (
            self.gen.greedy_verify_rounds if self.verified_acceptance else 0
        )
        tree_before = self.gen.tree_verify_rounds if self.verified_acceptance else 0
        final = self._generate(pending)
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
            msg = "Native terminal event lacks complete text or usage"
            raise RuntimeError(msg)
        spec_rounds, spec_committed = speculative_counts(
            final, self.draft_window, completion_tokens
        )
        if (
            self.verified_acceptance
            and self.gen.greedy_verify_rounds - verified_before != spec_rounds
        ):
            msg = "A verify round bypassed the admitted acceptance decision"
            raise RuntimeError(msg)
        # With the dynamic tree on (and not forced to the chain) every greedy verify
        # round of this single-sequence argmax job is a tree round; otherwise none
        # is
        tree_rounds = (
            self.gen.tree_verify_rounds - tree_before if self.verified_acceptance else 0
        )
        if tree_rounds != (spec_rounds if self.tree_rounds_expected else 0):
            msg = "Tree verify rounds disagree with the EXL3_TREE configuration"
            raise RuntimeError(msg)
        if self.persist_debug:
            log_line(
                f"[persist] job prompt_tokens={prompt_tokens} "
                f"cached_pages={final.get('cached_pages')} "
                f"cached_tokens={final.get('cached_tokens')}"
            )
        return Result(
            text,
            prompt_tokens,
            completion_tokens,
            "length" if final.get("eos_reason") == "max_new_tokens" else "stop",
            spec_rounds,
            spec_committed,
        )

    def _worker(self) -> None:
        while True:
            pending = self._next_pending()
            if pending is None:
                break
            try:
                pending.result = self._run_job(pending)
            except ShutdownError as exc:
                pending.error = exc
            except Exception as exc:
                # Last resort: record the failure and return it to the waiting request.
                self.failure = exc
                pending.error = exc
                LOGGER.exception(
                    "[serve] native worker failure: %s", type(exc).__name__
                )
            finally:
                pending.event.set()
                self.queue.task_done()
                self.persist_dirty = True
                self.last_job_end = time.monotonic()
        self._save(final=True)
        self.stopped.set()


MAX_KEY_BYTES = 4096
FIRST_PRINTABLE = ord("!")
LAST_PRINTABLE = ord("~")


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
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) not in {
            0o400,
            0o600,
        }:
            msg = "API key must be a private regular file"
            raise ValueError(msg)
        key = key_file.read(4098).removesuffix(b"\n")
    if not 1 <= len(key) <= MAX_KEY_BYTES or not all(
        FIRST_PRINTABLE <= byte <= LAST_PRINTABLE for byte in key
    ):
        msg = "Invalid API key bytes"
        raise ValueError(msg)
    return "Bearer " + key.decode("ascii")


MAX_LENGTH_DIGITS = 10
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


def stream_deltas(
    reasoning: str | None,
    content: str,
    calls: list[dict[str, JSON]],
    finish: str,
) -> list[tuple[dict[str, JSON], str | None]]:
    """List the chat-completion chunk deltas of one buffered response.

    Returns:
        Each delta with its finish reason, in stream order.

    """
    deltas: list[tuple[dict[str, JSON], str | None]] = [
        ({"role": "assistant", "content": ""}, None)
    ]
    if reasoning is not None:
        deltas.append(({"reasoning_content": reasoning}, None))
    if content:
        deltas.append(({"content": content}, None))
    deltas.extend(
        ({"tool_calls": [{"index": index, **call}]}, None)
        for index, call in enumerate(calls)
    )
    deltas.append(({}, finish))
    return deltas


def sse_frame(chunk: dict[str, JSON]) -> bytes:
    """Encode one server-sent event.

    Returns:
        The UTF-8 `data:` frame.

    """
    return (
        "data: " + json.dumps(chunk, ensure_ascii=False, allow_nan=False) + "\n\n"
    ).encode("utf-8")


def sse_frames(
    common: dict[str, JSON],
    deltas: list[tuple[dict[str, JSON], str | None]],
    usage: dict[str, JSON] | None,
) -> list[bytes]:
    """Encode a buffered chat-completion stream.

    Args:
        common: Fields shared by every chunk.
        deltas: Chunk deltas with their finish reasons.
        usage: The usage object when the client asked for it, else None.

    Returns:
        Every frame, ending with `[DONE]`.

    """
    frames: list[bytes] = []
    for delta, reason in deltas:
        chunk: dict[str, JSON] = {
            **common,
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": reason}],
        }
        if usage is not None:
            chunk["usage"] = None
        frames.append(sse_frame(chunk))
    if usage is not None:
        frames.append(
            sse_frame({
                **common,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": usage,
            })
        )
    frames.append(b"data: [DONE]\n\n")
    return frames


class DeadlineReader(io.RawIOBase):
    """Read a blocking client socket only until the current read deadline.

    The wait uses poll(), so the socket stays blocking and writes have no timeout.
    """

    def __init__(self, sock: socket.socket) -> None:
        """Start with an expired deadline, so reads fail until the handler sets one."""
        super().__init__()
        self.sock = sock
        self.poller = select.poll()
        self.poller.register(sock, select.POLLIN)
        self.deadline = 0.0

    @override
    def readable(self) -> bool:
        return True

    @override
    def readinto(self, buffer: Buffer) -> int:
        """Receive available bytes, or fail when none arrive before the deadline.

        Returns:
            The number of bytes received; 0 at end of stream.

        Raises:
            TimeoutError: The deadline passed before data arrived.

        """
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or not self.poller.poll(math.ceil(remaining * 1000)):
            message = "Request read deadline passed"
            raise TimeoutError(message)
        return self.sock.recv_into(buffer)


class Handler(BaseHTTPRequestHandler):
    """Expose authenticated OpenAI endpoints without executing arbitrary tools."""

    protocol_version = "HTTP/1.1"
    engine: ClassVar[Server]
    authorization: ClassVar[str]
    reader: DeadlineReader

    @override
    def setup(self) -> None:
        super().setup()
        # Close the stock reader so its socket reference does not keep the socket open.
        self.rfile.close()
        self.reader = DeadlineReader(self.connection)
        self.rfile = io.BufferedReader(self.reader)

    @override
    def handle_one_request(self) -> None:
        # The request line and headers, including keep-alive idle time.
        self.reader.deadline = time.monotonic() + HEADER_SECONDS
        super().handle_one_request()

    @override
    def log_message(self, format: str, *args: object) -> None:
        # Do not echo request targets, credentials, bodies, or model text.
        log_line(f"[http] {self.address_string()} request complete")

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
            if error.status == HTTPStatus.UNAUTHORIZED
            else "invalid_request_error"
            if error.status < HTTPStatus.INTERNAL_SERVER_ERROR
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
            msg = "Expected one Content-Length and no Transfer-Encoding"
            raise APIError(msg)
        length = lengths[0]
        if (
            not length.isascii()
            or not length.isdecimal()
            or len(length) > MAX_LENGTH_DIGITS
        ):
            msg = "Invalid Content-Length"
            raise APIError(msg)
        size = int(length)
        if size > MAX_BODY:
            msg = "Request body exceeds 32 MiB"
            raise APIError(msg, 413, "body_too_large")
        if size <= 0:
            msg = "Request body must be nonempty"
            raise APIError(msg)
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

    def read_framed(self) -> bytes:
        """Read exactly the framed body bytes before the body deadline.

        Returns:
            The raw body.

        Raises:
            APIError: The connection ended before the whole body arrived.

        """
        size = self.body_length()
        self.reader.deadline = time.monotonic() + BODY_SECONDS
        raw = self.rfile.read(size)
        if len(raw) != size:
            msg = "Incomplete request body"
            raise APIError(msg)
        return raw

    def read_body(self) -> dict[str, JSON]:
        """Read one framed UTF-8 JSON object, closing invalid request connections.

        Returns:
            The validated request object.

        Raises:
            APIError: The body is incomplete or is not a valid JSON object.

        """
        try:
            raw = self.read_framed()
            return object_value(load_json(raw.decode("utf-8")), "Request body")
        except TimeoutError as exc:
            self.close_connection = True
            msg = f"Request body did not arrive within {BODY_SECONDS} seconds"
            raise APIError(
                msg,
                408,
                "request_timeout",
            ) from exc
        except (UnicodeError, ValueError, RecursionError) as exc:
            self.close_connection = True
            msg = (
                "Request body must be valid UTF-8 JSON without duplicate keys or "
                "nonfinite numbers"
            )
            raise APIError(msg) from exc
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
        if self.path in {"/health", "/v1/health"}:
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
        if self.path not in {
            "/v1/chat/completions",
            "/v1/chat/completions/render",
            "/v1/completions",
        }:
            self.close_connection = True
            self.send_error_json(APIError("Not found", 404, "not_found"))
            return
        try:
            body = self.read_body()
            if self.path == "/v1/completions":
                self.completions(body)
            else:
                self.chat(body, render=self.path.endswith("/render"))
        except APIError as exc:
            self.send_error_json(exc)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            # Last resort: an unexpected request failure becomes a 500 response.
            LOGGER.exception("[serve] request failure: %s", type(exc).__name__)
            self.send_error_json(
                APIError("Internal server error", 500, "internal_error")
            )

    def chat(self, body: dict[str, JSON], *, render: bool) -> None:
        """Render or generate a chat response, including buffered tool-call SSE.

        Raises:
            APIError: Rendering was requested with streaming enabled.

        """
        only_fields(body, CHAT_FIELDS, "chat request")
        options = parse_options(body, self.engine.model_name, chat=True)
        chat = parse_chat(body)
        if render and options.stream:
            msg = "Rendering does not support stream=true"
            raise APIError(msg)
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
        reasoning, content = split_reasoning(result.text, thinking=chat.thinking)
        with pattern_budget():
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
        deltas = stream_deltas(reasoning, content, calls, finish)
        frames = sse_frames(common, deltas, usage if options.include_usage else None)
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
        options = parse_options(body, self.engine.model_name, chat=False)
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
                    msg = (
                        "prompt token IDs must be nonempty integers within the "
                        "tokenizer vocabulary"
                    )
                    raise APIError(msg)
                ids = torch.tensor([tokens], dtype=torch.long)
            else:
                msg = (
                    "prompt must be a string, a token-ID list, or one nested "
                    "token-ID list"
                )
                raise APIError(msg)
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


REQUIRED_CQ = 3


def main() -> None:
    """Start the authenticated adapter only with the fixed native model geometry."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=CONTEXT)
    parser.add_argument("--cache-tokens", type=int, default=CACHE_TOKENS)
    parser.add_argument("--cq", type=int, default=REQUIRED_CQ)
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
        or args.cq != REQUIRED_CQ
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
    log_line(
        f"[serve] listening on http://{args.host}:{args.port}; "
        "SSE transport is buffered"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    if stopping.is_set():
        Handler.engine.wait_stopped()
        log_line("[serve] stopped")


if __name__ == "__main__":
    main()
