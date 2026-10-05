#!/usr/bin/env python3
# Copyright (c) 2026 inference contributors
"""Prepare an exclusive transport-only copy of a frozen native evaluation config."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from verifiers.utils.eval_utils import load_toml_config

ALLOWED_URLS = (
    "http://127.0.0.1:18020/v1",
    "http://127.0.0.1:18021/v1",
)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("destination", type=Path)
parser.add_argument("--api-base-url", choices=ALLOWED_URLS)
arguments = parser.parse_args()
source_bytes = arguments.source.read_bytes()
source_text = source_bytes.decode("utf-8")
original_configs = load_toml_config(arguments.source)
original_urls = {config.get("api_base_url") for config in original_configs}
if len(original_urls) != 1:
    message = "The frozen config must have exactly one native transport URL."
    raise ValueError(message)
original_url = next(iter(original_urls))
if original_url not in ALLOWED_URLS:
    message = "The frozen transport URL is not an approved loopback endpoint."
    raise ValueError(message)
effective_url = (
    original_url if arguments.api_base_url is None else arguments.api_base_url
)
effective_text = source_text
if arguments.api_base_url is not None:
    source_line = f'api_base_url = "{original_url}"'
    if source_text.count(source_line) != 1:
        message = "Expected one exact frozen transport assignment; refusing rewrite."
        raise ValueError(message)
    effective_text = source_text.replace(
        source_line, f'api_base_url = "{effective_url}"', 1
    )
with arguments.destination.open("xb") as stream:
    stream.write(effective_text.encode("utf-8"))
effective_configs = load_toml_config(arguments.destination)
original_payload = json.dumps(
    [
        {key: value for key, value in config.items() if key != "api_base_url"}
        for config in original_configs
    ],
    sort_keys=True,
    separators=(",", ":"),
)
effective_payload = json.dumps(
    [
        {key: value for key, value in config.items() if key != "api_base_url"}
        for config in effective_configs
    ],
    sort_keys=True,
    separators=(",", ":"),
)
if original_payload != effective_payload:
    message = "Native normalized config changed outside the transport URL."
    raise ValueError(message)
if {config.get("api_base_url") for config in effective_configs} != {effective_url}:
    message = "Effective native transport URL does not match the approved override."
    raise ValueError(message)
if arguments.source.read_bytes() != source_bytes:
    message = "Frozen config changed during transport preparation."
    raise ValueError(message)
operator_recipe_identity = os.environ.get("QWEN_RECIPE_ID")
if not operator_recipe_identity:
    operator_recipe_identity = "unverified-running-deployment"
provenance = {
    "original_config": str(arguments.source),
    "effective_config": str(arguments.destination),
    "original_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "effective_sha256": hashlib.sha256(arguments.destination.read_bytes()).hexdigest(),
    "original_api_base_url": original_url,
    "effective_api_base_url": effective_url,
    "override_requested": arguments.api_base_url is not None,
    "native_normalized_nontransport_payload_identical": True,
    "native_normalized_nontransport_sha256": hashlib.sha256(
        original_payload.encode("utf-8")
    ).hexdigest(),
    "operator_recipe_identity": operator_recipe_identity,
    "recipe_identity_independently_verified": False,
}
with arguments.destination.with_suffix(".provenance.json").open("x") as stream:
    json.dump(provenance, stream, indent=2)
    stream.write("\n")
