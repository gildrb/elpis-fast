#!/usr/bin/env python3
# Copyright (c) 2026 Gil Rodrigues
"""Probe authenticated readiness once; Docker owns scheduling and process restarts."""

import http.client
import json
import logging
import signal
import sys
from pathlib import Path
from types import FrameType

MAX_KEY_BYTES = 4096
MIN_KEY_CHARACTER = 33
MAX_KEY_CHARACTER = 126
HTTP_OK = 200
PROBE_SECONDS = 310
MAX_BODY_BYTES = 65536


def deadline_expired(_signum: int, _frame: FrameType | None) -> None:
    """Interrupt the current probe when its whole-exchange budget expires.

    Raises:
        TimeoutError: The API exchange exceeded its deadline.

    """
    message = "API probe deadline exceeded"
    raise TimeoutError(message)


def _check_models(response: http.client.HTTPResponse, model: str) -> str | None:
    """Validate every inventory entry without exposing response content.

    Returns:
        A private failure description, or None on success.

    """
    body = response.read(MAX_BODY_BYTES + 1)
    if len(body) > MAX_BODY_BYTES:
        return "models: invalid or oversized JSON schema"
    payload: object = json.loads(body)
    if not isinstance(payload, dict):
        return "models: invalid or oversized JSON schema"
    models: object = payload.get("data")
    if not isinstance(models, list):
        return "models: invalid or oversized JSON schema"
    found = False
    for item in models:
        if not isinstance(item, dict):
            return "models: invalid or oversized JSON schema"
        identifier: object = item.get("id")
        if not isinstance(identifier, str):
            return "models: invalid or oversized JSON schema"
        if identifier == model:
            found = True
    return None if found else "models: expected model missing"


def _probe_endpoints(port: int, key: str, model: str, budget: float) -> str | None:
    """Close each connection before reporting its health or inventory failure.

    Returns:
        A private failure description, or None on success.

    """
    for path in ("/health", "/v1/models"):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=budget)
        try:
            connection.request("GET", path, headers={"Authorization": "Bearer " + key})
            response = connection.getresponse()
            if path == "/health":
                # Closing with unread bytes sends a TCP reset, which the server logs
                # as a ConnectionResetError traceback on every probe.
                response.read(MAX_BODY_BYTES + 1)
            if response.status != HTTP_OK:
                return f"{path}: HTTP {response.status}"
            if path == "/v1/models":
                failure = _check_models(response, model)
                if failure is not None:
                    return failure
        finally:
            connection.close()
    return None


def probe(port: int, key: str, model: str, budget: float) -> str | None:
    """Check scheduler health and the authenticated model inventory.

    Returns:
        A private failure description, or None on success.

    """
    # SIGALRM bounds the entire exchange, including slowly arriving bodies.
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        failure = _probe_endpoints(port, key, model, budget)
    except TimeoutError:
        return "API probe timeout"
    except OSError:
        return "API connection failure"
    except http.client.HTTPException:
        return "API HTTP protocol failure"
    except ValueError:
        return "models: invalid JSON"
    # Never expose response bodies, headers or exception text containing keys.
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return failure


def main() -> int:
    """Check the fixed in-container API without exposing credentials.

    Returns:
        Zero only after a successful authenticated readiness probe.

    Raises:
        ValueError: The credential has an invalid size or format.

    """
    logger = logging.getLogger(__name__)
    with Path("/app/api_key.txt").open("rb") as credential:
        raw_key = credential.read(MAX_KEY_BYTES + 2)
    # Same bound as the server: at most MAX_KEY_BYTES before one trailing newline.
    key_bytes = raw_key.removesuffix(b"\n")
    if len(key_bytes) > MAX_KEY_BYTES:
        message = "credential is too large"
        raise ValueError(message)
    key = key_bytes.decode("ascii")
    if len(key) == 0 or not all(
        MIN_KEY_CHARACTER <= ord(char) <= MAX_KEY_CHARACTER for char in key
    ):
        message = "invalid credential format"
        raise ValueError(message)
    signal.signal(signal.SIGALRM, deadline_expired)
    failure = probe(18020, key, "qwen3.8-27b", PROBE_SECONDS)
    if failure is not None:
        logger.error("Qwen probe failed: %s.", failure)
        return 1
    logger.info("Authenticated model API is ready.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.exit(main())
    except (OSError, ValueError):
        sys.exit("Qwen healthcheck configuration failed.")
