"""Content identity of the EXL3 base image, fail closed.

Runs in the image build with the venv's interpreter, and later inside the image:

    python -I -B manifest.py inputs <sources.lock> <wheels|apt> <directory>
    python -I -B manifest.py installed <sources.lock>
    python -I -B manifest.py normalise <sources.lock>
    python -I -B manifest.py listing
    python -I -B manifest.py write <record>
    python -I -B manifest.py check <record>

inputs requires <directory> to hold exactly the locked files of one group of
sources.lock, each with its pinned sha256. installed requires the venv to hold
exactly the locked distributions at their locked versions.

The listing has one line per installed Debian package (`p <package>=<version>`),
per regular file (`f <sha256> <mode> <path>`) and per symlink
(`l <target> <path>`) under /opt/uv-python and /opt/venv, sorted. It skips only
the PER_BUILD files. Modification times are not part of it.
The content sha256 is the sha256 of the listing. It is the same for every build
of the same inputs, so docker/base/engine-manifest.json pins it.

The nvcc output in the exllamav3_ext shared object is not byte-identical from
build to build, and the exllamav3 RECORD holds its hash. These PER_BUILD files are
recorded by sha256 in the per-build record, not pinned.

normalise runs once in the build stage. uv writes uv_cache.json with the install
time into every dist-info; normalise removes it and its RECORD line. It writes the
exllamav3 direct_url.json as the pinned commit tarball and updates its RECORD line.
It removes every __pycache__ directory: marshal output is not byte-identical from
run to run, and bytecode outside the listing could run in place of its source.
Imports then compile in memory.
write stores the record: the content sha256 and the PER_BUILD hashes.
check computes both again, requires the stored record to match and prints it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from importlib import metadata
from pathlib import Path
from typing import NoReturn

ROOTS = (Path("/opt/uv-python"), Path("/opt/venv"))
SITE = Path("/opt/venv/lib/python3.13/site-packages")
DIST = SITE / "exllamav3-1.5.0.dist-info"
PER_BUILD = (
    SITE / "exllamav3_ext.cpython-313-x86_64-linux-gnu.so",
    DIST / "RECORD",
)
DPKG_STATUS = Path("/var/lib/dpkg/status")


def fail(message: str) -> NoReturn:
    """Stop with an error."""
    raise SystemExit(f"manifest.py: {message}")


def require(condition: bool, message: str) -> None:
    """Stop unless the condition holds."""
    if not condition:
        fail(message)


def digest(path: Path) -> str:
    """Hash one regular file.

    Returns:
        The lowercase hex sha256.
    """
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def packages() -> list[str]:
    """Read the installed Debian packages from the dpkg status file.

    Returns:
        Sorted `p <package>=<version>` lines.
    """
    lines: list[str] = []
    for stanza in DPKG_STATUS.read_text(encoding="utf-8").split("\n\n"):
        fields: dict[str, str] = {}
        for line in stanza.splitlines():
            if line and not line[0].isspace() and ":" in line:
                key, _, value = line.partition(":")
                fields[key] = value.strip()
        if fields.get("Status") == "install ok installed":
            lines.append(f"p {fields['Package']}={fields['Version']}\n")
    require(bool(lines), "no installed Debian packages")
    return sorted(lines)


def files() -> list[str]:
    """List every file and symlink under the roots.

    Returns:
        Sorted `f` and `l` lines.
    """
    lines: list[str] = []
    for root in ROOTS:
        require(root.is_dir() and not root.is_symlink(), f"{root} is not a directory")
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(dirnames)
            for name in [*filenames, *dirnames]:
                path = Path(directory) / name
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    lines.append(f"l {os.readlink(path)} {path}\n")
                elif stat.S_ISREG(info.st_mode) and path not in PER_BUILD:
                    mode = stat.S_IMODE(info.st_mode)
                    lines.append(f"f {digest(path)} {mode:04o} {path}\n")
                else:
                    require(
                        stat.S_ISDIR(info.st_mode) or path in PER_BUILD,
                        f"unexpected file type: {path}",
                    )
    return sorted(lines)


def listing() -> bytes:
    """Build the full listing.

    Returns:
        The listing bytes.
    """
    return "".join(packages() + files()).encode()


def record() -> dict[str, object]:
    """Compute the record of this image.

    Returns:
        The content sha256 and the per-build hashes.
    """
    for path in PER_BUILD:
        require(path.is_file() and not path.is_symlink(), f"missing {path}")
    return {
        "schema": 1,
        "content_sha256": hashlib.sha256(listing()).hexdigest(),
        "per_build": {str(path): digest(path) for path in PER_BUILD},
    }


def encode(document: dict[str, object]) -> bytes:
    """Serialise a record canonically.

    Returns:
        The JSON bytes.
    """
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def record_line(path: Path) -> str:
    """Build the RECORD line of one file of the distribution.

    Returns:
        `<path>,sha256=<urlsafe base64>,<size>`.
    """
    data = path.read_bytes()
    sha = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return f"{path.relative_to(SITE).as_posix()},sha256={sha.decode()},{len(data)}"


def drop_cache(dist: Path) -> None:
    """Remove uv_cache.json (it holds the install time) and its RECORD line."""
    name = f"{dist.name}/uv_cache.json"
    lines = dist.joinpath("RECORD").read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if line.split(",", 1)[0] != name]
    require(len(kept) == len(lines) - 1, f"RECORD of {dist.name} lacks uv_cache.json")
    dist.joinpath("uv_cache.json").unlink()
    _ = dist.joinpath("RECORD").write_text("".join(f"{line}\n" for line in kept))


def normalise(sources: Path) -> None:
    """Remove install times, bytecode and the exllamav3 build path."""
    for root in ROOTS:
        for cache in sorted(root.rglob("__pycache__")):
            require(cache.is_dir() and not cache.is_symlink(), f"bad {cache}")
            shutil.rmtree(cache)
    caches = sorted(SITE.glob("*.dist-info/uv_cache.json"))
    require(
        len(caches) == len(list(SITE.glob("*.dist-info"))),
        "a distribution lacks uv_cache.json",
    )
    for cache in caches:
        drop_cache(cache.parent)
    archive = json.loads(sources.read_bytes())["archives"]["exllamav3"]
    url, sha = archive["url"], archive["sha256"]
    require(
        isinstance(url, str) and url.startswith("https://") and isinstance(sha, str),
        "bad exllamav3 archive record",
    )
    direct = DIST / "direct_url.json"
    require(
        json.loads(direct.read_bytes())
        == {"url": "file:///src/exllamav3", "dir_info": {}},
        "exllamav3 was not installed from /src/exllamav3",
    )
    _ = direct.write_text(
        json.dumps(
            {"archive_info": {"hashes": {"sha256": sha}}, "url": url},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    name = f"{DIST.name}/direct_url.json"
    lines = DIST.joinpath("RECORD").read_text(encoding="utf-8").splitlines()
    updated = [
        record_line(direct) if line.split(",", 1)[0] == name else line for line in lines
    ]
    require(updated != lines, "RECORD does not list direct_url.json")
    _ = DIST.joinpath("RECORD").write_text("".join(f"{line}\n" for line in updated))


def locked(sources: Path, group: str) -> dict[str, str]:
    """Read the file pins of one group of sources.lock.

    Returns:
        Path relative to the group directory, to sha256.
    """
    lock = json.loads(sources.read_bytes())
    require(lock["schema"] == 1, "unsupported sources.lock schema")
    records = {"wheels": lock["wheels"], "apt": lock["apt"]["packages"]}[group]
    pins = {
        str(item["file"]).split("/", 1)[-1]: str(item["sha256"]) for item in records
    }
    require(len(pins) == len(records), f"duplicate file in sources.lock {group}")
    return pins


def inputs(sources: Path, group: str, path: Path) -> None:
    """Require exactly the locked files of one group, each with its sha256."""
    pins = locked(sources, group)
    require(path.is_dir(), f"missing {path}")
    found = sorted(entry.name for entry in path.iterdir())
    require(found == sorted(pins), f"{path} does not hold exactly the locked files")
    for name, sha in pins.items():
        entry = path / name
        require(entry.is_file() and not entry.is_symlink(), f"{entry} is not a file")
        require(digest(entry) == sha, f"{entry} differs from its pin")


def installed(sources: Path) -> None:
    """Require the venv to hold exactly the locked distributions."""
    lock = json.loads(sources.read_bytes())
    pinned = sorted((str(w["name"]), str(w["version"])) for w in lock["wheels"])
    present = sorted(
        (re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower(), dist.version)
        for dist in metadata.distributions(path=[str(SITE)])
    )
    require(present == pinned, "installed distributions differ from sources.lock")


def main(argv: list[str]) -> None:
    """Dispatch one command."""
    if argv[1:] == ["listing"]:
        _ = sys.stdout.buffer.write(listing())
        return
    if len(argv) == 5 and argv[1] == "inputs":
        require(argv[3] in {"wheels", "apt"}, "unknown input group")
        inputs(Path(argv[2]), argv[3], Path(argv[4]))
        return
    if len(argv) != 3 or argv[1] not in {"installed", "normalise", "write", "check"}:
        fail("usage: see the docstring of docker/base/manifest.py")
    path = Path(argv[2])
    if argv[1] == "installed":
        installed(path)
        return
    if argv[1] == "normalise":
        normalise(path)
        return
    document = encode(record())
    if argv[1] == "write":
        require(not path.exists(), f"{path} already exists")
        _ = path.write_bytes(document)
        path.chmod(0o444)
        return
    require(
        path.read_bytes() == document, "the base differs from its recorded manifest"
    )
    _ = sys.stdout.buffer.write(document)


if __name__ == "__main__":
    main(sys.argv)
