"""Resolve the EXL3 base inputs to exact files and write the locks. Needs network.

    python3 -I -B docker/base/lock.py

Maintainer tool. The build never runs it. It reads docker/base/requirements.in,
docker/base/apt.in and serve/exl3-requirements.txt and writes
docker/base/requirements.lock and docker/base/sources.lock.

- Python: for each `name==version  # index: <index>` line it reads only that
  index's simple page and selects the one wheel whose tags have the best rank for
  CPython 3.13 on manylinux_2_39 x86_64 (the pip tag order). The index page gives
  the sha256.
- Serve wheels: each hash in serve/exl3-requirements.txt must name exactly one
  file on the PyPI page of that project.
- Debian packages: it runs the pinned CUDA runtime image with network access, points
  apt at the snapshot.ubuntu.com timestamp of apt.in and records the pool file and
  sha256 that the signed snapshot index gives for each package=version.
- Archives: it downloads the exllamav3 commit tarball and the CPython build that
  uv 0.9.15 installs. The CPython sha256 must equal the one in uv's own table.

Any missing, ambiguous or unexpected input stops the tool.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import NoReturn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEXES = {
    "cu130": "https://download.pytorch.org/whl/cu130/",
    "pypi": "https://pypi.org/simple/",
}
IMAGES = {
    "devel": "nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d",
    "runtime": "nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3",
    "uv": "ghcr.io/astral-sh/uv:0.9.15@sha256:4c1ad814fe658851f50ff95ecd6948673fffddb0d7994bdb019dcb58227abd52",
}
EXLLAMAV3 = (
    "https://codeload.github.com/r0b0tlab/exllamav3/tar.gz/"
    "355c6ee10fbd25b79070316a81ea0708cc18155a"
)
UV_TABLE = (
    "https://raw.githubusercontent.com/astral-sh/uv/0.9.15/"
    "crates/uv-python/download-metadata.json"
)
UV_KEY = "cpython-3.13.10-linux-x86_64-gnu"
UV_RELEASES = "https://github.com/astral-sh/python-build-standalone/releases/download/"
SNAPSHOT = "https://snapshot.ubuntu.com/ubuntu/"
PIN = re.compile(r"^([A-Za-z0-9._-]+)==(\S+)\s+# index: (\w+)$")
HASH = re.compile(r"^[0-9a-f]{64}$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9._+~%-]+$")
APT_SCRIPT = r"""set -euo pipefail
rm -f /etc/apt/sources.list.d/*
cat > /etc/apt/sources.list.d/snapshot.sources <<EOF
Types: deb
URIs: $SNAPSHOT_URL
Suites: noble noble-updates noble-security
Components: main universe
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF
apt-get update -qq >/dev/null
for spec in "$@"; do
  apt-cache show --no-all-versions "$spec" | awk -v s="$spec" \
    '/^Filename:/ {f = $2} /^SHA256:/ {h = $2} END {print s, f, h}'
done
"""


def fail(message: str) -> NoReturn:
    """Stop with an error."""
    raise SystemExit(f"lock.py: {message}")


def require(condition: bool, message: str) -> None:
    """Stop unless the condition holds."""
    if not condition:
        fail(message)


def get(url: str) -> bytes:
    """Read one HTTPS URL with certificate verification.

    Returns:
        The response body.
    """
    require(url.startswith("https://"), f"not an HTTPS URL: {url}")
    with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310
        return response.read()


class Links(HTMLParser):
    """Collect the href of every anchor in a simple index page."""

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record anchors."""
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.hrefs.append(href)


def canonical(name: str) -> str:
    """Normalise a project name (PEP 503).

    Returns:
        The normalised name.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def target_tags() -> dict[tuple[str, str, str], int]:
    """Rank the tags of CPython 3.13 on manylinux_2_39 x86_64 like pip does.

    Returns:
        Tag to rank; a lower rank is preferred.
    """
    legacy = {
        17: "manylinux2014_x86_64",
        12: "manylinux2010_x86_64",
        5: "manylinux1_x86_64",
    }
    expanded: list[str] = []
    for minor in range(39, 4, -1):
        expanded.append(f"manylinux_2_{minor}_x86_64")
        if minor in legacy:
            expanded.append(legacy[minor])
    platforms = [*expanded, "linux_x86_64"]
    order: list[tuple[str, str, str]] = []
    order += [("cp313", "cp313", p) for p in platforms]
    order += [("cp313", "abi3", p) for p in platforms]
    order += [("cp313", "none", p) for p in platforms]
    for minor in range(12, 1, -1):
        order += [(f"cp3{minor}", "abi3", p) for p in platforms]
    pythons = ["py313", "py3", *[f"py3{minor}" for minor in range(12, -1, -1)]]
    order += [(py, "none", p) for py in pythons for p in platforms]
    order.append(("cp313", "none", "any"))
    order += [(py, "none", "any") for py in pythons]
    return {tag: rank for rank, tag in enumerate(order)}


RANKS = target_tags()


def wheel(filename: str) -> tuple[str, str, int] | None:
    """Parse a wheel name and rank its best tag for the target.

    Returns:
        (canonical name, version, rank), or None if no tag fits.
    """
    if not filename.endswith(".whl"):
        return None
    parts = filename[: -len(".whl")].split("-")
    if len(parts) not in {5, 6}:
        return None
    pythons, abis, platforms = (part.split(".") for part in parts[-3:])
    ranks = [
        RANKS[(py, abi, plat)]
        for py in pythons
        for abi in abis
        for plat in platforms
        if (py, abi, plat) in RANKS
    ]
    if not ranks:
        return None
    return canonical(parts[0]), parts[1], min(ranks)


def index_files(index: str, project: str) -> list[tuple[str, str, str]]:
    """List the files of one project on one index.

    Returns:
        (file name, URL without fragment, sha256) for every file. The sha256 is
        empty when the index does not give one.
    """
    base = INDEXES[index]
    page = base + canonical(project) + "/"
    parser = Links()
    parser.feed(get(page).decode("utf-8"))
    files: list[tuple[str, str, str]] = []
    for href in parser.hrefs:
        url, _, fragment = urllib.parse.urljoin(page, href).partition("#")
        sha = fragment.removeprefix("sha256=") if fragment else ""
        require(sha == "" or HASH.match(sha) is not None, f"bad hash on {page}: {href}")
        name = urllib.parse.unquote(url.rsplit("/", 1)[1])
        files.append((name, url, sha))
    return files


def lock_wheels() -> list[dict[str, str]]:
    """Select one wheel per requirements.in line.

    Returns:
        The wheel records, sorted by name.
    """
    records: list[dict[str, str]] = []
    for line in (HERE / "requirements.in").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match = PIN.match(line)
        if match is None:
            fail(f"bad requirements.in line {line!r}")
        name, version, index = match.groups()
        require(index in INDEXES, f"unknown index {index!r}")
        best: list[tuple[int, str, str, str]] = []
        for filename, url, sha in index_files(index, name):
            parsed = wheel(filename)
            if parsed and parsed[0] == canonical(name) and parsed[1] == version:
                best.append((parsed[2], filename, url, sha))
        require(bool(best), f"no compatible wheel for {name}=={version} on {index}")
        best.sort()
        require(
            len(best) == 1 or best[0][0] != best[1][0],
            f"two wheels tie for {name}=={version}",
        )
        _, filename, url, sha = best[0]
        require(SAFE_FILE.match(filename) is not None, f"unsafe file name {filename}")
        if not sha:
            # The cu130 index gives no hash for some mirrored files: hash the download.
            sha = hashlib.sha256(get(url)).hexdigest()
        records.append({
            "name": canonical(name),
            "version": version,
            "index": index,
            "file": f"wheels/{filename}",
            "url": url,
            "sha256": sha,
        })
    return sorted(records, key=lambda record: record["name"])


def lock_serve_wheels() -> list[dict[str, str]]:
    """Find the PyPI file of every hash in serve/exl3-requirements.txt.

    Returns:
        The wheel records, in file order.
    """
    records: list[dict[str, str]] = []
    for line in (ROOT / "serve" / "exl3-requirements.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(\S+)==(\S+) --hash=sha256:([0-9a-f]{64})", line)
        if match is None:
            fail(f"bad serve requirement {line!r}")
        name, version, sha = match.groups()
        found = [entry for entry in index_files("pypi", name) if entry[2] == sha]
        require(len(found) == 1, f"no single PyPI file has the hash of {name}")
        filename, url, _ = found[0]
        parsed = wheel(filename)
        require(
            parsed is not None
            and parsed[0] == canonical(name)
            and parsed[1] == version,
            f"the pinned {name} file is not a compatible {version} wheel",
        )
        require(SAFE_FILE.match(filename) is not None, f"unsafe file name {filename}")
        records.append({
            "name": canonical(name),
            "version": version,
            "index": "pypi",
            "file": f"serve-wheels/{filename}",
            "url": url,
            "sha256": sha,
        })
    return records


def lock_apt() -> tuple[str, list[dict[str, str]]]:
    """Resolve apt.in at its snapshot inside the pinned runtime image.

    Returns:
        The snapshot URL and one record per package.
    """
    lines = [
        line
        for line in (HERE / "apt.in").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    require(lines[0].startswith("snapshot "), "apt.in must start with the snapshot")
    stamp = lines[0].split()[1]
    require(re.fullmatch(r"\d{8}T\d{6}Z", stamp) is not None, "bad snapshot stamp")
    specs = lines[1:]
    require(
        all(re.fullmatch(r"[a-z0-9.+-]+=[A-Za-z0-9.+:~-]+", spec) for spec in specs),
        "bad apt.in package line",
    )
    snapshot = f"{SNAPSHOT}{stamp}/"
    result = subprocess.run(
        [
            "docker", "run", "--rm", "--pull=never",
            "--env", f"SNAPSHOT_URL={snapshot}",
            "--entrypoint", "bash", IMAGES["runtime"],
            "-c", APT_SCRIPT, "lock-apt", *specs,
        ],
        check=True,
        capture_output=True,
        text=True,
    )  # fmt: skip
    packages: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        spec, pool, sha = line.split()
        package, version = spec.split("=", 1)
        filename = pool.rsplit("/", 1)[1]
        require(
            pool.startswith("pool/") and SAFE_FILE.match(filename) is not None,
            f"bad pool path {pool}",
        )
        require(HASH.match(sha) is not None, f"bad hash for {spec}")
        packages.append({
            "package": package,
            "version": version,
            "file": f"debs/{filename}",
            "url": snapshot + pool,
            "sha256": sha,
        })
    require(
        [f"{p['package']}={p['version']}" for p in packages] == specs,
        "apt did not resolve every apt.in line",
    )
    return snapshot, packages


def lock_archives() -> dict[str, dict[str, str]]:
    """Hash the exllamav3 tarball and check CPython against uv's table.

    Returns:
        The archive records.
    """
    table = json.loads(get(UV_TABLE))[UV_KEY]
    url = table["url"]
    require(url.startswith(UV_RELEASES), "uv names an unexpected CPython URL")
    python = get(url)
    python_sha = hashlib.sha256(python).hexdigest()
    require(python_sha == table["sha256"], "CPython archive differs from uv's table")
    relative = urllib.parse.unquote(url[len(UV_RELEASES) :])
    return {
        "exllamav3": {
            "url": EXLLAMAV3,
            "sha256": hashlib.sha256(get(EXLLAMAV3)).hexdigest(),
            "file": "exllamav3-355c6ee.tar.gz",
        },
        "cpython": {
            "url": url,
            "sha256": python_sha,
            "file": f"python/{relative}",
        },
    }


def main() -> None:
    """Resolve every input and write both locks."""
    wheels = lock_wheels()
    snapshot, packages = lock_apt()
    sources = {
        "schema": 1,
        "images": IMAGES,
        "archives": lock_archives(),
        "apt": {"snapshot": snapshot, "packages": packages},
        "wheels": wheels,
        "serve_wheels": lock_serve_wheels(),
    }
    lock = [
        "# Generated by docker/base/lock.py from requirements.in. Do not edit.\n",
        "# One wheel per line; the source URL of each is in sources.lock.\n",
    ]
    lock += [
        f"{w['name']}=={w['version']} --hash=sha256:{w['sha256']}\n" for w in wheels
    ]
    (HERE / "requirements.lock").write_text("".join(lock))
    (HERE / "sources.lock").write_text(json.dumps(sources, indent=2) + "\n")
    print(f"Locked {len(wheels)} wheels and {len(packages)} packages")


if __name__ == "__main__":
    if len(sys.argv) != 1:
        fail("usage: python3 -I -B docker/base/lock.py")
    main()
