"""Lay out the stock and the patched ExLlamaV3 trees for the bend/*_diff.py source links.

Usage (from the repository root):
    python3 -I -B bend/engine_trees.py OUT [--archive FILE] [--through PATCH]

Inputs:
- The ExLlamaV3 commit tarball of 355c6ee. Its URL and SHA-256 come from
  docker/base/sources.lock (key archives.exllamav3). With --archive, the script reads a
  local copy of that tarball (for example the file that docker/fetch-base.sh downloads).
  Without --archive, the script downloads the URL over HTTPS. In both cases the SHA-256
  must agree before the script reads the archive.
- patches/exl3/series and patches/exl3-ext/series, applied with the pinned checks of
  patches/exl3/apply.py and patches/exl3-ext/ext.py.

Output (OUT must not exist):
- OUT/stock: the stock exllamav3 package directory at 355c6ee.
- OUT/patched: the same package with patches/exl3/series, then patches/exl3-ext/series
  applied. Every patch hash, pre-image and post-image is checked.
- With --through PATCH: OUT/patched stops after the named patch of
  patches/exl3-ext/series. The script checks the patch hashes and the pre-images. The
  manifest has no post-images for a partial series, so the script does not check them.

The script fails closed: on any mismatch it stops with an error and a nonzero exit.
It builds in a temporary sibling directory and renames it to OUT only on success.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import NoReturn

REPO = Path(__file__).resolve().parent.parent
SOURCES_LOCK = REPO / "docker/base/sources.lock"
PACKAGE = "exllamav3"
REVISION = "355c6ee10fbd25b79070316a81ea0708cc18155a"
TOP = f"exllamav3-{REVISION}"
MAX_ARCHIVE = 64 << 20


def fail(message: str) -> NoReturn:
    """Stop with an error."""
    raise SystemExit(f"engine_trees: FAIL: {message}")


def load_ext() -> ModuleType:
    """Import patches/exl3-ext/ext.py (its directory name is not a package name).

    Returns:
        The ext module.
    """
    spec = importlib.util.spec_from_file_location(
        "elpis_exl3_ext", REPO / "patches/exl3-ext/ext.py"
    )
    if spec is None or spec.loader is None:
        fail("cannot load patches/exl3-ext/ext.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_pin() -> tuple[str, str]:
    """Read the tarball URL and SHA-256 from docker/base/sources.lock.

    Returns:
        (url, sha256).
    """
    record = json.loads(SOURCES_LOCK.read_bytes())["archives"]["exllamav3"]
    url, sha = record["url"], record["sha256"]
    if (
        not isinstance(url, str)
        or not url.startswith("https://")
        or REVISION not in url
    ):
        fail(
            f"{SOURCES_LOCK}: archives.exllamav3.url is not an https URL of {REVISION}"
        )
    if not isinstance(sha, str) or len(sha) != 64 or set(sha) - set("0123456789abcdef"):
        fail(f"{SOURCES_LOCK}: archives.exllamav3.sha256 is not a SHA-256")
    return url, sha


def fetch(url: str) -> bytes:
    """Download the tarball over HTTPS with certificate checks.

    Returns:
        The archive bytes.
    """
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310  https only
        data = response.read(MAX_ARCHIVE + 1)
    if len(data) > MAX_ARCHIVE:
        fail(f"{url}: archive larger than {MAX_ARCHIVE} bytes")
    return data


def extract(data: bytes, destination: Path) -> Path:
    """Extract the tarball's single top directory, with path checks.

    Returns:
        The extracted repository root.
    """
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            parts = PurePosixPath(member.name).parts
            if (
                not parts
                or parts[0] != TOP
                or ".." in parts
                or member.name.startswith("/")
            ):
                fail(f"archive member outside {TOP}/: {member.name!r}")
            if not (member.isdir() or member.isfile()):
                fail(f"archive member is not a file or directory: {member.name!r}")
        archive.extractall(destination, members=members, filter="data")
    return destination / TOP


def apply_through(ext: ModuleType, root: Path, through: str) -> None:
    """Apply patches/exl3-ext/series up to and including one patch, with pinned inputs."""
    manifest = ext.load()
    entries = ext.series(manifest)
    names = [name for name, _ in entries]
    if through not in names:
        fail(f"{through} is not in patches/exl3-ext/series")
    pins = ext.check_files(manifest)
    touched: set[str] = set()
    for name, sha in entries[: names.index(through) + 1]:
        path = ext.EXT_DIR / name
        ext.require(ext.digest(path) == sha, f"patch hash mismatch: {name}")
        for relative, creates, hunks in ext.parse_patch(
            path.read_text(encoding="utf-8")
        ):
            ext.require(relative in pins, f"patch touches unpinned file {relative}")
            if relative not in touched:
                pre = pins[relative][0]
                if pre is None:
                    ext.require(
                        creates, f"{relative}: created by the manifest, not by {name}"
                    )
                else:
                    ext.require(
                        ext.digest(root / relative) == pre,
                        f"pre-image mismatch: {relative}",
                    )
            ext.apply_file(root / relative, creates=creates, hunks=hunks)
            touched.add(relative)


def main(argv: list[str]) -> None:
    """Lay out OUT/stock and OUT/patched."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--archive", type=Path, help="local copy of the pinned tarball")
    parser.add_argument("--through", help="last patches/exl3-ext patch to apply")
    args = parser.parse_args(argv[1:])
    out: Path = args.out.resolve()
    if out.exists() or out.is_symlink():
        fail(f"{out} exists")
    url, sha = archive_pin()
    data = args.archive.read_bytes() if args.archive else fetch(url)
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        fail(f"archive sha256 {got} differs from the pin {sha}")
    ext = load_ext()
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        layout(ext, data, staging, args.through)
    except BaseException:
        shutil.rmtree(staging)
        raise
    staging.rename(out)
    print(f"stock   {out / 'stock'}")
    print(f"patched {out / 'patched'} (through {args.through or 'the full series'})")


def layout(ext: ModuleType, data: bytes, out: Path, through: str | None) -> None:
    """Write the stock and the patched package into the empty directory out."""
    root = extract(data, out / ".extract")
    stock = out / "stock"
    (root / PACKAGE).rename(stock)
    shutil.rmtree(out / ".extract")
    source = ext.load()["source"]
    tree = ext.safe_relative(ext.text_value(source["tree"], "source tree"))
    if ext.tree_digest(stock / tree) != ext.pinned(
        source["tree_sha256"], "tree_sha256"
    ):
        fail("stock extension sources differ from patches/exl3-ext/exl3-ext.json")
    patched = ext.engine_copy(stock, out / "patched")
    if through is None:
        ext.apply_series(ext.load(), patched)
    else:
        apply_through(ext, patched, through)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except ValueError as error:  # patches/exl3/apply.py fail()
        fail(str(error))
