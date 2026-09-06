"""Generate a deterministic SHA-256 manifest for the Paper 2 revision package."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_NAMES = {"CHECKSUMS_SHA256.txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    if not package.is_dir():
        raise NotADirectoryError(package)
    output = args.output.resolve() if args.output else package / "CHECKSUMS_SHA256.txt"
    files = sorted(
        path for path in package.rglob("*")
        if path.is_file() and path.name not in EXCLUDED_NAMES and path.resolve() != output
    )
    lines = [f"{sha256(path)}  {path.relative_to(package).as_posix()}" for path in files]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} SHA-256 entries to {output}")


if __name__ == "__main__":
    main()
