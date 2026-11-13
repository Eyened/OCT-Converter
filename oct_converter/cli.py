"""Command-line conversion of proprietary OCT files to DICOM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oct_converter.dicom import create_dicom_from_oct

SUPPORTED_SUFFIXES = {".fda", ".fds", ".e2e", ".img", ".oct"}


def collect_oct_inputs(
    paths: list[Path],
    *,
    recursive: bool = False,
) -> list[Path]:
    """Expand files and directories into a sorted list of OCT input paths."""
    found: list[Path] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            print(f"error: path not found: {path}", file=sys.stderr)
            raise SystemExit(2)
        if path.is_file():
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                print(
                    f"error: unsupported file type {path.suffix!r} "
                    f"(expected one of {sorted(SUPPORTED_SUFFIXES)}): {path}",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            found.append(path.resolve())
            continue
        pattern = "**/*" if recursive else "*"
        for child in sorted(path.glob(pattern)):
            if child.is_file() and child.suffix.lower() in SUPPORTED_SUFFIXES:
                found.append(child.resolve())
    if not found:
        print("error: no OCT files found in the given paths", file=sys.stderr)
        raise SystemExit(2)
    # Stable unique order
    return sorted(dict.fromkeys(found))


def output_dir_for(input_path: Path, output_root: Path, *, batch: bool) -> Path:
    """Target directory for one conversion.

    A single input writes into ``output_root``. Batches write each file into
    ``output_root / <stem>`` so stems from different folders do not collide
    on companion names like ``_octa.dcm``.
    """
    if batch:
        return output_root / input_path.stem
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oct-converter",
        description=(
            "Convert proprietary OCT volumes (.fda, .fds, .e2e, .img, .oct) "
            "to DICOM. OCTA FDA files also write a B-scan Volume Analysis "
            "companion (*_octa.dcm)."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="OCT file(s) and/or directories containing them",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for DICOM output (created if missing)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="When an input is a directory, search subfolders too",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Only print errors",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    inputs = collect_oct_inputs(args.inputs, recursive=args.recursive)
    batch = len(inputs) > 1
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    for src in inputs:
        dest = output_dir_for(src, output_root, batch=batch)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            written = create_dicom_from_oct(str(src), output_dir=str(dest))
        except Exception as exc:  # noqa: BLE001 - CLI reports and continues
            failures += 1
            print(f"error: {src}: {exc}", file=sys.stderr)
            continue
        if not args.quiet:
            print(f"{src} -> {dest}")
            for path in written:
                print(f"  {path}")
    return 1 if failures else 0


def run() -> None:
    """Console-script entry point (propagates exit status)."""
    raise SystemExit(main())


if __name__ == "__main__":
    run()
