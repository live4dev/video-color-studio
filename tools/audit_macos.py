"""Check every bundled Mach-O deployment target before promising macOS 13 compatibility."""

import json
import re
import subprocess
from pathlib import Path

MAGICS = {b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"}


def audit(bundle: Path, report_path: Path) -> None:
    binaries = []
    seen = set()
    for path in bundle.rglob("*"):
        if not path.is_file() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        with path.open("rb") as handle:
            if handle.read(4) not in MAGICS:
                continue
        output = subprocess.run(
            ["/usr/bin/otool", "-l", str(path)], check=True, capture_output=True, text=True
        ).stdout
        versions = re.findall(r"\bminos\s+(\d+(?:\.\d+)+)", output)
        versions += re.findall(
            r"LC_VERSION_MIN_MACOSX.*?\bversion\s+(\d+(?:\.\d+)+)", output, flags=re.DOTALL
        )
        binaries.append({"path": str(path.relative_to(bundle)), "minimum_macos": versions})
    violations = [
        item
        for item in binaries
        if any(tuple(map(int, value.split("."))) > (13, 0, 0) for value in item["minimum_macos"])
    ]
    record = {
        "checked_macho_files": len(binaries),
        "target": "13.0",
        "violations": violations,
        "binaries": binaries,
        "runtime_on_macos13_tested": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if violations:
        raise SystemExit(
            "Dependencies require macOS newer than 13. Sync with --python-platform "
            "aarch64-apple-darwin or x86_64-apple-darwin, then rebuild."
        )
    print(f"macOS 13 binary target audit passed: {len(binaries)} Mach-O files")
