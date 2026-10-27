import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    if sys.platform == "darwin":
        executable = ROOT / "dist/Freedive Color Studio.app/Contents/MacOS/Freedive Color Studio"
    elif sys.platform == "win32":
        executable = ROOT / "dist/Freedive Color Studio/Freedive Color Studio.exe"
    else:
        raise SystemExit("Only macOS and Windows bundles are configured")
    environment = os.environ.copy()
    # Exercise an isolated frozen executable without the development environment in its path.
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "QT_PLUGIN_PATH"):
        environment.pop(key, None)
    environment["PATH"] = (
        "/usr/bin:/bin" if sys.platform == "darwin" else environment.get("SystemRoot", "C:/Windows")
    )
    output = ROOT / "artifacts" / "bundle-smoke"
    index = 1
    while output.exists():
        index += 1
        output = ROOT / "artifacts" / f"bundle-smoke-{index}"
    command = [
        str(executable),
        "--smoke-test",
        str(ROOT / "tests/fixtures/freediver_moalboal.jpg"),
        "--evidence",
        str(output),
    ]
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        command = [
            "/usr/bin/sandbox-exec",
            "-p",
            "(version 1)(allow default)(deny network*)",
            *command,
        ]
    subprocess.run(
        command,
        cwd=ROOT.parent,
        env=environment,
        check=True,
        timeout=60,
    )
    record = json.loads((output / "gui_smoke.json").read_text(encoding="utf-8"))
    assert record["passed"] and record["source_unchanged"]
    (ROOT / "artifacts" / "latest-bundle-smoke.json").write_text(
        json.dumps(
            {
                "evidence": str(output),
                "passed": True,
                "network_denied": sys.platform == "darwin" and bool(shutil.which("sandbox-exec")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Frozen GUI import, controls and export passed: {executable}")


if __name__ == "__main__":
    main()
