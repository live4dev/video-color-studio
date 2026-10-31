import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the isolated standalone photo/video GUI")
    parser.add_argument("--executable", type=Path)
    arguments = parser.parse_args()
    if sys.platform == "darwin":
        executable = ROOT / "dist/Freedive Color Studio.app/Contents/MacOS/Freedive Color Studio"
    elif sys.platform == "win32":
        executable = ROOT / "dist/Freedive Color Studio/Freedive Color Studio.exe"
    else:
        raise SystemExit("Only macOS and Windows bundles are configured")
    if arguments.executable:
        executable = arguments.executable.resolve()
    environment = os.environ.copy()
    # Exercise an isolated frozen executable without the development environment in its path.
    for key in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "QT_PLUGIN_PATH",
        "QML2_IMPORT_PATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(key, None)
    environment["PATH"] = (
        "/usr/bin:/bin"
        if sys.platform == "darwin"
        else os.pathsep.join(
            (
                str(Path(environment.get("SystemRoot", "C:/Windows")) / "System32"),
                environment.get("SystemRoot", "C:/Windows"),
            )
        )
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
        "--show-mask",
        "skin",
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
    masks = record["output"]["masks"]
    assert "pphumanseg" in masks["method"] and masks["coverage"]["person"] > 0
    skin_model = json.loads(
        (ROOT / "src/freedive_color/assets/models/SELFIE_MULTICLASS_METADATA.json").read_text(
            encoding="utf-8"
        )
    )
    assert masks["skin_source"] == skin_model["model_id"]
    assert masks["skin_model_sha256"] == skin_model["onnx_sha256"]

    from generate_video_fixtures import generate
    from render_video_evidence import inspect_video, verify_full_export

    fixtures = generate(output / "inputs", duration=1.5, width=320, height=180, rate=8)
    video_directory = output / "video"
    video_command = [
        str(executable),
        "--smoke-video",
        str(fixtures["vfr_offset"]),
        "--evidence",
        str(video_directory),
    ]
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        video_command = [
            "/usr/bin/sandbox-exec",
            "-p",
            "(version 1)(allow default)(deny network*)",
            *video_command,
        ]
    subprocess.run(video_command, cwd=ROOT.parent, env=environment, check=True, timeout=90)
    video_record = json.loads((video_directory / "video_smoke.json").read_text(encoding="utf-8"))
    assert video_record["passed"] and video_record["source_unchanged"]
    assert video_record["skin_model_present"]
    assert video_record["masks"]["skin_model_sha256"] == skin_model["onnx_sha256"]
    assert video_record["playback_frame_count"] >= 3
    verification = verify_full_export(
        inspect_video(fixtures["vfr_offset"]), inspect_video(Path(video_record["output_path"]))
    )
    (video_directory / "independent_audit.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    (ROOT / "artifacts" / "latest-bundle-smoke.json").write_text(
        json.dumps(
            {
                "evidence": str(output),
                "passed": True,
                "photo_model_verified": True,
                "skin_model_sha256": skin_model["onnx_sha256"],
                "video_gui_playback_and_export_verified": True,
                "video_independent_audit": verification,
                "network_denied": sys.platform == "darwin" and bool(shutil.which("sandbox-exec")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Standalone photo masks, video playback, audio and VFR export passed: {executable}")


if __name__ == "__main__":
    main()
