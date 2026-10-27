#!/usr/bin/env python3
"""Verify an extracted portable payload; execute real GUI tests only on Windows."""

import argparse
import json
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

try:
    from build_windows_portable import MANIFEST, ROOT, file_hashes, native_audit, sha256
except ImportError:
    from tools.build_windows_portable import MANIFEST, ROOT, file_hashes, native_audit, sha256


def check_mcp_stdio(command: list[str], environment: dict, working_directory: Path) -> dict:
    """Packaged transport acceptance using host stdlib only; no desktop connection."""
    with tempfile.TemporaryDirectory(prefix="freedive-mcp-stdio-") as temporary:
        connection = Path(temporary) / "private connection"
        environment = {**environment, "FREEDIVE_COLOR_CONTROL_DIR": str(connection)}
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        received = queue.Queue(maxsize=64)
        diagnostics = bytearray()
        wire_lines = []

        def read_stdout():
            while True:
                line = process.stdout.readline(1_048_578)
                received.put(line)
                if not line:
                    break

        def read_stderr():
            while chunk := process.stderr.read(4096):
                if len(diagnostics) < 32768:
                    diagnostics.extend(chunk[: 32768 - len(diagnostics)])

        reader = threading.Thread(target=read_stdout, daemon=True)
        errors = threading.Thread(target=read_stderr, daemon=True)
        reader.start()
        errors.start()

        def send(value):
            process.stdin.write((json.dumps(value, ensure_ascii=False) + "\n").encode("utf-8"))
            process.stdin.flush()

        def parse(line):
            if not line or len(line) > 1_048_577 or not line.endswith(b"\n"):
                raise ValueError("Packaged MCP stdout has an incomplete/oversized message or EOF")
            value = json.loads(line.decode("utf-8", errors="strict"))
            if not isinstance(value, dict) or value.get("jsonrpc") != "2.0":
                raise ValueError("Packaged MCP stdout contains non-protocol output")
            wire_lines.append({"id": value.get("id"), "bytes": len(line)})
            return value

        def response(identifier):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    value = parse(received.get(timeout=max(0.001, deadline - time.monotonic())))
                except queue.Empty:
                    break
                if value.get("id") == identifier:
                    if "error" in value:
                        raise ValueError("Packaged MCP returned a protocol error")
                    return value["result"]
            raise TimeoutError("Packaged MCP stdio response timed out")

        try:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "Freedive portable check", "version": "1"},
                    },
                }
            )
            initialized = response(1)
            send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            tools = response(2)["tools"]
            expected = {
                "connect_session",
                "get_capabilities",
                "get_state",
                "list_media",
                "import_media",
                "select_media",
                "set_grade",
                "set_reference",
                "render_preview",
                "export_media",
                "get_job",
                "cancel_job",
                "save_project",
                "load_project",
                "show_window",
            }
            if {tool["name"] for tool in tools} != expected:
                raise ValueError("The packaged MCP adapter does not list all 15 tools")
            send(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "get_capabilities", "arguments": {}},
                }
            )
            capability_result = response(3)
            caps = capability_result.get("structuredContent", {})
            if (
                capability_result.get("isError")
                or caps.get("api_version") != 1
                or caps.get("privacy", {}).get("media_contents_in_responses") is not False
            ):
                raise ValueError("The packaged capabilities/privacy contract failed")
            if connection.exists():
                raise ValueError("Initialization/list/capabilities unexpectedly started the app")
            process.stdin.close()
            if process.wait(timeout=15) != 0:
                raise ValueError("The packaged MCP adapter did not exit cleanly after stdin EOF")
            reader.join(timeout=2)
            errors.join(timeout=2)
            while not received.empty():
                line = received.get_nowait()
                if line:
                    parse(line)
            return {
                "passed": True,
                "tools": sorted(expected),
                "tool_count": len(tools),
                "protocol_version": initialized["protocolVersion"],
                "api_version": caps["api_version"],
                "stdout_jsonrpc_only": True,
                "desktop_not_started": True,
                "messages": wire_lines,
                "stderr": diagnostics.decode("utf-8", errors="replace"),
            }
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            reader.join(timeout=2)
            errors.join(timeout=2)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


def check(stage: Path, output: Path, *, static_only=False, offscreen=False) -> dict:
    stage, output = stage.resolve(), output.resolve()
    if output.is_relative_to(stage):
        raise ValueError("Write test evidence outside the immutable application payload")
    if output.exists():
        raise FileExistsError("Native check report already exists; use a new output path")
    manifest = json.loads((stage / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("kind") != "freedive_windows_portable":
        raise ValueError("Not a generated Windows payload")
    if file_hashes(stage) != manifest["files"]:
        raise ValueError("Portable payload file hashes changed")
    audit = native_audit(stage)
    record = {
        "kind": "freedive_windows_check",
        "payload_sha256": manifest["payload_sha256"],
        "host_system": platform.system(),
        "static_integrity_passed": True,
        "native_files": len(audit["native_files"]),
        "windows_execution": "NOT RUN",
        "source_manifest_complete": manifest.get("sources_complete", False),
    }
    if not static_only:
        if platform.system() != "Windows":
            raise RuntimeError("Actual Windows execution is unavailable; use --static-only")
        if not manifest.get("opencv_native"):
            raise ValueError("Native acceptance requires the source-built IPP-free OpenCV")
        environment = os.environ.copy()
        for name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "VIRTUAL_ENV",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "QT_PLUGIN_PATH",
            "QML2_IMPORT_PATH",
        ):
            environment.pop(name, None)
        windows = Path(environment.get("SystemRoot", "C:/Windows"))
        environment["PATH"] = os.pathsep.join((str(windows / "System32"), str(windows)))
        if offscreen:
            environment["QT_QPA_PLATFORM"] = "offscreen"
        mcp_check = check_mcp_stdio([str(stage / "freedive-color-mcp.exe")], environment, stage)
        with tempfile.TemporaryDirectory(prefix="freedive-доказательство ") as temporary:
            diagnose_path = Path(temporary) / "проверка runtime.json"
            subprocess.run(
                [
                    str(stage / "Freedive Color Studio CLI.exe"),
                    "--diagnose-portable",
                    "--output",
                    str(diagnose_path),
                ],
                env=environment,
                check=True,
                timeout=60,
            )
            diagnose = json.loads(diagnose_path.read_text(encoding="utf-8"))
            if (
                diagnose.get("mcp_version") != "2.3.0"
                or diagnose.get("local_transport", {}).get("api_version") != 1
                or len(diagnose.get("windows_control_modules", [])) != 7
            ):
                raise ValueError("Packaged QtNetwork/pywin32/MCP dependencies are unavailable")
            photo = Path(temporary) / "подводный кадр с пробелами.jpg"
            photo_output = Path(temporary) / "результат 16 бит.png"
            shutil.copyfile(ROOT / "tests/fixtures/freediver_moalboal.jpg", photo)
            source_digest = sha256(photo)
            subprocess.run(
                [
                    str(stage / "Freedive Color Studio CLI.exe"),
                    "--input",
                    str(photo),
                    "--output",
                    str(photo_output),
                    "--preset",
                    "natural",
                    "--bit-depth",
                    "16",
                ],
                env=environment,
                check=True,
                timeout=60,
            )
            recipe = json.loads(
                photo_output.with_suffix(".png.freedive.json").read_text(encoding="utf-8")
            )
            if recipe["source"]["sha256"] != source_digest or sha256(photo) != source_digest:
                raise ValueError("Unicode source path or photo source integrity check failed")
            if recipe["output"]["bit_depth"] != 16:
                raise ValueError("The actual CLI did not export the requested 16-bit image")
        if not diagnose["isolated"] or not Path(diagnose["executable"]).resolve().is_relative_to(
            stage
        ):
            raise ValueError("The product used an external or non-isolated Python")
        if any(not Path(path).resolve().is_relative_to(stage) for path in diagnose["sys_path"]):
            raise ValueError("External Python module search path detected")
        native_modules = diagnose["opencv_native_modules"]
        if not native_modules or any(
            not Path(path).resolve().is_relative_to(stage) for path in native_modules.values()
        ):
            raise ValueError("Cannot confirm the packaged OpenCV extension was loaded")
        for line in diagnose["opencv_build_information"].splitlines():
            if ("Intel IPP:" in line or "FFMPEG:" in line) and line.split(":", 1)[
                1
            ].strip().upper() not in {"NO", "OFF", "DISABLED"}:
                raise ValueError("Unexpected IPP/FFmpeg enabled in the OpenCV extension")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/smoke_bundle.py"),
                "--executable",
                str(stage / "Freedive Color Studio.exe"),
            ],
            cwd=ROOT,
            env=environment,
            check=True,
            timeout=180,
        )
        smoke = json.loads(
            (ROOT / "artifacts/latest-bundle-smoke.json").read_text(encoding="utf-8")
        )
        if not smoke["passed"] or not smoke["video_gui_playback_and_export_verified"]:
            raise ValueError("Actual packaged photo/video GUI acceptance failed")
        record.update(
            {
                "windows_execution": "PASSED",
                "private_runtime_verified": True,
                "clean_opencv_verified": True,
                "unicode_cli_photo16_verified": True,
                "diagnostics": diagnose,
                "packaged_mcp_stdio": mcp_check,
                "photo_video_gui_acceptance": smoke,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument(
        "--offscreen", action="store_true", help="Native Windows CI without a display"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            check(args.stage, args.output, static_only=args.static_only, offscreen=args.offscreen),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
