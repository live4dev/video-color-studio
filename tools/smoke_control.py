"""Drive the actual desktop through SDK stdio and Qt IPC, without returning media.

Uses a fresh private control directory and three licensed synthetic clips. The
owned desktop is terminated only after the checks; no existing project is touched.
Run this with the development runtime, optionally targeting a frozen MCP executable.
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from contextlib import suppress
from pathlib import Path

import anyio
import av
from generate_video_fixtures import generate
from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"succeeded", "failed", "cancelled", "interrupted"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def media_clocks(path):
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        result = {
            "geometry": [video.width, video.height],
            "pts": [str(frame.pts * frame.time_base) for frame in container.decode(video)],
        }
    with av.open(str(path)) as container:
        audio = container.streams.audio[0]
        result["audio_start"] = str(audio.start_time * audio.time_base)
        result["audio_duration"] = str(audio.duration * audio.time_base)
        result["audio"] = [
            {
                "sha256": hashlib.sha256(bytes(packet)).hexdigest(),
                "pts": str(packet.pts * packet.time_base),
                "dts": str(packet.dts * packet.time_base),
                "duration": str(packet.duration * packet.time_base),
            }
            for packet in container.demux(audio)
            if packet.size
        ]
    return result


async def exercise(output: Path, adapter: Path | None, application: Path | None) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    inputs = list(generate(output / "inputs", duration=2.4, width=640, height=360, rate=8).values())
    checksums = {str(path): digest(path) for path in inputs}
    control = output / "private-control"
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "QT_PLUGIN_PATH"}
    }
    environment["FREEDIVE_COLOR_CONTROL_DIR"] = str(control)
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    if adapter and sys.platform == "darwin":
        environment["PATH"] = "/usr/bin:/bin"
    if application:
        environment["FREEDIVE_COLOR_APP"] = str(application.resolve())
    parameters = StdioServerParameters(
        command=str(adapter.resolve()) if adapter else sys.executable,
        args=[] if adapter else ["-m", "freedive_color.control.mcp_server"],
        env=environment,
        cwd=ROOT,
    )
    events = []
    process_id = None
    registry = control / "connection.json"

    async def call(client, method, params):
        started = time.monotonic()
        result = await client.call_tool(method, params)
        assert all(item.type == "text" for item in result.content), "Media entered MCP content"
        data = result.structured_content
        assert isinstance(data, dict) and not result.is_error, (method, data)
        assert data["ok"]
        events.append({"method": method, "seconds": round(time.monotonic() - started, 4)})
        return data

    async def wait_job(client, session, job_id):
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            state = await call(client, "get_job", {"session_id": session, "job_id": job_id})
            job = state["job"]
            if job["status"] in TERMINAL:
                assert job["status"] == "succeeded", job
                return job
            await anyio.sleep(0.1)
        raise AssertionError("Real desktop job did not finish before the smoke deadline")

    try:
        async with Client(parameters, read_timeout_seconds=60) as client:
            tools = await client.list_tools()
            assert len(tools.tools) == 15
            await call(client, "get_capabilities", {})
            assert not registry.exists(), "MCP initialization started the desktop"
            connected = await call(client, "connect_session", {})
            session = connected["session_id"]
            process_id = json.loads(registry.read_text())["pid"]
            imported = await call(
                client,
                "import_media",
                {
                    "session_id": session,
                    "paths": [str(path) for path in inputs],
                    "request_id": "smoke-import",
                },
            )
            await wait_job(client, session, imported["job_id"])
            listed = await call(client, "list_media", {"session_id": session})
            records = listed["media"]
            assert len(records) == 3
            ids = [record["media_id"] for record in records]
            await call(
                client,
                "set_grade",
                {
                    "session_id": session,
                    "media_ids": ids,
                    "expected_revisions": {r["media_id"]: r["revision"] for r in records},
                    "preset": "cinematic",
                    "grade": {"temperature": 0.08},
                },
            )
            records = (await call(client, "list_media", {"session_id": session}))["media"]
            await call(
                client,
                "set_grade",
                {
                    "session_id": session,
                    "media_ids": [ids[2]],
                    "expected_revisions": {ids[2]: records[2]["revision"]},
                    "local": {"water_saturation": -0.1},
                },
            )
            await call(
                client,
                "select_media",
                {
                    "session_id": session,
                    "media_id": ids[0],
                    "time_seconds": 0.5,
                },
            )
            preview = await call(
                client,
                "render_preview",
                {
                    "session_id": session,
                    "media_id": ids[0],
                    "duration_seconds": 0.75,
                    "start_seconds": 0.5,
                    "request_id": "smoke-preview",
                },
            )
            preview_job = await wait_job(client, session, preview["job_id"])
            assert Path(preview_job["result"]["path"]).is_file()
            export_params = {
                "session_id": session,
                "media_ids": ids,
                "directory": str(output / "exports"),
                "request_id": "smoke-export",
            }
            exported = await call(client, "export_media", export_params)
            duplicate = await call(client, "export_media", export_params)
            assert duplicate["job_id"] == exported["job_id"]
            # A later edit must not change a queued/running export's frozen settings.
            records = (await call(client, "list_media", {"session_id": session}))["media"]
            await call(
                client,
                "set_grade",
                {
                    "session_id": session,
                    "media_ids": [ids[2]],
                    "expected_revisions": {ids[2]: records[2]["revision"]},
                    "grade": {"temperature": 0.19},
                },
            )
            await call(client, "show_window", {"session_id": session, "visible": False})
            before_disconnect = await call(
                client,
                "get_job",
                {
                    "session_id": session,
                    "job_id": exported["job_id"],
                },
            )
        # The SDK closes the MCP subprocess here. The desktop owns the work.
        await anyio.sleep(0.2)
        assert json.loads(registry.read_text())["pid"] == process_id
        async with Client(parameters, read_timeout_seconds=60) as client:
            reconnected = await call(client, "connect_session", {})
            assert reconnected["session_id"] == session
            assert json.loads(registry.read_text())["pid"] == process_id
            export_job = await wait_job(client, session, exported["job_id"])
            paths = sorted((output / "exports").glob("*.mp4"))
            assert len(paths) == 3
            verifications = []
            for source in inputs:
                matching = [path for path in paths if path.stem.startswith(source.stem)]
                assert len(matching) == 1, (source, paths)
                result = matching[0]
                assert media_clocks(source) == media_clocks(result), source.name
                recipe = json.loads(Path(str(result) + ".freedive.json").read_text())
                assert recipe["settings"]["temperature"] == 0.08
                assert digest(source) == checksums[str(source)]
                verifications.append(
                    {
                        "source": str(source),
                        "output": str(result),
                        "sha256": digest(result),
                        "all_pts_and_audio_exact": True,
                        "frozen_temperature": recipe["settings"]["temperature"],
                    }
                )
            saved = await call(
                client,
                "save_project",
                {
                    "session_id": session,
                    "path": str(output / "project.fcsproj"),
                },
            )
            await wait_job(client, session, saved["job_id"])
            # Pending viewport refreshes share the queue, too.
            deadline = time.monotonic() + 60
            while True:
                state = await call(client, "get_state", {"session_id": session})
                if state["queue"]["active_count"] == 0:
                    break
                assert time.monotonic() < deadline
                await anyio.sleep(0.1)
            assert state["selected"]["media_id"] == ids[0]
        return {
            "passed": True,
            "adapter": str(adapter) if adapter else "source SDK module",
            "sdk": "2.3.0",
            "tool_count": 15,
            "pid": process_id,
            "session_id": session,
            "state_before_disconnect": before_disconnect["job"]["status"],
            "same_process_after_adapter_disconnect": True,
            "deduplicated_export": exported["job_id"],
            "export_status": export_job["status"],
            "verified_media": verifications,
            "media_entered_mcp": False,
            "events": events,
            "visual_quality_claim": False,
            "teardown": "Terminate only this isolated smoke desktop after jobs finish.",
        }
    finally:
        if process_id:
            # This PID came from the fresh smoke-only directory created above.
            with suppress(ProcessLookupError):
                os.kill(process_id, signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--application", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        result = anyio.run(exercise, output, args.adapter, args.application)
    except BaseException as error:
        result = {"passed": False, "error": str(error), "error_type": type(error).__name__}
        if output.exists():
            (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
