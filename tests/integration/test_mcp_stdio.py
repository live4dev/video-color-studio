"""Real SDK 2.x client/server over subprocess stdio, with an explicit test backend.

These tests exercise the actual MCP transport/schema/privacy boundary. Desktop
controller, QLocalSocket and real rendering acceptance have their own tests.
"""

import json
import sys
from pathlib import Path

import anyio
import pytest
from mcp import Client, StdioServerParameters, stdio_client
from mcp.types import jsonrpc_message_adapter

from freedive_color.control.mcp_server import SERVER_INSTRUCTIONS, create_server
from freedive_color.control.schemas import PARAM_MODELS

ROOT = Path(__file__).resolve().parents[2]


def _write_backend(tmp_path):
    calls = tmp_path / "calls.jsonl"
    driver = tmp_path / "metadata_backend.py"
    driver.write_text(
        """
import json,os
from pathlib import Path
import anyio
from freedive_color.control.errors import ControlError
from freedive_color.control.mcp_server import serve

class RemoteError(Exception):
    code='REVISION_CONFLICT'
    message='Ревизия файла изменилась.'
    hint='Перечитайте get_state и повторите правку с актуальной ревизией.'
    details={'actual_revision':4}

class MetadataBackend:
    def call(self,method,params):
        with Path(CALLS).open('a',encoding='utf8') as output:
            output.write(json.dumps({'method':method,'params':params},ensure_ascii=False)+'\\n')
        if params.get('session_id')=='remote-error':raise RemoteError()
        if params.get('session_id')=='private-buffer':return {'thumbnail':b'not real media'}
        if params.get('session_id')=='private-json':
            return {'nested':{'type':'image','data':'never-forward-this','mimeType':'image/png'}}
        if params.get('session_id')=='oversized':return {'metadata':'x'*1100000}
        if method=='connect_session':
            print('Python diagnostic deliberately directed at stdout',flush=True)
            os.write(1,b'Native diagnostic deliberately directed at stdout\\n')
        return {'session_id':'session-1','called_method':method,'received':params,
                'job_id':'job-1','message':'Локальная операция проверена.'}

if __name__=='__main__':anyio.run(serve,MetadataBackend)
""".replace("CALLS", repr(str(calls))),
        encoding="utf8",
    )
    return driver, calls


def _stdio_parameters(driver):
    return StdioServerParameters(command=sys.executable, args=[str(driver)], cwd=ROOT)


def test_real_sdk_stdio_routes_all_tools_and_returns_only_structured_metadata(tmp_path):
    driver, calls = _write_backend(tmp_path)
    arguments = {
        "connect_session": {},
        "get_capabilities": {},
        "get_state": {"session_id": "session-1"},
        "list_media": {"session_id": "session-1"},
        "import_media": {
            "session_id": "session-1",
            "paths": [str(tmp_path / "explicit.mp4")],
            "request_id": "import-1",
        },
        "select_media": {"session_id": "session-1", "media_id": "media-1", "time_seconds": 1},
        "set_grade": {
            "session_id": "session-1",
            "media_ids": ["media-1", "media-2"],
            "expected_revisions": {"media-1": 0, "media-2": 2},
            "preset": "cinematic",
            "grade": {"temperature": 0.1},
            "local": {"water_saturation": -0.1},
        },
        "set_reference": {"session_id": "session-1", "source": "clear"},
        "render_preview": {
            "session_id": "session-1",
            "media_id": "media-1",
            "request_id": "preview-1",
        },
        "export_media": {
            "session_id": "session-1",
            "media_ids": ["media-1"],
            "directory": str(tmp_path),
            "request_id": "export-1",
        },
        "get_job": {"session_id": "session-1", "job_id": "job-1"},
        "cancel_job": {"session_id": "session-1", "job_id": "job-1"},
        "save_project": {"session_id": "session-1", "path": str(tmp_path / "project.fcsproj")},
        "load_project": {"session_id": "session-1", "path": str(tmp_path / "project.fcsproj")},
        "show_window": {"session_id": "session-1", "visible": False},
    }

    async def exercise():
        async with Client(_stdio_parameters(driver), read_timeout_seconds=10) as client:
            listed = await client.list_tools()
            assert {tool.name for tool in listed.tools} == set(PARAM_MODELS)
            assert (
                not calls.exists()
            )  # initialize/list did not instantiate or call the desktop client
            caps = await client.call_tool("get_capabilities", {})
            assert caps.structured_content["privacy"]["media_contents_in_responses"] is False
            assert not calls.exists()
            for name, params in arguments.items():
                result = await client.call_tool(name, params)
                assert result.is_error is False
                assert result.structured_content["ok"] is True
                assert all(content.type == "text" for content in result.content)
                assert any("а" <= char <= "я" for char in result.content[0].text.lower())
            assert caps.structured_content["limits"]["preview_seconds"] == 15

    anyio.run(exercise)
    rows = [json.loads(line) for line in calls.read_text(encoding="utf8").splitlines()]
    assert {row["method"] for row in rows} == set(PARAM_MODELS) - {"get_capabilities"}
    patch = next(row["params"] for row in rows if row["method"] == "set_grade")
    assert patch["local"] == {"water_saturation": -0.1}
    assert patch["grade"] == {"temperature": 0.1}


def test_real_stdio_rejects_invalid_arguments_before_the_desktop_client_is_called(tmp_path):
    driver, calls = _write_backend(tmp_path)
    invalid = [
        ("connect_session", {"extra": True}),
        ("get_capabilities", {"extra": True}),
        ("show_window", {"session_id": "s", "visible": 1}),
        ("list_media", {"session_id": "s", "limit": "10"}),
        (
            "set_grade",
            {
                "session_id": "s",
                "media_ids": ["a"],
                "expected_revisions": {"a": 0},
                "grade": {"exposure": True},
            },
        ),
        (
            "render_preview",
            {"session_id": "s", "media_id": "a", "duration_seconds": 16},
        ),
    ]

    async def exercise():
        async with Client(_stdio_parameters(driver), read_timeout_seconds=10) as client:
            for name, args in invalid:
                result = await client.call_tool(name, args)
                assert result.is_error
                assert result.structured_content["error"]["code"] == "INVALID_PARAMS"
                assert result.structured_content["error"]["hint"]
            unknown = await client.call_tool("unsupported_operation", {})
            assert unknown.structured_content["error"]["code"] == "METHOD_NOT_FOUND"

    anyio.run(exercise)
    assert not calls.exists()


def test_remote_errors_remain_machine_readable_and_media_or_oversized_results_are_denied(tmp_path):
    driver, _ = _write_backend(tmp_path)

    async def exercise():
        async with Client(_stdio_parameters(driver), read_timeout_seconds=10) as client:
            remote = await client.call_tool("get_state", {"session_id": "remote-error"})
            assert remote.is_error
            assert remote.structured_content["error"]["code"] == "REVISION_CONFLICT"
            assert remote.structured_content["error"]["details"] == {"actual_revision": 4}
            for session in ("private-buffer", "private-json", "oversized"):
                denied = await client.call_tool("get_state", {"session_id": session})
                assert denied.is_error
                assert denied.structured_content["error"]["code"] == "INVALID_RESPONSE"
                text = denied.model_dump_json()
                assert "never-forward-this" not in text
                assert "not real media" not in text
                assert "thumbnail" not in text
                assert len(text) < 4096

    anyio.run(exercise)


def test_actual_sdk_stdout_contains_only_jsonrpc_despite_python_and_native_diagnostics(tmp_path):
    driver, _ = _write_backend(tmp_path)
    wire = tmp_path / "stdout-wire.jsonl"
    tee = tmp_path / "protocol_tee.py"
    tee.write_text(
        """
import os,subprocess,sys,threading
from pathlib import Path
child=subprocess.Popen([sys.executable,DRIVER],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
def forward_input():
    while True:
        chunk=os.read(0,65536)
        if not chunk:break
        child.stdin.write(chunk);child.stdin.flush()
    child.stdin.close()
threading.Thread(target=forward_input,daemon=True).start()
with Path(WIRE).open('wb') as log:
    for line in child.stdout:
        log.write(line);log.flush()
        sys.stdout.buffer.write(line);sys.stdout.buffer.flush()
raise SystemExit(child.wait())
""".replace("DRIVER", repr(str(driver))).replace("WIRE", repr(str(wire))),
        encoding="utf8",
    )
    stderr_path = tmp_path / "stderr.log"

    async def exercise():
        with stderr_path.open("w", encoding="utf8") as stderr:
            async with Client(
                stdio_client(_stdio_parameters(tee), errlog=stderr), read_timeout_seconds=10
            ) as client:
                await client.list_tools()
                result = await client.call_tool("connect_session", {})
                assert result.structured_content["ok"] is True

    anyio.run(exercise)
    lines = wire.read_text(encoding="utf8").splitlines()
    assert len(lines) >= 3
    for line in lines:
        jsonrpc_message_adapter.validate_json(line, by_name=False)
        assert json.loads(line)["jsonrpc"] == "2.0"
    stderr = stderr_path.read_text(encoding="utf8")
    assert "Python diagnostic deliberately directed at stdout" in stderr
    assert "Native diagnostic deliberately directed at stdout" in stderr
    assert "diagnostic" not in wire.read_text(encoding="utf8")


def test_initialization_capabilities_and_listing_are_lazy_even_if_client_constructor_would_fail():
    def forbidden_factory():
        pytest.fail(
            "MCP initialization/list/static capabilities must not construct ApplicationClient"
        )

    async def exercise():
        async with Client(create_server(forbidden_factory), read_timeout_seconds=5) as client:
            tools = await client.list_tools()
            assert len(tools.tools) == 15
            capabilities = await client.call_tool("get_capabilities", {})
            assert capabilities.structured_content["ok"] is True

    anyio.run(exercise)
    assert "Повторное подтверждение уже запрошенного экспорта не требуется" in SERVER_INSTRUCTIONS
    assert "Статистики анализа не доказывают" in SERVER_INSTRUCTIONS
