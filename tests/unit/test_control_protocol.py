import json
import os
import stat

import pytest

from freedive_color.control.protocol import (
    MAX_MESSAGE_BYTES,
    ControlPaths,
    LineDecoder,
    ProtocolError,
    decode_json,
    encode_json,
    private_directory,
    private_file,
    read_connection,
    validate_request,
    write_private,
)


def envelope(**changes):
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "get_state",
        "params": {},
        "api_version": 1,
        "token": "secret",
        **changes,
    }


def test_fragmented_utf8_and_coalesced_messages():
    first = encode_json({"name": "море 🐬"})
    second = encode_json({"id": 2})
    decoder = LineDecoder()
    split = first.index("🐬".encode()) + 2
    assert decoder.feed(first[:split]) == []
    assert decoder.feed(first[split:] + second) == [{"name": "море 🐬"}, {"id": 2}]
    assert decoder.pending == bytearray()


def test_message_limit_is_per_line_not_whole_read():
    payload = encode_json({"text": "x" * (MAX_MESSAGE_BYTES - 11)})
    assert len(payload) == MAX_MESSAGE_BYTES + 1
    assert len(LineDecoder().feed(payload * 2)) == 2
    with pytest.raises(ProtocolError, match="1 МиБ"):
        LineDecoder().feed(b"x" * (MAX_MESSAGE_BYTES + 1))


@pytest.mark.parametrize(
    "raw",
    [b"[]", b"{}{}", b"\xff", b'{"x":NaN}', b'{"x":1e999}', b'{"x":1,"x":2}', b'{"x":"\\ud800"}'],
)
def test_invalid_json_is_rejected(raw):
    with pytest.raises(ProtocolError):
        decode_json(raw)


def test_strict_auth_version_and_envelope():
    assert validate_request(envelope(), "secret") == (1, "get_state", {})
    for changes, code in [
        ({"token": "wrong"}, "UNAUTHORIZED"),
        ({"token": "юникод"}, "UNAUTHORIZED"),
        ({"api_version": True}, "API_VERSION_MISMATCH"),
        ({"api_version": 2}, "API_VERSION_MISMATCH"),
        ({"id": True}, "INVALID_REQUEST"),
        ({"extra": 1}, "INVALID_REQUEST"),
        ({"params": []}, "INVALID_REQUEST"),
    ]:
        with pytest.raises(ProtocolError) as error:
            validate_request(envelope(**changes), "secret")
        assert error.value.code == code


def test_private_registry_and_no_symlink_follow(tmp_path):
    paths = ControlPaths(tmp_path / "Канал с пробелом")
    private_directory(paths.directory)
    write_private(paths.token, b"s" * 43)
    registry = {
        "api_version": 1,
        "endpoint": paths.endpoint(),
        "pid": os.getpid(),
        "instance_id": "random",
    }
    write_private(paths.registry, json.dumps(registry).encode())
    assert read_connection(paths) == (registry, "s" * 43)
    if os.name != "nt":
        assert stat.S_IMODE(paths.directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths.token.stat().st_mode) == 0o600
        paths.token.chmod(0o644)
        with pytest.raises(PermissionError):
            private_file(paths.token)
        paths.token.unlink()
        paths.token.symlink_to(paths.registry)
        with pytest.raises((PermissionError, OSError)):
            private_file(paths.token)
        with pytest.raises(PermissionError):
            write_private(paths.token, b"changed")


def test_registry_version_never_silently_reused(tmp_path):
    paths = ControlPaths(tmp_path / "channel")
    private_directory(paths.directory)
    write_private(paths.token, b"s" * 43)
    write_private(paths.registry, b'{"api_version":2,"endpoint":"x","pid":1,"instance_id":"x"}')
    with pytest.raises(ProtocolError) as error:
        read_connection(paths)
    assert error.value.code == "API_VERSION_MISMATCH"
