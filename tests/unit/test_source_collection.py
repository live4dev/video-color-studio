from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "source_collector", Path(__file__).resolve().parents[2] / "tools/collect_media_sources.py"
)
assert SPEC is not None and SPEC.loader is not None
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


@pytest.mark.parametrize("filename", ["../secret", "/absolute", "nested/file", ".."])
def test_source_records_cannot_escape_archive_cache(filename):
    with pytest.raises(ValueError):
        collector.validate_entry(
            {
                "name": "example",
                "filename": filename,
                "sha256": "0" * 64,
                "urls": ["https://example.com/source"],
            }
        )


def test_offline_collection_checks_cached_bytes_and_never_calls_network(tmp_path, monkeypatch):
    archive = tmp_path / "archives/example.tar.gz"
    archive.parent.mkdir()
    archive.write_bytes(b"known source bytes")
    entry = {
        "name": "example",
        "filename": archive.name,
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "urls": ["https://example.com/source"],
    }
    monkeypatch.setattr(collector.subprocess, "run", lambda *a, **k: pytest.fail("network called"))
    assert collector.acquire(entry, tmp_path, offline=True)["download"] == "verified-cache"
    archive.write_bytes(b"corrupt source")
    with pytest.raises(ValueError, match="hash mismatch"):
        collector.acquire(entry, tmp_path, offline=True)
    archive.unlink()
    with pytest.raises(FileNotFoundError, match="not cached"):
        collector.acquire(entry, tmp_path, offline=True)


def test_notice_reader_rejects_symlinks_and_never_extracts_member_paths(tmp_path):
    archive = tmp_path / "untrusted.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, data in [
            ("../../LICENSE.txt", b"permissive license"),
            ("src/runtime.py", b"must never execute"),
        ]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            output.addfile(member, io.BytesIO(data))
        symlink = tarfile.TarInfo("COPYING.link")
        symlink.type = tarfile.SYMTYPE
        symlink.linkname = "/etc/passwd"
        output.addfile(symlink)
    notices = collector.extract_notices(archive, tmp_path / "notices")
    assert len(notices) == 1
    assert notices[0]["member"] == "../../LICENSE.txt"
    saved = tmp_path / "notices" / notices[0]["path"]
    assert saved.read_bytes() == b"permissive license"
    assert saved.parent == tmp_path / "notices"
    assert not (tmp_path / "runtime.py").exists()


def test_qt_bundle_unknown_module_blocks_incomplete_source_claim(tmp_path):
    (tmp_path / "QtCore.framework").mkdir()
    (tmp_path / "QtWebEngineCore.framework").mkdir()
    with pytest.raises(RuntimeError, match="Missing corresponding Qt"):
        collector.audit_qt(tmp_path)


def test_mcp_source_lock_includes_crypto_extras_rust_sources_and_exact_openssl():
    lock = json.loads((collector.RECORDS / "mcp-sources-lock.json").read_text(encoding="utf-8"))
    for entry in lock["sources"]:
        collector.validate_entry(entry)
    assert lock["required_versions"]["mcp"] == "2.3.0"
    assert lock["required_versions"]["cryptography"] == "50.0.2"
    assert lock["required_versions"]["pydantic-core"] == "2.50.1"
    assert lock["required_versions"]["rpds-py"] == "2026.9.1"
    assert lock["required_versions"]["cffi"] == "2.1.1"
    openssl = next(e for e in lock["sources"] if e["name"] == "mcp-openssl")
    assert openssl["version"] == lock["openssl_version"] == "4.0.3"
    assert openssl["sha256"] == "325b5c806167c13b40b1ffeadfe0248197c00eccc4cf123ec1e28d2d2fd216d9"
    rust = [e for e in lock["sources"] if e["name"].startswith("mcp-rust-")]
    assert {root for entry in rust for root in entry["used_by"]} == {
        "cryptography",
        "pydantic-core",
        "rpds-py",
    }


def test_mcp_version_mismatch_and_missing_source_lock_block_complete_claim(tmp_path, monkeypatch):
    original = collector.importlib.metadata.version
    monkeypatch.setattr(
        collector.importlib.metadata,
        "version",
        lambda name: "unexpected" if name == "mcp" else original(name),
    )
    with pytest.raises(RuntimeError, match="MCP source lock mismatch: mcp"):
        collector.merge_mcp_sources({"provenance": {}, "sources": []}, {})
    monkeypatch.setattr(collector, "RECORDS", tmp_path)
    with pytest.raises(RuntimeError, match="MCP source lock is missing"):
        collector.merge_mcp_sources({}, {})


def test_windows_mcp_gate_checks_staged_versions_and_openssl_before_merging(tmp_path):
    sdk = json.loads((collector.RECORDS / "mcp-sources-lock.json").read_text(encoding="utf-8"))
    inventory = {
        "wheels": [
            {"package": name, "version": version}
            for name, version in sdk["required_versions"].items()
        ]
        + [{"package": "pywin32", "version": "312"}]
    }
    native = tmp_path / "runtime/Lib/site-packages/cryptography/hazmat/bindings/_rust.pyd"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"fixture OpenSSL 4.0.3 29 Sep 2026\0")
    base = {"provenance": {"existing_native_gate": True}, "sources": []}
    merged = collector.merge_mcp_sources(base, inventory, tmp_path)
    assert merged["provenance"]["existing_native_gate"] is True
    assert len(merged["sources"]) == len(sdk["sources"])
    assert base["sources"] == []
    native.write_bytes(b"fixture OpenSSL 4.1.0 1 Jan 2027\0")
    with pytest.raises(RuntimeError, match="OpenSSL source version is unverified"):
        collector.merge_mcp_sources(base, inventory, tmp_path)
    next(row for row in inventory["wheels"] if row["package"] == "pywin32")["version"] = "311"
    with pytest.raises(RuntimeError, match="pywin32 source lock mismatch"):
        collector.merge_mcp_sources(base, inventory, tmp_path)


@pytest.mark.parametrize("timeout", [True, "300", 0, 301])
def test_source_download_timeout_is_bounded_without_relaxing_checksum_gate(timeout):
    with pytest.raises(ValueError, match="timeout"):
        collector.validate_entry(
            {
                "name": "example",
                "filename": "source.tar.gz",
                "sha256": "0" * 64,
                "urls": ["https://example.com/source"],
                "download_timeout": timeout,
            }
        )
