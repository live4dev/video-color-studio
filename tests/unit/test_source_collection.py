from __future__ import annotations

import hashlib
import importlib.util
import io
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
