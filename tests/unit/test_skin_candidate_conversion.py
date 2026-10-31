"""Model conversion input/download integrity without accessing the network or torch."""

import hashlib
import io
import zipfile

import pytest

from tools.convert_skin_candidates import download_public_fcn, verify_inputs
from tools.skin_candidates import UPSTREAM


class Response(io.BytesIO):
    def __init__(self, data, *, status=206, content_range=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Range": content_range}


def setup_download(monkeypatch, payload):
    source = UPSTREAM["fcn-skin"].copy()
    source.update(
        checkpoint_bytes=len(payload), checkpoint_sha256=hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setitem(UPSTREAM, "fcn-skin", source)
    monkeypatch.setattr("tools.convert_skin_candidates.time.sleep", lambda _: None)


def test_public_download_requires_exact_range_and_pinned_full_sha(monkeypatch, tmp_path):
    payload = b"official checkpoint fixture"
    setup_download(monkeypatch, payload)

    def download(request, timeout):
        assert request.get_header("Range") == f"bytes=0-{len(payload) - 1}"
        assert request.full_url == UPSTREAM["fcn-skin"]["checkpoint_url"]
        return Response(payload, content_range=f"bytes 0-{len(payload) - 1}/{len(payload)}")

    monkeypatch.setattr("tools.convert_skin_candidates.urllib.request.urlopen", download)
    target = tmp_path / "model.pth"
    download_public_fcn(target)
    assert target.read_bytes() == payload
    assert not target.with_suffix(".part").exists()


@pytest.mark.parametrize(
    "status,content_range,data",
    [
        (200, "bytes 0-9/10", b"0123456789"),
        (206, "bytes 1-10/10", b"0123456789"),
        (206, "bytes 0-9/10", b"short"),
    ],
)
def test_bad_or_incomplete_http_range_never_publishes(
    monkeypatch, tmp_path, status, content_range, data
):
    setup_download(monkeypatch, b"0123456789")
    monkeypatch.setattr(
        "tools.convert_skin_candidates.urllib.request.urlopen",
        lambda *_args, **_kwargs: Response(data, status=status, content_range=content_range),
    )
    target = tmp_path / "model.pth"
    with pytest.raises(ValueError, match="range"):
        download_public_fcn(target)
    assert not target.exists()


def test_corrupted_resume_part_fails_full_sha_and_is_discarded(monkeypatch, tmp_path):
    setup_download(monkeypatch, b"0123456789")
    parts = tmp_path / "fcn-download-parts"
    parts.mkdir()
    part = parts / "000.part"
    part.write_bytes(b"xxxxxxxxxx")
    monkeypatch.setattr(
        "tools.convert_skin_candidates.urllib.request.urlopen",
        lambda *_args, **_kwargs: pytest.fail("Completed ranges should be reused"),
    )
    target = tmp_path / "model.pth"
    with pytest.raises(ValueError, match="SHA"):
        download_public_fcn(target)
    assert not target.exists() and not part.exists()


def test_changed_extracted_source_rejected_even_when_source_zip_is_valid(monkeypatch, tmp_path):
    source = UPSTREAM["fcn-skin"].copy()
    directory_name = "SemanticSegmentation-" + source["commit"]
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    archive = upstream / f"fcn-{source['commit']}.zip"
    with zipfile.ZipFile(archive, "w") as files:
        files.writestr(directory_name + "/LICENSE", "MIT fixture")
        files.writestr(directory_name + "/model.py", "original source")
    source["source_zip_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    source["checkpoint_sha256"] = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setitem(UPSTREAM, "fcn-skin", source)
    checkpoint = upstream / "model_segmentation_skin_30.pth"
    checkpoint.write_bytes(b"weights")
    directory, path = verify_inputs(tmp_path, "fcn-skin")
    assert path == checkpoint
    (directory / "model.py").write_text("modified source")
    with pytest.raises(ValueError, match="source changed"):
        verify_inputs(tmp_path, "fcn-skin")
