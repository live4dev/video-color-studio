import json
import socket
from pathlib import Path

import numpy as np
import pytest

from freedive_color.cli import run
from freedive_color.core.errors import Cancelled, StudioError
from freedive_color.core.image_pipeline import CancellationToken, export_photo, make_preview
from freedive_color.core.media_io import load_photo
from freedive_color.core.presets import PRESETS
from freedive_color.core.project import load_recipe, load_settings, save_settings, sha256_file


def test_real_photo_export(tmp_path):
    source = Path(__file__).parents[1] / "fixtures" / "freediver_moalboal.jpg"
    assert source.exists(), "The licensed real-photo fixture must be present"
    digest = sha256_file(source)
    preview = make_preview(source, PRESETS["natural"])
    output = tmp_path / "natural.png"
    report = export_photo(source, output, PRESETS["natural"], analysis=preview.analysis)
    result = load_photo(output)
    assert (result.width, result.height) == (1125, 1500)
    assert report["source"]["sha256"] == digest == sha256_file(source)
    assert np.mean(np.abs(result.rgb - load_photo(source).rgb)) > 0.001


def test_cli_uses_core_and_recipe_is_reproducible(photo_path, tmp_path, capsys):
    output = tmp_path / "result.png"
    assert run(["--input", str(photo_path), "--output", str(output), "--preset", "cinematic"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["output"]["width"] == 128
    sidecar = output.with_suffix(".png.freedive.json")
    setting, analysis = load_recipe(sidecar, photo_path)
    repeat = tmp_path / "repeat.png"
    export_photo(photo_path, repeat, setting, analysis=analysis)
    np.testing.assert_array_equal(load_photo(output).rgb, load_photo(repeat).rgb)


def test_original_and_existing_output_protected(photo_path, tmp_path):
    digest = sha256_file(photo_path)
    with pytest.raises(StudioError, match="Оригинал"):
        export_photo(photo_path, photo_path, PRESETS["natural"])
    output = tmp_path / "exists.png"
    output.write_bytes(b"user data")
    with pytest.raises(StudioError, match="существует"):
        export_photo(photo_path, output, PRESETS["natural"])
    assert output.read_bytes() == b"user data"
    assert sha256_file(photo_path) == digest


def test_cancellation_leaves_no_output(photo_path, tmp_path):
    token = CancellationToken()
    output = tmp_path / "cancelled.png"

    def progress(value, _text):
        if value == 65:
            token.cancel()

    with pytest.raises(Cancelled):
        export_photo(photo_path, output, PRESETS["natural"], token=token, progress=progress)
    assert not output.exists()
    assert not list(tmp_path.glob(".freedive-*"))


def test_second_publication_failure_rolls_back(photo_path, tmp_path, monkeypatch):
    import freedive_color.core.image_pipeline as pipeline

    actual_link = pipeline.os.link

    def fail_sidecar(source, target):
        if str(target).endswith(".json"):
            raise PermissionError("test failure")
        actual_link(source, target)

    monkeypatch.setattr(pipeline.os, "link", fail_sidecar)
    output = tmp_path / "failed.png"
    with pytest.raises(StudioError):
        export_photo(photo_path, output, PRESETS["natural"])
    assert not output.exists()
    assert not list(tmp_path.glob(".freedive-*"))


def test_not_enough_space(photo_path, tmp_path, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        "freedive_color.core.image_pipeline.shutil.disk_usage", lambda _path: usage(1000, 999, 1)
    )
    with pytest.raises(StudioError, match="места"):
        export_photo(photo_path, tmp_path / "out.png", PRESETS["natural"])


def test_settings_roundtrip_and_recipe_source_check(photo_path, tmp_path):
    path = tmp_path / "custom.json"
    save_settings(path, PRESETS["vivid"])
    assert load_settings(path) == PRESETS["vivid"]
    with pytest.raises(StudioError):
        save_settings(path, PRESETS["natural"])
    output = tmp_path / "out.png"
    export_photo(photo_path, output, PRESETS["natural"])
    with pytest.raises(StudioError, match="сумма"):
        load_recipe(output.with_suffix(".png.freedive.json"), output)


def test_changed_original_rejected(photo_path, tmp_path):
    preview = make_preview(photo_path, PRESETS["natural"])
    photo_path.write_bytes(photo_path.read_bytes() + b"changed")
    with pytest.raises(StudioError, match="изменился"):
        export_photo(
            photo_path,
            tmp_path / "out.png",
            PRESETS["natural"],
            expected_signature=preview.photo.stat_signature,
        )


def test_offline_workflow(photo_path, tmp_path, monkeypatch):
    def no_network(*_args, **_kwargs):
        raise AssertionError("Runtime attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    export_photo(photo_path, tmp_path / "offline.png", PRESETS["natural"])


def test_preview_uses_same_parameters_as_export(photo_path, tmp_path):
    preview = make_preview(photo_path, PRESETS["natural"])
    output = tmp_path / "out.png"
    export_photo(photo_path, output, PRESETS["natural"], analysis=preview.analysis)
    np.testing.assert_allclose(preview.result, load_photo(output).rgb, atol=1.1 / 255)


def test_export_to_filesystem_without_hard_links(photo_path, tmp_path, monkeypatch):
    import errno

    def unsupported_link(*_args):
        raise OSError(errno.EOPNOTSUPP, "No hard links on this filesystem")

    monkeypatch.setattr("freedive_color.core.image_pipeline.os.link", unsupported_link)
    output = tmp_path / "exfat.png"
    export_photo(photo_path, output, PRESETS["natural"])
    assert load_photo(output).rgb.shape == (96, 128, 3)
    assert output.with_suffix(".png.freedive.json").exists()
    assert not list(tmp_path.glob(".freedive-*"))
