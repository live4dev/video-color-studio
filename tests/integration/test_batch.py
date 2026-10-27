from dataclasses import replace

import pytest
from PIL import Image

from freedive_color.core.batch import export_batch, plan_batch
from freedive_color.core.errors import Cancelled, StudioError
from freedive_color.core.image_pipeline import CancellationToken, export_photo
from freedive_color.core.media_io import load_photo
from freedive_color.core.presets import get_preset
from freedive_color.core.project import sha256_file
from freedive_color.core.session import SessionItem


def make_photos(directory, count):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = directory / f"{index:02}.png"
        Image.new("RGB", (96, 64), (30 + index, 95 + index, 150 + index)).save(path)
        paths.append(path)
    return paths


def test_twenty_photos_export_individually_without_changing_originals(tmp_path):
    sources = make_photos(tmp_path / "input", 20)
    digests = {path: sha256_file(path) for path in sources}
    progress = []
    result = export_batch(
        tuple(SessionItem(path) for path in sources),
        tmp_path / "output",
        progress=lambda done, total, text: progress.append((done, total, text)),
    )
    assert result.completed == 20 and result.failed == 0 and not result.cancelled
    assert len(list((tmp_path / "output").glob("*.png"))) == 20
    assert len(list((tmp_path / "output").glob("*.png.freedive.json"))) == 20
    assert all(sha256_file(path) == digest for path, digest in digests.items())
    assert all(load_photo(record.destination).rgb.shape == (64, 96, 3) for record in result.records)
    assert progress[-1][:2] == (20, 20)


def test_per_image_overrides_and_exclusions_are_respected(tmp_path):
    sources = make_photos(tmp_path / "input", 3)
    settings = replace(get_preset("natural"), exposure=0.6)
    items = (
        SessionItem(sources[0]),
        SessionItem(sources[1], settings),
        SessionItem(sources[2], included=False),
    )
    result = export_batch(items, tmp_path / "output")
    assert (result.completed, result.excluded) == (2, 1)
    assert result.records[1].report["settings"]["exposure"] == 0.6
    assert (
        load_photo(result.records[1].destination).rgb.mean()
        > load_photo(result.records[0].destination).rgb.mean()
    )
    assert not (tmp_path / "output" / "02_natural.png").exists()


def test_duplicate_basenames_have_portable_deterministic_names(tmp_path):
    a = make_photos(tmp_path / "session" / "a", 1)[0]
    b = make_photos(tmp_path / "session" / "b", 1)[0]
    items = (SessionItem(a), SessionItem(b))
    names = [path.name for path in plan_batch(items, tmp_path / "output")]
    assert len(set(names)) == 2
    assert [path.name for path in plan_batch(items, tmp_path / "other")] == names
    reversed_names = [path.name for path in plan_batch(items[::-1], tmp_path / "output")]
    assert reversed_names == names[::-1]
    relocated = tuple(
        SessionItem(tmp_path / "moved" / item.path.relative_to(tmp_path / "session"))
        for item in items
    )
    assert [path.name for path in plan_batch(relocated, tmp_path / "output")] == names
    result = export_batch(items, tmp_path / "output")
    assert result.completed == 2


def test_prior_outputs_and_sidecars_are_not_overwritten(tmp_path):
    source = make_photos(tmp_path / "input", 1)[0]
    result = export_batch((SessionItem(source),), tmp_path / "output")
    destination = result.records[0].destination
    original_output = destination.read_bytes()
    second = export_batch((SessionItem(source),), tmp_path / "output")
    assert second.completed == 0 and second.failed == 1
    assert "существует" in second.records[0].error
    assert destination.read_bytes() == original_output


def test_corrupt_file_does_not_hide_or_abort_successful_siblings(tmp_path):
    sources = make_photos(tmp_path / "input", 3)
    sources[1].write_bytes(b"corrupt image")
    result = export_batch(tuple(SessionItem(path) for path in sources), tmp_path / "output")
    assert (result.completed, result.failed) == (2, 1)
    assert [record.status for record in result.records] == ["completed", "failed", "completed"]
    assert "повреждён" in result.records[1].error


def test_cancellation_preserves_completed_files_and_leaves_no_partial_current_file(tmp_path):
    sources = make_photos(tmp_path / "input", 5)
    token = CancellationToken()

    def cancel_second(source, destination, settings, **kwargs):
        if source == sources[1]:
            token.cancel()
        return export_photo(source, destination, settings, **kwargs)

    result = export_batch(
        tuple(SessionItem(path) for path in sources),
        tmp_path / "output",
        token=token,
        exporter=cancel_second,
    )
    assert result.cancelled and result.completed == 1 and result.failed == 0
    assert [record.status for record in result.records] == [
        "completed",
        "cancelled",
        "not_started",
        "not_started",
        "not_started",
    ]
    assert len(list((tmp_path / "output").glob("*.png"))) == 1
    assert not list((tmp_path / "output").glob(".freedive-*"))


def test_pre_cancelled_batch_returns_a_cancelled_summary(tmp_path):
    token = CancellationToken()
    token.cancel()
    source = make_photos(tmp_path / "input", 1)[0]
    result = export_batch(
        (SessionItem(source),),
        tmp_path / "output",
        token=token,
        reference_path=tmp_path / "missing-reference.png",
    )
    assert result.cancelled and result.records[0].status == "not_started"
    assert not (tmp_path / "output").exists()


def test_cancelling_from_progress_after_success_does_not_mark_success_as_cancelled(tmp_path):
    sources = make_photos(tmp_path / "input", 3)

    def progress(done, _total, _text):
        if done == 1:
            raise Cancelled()

    result = export_batch(
        tuple(SessionItem(path) for path in sources), tmp_path / "output", progress=progress
    )
    assert result.cancelled and result.completed == 1
    assert [record.status for record in result.records] == [
        "completed",
        "not_started",
        "not_started",
    ]


def test_cancel_during_reference_load_returns_a_cancelled_summary(tmp_path, monkeypatch):
    token = CancellationToken()
    source = make_photos(tmp_path / "input", 1)[0]

    def cancelled_reference(_path):
        token.cancel()
        return object()

    monkeypatch.setattr(
        "freedive_color.core.image_pipeline.create_image_reference",
        cancelled_reference,
        raising=False,
    )
    result = export_batch(
        (SessionItem(source),),
        tmp_path / "output",
        token=token,
        reference_path=tmp_path / "reference.png",
    )
    assert result.cancelled and result.records[0].status == "not_started"
    assert not (tmp_path / "output").exists()


def test_no_input_can_be_replaced_by_another_inputs_result(tmp_path):
    source = make_photos(tmp_path / "input", 1)[0]
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    other_source = output_dir / "00_natural.png"
    Image.new("RGB", (20, 30), (20, 30, 40)).save(other_source)
    digests = [sha256_file(path) for path in (source, other_source)]
    result = export_batch((SessionItem(source), SessionItem(other_source)), output_dir)
    assert result.records[0].status == "failed"
    assert "исходником другого" in result.records[0].error
    assert [sha256_file(path) for path in (source, other_source)] == digests


def test_saved_source_digest_is_checked_before_export(tmp_path):
    source = make_photos(tmp_path / "input", 1)[0]
    result = export_batch((SessionItem(source, source_sha256="0" * 64),), tmp_path / "output")
    assert result.failed == 1 and "изменился" in result.records[0].error


def test_duplicate_sources_and_wrong_options_are_rejected(tmp_path):
    source = make_photos(tmp_path / "input", 1)[0]
    item = SessionItem(source)
    with pytest.raises(StudioError, match="дважды"):
        plan_batch((item, item), tmp_path / "output")
    with pytest.raises(StudioError):
        plan_batch((item,), tmp_path / "output", options={"suffix": ".png"})


def test_local_settings_masks_and_frozen_reference_reach_real_export(tmp_path):
    import json

    from freedive_color.core.local_pipeline import LocalSettings
    from freedive_color.core.masks import BrushStroke, MaskEdits
    from freedive_color.core.reference import PaletteReference

    source = make_photos(tmp_path / "input", 1)[0]
    local = LocalSettings(water_hue=0.3, skin_warmth=0.23456, reference_strength=0.2)
    edits = MaskEdits(
        strokes=(BrushStroke(target="water", points=((0.5, 0.5),), radius=0.1),),
        waterline=((0.0, 0.2), (1.0, 0.25)),
    )
    reference = PaletteReference(water_ab=(-8.0, -18.0), source_name="Frozen edited frame")
    item = SessionItem(source, local_settings=local.to_dict(), mask_edits=edits.to_dict())
    digest = sha256_file(source)
    result = export_batch((item,), tmp_path / "output", reference=reference)
    assert result.completed == 1 and result.failed == 0
    record = result.records[0]
    report = json.loads(record.destination.with_suffix(".png.freedive.json").read_text())
    assert report["local_settings"]["skin_warmth"] == 0.23456
    assert report["mask_edits"]["waterline"] == [[0.0, 0.2], [1.0, 0.25]]
    assert report["reference"]["source_name"] == "Frozen edited frame"
    assert load_photo(record.destination).rgb.shape == (64, 96, 3)
    assert sha256_file(source) == digest


def test_reference_file_is_analyzed_once_for_whole_series(tmp_path, monkeypatch):
    from freedive_color.core.reference import PaletteReference

    sources = make_photos(tmp_path / "input", 3)
    reference = PaletteReference(water_ab=(-8.0, -18.0))
    calls = []

    def one_reference(path):
        calls.append(path)
        return reference

    monkeypatch.setattr("freedive_color.core.image_pipeline.create_image_reference", one_reference)
    result = export_batch(
        tuple(SessionItem(path) for path in sources), tmp_path / "output", reference_path=sources[0]
    )
    assert result.completed == 3
    assert calls == [sources[0]]
