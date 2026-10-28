import json
from dataclasses import replace

import numpy as np
import pytest

from freedive_color.core.analyze import analyze_scene
from freedive_color.core.errors import StudioError
from freedive_color.core.presets import get_preset
from freedive_color.core.session import (
    ExportOptions,
    Session,
    SessionItem,
    load_session,
    save_session,
)


def test_session_roundtrip_relative_paths_and_exact_values(tmp_path, monkeypatch):
    media = tmp_path / "photos" / "погружение.png"
    reference = tmp_path / "photos" / "reference.jpg"
    settings = replace(get_preset("natural"), exposure=0.12345)
    analysis = analyze_scene(np.full((16, 24, 3), [0.1, 0.4, 0.6], np.float32)).to_dict()
    item = SessionItem(
        media,
        settings,
        included=False,
        analysis=analysis,
        media_info={"decoder": "Pillow", "width": 24, "height": 16},
        source_sha256="a" * 64,
    )
    project = tmp_path / "dive.fcsproj"
    session = Session(
        (item,),
        reference_path=reference,
        temp_directory=tmp_path / "cache",
        output_directory=tmp_path / "output",
        selected_index=0,
        export_options=ExportOptions(".tiff", 16, 93, True),
    )
    save_session(project, session)
    raw = json.loads(project.read_text(encoding="utf-8"))
    assert raw["items"][0]["path"] == "photos/погружение.png"
    assert raw["reference_path"] == "photos/reference.jpg"
    other = tmp_path / "other-cwd"
    other.mkdir()
    monkeypatch.chdir(other)
    restored = load_session(project)
    assert restored == session
    assert restored.items[0].settings.exposure == 0.12345
    assert not media.exists(), "Projects may be loaded while offline media are missing"


def test_project_can_be_saved_again_without_touching_sources(tmp_path, photo_path):
    original = photo_path.read_bytes()
    path = tmp_path / "project.json"
    session = Session((SessionItem(photo_path),), selected_index=0)
    save_session(path, session)
    changed = replace(session, items=(replace(session.items[0], included=False),))
    save_session(path, changed, overwrite=True)
    assert not load_session(path).items[0].included
    assert photo_path.read_bytes() == original
    assert not list(tmp_path.glob(".freedive-project-*"))


def test_manual_masks_local_controls_and_frozen_reference_roundtrip(tmp_path, photo_path):
    from freedive_color.core.local_pipeline import LocalSettings
    from freedive_color.core.masks import BrushStroke, MaskEdits
    from freedive_color.core.reference import PaletteReference

    edits = MaskEdits(
        strokes=(BrushStroke(points=((0.25, 0.75), (0.3, 0.7)), radius=0.023),),
        waterline=((0.0, 0.2), (0.5, 0.3), (1.0, 0.25)),
        person_box=(0.1, 0.1, 0.9, 0.9),
    )
    local = LocalSettings(skin_warmth=0.23456, water_hue=-0.15)
    reference = PaletteReference(
        water_ab=(-10.5, -20.25), skin_ab=(12.5, 10.75), source_name="Edited reference"
    )
    project = tmp_path / "local.json"
    session = Session(
        (SessionItem(photo_path, local_settings=local.to_dict(), mask_edits=edits.to_dict()),),
        reference_path=photo_path,
        reference_data=reference.to_dict(),
    )
    save_session(project, session)
    restored = load_session(project)
    assert restored == session
    assert LocalSettings.from_dict(restored.items[0].local_settings) == local
    assert MaskEdits.from_dict(restored.items[0].mask_edits) == edits
    assert PaletteReference.from_dict(restored.reference_data) == reference


def test_invalid_local_mask_state_is_rejected(photo_path):
    with pytest.raises(StudioError):
        SessionItem(photo_path, local_settings={"skin_strength": 2})
    with pytest.raises(StudioError):
        SessionItem(photo_path, mask_edits={"strokes": [{"points": [[2, 0.5]]}]})


def test_local_state_is_not_shared_with_mutable_caller_dictionaries(photo_path):
    original = {"skin_warmth": 0.23456}
    item = SessionItem(photo_path, local_settings=original)
    original["skin_warmth"] = 0.99
    assert item.local_settings["skin_warmth"] == 0.23456


def test_save_as_does_not_overwrite_existing_project_or_foreign_json(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text('{"kind": "preset", "schema_version": 1, "settings": {}}', encoding="utf-8")
    original = path.read_bytes()
    with pytest.raises(StudioError):
        save_session(path, Session())
    with pytest.raises(StudioError, match="не проект"):
        save_session(path, Session(), overwrite=True)
    assert path.read_bytes() == original


def test_failed_atomic_update_preserves_saved_project(tmp_path, monkeypatch):
    path = tmp_path / "project.json"
    save_session(path, Session())
    original = path.read_bytes()

    def fail_replace(*_args):
        raise PermissionError("simulated write failure")

    monkeypatch.setattr("freedive_color.core.session.os.replace", fail_replace)
    with pytest.raises(StudioError):
        save_session(path, Session(), overwrite=True)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".freedive-project-*"))


def test_project_path_cannot_replace_a_declared_source(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        '{"schema_version": 1, "kind": "freedive_project", "items": []}', encoding="utf-8"
    )
    original = source.read_bytes()
    with pytest.raises(StudioError, match="защищён"):
        save_session(source, Session((SessionItem(source),)), overwrite=True)
    assert source.read_bytes() == original


@pytest.mark.parametrize("version", (2, True, 1.0, "1"))
def test_unknown_or_wrongly_typed_schema_is_rejected(tmp_path, version):
    path = tmp_path / "project.json"
    path.write_text(
        json.dumps({"schema_version": version, "kind": "freedive_project", "items": []}),
        encoding="utf-8",
    )
    with pytest.raises(StudioError, match="Версия"):
        load_session(path)


@pytest.mark.parametrize("extra", ({"script": "do not execute"}, {"items": "not a list"}))
def test_unknown_fields_and_bad_item_types_rejected(tmp_path, extra):
    path = tmp_path / "project.json"
    path.write_text(
        json.dumps({"schema_version": 1, "kind": "freedive_project", "items": [], **extra}),
        encoding="utf-8",
    )
    with pytest.raises(StudioError):
        load_session(path)


@pytest.mark.parametrize(
    "path", ("https://example.com/photo.jpg", "file:///photo.jpg", "//server/share/photo.jpg")
)
def test_remote_paths_are_never_resolved(path):
    with pytest.raises(StudioError, match="Сетевые"):
        SessionItem(path)


@pytest.mark.parametrize(
    "metadata", ({"bad": float("nan")}, {1: "not a string key"}, {"bad": object()})
)
def test_metadata_must_be_finite_json(photo_path, metadata):
    with pytest.raises(StudioError):
        SessionItem(photo_path, media_info=metadata)


def test_invalid_selection_duplicate_items_and_hash_rejected(photo_path):
    item = SessionItem(photo_path)
    with pytest.raises(StudioError):
        Session((item, item))
    with pytest.raises(StudioError):
        Session((item,), selected_index=True)
    with pytest.raises(StudioError):
        Session((item,), selected_index=1)
    with pytest.raises(StudioError):
        SessionItem(photo_path, source_sha256="invalid")


@pytest.mark.parametrize(
    "options",
    (
        {"bit_depth": True},
        {"suffix": ".jpg", "bit_depth": 16},
        {"quality": 101},
        {"keep_gps": 1},
        {"suffix": ".mp4"},
    ),
)
def test_export_options_validate_types_and_format(options):
    with pytest.raises(StudioError):
        ExportOptions(**options)
