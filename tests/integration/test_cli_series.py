"""Regression checks for explicit CLI overrides and reproducible project/recipe exports."""

import json
from dataclasses import replace

import numpy as np
import pytest

from freedive_color.cli import run
from freedive_color.core.image_pipeline import export_photo
from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.masks import BrushStroke, MaskEdits
from freedive_color.core.media_io import load_photo
from freedive_color.core.presets import get_preset
from freedive_color.core.reference import PaletteReference
from freedive_color.core.session import Session, SessionItem, save_session


@pytest.fixture
def local_edits_file(tmp_path):
    path = tmp_path / "masks.json"
    edits = MaskEdits(strokes=(BrushStroke(target="water", points=((0.7, 0.3),)),))
    path.write_text(json.dumps(edits.to_dict()), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "conflict", ("--global-only", "--water-hue", "--mask-edits", "--reference", "--preset")
)
def test_recipe_rejects_conflicting_pixel_options(
    photo_path, tmp_path, capsys, local_edits_file, conflict
):
    first = tmp_path / "original.png"
    export_photo(photo_path, first, get_preset("natural"))
    output = tmp_path / "conflicting.png"
    values = {
        "--water-hue": "0.5",
        "--mask-edits": str(local_edits_file),
        "--reference": str(photo_path),
        "--preset": "vivid",
    }
    arguments = [
        "--input",
        str(photo_path),
        "--output",
        str(output),
        "--recipe",
        str(first.with_suffix(".png.freedive.json")),
        conflict,
    ]
    if conflict in values:
        arguments.append(values[conflict])
    assert run(arguments) == 2
    assert "error" in json.loads(capsys.readouterr().out)
    assert not output.exists()


@pytest.mark.parametrize(
    "conflict", ("--skin-warmth", "--skin-strength", "--water-hue", "--reference", "--mask-edits")
)
def test_global_only_rejects_conflicting_local_options(
    photo_path, tmp_path, capsys, local_edits_file, conflict
):
    output = tmp_path / "conflicting.png"
    value = (
        str(photo_path)
        if conflict == "--reference"
        else (str(local_edits_file) if conflict == "--mask-edits" else "0.5")
    )
    assert (
        run(["--input", str(photo_path), "--output", str(output), "--global-only", conflict, value])
        == 2
    )
    assert "error" in json.loads(capsys.readouterr().out)
    assert not output.exists()


def test_project_global_only_disables_saved_local_state_and_reference(photo_path, tmp_path, capsys):
    project = tmp_path / "dive.fcsproj"
    settings = replace(get_preset("natural"), exposure=0.25)
    local = LocalSettings(water_hue=0.6, skin_strength=0.8)
    edits = MaskEdits(waterline=((0.0, 0.3), (1.0, 0.3)))
    reference = PaletteReference(water_ab=(-3, -8), source_name="Saved reference")
    save_session(
        project,
        Session(
            (
                SessionItem(
                    photo_path, settings, local_settings=local.to_dict(), mask_edits=edits.to_dict()
                ),
            ),
            reference_path=photo_path,
            reference_data=reference.to_dict(),
        ),
    )
    assert (
        run(["--project", str(project), "--output", str(tmp_path / "output"), "--global-only"]) == 0
    )
    record = json.loads(capsys.readouterr().out)["records"][0]
    assert record["report"]["local_settings"]["enabled"] is False
    assert record["report"]["reference"] is None
    expected = tmp_path / "global.png"
    export_photo(photo_path, expected, settings)
    np.testing.assert_array_equal(load_photo(record["destination"]).rgb, load_photo(expected).rgb)


def test_project_reference_path_fallback_is_used_when_frozen_palette_is_absent(
    photo_path, tmp_path, capsys
):
    project = tmp_path / "dive.fcsproj"
    save_session(project, Session((SessionItem(photo_path),), reference_path=photo_path))
    assert run(["--project", str(project), "--output", str(tmp_path / "output")]) == 0
    report = json.loads(capsys.readouterr().out)["records"][0]["report"]
    assert report["reference"] is not None
    assert report["reference"]["source_name"] == photo_path.name


def test_project_preset_and_set_overrides_apply_intentionally(photo_path, tmp_path, capsys):
    project = tmp_path / "dive.fcsproj"
    source_settings = replace(get_preset("natural"), exposure=-0.7, contrast=-0.2)
    save_session(project, Session((SessionItem(photo_path, source_settings),)))
    assert (
        run(
            [
                "--project",
                str(project),
                "--output",
                str(tmp_path / "output"),
                "--preset",
                "cinematic",
                "--set",
                "exposure=0.125",
            ]
        )
        == 0
    )
    settings = json.loads(capsys.readouterr().out)["records"][0]["report"]["settings"]
    assert settings["preset"] == "cinematic"
    assert settings["contrast"] == get_preset("cinematic").contrast
    assert settings["exposure"] == 0.125


def test_project_default_preserves_saved_per_file_settings(photo_path, tmp_path, capsys):
    project = tmp_path / "dive.fcsproj"
    source_settings = replace(get_preset("vivid"), exposure=-0.37, contrast=-0.2)
    save_session(project, Session((SessionItem(photo_path, source_settings),)))
    assert run(["--project", str(project), "--output", str(tmp_path / "output")]) == 0
    settings = json.loads(capsys.readouterr().out)["records"][0]["report"]["settings"]
    assert settings == source_settings.to_dict()


def test_local_recipe_replay_is_pixel_identical(photo_path, tmp_path, capsys):
    source_output = tmp_path / "first.png"
    edits = MaskEdits(strokes=(BrushStroke(target="water", points=((0.7, 0.3),)),))
    export_photo(
        photo_path,
        source_output,
        replace(get_preset("natural"), exposure=0.2),
        local_settings=LocalSettings(water_hue=0.2),
        mask_edits=edits,
        reference=PaletteReference(water_ab=(-8, -18)),
    )
    repeated = tmp_path / "repeated.png"
    assert (
        run(
            [
                "--input",
                str(photo_path),
                "--output",
                str(repeated),
                "--recipe",
                str(source_output.with_suffix(".png.freedive.json")),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["settings"]["exposure"] == 0.2
    np.testing.assert_array_equal(load_photo(source_output).rgb, load_photo(repeated).rgb)


def test_bad_mask_json_produces_an_actionable_cli_error(photo_path, tmp_path, capsys):
    path = tmp_path / "bad-mask.json"
    path.write_text("[]", encoding="utf-8")
    assert (
        run(
            [
                "--input",
                str(photo_path),
                "--output",
                str(tmp_path / "out.png"),
                "--mask-edits",
                str(path),
            ]
        )
        == 2
    )
    assert "error" in json.loads(capsys.readouterr().out)


def test_custom_preset_keeps_local_controls(photo_path, tmp_path, capsys):
    from freedive_color.cli import run
    from freedive_color.core.local_pipeline import LocalSettings
    from freedive_color.core.presets import get_preset
    from freedive_color.core.project import save_settings

    local = LocalSettings(water_hue=0.25, water_saturation=0.1, skin_warmth=0.2)
    preset = tmp_path / "local_preset.json"
    save_settings(preset, get_preset("natural"), local_settings=local)
    assert (
        run(
            [
                "--input",
                str(photo_path),
                "--output",
                str(tmp_path / "preset.png"),
                "--settings",
                str(preset),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["local_settings"] == local.to_dict()
