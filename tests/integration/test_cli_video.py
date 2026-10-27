"""CLI wiring: replay video clocks/analysis and preserve mixed-series options."""

import json

from freedive_color.cli import run
from freedive_color.core import video_pipeline
from freedive_color.core.presets import get_preset
from freedive_color.core.project import sha256_file
from freedive_color.core.session import ExportOptions, Session, SessionItem, save_session


def test_video_recipe_reuses_plan_and_preview_range(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"fixture-only; exporter is replaced in this wiring test")
    recipe = tmp_path / "recipe.json"
    data = {
        "schema_version": 1,
        "kind": "video_recipe",
        "source": {"sha256": sha256_file(source)},
        "settings": get_preset("natural").to_dict(),
        "plan": {"fixed_shot_analysis": True},
        "output": {"proxy_edge": 1280},
        "preview": {"enabled": True, "requested_start": 0.75, "requested_seconds": 1.5},
    }
    recipe.write_text(json.dumps(data))
    calls = []

    def export(source, destination, settings, **kwargs):
        calls.append((source, destination, settings, kwargs))
        return {"verified_wiring": True}

    monkeypatch.setattr(video_pipeline, "export_video", export)
    assert (
        run(
            ["--input", str(source), "--output", str(tmp_path / "out.mp4"), "--recipe", str(recipe)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["verified_wiring"]
    assert calls[0][3]["plan"] == data["plan"]
    assert calls[0][3]["preview_start"] == 0.75
    assert calls[0][3]["preview_seconds"] == 1.5
    assert calls[0][3]["proxy_edge"] == 1280
    source.write_bytes(b"changed original")
    assert (
        run(
            [
                "--input",
                str(source),
                "--output",
                str(tmp_path / "other.mp4"),
                "--recipe",
                str(recipe),
            ]
        )
        == 2
    )
    assert "другого исходника" in json.loads(capsys.readouterr().out)["error"]
    assert len(calls) == 1


def test_project_video_export_keeps_gps_and_temp_directory(tmp_path, monkeypatch, capsys):
    source = tmp_path / "source.mov"
    source.write_bytes(b"fixture-only; exporter is replaced in this wiring test")
    project, temporary = tmp_path / "dive.fcsproj", tmp_path / "temporary"
    save_session(
        project,
        Session(
            (SessionItem(source),),
            temp_directory=temporary,
            export_options=ExportOptions(keep_gps=True),
        ),
    )
    calls = []
    monkeypatch.setattr(
        video_pipeline, "export_video", lambda *args, **kwargs: calls.append((args, kwargs)) or {}
    )
    assert run(["--project", str(project), "--output", str(tmp_path / "exports")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["completed"] == 1
    assert calls[0][0][1].suffix == ".mov"
    assert calls[0][1]["keep_gps"] is True
    assert calls[0][1]["temp_directory"] == temporary
    assert "bit_depth" not in calls[0][1]


def test_series_rejects_preview_range(tmp_path, capsys):
    assert (
        run(
            [
                "--input",
                str(tmp_path),
                "--output",
                str(tmp_path / "exports"),
                "--preview-seconds",
                "2",
            ]
        )
        == 2
    )
    assert "серии" in json.loads(capsys.readouterr().out)["error"]
