"""Real Qt queue/state tests. Controller has no window and never returns media buffers."""

import json
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from freedive_color.control.controller import ProjectController
from freedive_color.control.errors import ControlError
from freedive_color.core.image_pipeline import Preview
from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.masks import BrushStroke, MaskEdits
from freedive_color.core.media_io import load_photo
from freedive_color.core.presets import GradeSettings
from freedive_color.core.project import sha256_file
from freedive_color.core.session import Session, SessionItem


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(app, tmp_path):
    value = ProjectController(journal_path=tmp_path / "jobs.json", temp_root=tmp_path / "temp")
    yield value
    value.shutdown()
    deadline = time.monotonic() + 10
    while not value.is_idle() and time.monotonic() < deadline:
        app.processEvents()
        threading.Event().wait(0.005)
    assert value.is_idle()


def wait(app, controller, job_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        job = controller.get_job(job_id)
        if job["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return job
        threading.Event().wait(0.005)
    pytest.fail(f"Controller queue timed out: {controller.get_job(job_id)}")


def call(controller, method, **params):
    return controller.execute(method, {"session_id": controller.session_id, **params})


def photos(tmp_path, count=3):
    paths = []
    for i in range(count):
        path = tmp_path / f"Подводный кадр {i}.png"
        rgb = np.zeros((64, 96, 3), np.uint8)
        rgb[:] = (30 + i * 5, 115, 145)
        rgb[16:45, 35:60] = (90, 100, 95)
        Image.fromarray(rgb).save(path)
        paths.append(path)
    return paths


def add(controller, paths):
    return [
        controller.add_item(
            SessionItem(path=p, local_settings=LocalSettings(enabled=False).to_dict())
        )
        for p in paths
    ]


def test_atomic_group_revision_and_preserve_individual_masks(controller, tmp_path):
    first, second = add(controller, photos(tmp_path, 2))
    edits = MaskEdits(strokes=(BrushStroke(target="skin", points=((0.5, 0.5),), radius=0.05),))
    old = controller.read_record(first)
    controller.replace_item(first, replace(old.item, mask_edits=edits.to_dict()))
    before = controller.records()
    with pytest.raises(ControlError, match="изменены") as exc:
        call(
            controller,
            "set_grade",
            media_ids=[first, second],
            expected_revisions={first: 0, second: 0},
            preset="cinematic",
        )
    assert exc.value.code == "REVISION_CONFLICT"
    assert controller.records() == before
    with pytest.raises(ControlError):
        call(
            controller,
            "set_grade",
            media_ids=[first, "unknown"],
            expected_revisions={first: 1, "unknown": 0},
            preset="vivid",
        )
    assert controller.records() == before
    observed = []
    controller.state_changed.connect(
        lambda _event: observed.append([r.item.settings.preset for r in controller.records()])
    )
    call(
        controller,
        "set_grade",
        media_ids=[first, second],
        expected_revisions={first: 1, second: 0},
        preset="cinematic",
        grade={"exposure": 0.2},
    )
    assert observed == [["cinematic", "cinematic"]]
    assert controller.read_record(first).item.mask_edits == edits.to_dict()
    assert controller.read_record(first).item.local_settings == before[0].item.local_settings
    assert [r.revision for r in controller.records()] == [2, 1]


def test_records_copied_metadata_does_not_advance_processing_revision(controller, tmp_path):
    key = add(controller, photos(tmp_path, 1))[0]
    external = controller.read_record(key)
    external.item.local_settings["enabled"] = True
    assert controller.read_record(key).item.local_settings["enabled"] is False
    controller.replace_item(
        key, replace(controller.read_record(key).item, media_info={"width": 96})
    )
    assert controller.read_record(key).revision == 0
    controller.replace_item(
        key, replace(controller.read_record(key).item, settings=GradeSettings(exposure=0.4))
    )
    assert controller.read_record(key).revision == 1
    state = call(controller, "get_state")
    assert state["selected_media_id"] == key
    call(controller, "select_media", media_id=key, time_seconds=0.0)
    assert controller.time_seconds == 0


@pytest.mark.parametrize(
    "patch",
    [
        {"grade": {"saturation": 2}},
        {"local": {"enabled": 1}},
        {"grade": {"fake": 0.5}},
        {"extra": True},
    ],
)
def test_strict_errors_before_any_mutation(controller, tmp_path, patch):
    key = add(controller, photos(tmp_path, 1))[0]
    before = controller.records()
    with pytest.raises(ControlError) as exc:
        call(controller, "set_grade", media_ids=[key], expected_revisions={key: 0}, **patch)
    assert exc.value.code == "INVALID_PARAMS"
    assert controller.records() == before


def test_same_queue_cancel_waiting_does_not_start_next_while_running(controller, app):
    entered, release = threading.Event(), threading.Event()
    activity = []
    finished = []

    def first(token, progress):
        activity.append("first-start")
        entered.set()
        assert release.wait(5)
        activity.append("first-end")
        return {"done": True}

    one = controller.submit_ui_job("export", first, on_finished=lambda: finished.append("one"))
    assert entered.wait(3)
    two = controller.submit_ui_job(
        "preview",
        lambda *_: activity.append("CANCELLED-MUST-NOT-RUN"),
        on_finished=lambda: finished.append("two"),
    )
    three = controller.submit_ui_job("export", lambda *_: activity.append("third"))
    call(controller, "cancel_job", job_id=two)
    assert controller.get_job(two)["status"] == "cancelled"
    assert controller.get_job(three)["status"] == "queued"
    assert call(controller, "get_state")["queue"]["running"] == one
    release.set()
    assert wait(app, controller, one)["status"] == "succeeded"
    assert wait(app, controller, three)["status"] == "succeeded"
    assert activity == ["first-start", "first-end", "third"]
    assert sorted(finished) == ["one", "two"]


def test_frozen_snapshot_and_stale_preview_no_callback(controller, app, tmp_path):
    key = add(controller, photos(tmp_path, 1))[0]
    entered, release = threading.Event(), threading.Event()
    success, finished = [], []

    def preview(token, progress):
        entered.set()
        assert release.wait(5)
        return {"path": "/not-media-bytes", "settings": {"exposure": 0}}

    job_id = controller.submit_ui_job(
        "preview",
        preview,
        media_ids=[key],
        on_success=success.append,
        on_finished=lambda: finished.append(True),
    )
    assert entered.wait(3)
    call(
        controller,
        "set_grade",
        media_ids=[key],
        expected_revisions={key: 0},
        grade={"exposure": 0.5},
    )
    release.set()
    job = wait(app, controller, job_id)
    assert job["stale_preview"] is True and success == [] and finished == [True]
    assert job["snapshot"]["media"][0]["settings"]["exposure"] == 0
    assert controller.read_record(key).item.settings.exposure == 0.5


def test_live_preview_rgb_never_in_public_job_or_journal(controller, app, tmp_path):
    path = photos(tmp_path, 1)[0]
    photo = load_photo(path)
    result = Preview(photo, None, photo.rgb)
    got = []
    job_id = controller.submit_ui_job("view", lambda *_: result, on_success=got.append)
    job = wait(app, controller, job_id)
    assert got == [result]
    assert job["result"]["width"] == 96
    encoded = json.dumps(job)
    assert '"rgb"' not in encoded and '"alpha"' not in encoded
    journal = (tmp_path / "jobs.json").read_text(encoding="utf-8")
    assert '"rgb"' not in journal and "ndarray" not in journal


def test_real_photo_import_preview_export_dedupe_and_exact_recipe(controller, app, tmp_path):
    paths = photos(tmp_path)
    hashes = [sha256_file(p) for p in paths]
    imported = call(
        controller, "import_media", paths=list(map(str, paths)), request_id="import-three"
    )
    job = wait(app, controller, imported["job_id"])
    assert job["status"] == "succeeded"
    assert (
        call(controller, "import_media", paths=list(map(str, paths)), request_id="import-three")[
            "job_id"
        ]
        == imported["job_id"]
    )
    ids = imported["media_ids"]
    call(
        controller,
        "set_grade",
        media_ids=ids,
        expected_revisions=dict.fromkeys(ids, 0),
        preset="cinematic",
        local={"enabled": False},
    )
    call(
        controller,
        "set_grade",
        media_ids=[ids[1]],
        expected_revisions={ids[1]: 1},
        grade={"exposure": 0.35},
    )
    preview = call(
        controller,
        "render_preview",
        media_id=ids[1],
        duration_seconds=1.0,
        request_id="preview-one",
    )
    completed = wait(app, controller, preview["job_id"])
    assert completed["status"] == "succeeded", completed
    assert Path(completed["result"]["path"]).is_file()
    assert completed["result"]["recipe"]["settings"]["exposure"] == 0.35
    export = call(
        controller,
        "export_media",
        media_ids=ids,
        directory=str(tmp_path / "exports"),
        request_id="export-three",
    )
    completed = wait(app, controller, export["job_id"])
    assert completed["status"] == "succeeded", completed
    assert completed["result"]["completed"] == 3
    assert [sha256_file(p) for p in paths] == hashes
    assert (
        call(
            controller,
            "export_media",
            media_ids=ids,
            directory=str(tmp_path / "exports"),
            request_id="export-three",
        )["job_id"]
        == export["job_id"]
    )
    for row, expected in zip(completed["result"]["records"], [0.0, 0.35, 0.0], strict=True):
        recipe = json.loads(Path(row["recipe_path"]).read_text(encoding="utf-8"))
        assert recipe["settings"]["exposure"] == expected


def test_restart_marks_unfinished_interrupted_no_auto_export(app, tmp_path):
    path = tmp_path / "jobs.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "freedive_job_journal",
                "jobs": [
                    {
                        "job_id": "old",
                        "session_id": "prior",
                        "kind": "export_media",
                        "status": "running",
                        "result": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    value = ProjectController(journal_path=path)
    assert value.get_job("old")["status"] == "interrupted"
    assert value.is_idle() and value.pool.activeThreadCount() == 0
    assert value.session_id != "prior"
    value.shutdown()


def test_project_save_load_new_ids_compatible_and_busy_rejected(controller, app, tmp_path):
    keys = add(controller, photos(tmp_path, 2))
    source_session = controller.session_id
    path = tmp_path / "project.fcsproj"
    job = call(controller, "save_project", path=str(path))["job_id"]
    assert wait(app, controller, job)["status"] == "succeeded"
    saved = path.read_text(encoding="utf-8")
    assert "media_id" not in saved and "session_id" not in saved
    entered, release = threading.Event(), threading.Event()

    def block(*_):
        entered.set()
        assert release.wait(5)

    busy = controller.submit_ui_job("view", block)
    assert entered.wait(3)
    with pytest.raises(ControlError) as exc:
        call(controller, "load_project", path=str(path))
    assert exc.value.code == "PROJECT_BUSY"
    release.set()
    wait(app, controller, busy)
    loaded = call(controller, "load_project", path=str(path))["job_id"]
    assert wait(app, controller, loaded)["status"] == "succeeded"
    assert controller.session_id != source_session
    assert not set(keys) & {r.media_id for r in controller.records()}
    assert (
        controller.get_session().items
        == Session(items=tuple(r.item for r in controller.records())).items
    )


def test_three_real_videos_group_individual_preview_and_export(controller, app, tmp_path):
    from tools.generate_video_fixtures import generate

    paths = list(
        generate(tmp_path / "videos", duration=1.0, width=192, height=108, rate=4).values()
    )
    hashes = [sha256_file(path) for path in paths]
    imported = call(
        controller, "import_media", paths=list(map(str, paths)), request_id="three-videos"
    )
    assert wait(app, controller, imported["job_id"])["status"] == "succeeded"
    ids = imported["media_ids"]
    call(
        controller,
        "set_grade",
        media_ids=ids,
        expected_revisions=dict.fromkeys(ids, 0),
        preset="cinematic",
        local={"enabled": False},
    )
    call(
        controller,
        "set_grade",
        media_ids=[ids[1]],
        expected_revisions={ids[1]: 1},
        grade={"exposure": 0.25},
    )
    preview = call(
        controller,
        "render_preview",
        media_id=ids[1],
        duration_seconds=0.5,
        request_id="video-preview",
    )
    result = wait(app, controller, preview["job_id"], timeout=60)
    assert result["status"] == "succeeded", result
    assert result["result"]["recipe"]["output"]["width"] == 192
    assert controller.read_record(ids[1]).preview["revision"] == 2
    exported = call(
        controller,
        "export_media",
        media_ids=ids,
        directory=str(tmp_path / "exports"),
        request_id="three-video-export",
    )
    job = wait(app, controller, exported["job_id"], timeout=60)
    assert job["status"] == "succeeded", job
    assert job["result"]["completed"] == 3
    for row, exposure in zip(job["result"]["records"], [0.0, 0.25, 0.0], strict=True):
        report = row["recipe"]
        assert report["verification"]["audio_verified"]
        assert report["settings"]["exposure"] == exposure
        assert report["output"]["width"] == 192 and report["output"]["height"] == 108
    assert hashes == [sha256_file(path) for path in paths]


def test_cancel_real_export_series_keeps_completed_files(controller, app, tmp_path, monkeypatch):
    import freedive_color.control.controller as module

    paths = photos(tmp_path)
    hashes = [sha256_file(path) for path in paths]
    keys = add(controller, paths)
    original = module.export_photo

    def stop_after_first(*args, **kwargs):
        result = original(*args, **kwargs)
        kwargs["token"].cancel()
        return result

    monkeypatch.setattr(module, "export_photo", stop_after_first)
    response = call(controller, "export_media", media_ids=keys, directory=str(tmp_path / "exports"))
    job = wait(app, controller, response["job_id"])
    assert job["status"] == "cancelled" and job["result"]["completed"] == 1
    first = Path(job["result"]["records"][0]["path"])
    assert first.is_file() and first.with_suffix(".png.freedive.json").is_file()
    assert len(list((tmp_path / "exports").glob("*.png"))) == 1
    assert hashes == [sha256_file(path) for path in paths]


def test_generic_ui_batch_cancelled_state_keeps_success_callback(controller, app):
    from freedive_color.core.batch import BatchResult

    value = BatchResult((), cancelled=True)
    results = []
    job = controller.submit_ui_job("export", lambda *_: value, on_success=results.append)
    assert wait(app, controller, job)["status"] == "cancelled"
    assert results == [value]


def test_wire_large_masks_and_byte_limited_media_pages(controller, tmp_path):
    points = ((0.1, 0.2),) * 10_000
    key = controller.add_item(
        SessionItem(
            path=tmp_path / "mask.png",
            mask_edits=MaskEdits(strokes=(BrushStroke(points=points),)).to_dict(),
        )
    )
    state = call(controller, "get_state")
    assert state["selected"]["mask_edits"]["summary_only"]
    assert len(controller.read_record(key).item.mask_edits["strokes"][0]["points"]) == 10_000
    for i in range(30):
        controller.add_item(
            SessionItem(path=tmp_path / f"large-{i}.png", media_info={"log": ["sample"] * 5000})
        )
    page = call(controller, "list_media", limit=50)
    assert page["next_cursor"] is not None and len(page["media"]) < 31
    assert len(json.dumps(page, ensure_ascii=False).encode()) < 1_048_576
    collected = [row["media_id"] for row in page["media"]]
    while page["next_cursor"]:
        page = call(controller, "list_media", limit=50, cursor=page["next_cursor"])
        collected.extend(row["media_id"] for row in page["media"])
    assert collected == [record.media_id for record in controller.records()]


def test_reference_clear_wins_over_older_pending_reference(controller, app, tmp_path, monkeypatch):
    import freedive_color.control.controller as module

    path = photos(tmp_path, 1)[0]
    entered, release = threading.Event(), threading.Event()
    original = module.create_image_reference

    def blocked(path):
        entered.set()
        assert release.wait(5)
        return original(path)

    monkeypatch.setattr(module, "create_image_reference", blocked)
    pending = call(controller, "set_reference", source="photo", path=str(path))["job_id"]
    assert entered.wait(3)
    call(controller, "set_reference", source="clear")
    release.set()
    job = wait(app, controller, pending)
    assert job["status"] == "succeeded" and job["result"]["applied"] is False
    assert controller.get_session().reference_data is None


def test_legacy_project_reference_path_resolved_once_for_real_preview(app, tmp_path):
    path = photos(tmp_path, 1)[0]
    item = SessionItem(path=path, local_settings=LocalSettings(enabled=False).to_dict())
    controller = ProjectController(
        Session(items=(item,), reference_path=path), temp_root=tmp_path / "temp"
    )
    key = controller.records()[0].media_id
    job = call(controller, "render_preview", media_id=key)["job_id"]
    result = wait(app, controller, job)
    assert result["status"] == "succeeded", result
    assert result["result"]["recipe"]["reference"]["source_sha256"] == sha256_file(path)
    controller.shutdown()


def test_ui_save_updates_shared_project_path_immediately(controller, tmp_path):
    add(controller, photos(tmp_path, 1))
    path = tmp_path / "saved.fcsproj"
    controller.save_project_from_ui(path)
    assert path.is_file() and controller.project_path == path
    assert call(controller, "get_state")["project_path"] == str(path)


def test_widget_finalization_error_cannot_stall_shared_queue(controller, app):
    def broken_widget():
        raise RuntimeError("Deleted UI widget")

    first = controller.submit_ui_job("view", lambda *_: None, on_finished=broken_widget)
    second = controller.submit_ui_job("export", lambda *_: {"completed": True})
    assert wait(app, controller, first)["status"] == "succeeded"
    assert wait(app, controller, second)["status"] == "succeeded"


def test_default_widget_normalization_is_not_a_processing_edit(controller, tmp_path):
    key = controller.add_item(SessionItem(path=tmp_path / "source.png"))
    record = controller.read_record(key)
    controller.replace_item(
        key,
        replace(
            record.item, local_settings=LocalSettings().to_dict(), mask_edits=MaskEdits().to_dict()
        ),
    )
    assert controller.read_record(key).revision == 0
    call(
        controller,
        "set_grade",
        media_ids=[key],
        expected_revisions={key: 0},
        local={"water_hue": 0.0},
    )
    assert controller.read_record(key).revision == 0
    call(
        controller,
        "set_grade",
        media_ids=[key],
        expected_revisions={key: 0},
        local={"water_hue": 0.1},
    )
    assert controller.read_record(key).revision == 1


def test_trusted_view_telemetry_is_silent_and_cannot_select_another_media(controller, tmp_path):
    first, second = add(controller, photos(tmp_path, 2))
    events = []
    controller.state_changed.connect(events.append)
    controller.update_view_position(first, 1.25)
    assert controller.time_seconds == 1.25 and events == []
    controller.update_view_position(second, 2.5)
    assert controller.time_seconds == 1.25 and events == []
