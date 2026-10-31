"""The real window and remote project operations observe one authoritative state."""

import threading
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from freedive_color.control.controller import ProjectController
from freedive_color.core.masks import BrushStroke
from freedive_color.core.session import load_session
from freedive_color.ui.main_window import MainWindow


def _ready(qtbot, window):
    qtbot.waitUntil(
        lambda: (
            window.preview is not None and not window._render_pending and window._view_job is None
        ),
        timeout=60_000,
    )


@pytest.fixture
def shared_window(qapp, qtbot, tmp_path):
    controller = ProjectController(temp_root=tmp_path / "previews")
    window = MainWindow(controller=controller, integration_mode=True)
    qtbot.addWidget(window)
    window.show()
    yield window, controller
    window.request_quit()
    qapp.processEvents()


def test_group_remote_grade_and_manual_edits_share_revisions(shared_window, qtbot, tmp_path):
    window, controller = shared_window
    paths = []
    for index in range(3):
        path = tmp_path / f"input-{index}.png"
        Image.new("RGB", (128, 96), (35 + 10 * index, 100, 150)).save(path)
        paths.append(path)
    window.import_paths(paths)
    _ready(qtbot, window)
    records = controller.records()
    assert len(records) == 3
    session = controller.session_id
    first = records[0].media_id
    window.controls.set_settings(replace(window.controls.settings(), exposure=0.17))
    _ready(qtbot, window)
    current = controller.read_record(first)
    assert current.item.settings.exposure == 0.17
    assert current.revision > records[0].revision
    window.viewer.stroke_created.emit(
        BrushStroke(points=((0.3, 0.4), (0.4, 0.5)), radius=0.03).to_dict()
    )
    _ready(qtbot, window)
    before_group = controller.records()
    controller.execute(
        "set_grade",
        {
            "session_id": session,
            "media_ids": [record.media_id for record in before_group],
            "expected_revisions": {record.media_id: record.revision for record in before_group},
            "preset": "cinematic",
            "grade": {"temperature": 0.12},
        },
    )
    _ready(qtbot, window)
    assert window.controls.settings().temperature == 0.12
    assert all(record.item.settings.preset == "cinematic" for record in controller.records())
    assert controller.read_record(first).item.mask_edits == before_group[0].item.mask_edits
    assert len(window.mask_edits.strokes) == 1
    third = controller.records()[2]
    controller.execute(
        "select_media",
        {
            "session_id": session,
            "media_id": third.media_id,
            "time_seconds": 0,
        },
    )
    _ready(qtbot, window)
    assert window.preview.photo.path == paths[2]
    assert window.mask_edits.strokes == ()
    state = controller.execute("get_state", {"session_id": session})
    assert state["selected_media_id"] == third.media_id
    project = tmp_path / "shared.fcsproj"
    window.save_project(project)
    saved = load_session(project)
    assert saved.items == controller.get_session().items
    assert "media_id" not in project.read_text(encoding="utf-8")
    old_ids = {record.media_id for record in controller.records()}
    window.load_project(project)
    _ready(qtbot, window)
    assert controller.session_id != session
    assert old_ids.isdisjoint({record.media_id for record in controller.records()})


def test_remote_edit_while_rendering_never_displays_stale_grade(
    shared_window, qtbot, tmp_path, monkeypatch
):
    import freedive_color.ui.main_window as ui

    window, controller = shared_window
    path = tmp_path / "source.png"
    Image.new("RGB", (120, 80), (40, 90, 140)).save(path)
    window.import_paths([path])
    _ready(qtbot, window)
    entered, release = threading.Event(), threading.Event()
    original = ui.render_rgb

    def held_render(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(ui, "render_rgb", held_render)
    window.controls.set_settings(replace(window.controls.settings(), exposure=0.1))
    qtbot.waitUntil(entered.is_set, timeout=5_000)
    record = controller.records()[0]
    controller.execute(
        "set_grade",
        {
            "session_id": controller.session_id,
            "media_ids": [record.media_id],
            "expected_revisions": {record.media_id: record.revision},
            "grade": {"exposure": 0.31},
        },
    )
    release.set()
    _ready(qtbot, window)
    assert window.controls.settings().exposure == 0.31
    assert window.rendered_settings.exposure == 0.31
    assert controller.read_record(record.media_id).item.settings.exposure == 0.31


def test_close_hides_window_and_preserves_a_running_queue(shared_window, qtbot):
    window, controller = shared_window
    entered, release = threading.Event(), threading.Event()

    def held_job(token, progress):
        entered.set()
        while not release.wait(0.01):
            token.check()
        progress(100, "Готово")
        return {"completed": True}

    job = controller.submit_ui_job("export", held_job)
    qtbot.waitUntil(entered.is_set, timeout=5_000)
    window.close()
    assert not window.isVisible()
    assert not controller.is_idle()
    release.set()
    qtbot.waitUntil(lambda: controller.get_job(job)["status"] == "succeeded", timeout=5_000)
    window.show()
    assert window.isVisible()
    assert controller.get_job(job)["result"] == {"completed": True}


def test_copy_configuration_does_not_edit_codex_settings(shared_window, monkeypatch):
    import tomllib

    from PySide6.QtWidgets import QApplication

    window, _controller = shared_window
    writes = []
    monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: writes.append(args))
    window._copy_mcp_configuration()
    config = tomllib.loads(QApplication.clipboard().text())["mcp_servers"]["freedive_color"]
    assert Path(config["command"]).is_absolute()
    assert config["args"] == ["-m", "freedive_color.control.mcp_server"]
    assert not writes


def test_gui_batch_journal_targets_all_included_files(shared_window, qtbot, tmp_path):
    from PySide6.QtCore import Qt

    window, controller = shared_window
    paths = []
    for index in range(3):
        path = tmp_path / f"batch-{index}.png"
        Image.new("RGB", (96, 64), (30, 100, 140)).save(path)
        paths.append(path)
    window.import_paths(paths)
    _ready(qtbot, window)
    window.library.item(2).setCheckState(Qt.CheckState.Unchecked)
    window.library.setCurrentRow(2)
    _ready(qtbot, window)
    expected = [record.media_id for record in controller.records() if record.item.included]
    output = tmp_path / "batch-output"
    window.start_batch_export(output)
    job_id = window._jobs[window._export_job][0].job_id
    captured = controller.get_job(job_id)
    assert captured["media_ids"] == expected
    assert {record["media_id"] for record in captured["snapshot"]["media"]} == set(expected)
    assert captured["snapshot"]["operation"]["directory"] == str(output)
    assert captured["snapshot"]["operation"]["excluded_paths"] == [str(paths[2])]
    assert controller.get_session().output_directory == output
    qtbot.waitUntil(lambda: controller.get_job(job_id)["status"] == "succeeded", timeout=15_000)
    result = controller.get_job(job_id)["result"]
    assert result["completed"] == 2 and result["excluded"] == 1


@pytest.mark.parametrize("change_selection", [False, True])
def test_remote_selection_or_grade_during_gui_export_refreshes_correct_source(
    shared_window, qtbot, tmp_path, monkeypatch, change_selection
):
    import freedive_color.ui.main_window as ui

    window, controller = shared_window
    paths = []
    for index in range(2):
        path = tmp_path / f"export-source-{index}.png"
        Image.new("RGB", (96, 64), (35 + 20 * index, 110, 150)).save(path)
        paths.append(path)
    window.import_paths(paths)
    _ready(qtbot, window)
    entered, release = threading.Event(), threading.Event()
    original = ui.export_photo

    def held(*args, **kwargs):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(ui, "export_photo", held)
    window.start_export(tmp_path / "first.png")
    qtbot.waitUntil(entered.is_set, timeout=5_000)
    target = controller.records()[1 if change_selection else 0]
    if change_selection:
        controller.execute(
            "select_media",
            {"session_id": controller.session_id, "media_id": target.media_id, "time_seconds": 0},
        )
    target = controller.read_record(target.media_id)
    controller.execute(
        "set_grade",
        {
            "session_id": controller.session_id,
            "media_ids": [target.media_id],
            "expected_revisions": {target.media_id: target.revision},
            "grade": {"exposure": 0.37},
        },
    )
    assert window.preview is None and window._render_pending
    release.set()
    _ready(qtbot, window)
    assert window.preview.photo.path == target.item.path
    assert window.rendered_settings.exposure == 0.37
    assert window.last_export["source"]["path"] == str(paths[0])
    assert window.last_export["settings"]["exposure"] == 0
    if change_selection:
        assert "Экспортировано" not in window.library.currentItem().text()
    window.start_export(tmp_path / "second.png")
    qtbot.waitUntil(lambda: window._export_job is None, timeout=15_000)
    assert window.last_export["source"]["path"] == str(target.item.path)
    assert window.last_export["settings"]["exposure"] == 0.37


def test_playback_updates_shared_position_and_grade_uses_visible_frame(
    shared_window, qtbot, tmp_path
):
    from tools.generate_video_fixtures import generate

    window, controller = shared_window
    source = generate(tmp_path / "clips", duration=1.5, width=192, height=108, rate=6)[
        "depth_change"
    ]
    window.import_paths([source])
    _ready(qtbot, window)
    window.create_video_preview(0.0, 1.2)
    qtbot.waitUntil(lambda: window._export_job is None, timeout=30_000)
    window.toggle_video_playback()
    qtbot.waitUntil(lambda: controller.time_seconds >= 0.3, timeout=5_000)
    visible_time = window.video_frame_seconds
    assert controller.time_seconds == visible_time
    record = controller.records()[0]
    controller.execute(
        "set_grade",
        {
            "session_id": controller.session_id,
            "media_ids": [record.media_id],
            "expected_revisions": {record.media_id: record.revision},
            "grade": {"exposure": 0.21},
        },
    )
    _ready(qtbot, window)
    assert not window._video_playing
    assert abs(window.preview.timestamp - visible_time) <= 1 / 6 + 1e-6
    assert window.rendered_settings.exposure == 0.21


def test_switch_video_during_gui_proxy_never_assigns_old_plan_to_new_file(
    shared_window, qtbot, tmp_path, monkeypatch
):
    import freedive_color.core.video_pipeline as pipeline
    from tools.generate_video_fixtures import generate

    window, controller = shared_window
    clips = generate(tmp_path / "clips", duration=1.0, width=192, height=108, rate=4)
    paths = [clips["static"], clips["vfr_offset"]]
    # Distinct clocks make copying the previous file's plan observable even if both
    # clips show the same scene. The selected file now legitimately creates its own plan.
    plans = [pipeline.analyze_video(path) for path in paths]
    assert plans[0] != plans[1]
    window.import_paths(paths)
    _ready(qtbot, window)
    entered, release = threading.Event(), threading.Event()
    original = pipeline.export_video

    def held(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "export_video", held)
    window.create_video_preview(0.0, 0.5)
    qtbot.waitUntil(entered.is_set, timeout=5_000)
    first, second = controller.records()
    controller.execute(
        "select_media",
        {"session_id": controller.session_id, "media_id": second.media_id, "time_seconds": 0},
    )
    assert window.preview is None
    release.set()
    _ready(qtbot, window)
    assert window.preview.photo.path == paths[1]
    assert window.video_preview_path is None
    assert (
        pipeline.ShotPlan.from_dict(
            controller.read_record(second.media_id).item.media_info["shot_plan"]
        )
        == plans[1]
    )
    assert (
        pipeline.ShotPlan.from_dict(
            controller.read_record(first.media_id).item.media_info["shot_plan"]
        )
        == plans[0]
    )
    assert window._video_plan == plans[1]
