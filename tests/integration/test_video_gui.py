"""Milestone C: real GUI video seek, paired playback, previews and media export."""

import importlib.util
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from freedive_color.core.masks import BrushStroke, MaskEdits, MaskSet
from freedive_color.core.project import sha256_file
from freedive_color.core.reference import create_reference
from freedive_color.core.session import load_session
from freedive_color.ui.main_window import MainWindow
from freedive_color.ui.theme import apply_theme

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def video_fixtures(tmp_path_factory):
    # This developer-only generator creates real encoded MP4/AAC fixtures offline.
    spec = importlib.util.spec_from_file_location(
        "video_fixture_generator", ROOT / "tools" / "generate_video_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path_factory.mktemp("gui_video")
    module.generate(directory, duration=2.4, width=320, height=180, rate=8)
    module._one(directory / "large.mp4", module._photo_canvas(1800, 1000), 0.5, 2)
    return directory


@pytest.fixture
def window(qapp, qtbot):
    apply_theme(qapp)
    widget = MainWindow()
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    widget.close()


def _ready(qtbot, window, *, video=True):
    qtbot.waitUntil(
        lambda: (
            window.preview is not None
            and not window._render_pending
            and window._view_job is None
            and (not video or window.video_info is not None)
        ),
        timeout=60_000,
    )


def _media(path: Path):
    with av.open(str(path)) as container:
        geometry = (container.streams.video[0].width, container.streams.video[0].height)
        timestamps = [
            frame.pts * frame.time_base for frame in container.decode(container.streams.video[0])
        ]
    with av.open(str(path)) as container:
        audio = [bytes(packet) for packet in container.demux(audio=0) if packet.size]
    return geometry, timestamps, audio


def _operation(qtbot, window, signal, start):
    outcomes = []

    def success(report):
        outcomes.append((True, report))

    def failed(message):
        outcomes.append((False, message))

    signal.connect(success)
    window.operation_failed.connect(failed)
    try:
        start()
        assert window._export_job is not None, "A ready video command did not start its worker"
        qtbot.waitUntil(lambda: bool(outcomes), timeout=120_000)
    finally:
        signal.disconnect(success)
        window.operation_failed.disconnect(failed)
    assert outcomes[0][0], outcomes[0][1]
    qtbot.waitUntil(lambda: window._export_job is None, timeout=10_000)
    return outcomes[0][1]


def test_gui_video_seek_preview_playback_and_full_export(window, qtbot, video_fixtures, tmp_path):
    source = video_fixtures / "depth_change.mp4"
    digest = sha256_file(source)
    window.import_paths([source])
    _ready(qtbot, window)
    assert window.video_panel.isVisible()
    assert window.video_timeline.maximum() > 2_000
    assert window.video_duration_spin.value() == pytest.approx(min(10, window.video_info.duration))
    assert window.preview.source_sha256 == digest
    assert not window.viewer.before.isNull() and not window.viewer.after.isNull()
    window.seek_video(0.75)
    _ready(qtbot, window)
    assert window.video_frame_seconds == pytest.approx(0.75, abs=0.15)

    destination = tmp_path / "preview.mp4"
    _operation(
        qtbot,
        window,
        window.video_preview_completed,
        lambda: window.create_video_preview(0.0, 2.0, destination),
    )
    assert window.video_preview_path == destination
    assert destination.is_file()
    assert window.video_preview_start == 0.0
    assert window.video_preview_duration == 2.0
    assert _media(destination)[2], "The real preview must contain source audio packets"

    window.toggle_video_playback()
    qtbot.waitUntil(lambda: window._video_playing, timeout=5_000)
    playback_frames = []
    window.video_playback.frame_ready.connect(
        lambda _before, _after, seconds: playback_frames.append(seconds)
    )
    qtbot.waitUntil(lambda: any(point > 0.3 for point in playback_frames), timeout=15_000)
    window._one_to_one()
    qtbot.waitUntil(lambda: not window._video_playing, timeout=5_000)
    _ready(qtbot, window)
    assert window.video_frame_seconds >= 0.3

    output = tmp_path / "full_export.mp4"
    _operation(
        qtbot, window, window.video_export_completed, lambda: window.start_video_export(output)
    )
    assert output.is_file()
    assert _media(output) == _media(source), "Geometry, video clocks and encoded audio must survive"
    sidecar = Path(f"{output}.freedive.json")
    report = json.loads(sidecar.read_text(encoding="utf-8"))
    assert digest in json.dumps(report)
    assert sha256_file(source) == digest


def test_gui_vfr_offset_seek_is_relative_and_project_reopens(
    window, qtbot, video_fixtures, tmp_path
):
    source = video_fixtures / "vfr_offset.mp4"
    digest = sha256_file(source)
    window.import_paths([source])
    _ready(qtbot, window)
    assert window.video_info.video_start > 0
    window.seek_video(0.0)
    _ready(qtbot, window)
    assert window.video_frame_seconds == pytest.approx(0, abs=0.15)
    window.seek_video(0.65)
    _ready(qtbot, window)
    assert window.video_frame_seconds == pytest.approx(0.65, abs=0.25)
    assert 0 <= window.video_frame_seconds < window.video_info.duration
    project = tmp_path / "video.fcsproj"
    window.save_project(project)
    window.load_project(project)
    _ready(qtbot, window)
    assert window.library.count() == 1
    assert window.video_info.path == source
    assert window.preview.source_sha256 == digest
    assert sha256_file(source) == digest


def test_vfr_playback_pairs_the_exact_first_frame_after_nonaligned_start(
    window, qtbot, video_fixtures, tmp_path, monkeypatch
):
    from freedive_color.ui import main_window as ui

    source = video_fixtures / "vfr_offset.mp4"
    window.import_paths([source])
    _ready(qtbot, window)
    report = _operation(
        qtbot,
        window,
        window.video_preview_completed,
        lambda: window.create_video_preview(0.19, 1.0, tmp_path / "vfr_preview.mp4"),
    )
    actual = report["preview"]["actual_source_start"]
    expected = float(
        Fraction(actual["numerator"], actual["denominator"]) - window.video_info.video_start
    )
    assert expected > 0.19
    player_type = ui.VideoPlayback
    displayed = []

    def observed_player(*args, **kwargs):
        player = player_type(*args, **kwargs)
        player.frame_ready.connect(lambda _before, _after, seconds: displayed.append(seconds))
        return player

    monkeypatch.setattr(ui, "VideoPlayback", observed_player)
    window.toggle_video_playback()
    qtbot.waitUntil(lambda: bool(displayed), timeout=15_000)
    window.stop_video_playback()
    assert displayed[0] == pytest.approx(expected, abs=1e-6)


def test_current_video_result_becomes_a_persistent_frame_reference(
    window, qtbot, video_fixtures, tmp_path
):
    source = video_fixtures / "depth_change.mp4"
    digest = sha256_file(source)
    window.import_paths([source])
    _ready(qtbot, window)
    window.seek_video(0.75)
    _ready(qtbot, window)
    timestamp = window.preview.timestamp
    name = f"{source.name} · {timestamp:.2f} с"
    expected = create_reference(
        window.preview.result,
        masks=window.preview.masks,
        source_name=name,
        source_sha256=digest,
    )
    window.use_frame_reference()
    _ready(qtbot, window)
    assert window.reference.to_dict() == expected.to_dict()
    assert window.reference_path is None
    assert f"{timestamp:.2f}" in window.local_controls.reference_label.text()
    project = tmp_path / "frame_reference.fcsproj"
    window.save_project(project)
    assert load_session(project).reference_data == expected.to_dict()
    window.load_project(project)
    _ready(qtbot, window)
    assert window.reference.to_dict() == expected.to_dict()
    assert window.reference.source_sha256 == digest
    assert name == window.reference.source_name
    assert window.reference_path is None
    assert sha256_file(source) == digest


def test_video_selection_cannot_replace_a_later_photo(window, qtbot, video_fixtures, photo_path):
    window.import_paths([video_fixtures / "static.mp4", photo_path])
    window.library.setCurrentRow(1)
    _ready(qtbot, window, video=False)
    assert window.preview.photo.path == photo_path
    assert window.video_info is None
    assert not window.video_panel.isVisible()
    assert not window._video_playing


@pytest.mark.parametrize("partial", [False, True])
def test_gui_passes_tracked_masks_and_reuses_raw_alpha_only_for_color_changes(
    window, qtbot, video_fixtures, monkeypatch, partial
):
    from freedive_color.core import video_pipeline
    from freedive_color.ui import main_window as ui

    calls, render_masks = [], []
    original_preview, original_render = video_pipeline.read_video_preview_frame, ui.render_rgb

    def semantic(rgb, *_args, **_kwargs):
        person = np.ones(rgb.shape[:2], np.float32)
        return MaskSet(
            person,
            person * 0.9,
            person * 0,
            person,
            0.95,
            0.95,
            skin_source="selfie-multiclass-v1",
            skin_status="needs-review" if partial else "automatic",
            skin_diagnostics={
                "automatic_accepted": True,
                "accepted_components": 1,
                "rejected_components": int(partial),
            },
        )

    monkeypatch.setattr("freedive_color.core.temporal.build_masks", semantic)

    def record_preview(*args, **kwargs):
        result = original_preview(*args, **kwargs)
        calls.append((kwargs, result))
        return result

    def record_render(*args, **kwargs):
        render_masks.append(kwargs.get("masks"))
        return original_render(*args, **kwargs)

    monkeypatch.setattr(video_pipeline, "read_video_preview_frame", record_preview)
    monkeypatch.setattr(ui, "render_rgb", record_render)
    window.import_paths([video_fixtures / "vfr_offset.mp4"])
    _ready(qtbot, window)
    assert len(calls) == 1
    assert window.preview.video_masks is calls[-1][1].masks is render_masks[-1]
    assert window._video_plan is calls[-1][1].plan
    assert np.any(window.preview.masks.skin > 0)
    assert (
        "Уверенные участки кожи обрабатываются" in window.local_controls.confidence_label.text()
    ) == partial

    # A soft waterline is applied during rendering, never accumulated in cached alpha.
    edits = MaskEdits(waterline=((0, 0.5), (1, 0.5)))
    window._mask_changed(edits)
    _ready(qtbot, window)
    assert len(calls) == 2
    assert calls[-1][0]["edits"] == edits
    raw = window.preview.video_masks
    expected = window.preview.masks.skin.copy()
    assert np.any((raw.underwater > 0) & (raw.underwater < 1))
    assert np.any(expected < raw.skin)
    for exposure in (0.1, 0.2, -0.1):
        window.controls.set_settings(replace(window.controls.settings(), exposure=exposure))
        _ready(qtbot, window)
        assert len(calls) == 2
        assert render_masks[-1] is raw
        np.testing.assert_array_equal(window.preview.masks.skin, expected)

    window.local_controls.set_settings(replace(window.local_controls.settings(), feather=0.012))
    _ready(qtbot, window)
    assert len(calls) == 3
    assert calls[-1][0]["feather"] == pytest.approx(0.012)
    window._mask_changed(replace(edits, strokes=(BrushStroke("skin", "add", ((0.5, 0.8),), 0.08),)))
    _ready(qtbot, window)
    assert len(calls) == 4
    assert window.preview.masks.skin_status == "manual"
    assert render_masks[-1] is window.preview.video_masks
    window.seek_video(0.65)
    _ready(qtbot, window)
    assert len(calls) == 5
    assert calls[-1][0]["plan"] is window._video_plan
    assert window.preview.timestamp >= 0.65
    assert all(mask is not None for mask in render_masks)


def test_video_one_to_one_loads_original_frame_resolution(window, qtbot, video_fixtures):
    window.import_paths([video_fixtures / "large.mp4"])
    _ready(qtbot, window)
    assert window.preview.photo.rgb.shape[1] < 1800
    QTest.mouseClick(window.one_button, Qt.MouseButton.LeftButton)
    _ready(qtbot, window)
    assert window.preview.photo.rgb.shape == (1000, 1800, 3)
    assert window.viewer.before.width() == 1800
    assert window.viewer.fitting is False


def test_cancelled_video_export_restores_actions_and_leaves_no_output(
    window, qtbot, video_fixtures, tmp_path, monkeypatch
):
    import threading

    from freedive_color.core import video_pipeline

    source = video_fixtures / "static.mp4"
    window.import_paths([source])
    _ready(qtbot, window)
    digest = sha256_file(source)
    original_export = video_pipeline.export_video

    def slow_export(*args, **kwargs):
        # The genuine backend still runs if cancellation does not arrive. The pause only
        # makes a cancellation during an active worker reproducible on fast machines.
        for _ in range(40):
            kwargs["token"].check()
            threading.Event().wait(0.01)
        return original_export(*args, **kwargs)

    monkeypatch.setattr(video_pipeline, "export_video", slow_export)
    destination = tmp_path / "cancelled.mp4"
    successes = []
    window.video_export_completed.connect(successes.append)
    window.start_video_export(destination)
    qtbot.waitUntil(lambda: window._export_job is not None, timeout=5_000)
    window.cancel_operation()
    qtbot.waitUntil(lambda: window._export_job is None, timeout=10_000)
    assert not successes
    assert not destination.exists()
    assert not Path(f"{destination}.freedive.json").exists()
    assert window.export_button.isEnabled()
    assert sha256_file(source) == digest
