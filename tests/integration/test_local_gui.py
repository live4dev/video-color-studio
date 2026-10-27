"""Milestone B acceptance: edits refer to source geometry, not a transient viewport."""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest

from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.masks import BrushStroke, rasterize_stroke
from freedive_color.core.media_io import PHOTO_EXTENSIONS, load_photo
from freedive_color.core.project import sha256_file
from freedive_color.core.session import ExportOptions, Session, SessionItem, save_session
from freedive_color.ui.viewer import CompareViewer, to_qimage


@pytest.fixture
def viewer(qtbot):
    widget = CompareViewer()
    qtbot.addWidget(widget)
    widget.resize(900, 620)
    pixels = np.full((400, 800, 3), [0.08, 0.35, 0.6], np.float32)
    widget.set_images(to_qimage(pixels), to_qimage(pixels), source_width=1600)
    widget.show()
    widget.fit()
    return widget


def _viewport_point(viewer, normalized: tuple[float, float]) -> QPoint:
    rect = viewer.image_rect()
    return QPoint(
        round(rect.left() + rect.width() * normalized[0]),
        round(rect.top() + rect.height() * normalized[1]),
    )


def _drag(viewer, start=(0.35, 0.4), end=(0.6, 0.55)) -> None:
    QTest.mousePress(viewer, Qt.MouseButton.LeftButton, pos=_viewport_point(viewer, start))
    QTest.mouseMove(
        viewer, _viewport_point(viewer, ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2))
    )
    QTest.mouseMove(viewer, _viewport_point(viewer, end))
    QTest.mouseRelease(viewer, Qt.MouseButton.LeftButton, pos=_viewport_point(viewer, end))


def test_normalized_brush_survives_fit_pan_zoom_and_full_resolution(viewer):
    strokes = []
    viewer.stroke_created.connect(strokes.append)
    viewer.set_tool("skin_add")
    viewer.brush_radius = 0.027
    original_split = viewer.split
    for mode in ("fit", "pan_zoom", "full_buffer", "one_to_one"):
        if mode == "fit":
            viewer.fit()
        elif mode == "pan_zoom":
            viewer.fitting = False
            viewer.scale = 1.3
            viewer.offset = QPointF(22, -14)
        elif mode == "full_buffer":
            pixels = np.full((800, 1600, 3), [0.08, 0.35, 0.6], np.float32)
            viewer.set_images(to_qimage(pixels), to_qimage(pixels), source_width=1600)
        else:
            viewer.one_to_one()
        point = (0.35, 0.4)
        position = QPointF(
            viewer.image_rect().left() + viewer.image_rect().width() * point[0],
            viewer.image_rect().top() + viewer.image_rect().height() * point[1],
        )
        np.testing.assert_allclose(viewer.normalized_point(position), point, atol=1e-9)
        _drag(viewer)
        stroke = strokes[-1]
        assert stroke["target"] == "skin" and stroke["operation"] == "add"
        assert stroke["radius"] == 0.027
        tolerance = 2 / min(viewer.image_rect().width(), viewer.image_rect().height())
        np.testing.assert_allclose(stroke["points"][0], (0.35, 0.4), atol=tolerance)
        np.testing.assert_allclose(stroke["points"][-1], (0.6, 0.55), atol=tolerance)
    assert len(strokes) == 4
    assert viewer.split == original_split, "Painting across the divider must not move comparison"


def test_letterbox_click_does_not_create_a_mask_stroke(viewer):
    strokes = []
    viewer.stroke_created.connect(strokes.append)
    viewer.set_tool("water_add")
    letterbox = QPoint(20, 5)
    assert not viewer.image_rect().contains(QPointF(letterbox))
    assert viewer.normalized_point(QPointF(letterbox)) is None
    QTest.mouseClick(viewer, Qt.MouseButton.LeftButton, pos=letterbox)
    assert strokes == []


def test_waterline_and_person_box_use_normalized_source_coordinates(viewer):
    lines, boxes = [], []
    viewer.waterline_changed.connect(lines.append)
    viewer.person_box_changed.connect(boxes.append)
    viewer.set_tool("waterline")
    _drag(viewer, (0.15, 0.3), (0.85, 0.34))
    assert len(lines) == 1
    np.testing.assert_allclose(lines[0][0], (0.15, 0.3), atol=0.005)
    np.testing.assert_allclose(lines[0][-1], (0.85, 0.34), atol=0.005)
    viewer.set_tool("person_box")
    _drag(viewer, (0.75, 0.8), (0.25, 0.2))
    assert len(boxes) == 1
    np.testing.assert_allclose(boxes[0], (0.25, 0.2, 0.75, 0.8), atol=0.005)


def test_mask_overlay_never_changes_before_after_pixels(viewer):
    before_key, after_key = viewer.before.cacheKey(), viewer.after.cacheKey()
    source_pixel = viewer.before.pixelColor(250, 150).getRgb()
    mask = np.zeros((400, 800), np.float32)
    mask[120:220, 200:400] = 1
    viewer.set_mask_overlay(mask, "skin")
    assert not viewer.overlay.isNull()
    viewer.grab()
    assert viewer.before.cacheKey() == before_key and viewer.after.cacheKey() == after_key
    assert viewer.before.pixelColor(250, 150).getRgb() == source_pixel
    viewer.set_mask_overlay(None)
    assert viewer.overlay.isNull()


@pytest.mark.parametrize("orientation", (6, 8))
def test_paint_on_exif_rotated_photo_matches_display_landmark(qtbot, tmp_path, orientation):
    pixels = np.full((80, 120, 3), [20, 80, 150], np.uint8)
    pixels[10:30, 20:40] = [180, 110, 80]
    exif = Image.Exif()
    exif[274] = orientation
    source = tmp_path / f"orientation_{orientation}.png"
    Image.fromarray(pixels).save(source, exif=exif)
    photo = load_photo(source)
    assert (photo.width, photo.height) == (80, 120)
    landmark = photo.rgb[..., 0] > 0.5
    ys, xs = np.nonzero(landmark)
    center = (
        (xs.min() + xs.max()) / 2 / (photo.width - 1),
        (ys.min() + ys.max()) / 2 / (photo.height - 1),
    )
    widget = CompareViewer()
    qtbot.addWidget(widget)
    widget.resize(750, 650)
    widget.set_images(to_qimage(photo.rgb), to_qimage(photo.rgb), photo.width)
    widget.show()
    widget.fit()
    widget.set_tool("skin_add")
    widget.brush_radius = 0.025
    strokes = []
    widget.stroke_created.connect(strokes.append)
    QTest.mouseClick(widget, Qt.MouseButton.LeftButton, pos=_viewport_point(widget, center))
    assert len(strokes) == 1
    raster = rasterize_stroke(BrushStroke.from_dict(strokes[0]), photo.rgb.shape[:2])
    assert raster[round(ys.mean()), round(xs.mean())] > 0.99
    assert float(raster[~landmark].sum()) < float(raster[landmark].sum())


def _wait_ready(qtbot, window):
    qtbot.waitUntil(
        lambda: (
            window.preview is not None and not window._render_pending and window._view_job is None
        ),
        timeout=45_000,
    )


@pytest.fixture
def local_window(qapp, qtbot):
    from freedive_color.ui.main_window import MainWindow
    from freedive_color.ui.theme import apply_theme

    apply_theme(qapp)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    yield window
    window.close()


def test_cancel_debounce_restores_local_controls_masks_and_item_state(
    local_window, qtbot, photo_path
):
    window = local_window
    window.import_paths([photo_path])
    _wait_ready(qtbot, window)
    local = window.local_controls.settings()
    edits = window.mask_edits
    rendered = window.preview.result.copy()
    window.local_controls.set_settings(replace(local, skin_warmth=0.41, water_hue=0.23))
    window.viewer.stroke_created.emit(
        BrushStroke(points=((0.35, 0.4), (0.55, 0.6)), radius=0.03).to_dict()
    )
    assert window._debounce.isActive() and window.mask_edits != edits
    window.cancel_operation()
    qtbot.wait(180)
    assert not window._debounce.isActive() and not window._render_pending
    assert window.local_controls.settings() == local
    assert window.mask_edits == edits
    item = window.library.currentItem()
    assert item.data(Qt.ItemDataRole.UserRole + 2) == local.to_dict()
    assert item.data(Qt.ItemDataRole.UserRole + 3) == edits.to_dict()
    np.testing.assert_array_equal(window.preview.result, rendered)


@pytest.mark.parametrize("suffix,canonical", ((".jpeg", ".jpg"), (".tif", ".tiff")))
def test_project_export_suffix_alias_has_valid_ui_format(
    local_window, qtbot, photo_path, tmp_path, suffix, canonical
):
    project = tmp_path / "export-alias.fcsproj"
    save_session(
        project,
        Session(
            items=(SessionItem(photo_path),),
            selected_index=0,
            export_options=ExportOptions(suffix=suffix),
        ),
    )
    local_window.load_project(project)
    _wait_ready(qtbot, local_window)
    assert local_window.format_combo.currentIndex() >= 0
    assert local_window.format_combo.currentData() == canonical
    assert local_window._export_options().suffix == canonical


def test_changed_project_source_requires_explicit_reimport(
    local_window, qtbot, photo_path, tmp_path
):
    window = local_window
    window.import_paths([photo_path])
    _wait_ready(qtbot, window)
    original_hash = sha256_file(photo_path)
    window.viewer.stroke_created.emit(BrushStroke(points=((0.4, 0.5),), radius=0.03).to_dict())
    _wait_ready(qtbot, window)
    assert window.mask_edits.strokes
    project = tmp_path / "source-fingerprint.fcsproj"
    window.save_project(project)
    Image.new("RGB", (128, 96), (30, 110, 160)).save(photo_path)
    new_hash = sha256_file(photo_path)
    assert new_hash != original_hash
    with qtbot.waitSignal(window.operation_failed, timeout=45_000) as failed:
        window.load_project(project)
    assert "измен" in failed.args[0].lower()
    assert window.preview is None, "A changed source must not silently reuse cached edits"
    window.reimport_selected()
    _wait_ready(qtbot, window)
    assert window.preview.source_sha256 == new_hash
    assert window.mask_edits.strokes == () and window.mask_edits.waterline == ()
    assert window.mask_edits.person_box is None
    np.testing.assert_array_equal(window.preview.photo.rgb, load_photo(photo_path).rgb)
    assert sha256_file(photo_path) == new_hash


def test_manual_masks_undo_reset_project_and_batch_roundtrip(qapp, qtbot, photo_path, tmp_path):
    from freedive_color.ui.main_window import MainWindow
    from freedive_color.ui.theme import apply_theme

    apply_theme(qapp)
    window = MainWindow()
    qtbot.addWidget(window)
    window.show()
    other = tmp_path / "second.png"
    Image.new("RGB", (160, 120), (30, 100, 150)).save(other)
    source_hash = sha256_file(photo_path)
    try:
        window.import_paths([photo_path, other])
        _wait_ready(qtbot, window)
        assert window.preview.masks is not None
        tool = window.local_controls.tool_combo
        tool.setCurrentIndex(tool.findData("skin_add"))
        _drag(window.viewer)
        _wait_ready(qtbot, window)
        assert len(window.mask_edits.strokes) == 1
        window.undo_mask_edit()
        _wait_ready(qtbot, window)
        assert window.mask_edits.strokes == ()
        _drag(window.viewer)
        _wait_ready(qtbot, window)
        window.reset_mask_edits()
        _wait_ready(qtbot, window)
        assert window.mask_edits.strokes == () and window.mask_edits.waterline == ()

        _drag(window.viewer)
        _wait_ready(qtbot, window)
        tool.setCurrentIndex(tool.findData("waterline"))
        _drag(window.viewer, (0.1, 0.3), (0.9, 0.34))
        _wait_ready(qtbot, window)
        assert len(window.mask_edits.waterline) >= 2
        local = replace(LocalSettings(), water_hue=0.17, water_saturation=0.11)
        window.local_controls.set_settings(local)
        _wait_ready(qtbot, window)
        saved_edits = window.mask_edits.to_dict()
        window.library.item(1).setCheckState(Qt.CheckState.Unchecked)
        project = tmp_path / "manual-mask-project.fcsproj"
        window.save_project(project)
        assert project.is_file()

        window.library.setCurrentRow(1)
        _wait_ready(qtbot, window)
        assert window.mask_edits.strokes == ()
        window.load_project(project)
        _wait_ready(qtbot, window)
        assert window.library.count() == 2
        assert window.mask_edits.to_dict() == saved_edits
        assert window.local_controls.settings() == local
        assert window.library.item(1).checkState() == Qt.CheckState.Unchecked
        results = []
        window.batch_completed.connect(results.append)
        output = tmp_path / "series-export"
        with qtbot.waitSignal(window.batch_completed, timeout=45_000):
            window.start_batch_export(output)
        result = results[-1]
        assert result.completed == 1 and result.failed == 0 and result.excluded == 1
        photos = [path for path in output.iterdir() if path.suffix.lower() in PHOTO_EXTENSIONS]
        assert len(photos) == 1
        assert load_photo(photos[0]).rgb.shape == load_photo(photo_path).rgb.shape
        assert sha256_file(photo_path) == source_hash
    finally:
        window.close()
