"""Float32 SDR color transforms. This module never resizes or remaps source pixels."""

from collections.abc import Callable

import cv2
import numpy as np

from .analyze import SceneAnalysis, analyze_scene, linear_to_srgb, srgb_to_linear, validate_rgb
from .presets import GradeSettings

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _gamut_limit(rgb: np.ndarray, luminance: np.ndarray) -> np.ndarray:
    """Compress chroma toward the existing luminance instead of clipping hue channels."""
    delta = rgb - luminance[..., None]
    upper = np.max(delta, axis=2)
    lower = np.min(delta, axis=2)
    scale = np.minimum(1, (1 - luminance) / np.maximum(upper, 1e-6))
    scale = np.minimum(scale, luminance / np.maximum(-lower, 1e-6))
    return np.clip(luminance[..., None] + delta * scale[..., None], 0, 1).astype(np.float32)


def grade_image(
    rgb: np.ndarray,
    settings: GradeSettings,
    analysis: SceneAnalysis | None = None,
    check_cancelled: Callable[[], None] = lambda: None,
) -> np.ndarray:
    validate_rgb(rgb)
    check_cancelled()
    analysis = analysis or analyze_scene(rgb)
    linear = srgb_to_linear(rgb)
    # WB is multiplicative in linear light. Red recovery alone cannot create missing red.
    red_gain = 1 + (analysis.red_gain - 1) * settings.auto_strength * settings.red_recovery
    green_gain = 1 + (analysis.green_gain - 1) * settings.auto_strength
    gains = np.array(
        [
            red_gain * np.exp(settings.temperature * 0.32 + settings.tint * 0.06),
            green_gain * np.exp(-settings.tint * 0.23),
            np.exp(-settings.temperature * 0.27 + settings.tint * 0.06),
        ],
        np.float32,
    )
    linear *= gains * np.float32(2**settings.exposure)
    # A continuous shoulder is only active above SDR white; zero settings remain identity.
    if settings.exposure > 0:
        shoulder = np.maximum(linear - 0.8, 0)
        linear /= 1 + shoulder * np.float32(settings.exposure * 0.15)
    out = np.clip(linear_to_srgb(linear), 0, 1)
    del linear
    check_cancelled()
    y = out @ LUMA
    toned = y.copy()
    toned += np.float32(settings.shadows * 0.7) * y * (1 - y) ** 3
    toned += np.float32(settings.highlights * 0.3) * np.maximum(y - 0.45, 0) ** 2
    toned += np.float32(settings.contrast * 0.45) * (y - 0.5) * (1 - (2 * y - 1) ** 2)
    if settings.dehaze > 0:
        # Broad luminance contrast, capped at 0.35. No estimated/synthesized backscatter.
        sigma = max(2, min(y.shape) * 0.018)
        local = cv2.GaussianBlur(y, (0, 0), sigma)
        toned += np.float32(settings.dehaze * 0.6) * (y - local)
    toned = np.clip(toned, 0, 1)
    out *= (toned / np.maximum(y, 1e-6))[..., None]
    out = toned[..., None] + (out - toned[..., None]) * np.float32(1 + settings.saturation * 0.65)
    out = _gamut_limit(out, toned)
    check_cancelled()
    if settings.denoise > 0:
        # Small bilateral neighborhood; blend only chroma, keep original luminance texture.
        filtered = cv2.bilateralFilter(out, 5, 0.025 + settings.denoise * 0.025, 2)
        chroma = filtered - (filtered @ LUMA)[..., None]
        out = (
            toned[..., None]
            + (1 - settings.denoise * 0.7) * (out - toned[..., None])
            + settings.denoise * 0.7 * chroma
        )
    check_cancelled()
    if settings.detail > 0:
        current_y = out @ LUMA
        # Resolution-relative radius keeps preview and final rendering visually consistent.
        sigma = max(0.35, min(out.shape[:2]) / 1000 * 0.8)
        highpass = current_y - cv2.GaussianBlur(current_y, (0, 0), sigma)
        highpass = np.sign(highpass) * np.maximum(np.abs(highpass) - 0.003, 0)
        out += (np.float32(settings.detail * 0.35) * highpass)[..., None]
    check_cancelled()
    return np.clip(out, 0, 1).astype(np.float32)
