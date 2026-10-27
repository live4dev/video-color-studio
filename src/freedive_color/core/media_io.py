"""SDR image decoding, explicit color management and precision-preserving encoding."""

import io
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import tifffile
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from .errors import StudioError

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
MAX_PIXELS = 40_000_000
SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
# Preserve useful photographic metadata, not stale dimensions, thumbnails or opaque maker notes.
EXIF_TAGS = {271, 272, 274, 306, 315, 33432, 33434, 33437, 34855, 36867, 36868, 37386}


@dataclass
class Photo:
    path: Path
    rgb: np.ndarray
    alpha: np.ndarray | None
    width: int
    height: int
    bit_depth: int
    source_format: str
    profile_status: str
    warnings: list[str] = field(default_factory=list)
    exif_bytes: bytes = b""
    stat_signature: tuple[int, int] = (0, 0)

    def metadata(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "bit_depth": self.bit_depth,
            "format": self.source_format,
            "color": self.profile_status,
            "decoder": "Pillow/LittleCMS + OpenCV/tifffile (16-bit)",
            "orientation_applied": True,
        }


def signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _orient(array: np.ndarray, orientation: int) -> np.ndarray:
    # The same lossless display orientation used by ImageOps.exif_transpose.
    operations = {
        2: lambda x: np.flip(x, 1),
        3: lambda x: np.rot90(x, 2),
        4: lambda x: np.flip(x, 0),
        5: lambda x: np.swapaxes(x, 0, 1),
        6: lambda x: np.rot90(x, -1),
        7: lambda x: np.flip(np.swapaxes(x, 0, 1), (0, 1)),
        8: lambda x: np.rot90(x, 1),
    }
    return np.ascontiguousarray(operations.get(orientation, lambda x: x)(array))


def _safe_exif(source: Image.Exif, keep_gps: bool) -> bytes:
    clean = Image.Exif()
    for tag in EXIF_TAGS - {274}:
        if tag in source:
            clean[tag] = source[tag]
    # Some camera EXIF lives in the ExifIFD, not the top-level TIFF IFD.
    if 34665 in source:
        for tag, value in source.get_ifd(34665).items():
            if tag in EXIF_TAGS:
                clean[tag] = value
    if keep_gps and 34853 in source:
        clean[34853] = source.get_ifd(34853)
    clean[274] = 1
    clean[305] = "Freedive Color Studio 0.1"
    return clean.tobytes()


def _png_depth_and_hdr(path: Path) -> tuple[int, bool]:
    depth, hdr = 8, False
    with path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            return depth, hdr
        while True:
            header = handle.read(8)
            if len(header) != 8:
                break
            length, kind = struct.unpack(">I4s", header)
            if kind == b"IHDR":
                data = handle.read(length)
                depth = data[8]
            elif kind == b"cICP":
                data = handle.read(length)
                hdr = len(data) >= 2 and data[1] in (16, 18)
            elif kind in (b"IDAT", b"IEND"):
                break
            else:
                handle.seek(length, os.SEEK_CUR)
            handle.seek(4, os.SEEK_CUR)
    return depth, hdr


def load_photo(path: str | Path, max_edge: int | None = None, keep_gps: bool = False) -> Photo:
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in PHOTO_EXTENSIONS:
        raise StudioError("Этот формат пока не поддерживается. Откройте JPEG, PNG или TIFF SDR.")
    if not path.is_file():
        raise StudioError(f"Файл не найден: {path.name}")
    if max_edge is not None and max_edge < 1:
        raise StudioError("Размер предпросмотра должен быть положительным.")
    try:
        before = signature(path)
        with Image.open(path) as image:
            width, height = image.size
            if width * height > MAX_PIXELS:
                raise StudioError("Фото больше 40 Мп. Уменьшите копию перед импортом в этот MVP.")
            if image.format not in ("JPEG", "PNG", "TIFF") or getattr(image, "n_frames", 1) != 1:
                raise StudioError("Многокадровые, animated и HDR gain-map фото не поддерживаются.")
            exif = image.getexif()
            orientation = exif.get(274, 1)
            xmp = str(image.info.get("xmp", b"")) + str(image.info.get("XML:com.adobe.xmp", ""))
            if any(token in xmp.lower() for token in ("hdrgm:", "hdrgainmap", "hdr-gain-map")):
                raise StudioError("HDR gain-map не поддерживается. Сначала экспортируйте SDR sRGB.")
            depth, hdr = _png_depth_and_hdr(path) if image.format == "PNG" else (8, False)
            if image.format == "TIFF":
                bits = image.tag_v2.get(258, (8,))
                depth = max(bits) if isinstance(bits, tuple) else bits
                if image.tag_v2.get(339, (1,)) not in ((1,), (1, 1, 1), (1, 1, 1, 1), 1):
                    raise StudioError("Float/HDR TIFF не поддерживается. Нужен целочисленный SDR.")
                if image.tag_v2.get(262) not in (0, 1, 2, 3, 5, 8):
                    raise StudioError("Это цветовое пространство TIFF пока не поддерживается.")
            if hdr or depth not in (1, 2, 4, 8, 16):
                raise StudioError("PQ/HLG/HDR пока не поддерживается. Нужен SDR sRGB.")
            profile_bytes = image.info.get("icc_profile")
            profile = None
            warnings = []
            if profile_bytes:
                try:
                    profile = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
                    description = ImageCms.getProfileDescription(profile).strip()
                    if any(token in description.lower() for token in ("pq", "hlg", "hdr", "log")):
                        raise StudioError("HDR/Log ICC не поддерживается. Нужен SDR sRGB.")
                except (OSError, ValueError, TypeError) as exc:
                    raise StudioError(
                        "Повреждённый ICC-профиль. Пересохраните копию в sRGB."
                    ) from exc
                profile_status = f"ICC → sRGB ({description})"
            else:
                profile_status = "sRGB (профиль не указан)"
                warnings.append("ICC-профиль отсутствует: используется sRGB.")
            safe_exif = _safe_exif(exif, keep_gps)
            source_format = image.format
            if depth == 16:
                if profile is not None and (
                    profile_bytes[128:] != SRGB_PROFILE[128:]
                    or profile_bytes[16:24] != SRGB_PROFILE[16:24]
                ):
                    raise StudioError(
                        "16-битный файл с ICC пока не преобразуется без потери точности. "
                        "Используйте 8-битную копию с ICC или 16-битный sRGB без ICC. "
                        "Собственный стандартный sRGB-профиль приложения поддерживается."
                    )
                if image.format == "TIFF":
                    photometric = image.tag_v2.get(262)
                    if photometric not in (0, 1, 2):
                        raise StudioError("TIFF 16 бит: поддерживается только RGB или grayscale.")
                    # tifffile preserves uint16 and leaves orientation for our explicit transform.
                    pixels = tifffile.imread(path)
                    if pixels.dtype != np.uint16:
                        raise StudioError("Этот вариант 16-битного TIFF не поддерживается.")
                    if photometric == 0:
                        pixels = 65535 - pixels
                    if image.tag_v2.get(284, 1) == 2:
                        pixels = np.moveaxis(pixels, 0, -1)
                else:
                    pixels = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_UNCHANGED)
                    if pixels is None or pixels.dtype != np.uint16:
                        raise StudioError("Декодер не смог прочитать 16-битные пиксели.")
                    if pixels.ndim == 3:
                        pixels = pixels[..., [2, 1, 0, 3] if pixels.shape[2] == 4 else [2, 1, 0]]
                if pixels.ndim == 2:
                    pixels = np.repeat(pixels[..., None], 3, axis=2)
                if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
                    raise StudioError("Поддерживается только RGB/RGBA/grayscale TIFF или PNG.")
                pixels = _orient(pixels, orientation)
            else:
                if image.mode == "CMYK" and profile is None:
                    raise StudioError(
                        "CMYK без ICC нельзя достоверно преобразовать. Нужен RGB sRGB."
                    )
                image = ImageOps.exif_transpose(image)
                alpha_image = image.getchannel("A") if "A" in image.getbands() else None
                if image.mode == "P" and "transparency" in image.info:
                    alpha_image = image.convert("RGBA").getchannel("A")
                if profile is not None:
                    base = (
                        image if image.mode in ("RGB", "CMYK", "L", "LAB") else image.convert("RGB")
                    )
                    try:
                        converted = ImageCms.profileToProfile(
                            base,
                            profile,
                            ImageCms.ImageCmsProfile(io.BytesIO(SRGB_PROFILE)),
                            outputMode="RGB",
                            renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
                        )
                    except (ImageCms.PyCMSError, OSError, ValueError) as exc:
                        raise StudioError(
                            "ICC не соответствует пикселям. Пересохраните копию в sRGB."
                        ) from exc
                else:
                    converted = image.convert("RGB")
                if max_edge:
                    converted.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                    if alpha_image is not None:
                        alpha_image = alpha_image.resize(converted.size, Image.Resampling.LANCZOS)
                pixels = np.asarray(converted)
                if alpha_image is not None:
                    pixels = np.dstack((pixels, np.asarray(alpha_image)))
            full_width, full_height = (
                (height, width) if orientation in (5, 6, 7, 8) else (width, height)
            )
            if max_edge and depth == 16 and max(pixels.shape[:2]) > max_edge:
                scale = max_edge / max(pixels.shape[:2])
                pixels = cv2.resize(
                    pixels,
                    (round(pixels.shape[1] * scale), round(pixels.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            divisor = 65535 if depth == 16 else 255
            rgb = np.ascontiguousarray(pixels[..., :3], dtype=np.float32) / np.float32(divisor)
            alpha = pixels[..., 3].astype(np.float32) / divisor if pixels.shape[2] == 4 else None
        if before != signature(path):
            raise StudioError("Файл изменился во время чтения. Импортируйте его ещё раз.")
        return Photo(
            path,
            rgb,
            alpha,
            full_width,
            full_height,
            16 if depth == 16 else 8,
            source_format,
            profile_status,
            warnings,
            safe_exif,
            before,
        )
    except StudioError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        cv2.error,
        tifffile.TiffFileError,
        Image.DecompressionBombError,
    ) as exc:
        raise StudioError(
            f"Не удалось прочитать «{path.name}»: файл повреждён или не поддерживается."
        ) from exc


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def encode_photo(
    path: Path,
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    exif_bytes: bytes,
    output_format: str,
    bit_depth: int = 8,
    quality: int = 95,
) -> None:
    """Encode into a private staging file. Publication is handled by image_pipeline."""
    if output_format not in ("JPEG", "PNG", "TIFF") or bit_depth not in (8, 16):
        raise StudioError("Неподдерживаемый формат или разрядность экспорта.")
    if not 1 <= quality <= 100:
        raise StudioError("Качество JPEG должно быть от 1 до 100.")
    if output_format == "JPEG" and (bit_depth != 8 or alpha is not None):
        raise StudioError("JPEG поддерживает 8 бит без прозрачности. Выберите PNG или TIFF.")
    maximum = 65535 if bit_depth == 16 else 255
    pixels = np.rint(np.clip(rgb, 0, 1) * maximum).astype(
        np.uint16 if bit_depth == 16 else np.uint8
    )
    if alpha is not None:
        pixels = np.dstack((pixels, np.rint(np.clip(alpha, 0, 1) * maximum).astype(pixels.dtype)))
    if bit_depth == 16 and output_format == "TIFF":
        # TIFF ICC tag is standard; EXIF subset is supplied through baseline TIFF tags.
        exif = Image.Exif()
        exif.load(exif_bytes)
        extras = [(34675, "B", len(SRGB_PROFILE), SRGB_PROFILE, False)]
        for tag in (271, 272, 305, 306, 315, 33432):
            if tag in exif:
                extras.append((tag, "s", 0, str(exif[tag]), False))
        tifffile.imwrite(
            path,
            pixels,
            photometric="rgb",
            metadata=None,
            extratags=extras,
            extrasamples="unassalpha" if alpha is not None else None,
        )
    elif bit_depth == 16:
        bgr = pixels[..., [2, 1, 0, 3] if pixels.shape[2] == 4 else [2, 1, 0]]
        success, encoded = cv2.imencode(".png", bgr)
        if not success:
            raise StudioError("Не удалось закодировать PNG.")
        raw = encoded.tobytes()
        # Insert ICC and normalized EXIF after IHDR without re-quantizing 16-bit RGB.
        extra = _png_chunk(b"iCCP", b"sRGB\0\0" + zlib.compress(SRGB_PROFILE))
        if exif_bytes:
            extra += _png_chunk(b"eXIf", exif_bytes.removeprefix(b"Exif\0\0"))
        path.write_bytes(raw[:33] + extra + raw[33:])
    else:
        options = {"icc_profile": SRGB_PROFILE, "exif": exif_bytes}
        if output_format == "JPEG":
            options.update(quality=quality, subsampling=0, optimize=True)
        if output_format == "TIFF":
            options["compression"] = "tiff_lzw"
        Image.fromarray(pixels).save(path, format=output_format, **options)
