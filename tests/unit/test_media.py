import cv2
import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms, ImageOps, PngImagePlugin

from freedive_color.core.errors import StudioError
from freedive_color.core.media_io import SRGB_PROFILE, encode_photo, load_photo


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_display_orientation_once(tmp_path, orientation):
    image = Image.fromarray(np.arange(18 * 24 * 3, dtype=np.uint8).reshape(18, 24, 3))
    exif = Image.Exif()
    exif[274] = orientation
    path = tmp_path / "orientation.png"
    image.save(path, exif=exif)
    with Image.open(path) as stored:
        expected = np.asarray(ImageOps.exif_transpose(stored))
    photo = load_photo(path)
    np.testing.assert_allclose(photo.rgb * 255, expected, atol=2e-5)
    assert (photo.width, photo.height) == (expected.shape[1], expected.shape[0])


@pytest.mark.parametrize("extension,fmt", [("jpg", "JPEG"), ("png", "PNG"), ("tiff", "TIFF")])
def test_eight_bit_export_has_srgb_and_orientation(tmp_path, rgb, extension, fmt):
    path = tmp_path / ("out." + extension)
    exif = Image.Exif()
    exif[274], exif[271] = 1, "Camera"
    encode_photo(path, rgb, None, exif.tobytes(), fmt)
    with Image.open(path) as image:
        assert image.size == (128, 96)
        assert image.info["icc_profile"]
        assert image.getexif().get(274, 1) == 1
    assert load_photo(path).rgb.shape == rgb.shape


@pytest.mark.parametrize("fmt,extension", [("PNG", "png"), ("TIFF", "tiff")])
def test_sixteen_bit_precision_roundtrip(tmp_path, fmt, extension):
    ramp = np.linspace(0.01, 0.99, 2048, dtype=np.float32)
    rgb = np.repeat(np.repeat(ramp[None, :, None], 3, axis=2), 8, axis=0)
    path = tmp_path / ("high_precision." + extension)
    encode_photo(path, rgb, None, Image.Exif().tobytes(), fmt, bit_depth=16)
    image = load_photo(path)
    assert image.bit_depth == 16
    np.testing.assert_allclose(image.rgb, rgb, atol=1 / 65535)
    assert len(np.unique(image.rgb[..., 0])) == 2048


def test_untagged_uint16_tiff_import(tmp_path):
    pixels = np.arange(200 * 100 * 3, dtype=np.uint16).reshape(100, 200, 3)
    path = tmp_path / "original.tif"
    tifffile.imwrite(path, pixels, photometric="rgb", metadata=None)
    image = load_photo(path)
    np.testing.assert_allclose(image.rgb * 65535, pixels, atol=0.004)


def test_untagged_uint16_png_import(tmp_path):
    pixels = np.arange(200 * 100 * 3, dtype=np.uint16).reshape(100, 200, 3)
    path = tmp_path / "original.png"
    ok, encoded = cv2.imencode(".png", pixels[..., ::-1])
    assert ok
    path.write_bytes(encoded.tobytes())
    np.testing.assert_allclose(load_photo(path).rgb * 65535, pixels, atol=0.004)


def test_alpha_roundtrip_and_jpeg_rejection(tmp_path, rgb):
    alpha = np.linspace(0, 1, rgb.shape[1], dtype=np.float32)[None, :].repeat(rgb.shape[0], 0)
    path = tmp_path / "alpha.png"
    encode_photo(path, rgb, alpha, b"", "PNG", bit_depth=16)
    np.testing.assert_allclose(load_photo(path).alpha, alpha, atol=1 / 65535)
    with pytest.raises(StudioError, match="прозрачности"):
        encode_photo(tmp_path / "bad.jpg", rgb, alpha, b"", "JPEG")


def test_invalid_profile_rejected(tmp_path):
    path = tmp_path / "bad_icc.png"
    Image.new("RGB", (16, 16)).save(path, icc_profile=b"broken ICC")
    with pytest.raises(StudioError, match="ICC"):
        load_photo(path)


def test_valid_icc_is_transformed(tmp_path):
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB"))
    path = tmp_path / "lab.tiff"
    Image.new("LAB", (16, 16), (130, 130, 130)).save(path, icc_profile=profile.tobytes())
    photo = load_photo(path)
    assert photo.profile_status.startswith("ICC → sRGB")
    assert photo.rgb.shape == (16, 16, 3)


def test_float_tiff_rejected(tmp_path):
    path = tmp_path / "hdr.tiff"
    tifffile.imwrite(path, np.ones((10, 10, 3), dtype=np.float32), photometric="rgb")
    with pytest.raises(StudioError):
        load_photo(path)


def test_pq_png_rejected(tmp_path):
    metadata = PngImagePlugin.PngInfo()
    metadata.add(b"cICP", bytes([9, 16, 0, 1]))
    path = tmp_path / "hdr.png"
    Image.new("RGB", (16, 16)).save(path, pnginfo=metadata)
    with pytest.raises(StudioError, match="HDR"):
        load_photo(path)


def test_gain_map_rejected(tmp_path):
    path = tmp_path / "gainmap.jpg"
    Image.new("RGB", (16, 16)).save(path, xmp=b'<rdf hdrgm:Version="1.0" />')
    with pytest.raises(StudioError, match="HDR"):
        load_photo(path)


def test_corrupt_and_unsupported(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"This is not a photo")
    with pytest.raises(StudioError, match="повреждён"):
        load_photo(path)
    with pytest.raises(StudioError, match="формат"):
        load_photo(tmp_path / "dive.mov")


def test_export_gps_opt_in(tmp_path):
    exif = Image.Exif()
    exif[34853] = {1: "N", 2: (1, 2, 3), 3: "E", 4: (4, 5, 6)}
    path = tmp_path / "gps.jpg"
    Image.new("RGB", (10, 10)).save(path, exif=exif, icc_profile=SRGB_PROFILE)
    no_gps = Image.Exif()
    no_gps.load(load_photo(path).exif_bytes)
    with_gps = Image.Exif()
    with_gps.load(load_photo(path, keep_gps=True).exif_bytes)
    assert 34853 not in no_gps
    assert 34853 in with_gps


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("bits", (8, 16))
def test_tiff_orientation_matches_source_pixels(tmp_path, orientation, bits):
    pixels = np.arange(12 * 18 * 3, dtype=np.uint16).reshape(12, 18, 3)
    pixels = pixels.astype(np.uint8) if bits == 8 else pixels * 60
    path = tmp_path / "oriented.tiff"
    tifffile.imwrite(
        path,
        pixels,
        photometric="rgb",
        metadata=None,
        extratags=[(274, "H", 1, orientation, False)],
    )
    methods = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    expected = pixels.copy()
    if orientation in methods:
        expected = np.stack(
            [
                np.asarray(Image.fromarray(pixels[..., channel]).transpose(methods[orientation]))
                for channel in range(3)
            ],
            axis=2,
        )
    photo = load_photo(path)
    np.testing.assert_allclose(photo.rgb * (255 if bits == 8 else 65535), expected, atol=0.004)
    assert (photo.width, photo.height) == (expected.shape[1], expected.shape[0])


def test_rgb_png_transparency_key_is_preserved(tmp_path):
    path = tmp_path / "color_key.png"
    Image.new("RGB", (10, 10), (20, 30, 40)).save(path, transparency=(20, 30, 40))
    photo = load_photo(path)
    assert photo.alpha is not None and photo.alpha.max() == 0


def test_non_srgb_gamma_is_not_graded_silently(tmp_path):
    metadata = PngImagePlugin.PngInfo()
    import struct

    metadata.add(b"gAMA", struct.pack(">I", 100000))
    path = tmp_path / "linear.png"
    Image.new("RGB", (10, 10)).save(path, pnginfo=metadata)
    with pytest.raises(StudioError, match="gamma"):
        load_photo(path)
