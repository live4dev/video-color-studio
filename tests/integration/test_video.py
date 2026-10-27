"""Real codec round trips: VFR clocks, AAC payloads, boundaries and protected files."""

import importlib.util
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from freedive_color.core.errors import Cancelled, StudioError
from freedive_color.core.image_pipeline import CancellationToken
from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.presets import get_preset
from freedive_color.core.project import sha256_file
from freedive_color.core.reference import PaletteReference
from freedive_color.core.temporal import ShotPlan
from freedive_color.core.video_pipeline import (
    analyze_video,
    decode_video_frame,
    encode_video_frame,
    export_video,
    probe_video,
    read_video_frame,
)
from freedive_color.integrations.ffmpeg import open_local_media


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    path = Path(__file__).resolve().parents[2] / "tools/generate_video_fixtures.py"
    spec = importlib.util.spec_from_file_location("video_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate(
        tmp_path_factory.mktemp("video-inputs"), duration=0.75, width=320, height=180, rate=12
    )


def frame_pts(path):
    with open_local_media(path) as container:
        return [Fraction(frame.pts) * frame.time_base for frame in container.decode(video=0)]


def audio_packets(path):
    with open_local_media(path) as container:
        return [
            (
                packet.stream.index,
                packet.pts * Fraction(packet.time_base),
                packet.dts * Fraction(packet.time_base),
                bytes(packet),
            )
            for packet in container.demux(*container.streams.audio)
            if packet.size
        ]


def track_end(track):
    return track.start_time + Fraction(str(track.duration)).limit_denominator(
        track.time_base.denominator
    )


def test_probe_relative_seeking_and_rational_nonzero_origin(videos):
    info = probe_video(videos["vfr_offset"])
    assert (info.width, info.height) == (320, 180)
    assert info.video_start == Fraction(1, 4)
    assert info.time_base == Fraction(1, 12000)
    rgb, actual = read_video_frame(videos["vfr_offset"], 0.2, max_edge=160)
    assert rgb.dtype == np.float32 and rgb.shape == (90, 160, 3)
    assert 0.2 <= actual < 0.3 and 0 <= rgb.min() <= rgb.max() <= 1
    json.dumps(info.to_dict(), allow_nan=False)


def test_vfr_full_export_preserves_every_pts_audio_packet_offset_and_tail(videos, tmp_path):
    source = videos["vfr_offset"]
    before = sha256_file(source)
    output = tmp_path / "vfr.mp4"
    report = export_video(source, output, get_preset("natural"))
    source_info, result_info = probe_video(source), probe_video(output)
    assert frame_pts(source) == frame_pts(output)
    assert len(set(np.diff([float(v) for v in frame_pts(output)]).round(6))) > 1
    assert audio_packets(source) == audio_packets(output)
    assert result_info.duration == source_info.duration
    assert track_end(result_info.audio_streams[0]) == track_end(source_info.audio_streams[0])
    assert track_end(result_info.audio_streams[0]) > (
        result_info.video_start + Fraction(str(result_info.duration))
    )
    assert result_info.audio_streams[0].start_time == source_info.audio_streams[0].start_time
    assert (
        result_info.color_space,
        result_info.color_trc,
        result_info.color_primaries,
        result_info.color_range,
    ) == (1, 1, 1, 1)
    assert report["verification"]["pts_exact"]
    assert report["verification"]["audio_verified"]
    assert report["verification"]["audio_tracks"][0]["packets_identical"]
    assert sha256_file(source) == before
    sidecar = output.with_suffix(".mp4.freedive.json")
    assert ShotPlan.from_dict(json.loads(sidecar.read_text())["plan"]).to_dict() == report["plan"]
    assert not list(tmp_path.glob(".freedive-*"))


def test_preview_rebases_both_tracks_to_same_actual_video_origin(videos, tmp_path):
    source = videos["vfr_offset"]
    output = tmp_path / "preview.mp4"
    report = export_video(
        source,
        output,
        get_preset("cinematic"),
        preview_start=0.2,
        preview_seconds=0.35,
        proxy_edge=160,
    )
    shift = Fraction(
        report["preview"]["common_pts_shift"]["numerator"],
        report["preview"]["common_pts_shift"]["denominator"],
    )
    info = probe_video(source)
    selected = [
        t
        for t in frame_pts(source)
        if info.video_start + Fraction(1, 5) <= t < info.video_start + Fraction(11, 20)
    ]
    assert frame_pts(output) == [t - shift for t in selected]
    assert frame_pts(output)[0] == 0
    assert shift == selected[0]
    assert (probe_video(output).width, probe_video(output).height) == (160, 90)
    source_audio = {payload: (pts, dts) for _, pts, dts, payload in audio_packets(source)}
    for _, pts, dts, payload in audio_packets(output):
        assert (pts + shift, dts + shift) == source_audio[payload]
    assert probe_video(output).audio_streams[0].start_time > 0
    assert report["verification"]["audio_verified"]


def test_static_local_reference_grade_has_no_temporal_pumping_and_plan_replays(videos, tmp_path):
    source = videos["static"]
    plan = analyze_video(source)
    assert isinstance(plan, ShotPlan) and plan.frame_count == 9
    assert not plan.cuts
    settings = replace(get_preset("natural"), exposure=0.12)
    kwargs = {
        "local_settings": LocalSettings(),
        "reference": PaletteReference(water_ab=(-8, -18)),
        "plan": plan.to_dict(),
    }
    first, second = tmp_path / "one.mp4", tmp_path / "replay.mp4"
    report = export_video(source, first, settings, **kwargs)
    export_video(source, second, settings, **kwargs)
    assert report["reference_transforms"]
    assert report["temporal_masks"]["semantic_refreshes"] < report["output"]["frame_count"]
    assert sha256_file(first) == sha256_file(second)
    with open_local_media(first) as container:
        frames = [decode_video_frame(f) for f in container.decode(video=0)]
    # H.264 compression has small residual P-frame differences; grading itself is fixed.
    assert np.ptp([float(f.mean()) for f in frames]) < 0.003


def test_cancel_during_encode_removes_partial_output_and_uses_selected_temp_folder(
    videos, tmp_path
):
    token = CancellationToken()
    directory = tmp_path / "temporary"
    output = tmp_path / "cancelled.mp4"
    before = sha256_file(videos["static"])

    def progress(value, message):
        if value >= 20:
            token.cancel()

    with pytest.raises(Cancelled):
        export_video(
            videos["static"],
            output,
            get_preset("natural"),
            token=token,
            progress=progress,
            temp_directory=directory,
        )
    assert not output.exists() and not output.with_suffix(".mp4.freedive.json").exists()
    assert directory.is_dir() and not list(directory.iterdir())
    assert sha256_file(videos["static"]) == before


def test_precancelled_export_never_creates_directory(videos, tmp_path):
    token = CancellationToken()
    token.cancel()
    output = tmp_path / "new" / "cancelled.mp4"
    with pytest.raises(Cancelled):
        export_video(videos["static"], output, get_preset("natural"), token=token)
    assert not output.parent.exists()


def test_outputs_original_and_json_sidecar_are_protected(videos, tmp_path):
    source = videos["static"]
    with pytest.raises(StudioError, match="Оригинал"):
        export_video(source, source, get_preset("natural"))
    output = tmp_path / "protected.mp4"
    output.write_bytes(b"user file")
    with pytest.raises(StudioError, match="существует"):
        export_video(source, output, get_preset("natural"))
    assert output.read_bytes() == b"user file"
    output.unlink()
    output.with_suffix(".mp4.freedive.json").write_text("user recipe")
    with pytest.raises(StudioError, match="существует"):
        export_video(source, output, get_preset("natural"))
    assert not output.exists()


def test_identity_rec709_range_matrix_and_transfer_round_trip():
    for value in (0.03, 0.18, 0.5, 0.9):
        rgb = np.full((32, 48, 3), value, np.float32)
        decoded = decode_video_frame(encode_video_frame(rgb))
        assert np.max(np.abs(decoded - rgb)) < 0.012


def _tagged_video(path, *, trc=1, sar=Fraction(1), with_audio=False):
    with av.open(str(path), "w", format="mov") as container:
        stream = container.add_stream("libx264", rate=12)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.codec_context.color_primaries = stream.codec_context.colorspace = 1
        stream.codec_context.color_trc = trc
        stream.codec_context.color_range = 1
        stream.codec_context.sample_aspect_ratio = sar
        stream.options = {"bf": "0", "tune": "zerolatency"}
        for index in range(3):
            frame = encode_video_frame(np.full((64, 96, 3), 0.2, np.float32))
            frame.pts, frame.time_base = index, Fraction(1, 12)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


@pytest.mark.parametrize("trc", [16, 18])
def test_hdr_pq_hlg_rejected_explicitly_without_partial_output(tmp_path, trc):
    source = tmp_path / "hdr.mov"
    _tagged_video(source, trc=trc)
    assert probe_video(source).unsupported_reason
    with pytest.raises(StudioError, match="PQ/HLG/HDR"):
        export_video(source, tmp_path / "out.mp4", get_preset("natural"))
    assert not (tmp_path / "out.mp4").exists()


def test_anamorphic_footage_rejected_without_geometry_changes(tmp_path):
    source = tmp_path / "anamorphic.mov"
    _tagged_video(source, sar=Fraction(4, 3))
    assert probe_video(source).sample_aspect_ratio == Fraction(4, 3)
    with pytest.raises(StudioError, match="Анаморфное"):
        export_video(source, tmp_path / "out.mp4", get_preset("natural"))


def test_silent_video_remains_silent(tmp_path):
    source, output = tmp_path / "silent.mov", tmp_path / "silent.mp4"
    _tagged_video(source)
    report = export_video(source, output, get_preset("natural"))
    assert probe_video(output).audio_streams == ()
    assert report["verification"]["audio_tracks"] == []


def test_offline_input_wrapper_restricts_demuxers_and_protocols(tmp_path, monkeypatch):
    source = tmp_path / "fake.mp4"
    source.write_bytes(b"local")
    calls = []
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: calls.append((args, kwargs)))
    open_local_media(source)
    assert calls[0][1]["options"] == {
        "protocol_whitelist": "file,pipe",
        "format_whitelist": "mov,mp4,m4a,3gp,3g2,mj2",
    }
    with pytest.raises(StudioError, match="локальный"):
        open_local_media("https://example.invalid/video.mp4")


def test_local_playlist_disguised_as_mp4_is_rejected_before_protocol_access(tmp_path):
    source = tmp_path / "playlist.mp4"
    source.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n"
        "https://example.invalid/segment.ts\n#EXT-X-ENDLIST\n"
    )
    with pytest.raises(StudioError):
        probe_video(source)


def test_plan_origin_validation_and_invalid_preview_options(videos, tmp_path):
    source = videos["static"]
    plan = analyze_video(source).to_dict()
    plan["start_time"] = 5
    with pytest.raises(StudioError):
        export_video(source, tmp_path / "out.mp4", get_preset("natural"), plan=plan)
    for kwargs in (
        {"preview_start": -1},
        {"preview_seconds": 0},
        {"proxy_edge": 63},
        {"preview_start": float("nan")},
        {"preview_start": 1000},
    ):
        with pytest.raises(StudioError):
            export_video(source, tmp_path / "out.mp4", get_preset("natural"), **kwargs)


def _matrix_copy(source, destination, angle):
    """Set a generated test MP4's standard tkhd rotation without recoding its pixels."""
    import struct

    data = bytearray(source.read_bytes())
    at = data.find(b"tkhd")
    assert at >= 4 and data[at + 4] in (0, 1)
    offset = at + 44 + (12 if data[at + 4] == 1 else 0)
    matrices = {
        90: (0, -65536, 0, 65536, 0, 0, 0, 0, 1 << 30),
        -90: (0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30),
        180: (-65536, 0, 0, 0, -65536, 0, 0, 0, 1 << 30),
        "reflection": (-65536, 0, 0, 0, 65536, 0, 0, 0, 1 << 30),
    }
    data[offset : offset + 36] = struct.pack(">9i", *matrices[angle])
    destination.write_bytes(data)


@pytest.mark.parametrize("angle", [90, -90, 180])
def test_portrait_display_rotation_is_lossless_pixel_index_normalization(videos, tmp_path, angle):
    source, output = tmp_path / "rotated.mp4", tmp_path / "normalized.mp4"
    _matrix_copy(videos["static"], source, angle)
    digest = sha256_file(source)
    info = probe_video(source)
    assert not info.unsupported_reason
    assert info.coded_width == 320 and info.coded_height == 180
    assert abs(info.rotation) == abs(angle)
    assert (info.width, info.height) == ((180, 320) if abs(angle) == 90 else (320, 180))
    original, _ = read_video_frame(videos["static"])
    oriented, _ = read_video_frame(source)
    assert np.array_equal(oriented, np.rot90(original, round(info.rotation / 90)))
    report = export_video(source, output, get_preset("natural"))
    out = probe_video(output)
    assert (out.width, out.height, out.rotation) == (info.width, info.height, 0)
    assert frame_pts(source) == frame_pts(output)
    assert audio_packets(source) == audio_packets(output)
    assert report["output"]["source_rotation_applied"] == info.rotation
    assert sha256_file(source) == digest


def test_reflected_display_matrix_rejected_instead_of_silently_losing_geometry(videos, tmp_path):
    source = tmp_path / "reflection.mp4"
    _matrix_copy(videos["static"], source, "reflection")
    with pytest.raises(StudioError, match="display matrix"):
        export_video(source, tmp_path / "out.mp4", get_preset("natural"))


def test_all_original_audio_tracks_and_language_tags_are_remuxed(videos, tmp_path):
    source, output = tmp_path / "two-tracks.mov", tmp_path / "two-tracks.mp4"
    with (
        open_local_media(videos["static"]) as original,
        av.open(str(source), "w", format="mov", options={"movie_timescale": "48000"}) as container,
    ):
        video = container.add_stream_from_template(original.streams.video[0])
        audio = [container.add_stream_from_template(original.streams.audio[0]) for _ in range(2)]
        for track, language in zip(audio, ("eng", "rus"), strict=True):
            track.metadata["language"] = language
        source_audio = original.streams.audio[0]
        audio_end = (source_audio.start_time + source_audio.duration) * source_audio.time_base
        for packet in original.demux():
            if not packet.size:
                continue
            if packet.stream.type == "video":
                packet.stream = video
                container.mux(packet)
            elif packet.stream.type == "audio":
                duration = min(packet.duration, int(audio_end / packet.time_base) - packet.pts)
                for index, track in enumerate(audio):
                    copied = av.Packet(bytes(packet))
                    copied.pts = packet.pts + index * 2400
                    copied.dts = packet.dts + index * 2400
                    copied.duration, copied.time_base = duration, packet.time_base
                    copied.stream = track
                    container.mux(copied)
    report = export_video(source, output, get_preset("natural"))
    assert audio_packets(source) == audio_packets(output)
    assert len(report["verification"]["audio_tracks"]) == 2
    assert all(
        track["packets_identical"] and track["timestamps_identical"]
        for track in report["verification"]["audio_tracks"]
    )
    with open_local_media(output) as container:
        assert [stream.metadata["language"] for stream in container.streams.audio] == ["eng", "rus"]


def test_10bit_sdr_hevc_import_uses_the_same_processing_engine(tmp_path):
    if "libx265" not in av.codecs_available:
        pytest.skip("Fixture encoder libx265 unavailable; production only requires HEVC decoding")
    source, output = tmp_path / "hevc.mov", tmp_path / "graded.mp4"
    with av.open(str(source), "w", format="mov") as container:
        stream = container.add_stream("libx265", rate=12)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p10le"
        stream.codec_context.color_primaries = stream.codec_context.color_trc = 1
        stream.codec_context.colorspace = stream.codec_context.color_range = 1
        stream.options = {"x265-params": "pools=1:frame-threads=1:log-level=error:bframes=0"}
        for index in range(3):
            frame = encode_video_frame(np.full((64, 96, 3), [0.1, 0.3, 0.5], np.float32))
            frame.pts, frame.time_base = index, Fraction(1, 12)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    assert probe_video(source).codec == "hevc"
    assert "10" in probe_video(source).pixel_format
    assert probe_video(source).bit_depth == 10
    assert probe_video(source).to_dict()["bit_depth"] == 10
    rgb, _ = read_video_frame(source)
    assert rgb.shape == (64, 96, 3) and np.isfinite(rgb).all()
    report = export_video(source, output, get_preset("natural"))
    assert report["verification"]["frame_count"] == 3
    assert probe_video(output).codec == "h264"


def test_pcm_audio_requires_explicit_mov_destination_and_is_preserved(tmp_path):
    source = tmp_path / "pcm.mov"
    with av.open(str(source), "w", format="mov") as container:
        video = container.add_stream("libx264", rate=12)
        video.width, video.height, video.pix_fmt = 96, 64, "yuv420p"
        video.codec_context.color_primaries = video.codec_context.color_trc = 1
        video.codec_context.colorspace = video.codec_context.color_range = 1
        audio = container.add_stream("pcm_s16le", rate=48000)
        audio.layout = "stereo"
        for index in range(3):
            frame = encode_video_frame(np.full((64, 96, 3), 0.2, np.float32))
            frame.pts, frame.time_base = index, Fraction(1, 12)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        frame = av.AudioFrame.from_ndarray(
            np.zeros((1, 24000), np.int16), format="s16", layout="stereo"
        )
        frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)
    with pytest.raises(StudioError, match="Выберите MOV"):
        export_video(source, tmp_path / "bad.mp4", get_preset("natural"))
    output = tmp_path / "preserved.mov"
    report = export_video(source, output, get_preset("natural"))
    assert audio_packets(source) == audio_packets(output)
    assert report["verification"]["audio_verified"]
    assert not (tmp_path / "bad.mp4").exists()
    from freedive_color.core.batch import export_batch
    from freedive_color.core.session import SessionItem

    batch = export_batch((SessionItem(source),), tmp_path / "series")
    assert batch.completed == 1 and batch.failed == 0
    assert batch.records[0].destination.suffix == ".mov"
    assert audio_packets(source) == audio_packets(batch.records[0].destination)
