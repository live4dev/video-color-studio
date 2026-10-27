"""Independently verify real encoded milestone C outputs, clocks, audio and flicker.

Run generate_video_fixtures.py first. Its clips derive from one licensed photograph;
they are controlled temporal targets, not a claim about real moving dive footage.
Clean video exports and separate, visibly labelled QA frame comparisons are retained.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from freedive_color.core.errors import Cancelled
from freedive_color.core.image_pipeline import CancellationToken
from freedive_color.core.local_pipeline import LocalSettings
from freedive_color.core.presets import get_preset
from freedive_color.core.project import sha256_file
from freedive_color.core.video_pipeline import export_video, read_video_frame


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def _clock(value, time_base) -> str | None:
    return str(Fraction(value) * time_base) if value is not None else None


def inspect_video(path: Path) -> dict:
    """Keep metadata and small scalar summaries; never retain an entire decoded clip."""
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        audio = {
            stream.index: {
                "codec": stream.codec_context.name,
                "language": stream.metadata.get("language"),
                "time_base": str(stream.time_base),
                "stream_start": _clock(stream.start_time, stream.time_base),
                "stream_duration": _clock(stream.duration, stream.time_base),
                "packet_hashes": [],
                "pts": [],
                "dts": [],
                "duration": [],
            }
            for stream in container.streams.audio
        }
        result = {
            "path": str(path.resolve()),
            "width": video.codec_context.width,
            "height": video.codec_context.height,
            "time_base": str(video.time_base),
            "video_stream_start": _clock(video.start_time, video.time_base),
            "video_stream_duration": _clock(video.duration, video.time_base),
            "container_duration_seconds": (
                container.duration / av.time_base if container.duration is not None else None
            ),
            "video_pts": [],
            "frame_luminance": [],
            "frame_rgb_mean": [],
            "audio": list(audio.values()),
        }
        for packet in container.demux():
            if packet.stream.index in audio and packet.size:
                stream_audio = audio[packet.stream.index]
                stream_audio["packet_hashes"].append(hashlib.sha256(bytes(packet)).hexdigest())
                stream_audio["pts"].append(_clock(packet.pts, packet.time_base))
                stream_audio["dts"].append(_clock(packet.dts, packet.time_base))
                stream_audio["duration"].append(_clock(packet.duration, packet.time_base))
            elif packet.stream.index == video.index:
                for frame in packet.decode():
                    if (frame.width, frame.height) != (result["width"], result["height"]):
                        raise AssertionError("A decoded frame changed dimensions inside the clip")
                    if frame.pts is None:
                        raise AssertionError("A decoded video frame has no presentation timestamp")
                    result["video_pts"].append(_clock(frame.pts, frame.time_base))
                    proxy_width = min(384, frame.width)
                    proxy_height = max(2, round(frame.height * proxy_width / frame.width))
                    proxy = frame.reformat(
                        width=proxy_width, height=proxy_height, format="rgb24"
                    ).to_ndarray()
                    means = np.mean(proxy, axis=(0, 1), dtype=np.float64) / 255
                    result["frame_rgb_mean"].append(means.tolist())
                    result["frame_luminance"].append(
                        float(means @ np.array([0.2126, 0.7152, 0.0722]))
                    )
        result["frame_count"] = len(result["video_pts"])
        if not result["frame_count"]:
            raise AssertionError("No real video frames were decoded")
        luminance = np.asarray(result["frame_luminance"])
        result["luminance_span"] = float(np.ptp(luminance))
        result["maximum_luminance_step"] = (
            float(np.max(np.abs(np.diff(luminance)))) if len(luminance) > 1 else 0.0
        )
        result["luminance_step_std"] = (
            float(np.std(np.diff(luminance))) if len(luminance) > 1 else 0.0
        )
        return result


def _maximum_clock_error(source: list, output: list, shift: Fraction = Fraction(0)) -> float:
    if len(source) != len(output):
        raise AssertionError(f"Timestamp count changed: {len(source)} -> {len(output)}")
    errors = []
    for before, after in zip(source, output, strict=True):
        if before is None or after is None:
            if before != after:
                raise AssertionError("A previously defined packet timestamp disappeared")
            continue
        errors.append(abs(Fraction(after) - (Fraction(before) - shift)))
    return float(max(errors, default=Fraction(0)))


def verify_full_export(source: dict, output: dict) -> dict:
    if (source["width"], source["height"]) != (output["width"], output["height"]):
        raise AssertionError("Export changed source frame dimensions")
    tolerance = float(Fraction(output["time_base"])) * 1.1 + 1e-9
    video_error = _maximum_clock_error(source["video_pts"], output["video_pts"])
    if video_error > tolerance:
        raise AssertionError(f"Full export changed source presentation clocks by {video_error}s")
    video_start_error = _maximum_clock_error(
        [source["video_stream_start"]], [output["video_stream_start"]]
    )
    if video_start_error > tolerance:
        raise AssertionError("The source video track's playable start offset changed")
    video_duration_error = None
    if source["video_stream_duration"] is not None and output["video_stream_duration"] is not None:
        video_duration_error = abs(
            float(
                Fraction(source["video_stream_duration"])
                - Fraction(output["video_stream_duration"])
            )
        )
        if video_duration_error > tolerance:
            raise AssertionError("The final video frame duration changed")
    if len(source["audio"]) != len(output["audio"]):
        raise AssertionError("Export changed the number of audio streams")
    audio_results = []
    for before, after in zip(source["audio"], output["audio"], strict=True):
        if before["packet_hashes"] != after["packet_hashes"]:
            raise AssertionError("Original encoded audio packet payloads changed")
        pts_error = _maximum_clock_error(before["pts"], after["pts"])
        dts_error = _maximum_clock_error(before["dts"], after["dts"])
        if max(pts_error, dts_error) > max(tolerance, 1 / 48000):
            raise AssertionError("Audio clock changed relative to the source video clock")
        if before["language"] != after["language"]:
            raise AssertionError("Original audio language metadata disappeared")
        audio_start_error = _maximum_clock_error([before["stream_start"]], [after["stream_start"]])
        if audio_start_error > max(tolerance, 1 / 48000):
            raise AssertionError("The source audio track's playable start offset changed")
        audio_duration_error = None
        if before["stream_duration"] is not None and after["stream_duration"] is not None:
            audio_duration_error = abs(
                float(Fraction(before["stream_duration"]) - Fraction(after["stream_duration"]))
            )
            audio_tolerance = max(float(Fraction(before["time_base"])), 1 / 48000) * 1.1
            if audio_duration_error > audio_tolerance:
                raise AssertionError("The original audio track's playable duration changed")
        audio_results.append(
            {
                "packet_count": len(before["packet_hashes"]),
                "all_payload_hashes_equal": True,
                "maximum_pts_error_seconds": pts_error,
                "maximum_dts_error_seconds": dts_error,
                "audio_track_duration_error_seconds": audio_duration_error,
                "audio_track_start_error_seconds": audio_start_error,
            }
        )
    return {
        "dimensions_unchanged": True,
        "frame_count_unchanged": True,
        "maximum_video_pts_error_seconds": video_error,
        "video_track_duration_error_seconds": video_duration_error,
        "video_track_start_error_seconds": video_start_error,
        "video_clock_tolerance_seconds": tolerance,
        "audio": audio_results,
    }


def verify_preview(source: dict, output: dict, start: float, duration: float) -> dict:
    first = Fraction(source["video_pts"][0])
    requested_origin = first + Fraction(str(start))
    end = requested_origin + Fraction(str(duration))
    expected_pts = [
        point for point in source["video_pts"] if requested_origin <= Fraction(point) < end
    ]
    if not expected_pts:
        raise AssertionError("The requested preview window has no source video frames")
    # The first actual frame is the common zero. A requested point between VFR frames
    # can land slightly earlier; audio must follow the same actual frame boundary.
    origin = Fraction(expected_pts[0])
    tolerance = float(Fraction(output["time_base"])) * 1.1 + 1e-9
    video_error = _maximum_clock_error(expected_pts, output["video_pts"], origin)
    if video_error > tolerance:
        raise AssertionError("Preview video did not use the requested common timeline origin")
    if len(source["audio"]) != len(output["audio"]):
        raise AssertionError("Preview lost an original audio stream")
    audio_results = []
    for before, after in zip(source["audio"], output["audio"], strict=True):
        hashes = after["packet_hashes"]
        if not hashes:
            raise AssertionError("Preview contains no actual audio packets")
        candidates = [
            index
            for index in range(len(before["packet_hashes"]) - len(hashes) + 1)
            if before["packet_hashes"][index : index + len(hashes)] == hashes
        ]
        matching = []
        for index in candidates:
            pts = before["pts"][index : index + len(hashes)]
            error = _maximum_clock_error(pts, after["pts"], origin)
            if error <= max(tolerance, 1 / 48000):
                matching.append((index, error))
        if not matching:
            raise AssertionError(
                "Preview audio is not an unmodified, correctly shifted source slice"
            )
        if max(Fraction(point) for point in after["pts"] if point is not None) >= (
            Fraction(str(duration)) + Fraction(1, 20)
        ):
            raise AssertionError("Preview retained audio outside its requested time range")
        audio_results.append(
            {
                "original_packet_start_index": matching[0][0],
                "packet_count": len(hashes),
                "payloads_are_contiguous_original_slice": True,
                "maximum_pts_error_seconds": matching[0][1],
            }
        )
    return {
        "requested_start_seconds": start,
        "requested_duration_seconds": duration,
        "common_source_origin": str(origin),
        "first_frame_delay_from_requested_start_seconds": float(origin - requested_origin),
        "maximum_video_pts_error_seconds": video_error,
        "audio": audio_results,
    }


def _comparison(source: Path, output: Path, destination: Path, seconds: float) -> None:
    original, _ = read_video_frame(source, seconds, max_edge=720)
    graded, _ = read_video_frame(output, seconds, max_edge=720)

    def image(pixels):
        return Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8))

    left, right = image(original), image(graded)
    canvas = Image.new("RGB", (left.width * 2 + 12, left.height + 40), "#20262b")
    canvas.paste(left, (0, 40))
    canvas.paste(right, (left.width + 12, 40))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    draw.text((12, 10), "Original - QA frame", fill="#dbe3e7", font=font)
    draw.text((left.width + 24, 10), "Natural - QA frame", fill="#dbe3e7", font=font)
    canvas.save(destination, quality=95)


def render(inputs: Path, directory: Path, *, preview_seconds: float = 10.0) -> dict:
    directory = directory.expanduser().resolve()
    inputs = inputs.expanduser().resolve()
    if directory.exists() and any(directory.iterdir()):
        raise SystemExit("Choose a new evidence folder; previous runs are never overwritten.")
    sources = {name: inputs / f"{name}.mp4" for name in ("static", "depth_change", "vfr_offset")}
    if (inputs / "short_4k.mp4").is_file():
        sources["short_4k"] = inputs / "short_4k.mp4"
    if (inputs / "real_motion.mp4").is_file():
        if not (inputs / "real_motion.license.json").is_file():
            raise SystemExit("Real motion QA input needs its author, source and license record.")
        sources["real_motion"] = inputs / "real_motion.mp4"
    for path in sources.values():
        if not path.is_file():
            raise SystemExit(f"Generate the controlled video fixture first: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    hashes = {str(path): sha256_file(path) for path in sources.values()}
    if "real_motion" in sources:
        motion_notice = json.loads(
            (inputs / "real_motion.license.json").read_text(encoding="utf-8")
        )
        raw_motion = inputs / "Backstroke_Underwater.webm"
        if raw_motion.is_file():
            raw_hash = sha256_file(raw_motion)
            if raw_hash != motion_notice["original_sha256"]:
                raise AssertionError("The licensed motion source does not match its provenance")
            hashes[str(raw_motion)] = raw_hash
    settings = get_preset("natural")
    local = LocalSettings()
    records = {}
    started = time.perf_counter()
    for name, source in sources.items():
        print(f"Processing actual video: {name}", flush=True)
        source_data = inspect_video(source)
        destination = directory / f"{name}_natural.mp4"
        export_started = time.perf_counter()
        report = export_video(source, destination, settings, local_settings=local)
        export_elapsed = time.perf_counter() - export_started
        output_data = inspect_video(destination)
        validation = verify_full_export(source_data, output_data)
        if name == "static":
            extra_step = (
                output_data["maximum_luminance_step"] - source_data["maximum_luminance_step"]
            )
            if extra_step > 0.003:
                raise AssertionError(f"Static-scene grading added a luminance step of {extra_step}")
            validation["extra_static_luminance_step"] = extra_step
            validation["static_luminance_step_allowance"] = 0.003
        qa_seconds = 4.0 if name == "real_motion" else 0.5
        _comparison(source, destination, directory / f"qa_{name}_frame.jpg", seconds=qa_seconds)
        records[name] = {
            "source": source_data,
            "output": output_data,
            "validation": validation,
            "export_elapsed_seconds": export_elapsed,
            "process_peak_rss_bytes_so_far": _peak_rss_bytes(),
            "processing_report": report,
            "qa_comparison_frame_seconds": qa_seconds,
        }
    preview_source = sources["vfr_offset"]
    preview_destination = directory / "preview_10s.mp4"
    preview_report = export_video(
        preview_source,
        preview_destination,
        settings,
        local_settings=local,
        preview_start=1.0,
        preview_seconds=preview_seconds,
    )
    preview_data = inspect_video(preview_destination)
    preview_validation = verify_preview(
        records["vfr_offset"]["source"], preview_data, 1.0, preview_seconds
    )
    cancel_destination = directory / "cancelled_must_not_exist.mp4"
    token = CancellationToken()
    progress_events = []

    def cancel_after_progress(percent, message):
        progress_events.append({"percent": percent, "message": message})
        if percent >= 22:
            token.cancel()

    try:
        export_video(
            sources["static"],
            cancel_destination,
            settings,
            token=token,
            progress=cancel_after_progress,
        )
    except Cancelled:
        cancelled = True
    else:
        raise AssertionError("A cancelled export completed instead of stopping")
    if cancel_destination.exists() or Path(f"{cancel_destination}.freedive.json").exists():
        raise AssertionError("Cancellation left a partial final export")
    if list(directory.glob(".freedive-video-*")):
        raise AssertionError("Cancellation left an application staging file")
    unchanged = all(sha256_file(Path(path)) == digest for path, digest in hashes.items())
    if not unchanged:
        raise AssertionError("An input video changed during processing")
    evidence = {
        "milestone": "C",
        "fixture_provenance": (
            "Controlled static / depth / VFR clips derived from one public-domain photograph; "
            "see tests/fixtures/ATTRIBUTION.md and fixture manifest. No generative media. "
            "Those clips use a synthetic audio test signal. Optional real motion is a licensed "
            "underwater pool clip with its own audio; its separate attribution applies to that "
            "video and its QA comparison frames. These checks do not establish semantic quality "
            "across open-water freedivers or all skin tones."
        ),
        "av_version": av.__version__,
        "av_library_versions": av.library_versions,
        "source_sha256": hashes,
        "originals_unchanged": unchanged,
        "clips": records,
        "preview": {
            "output": preview_data,
            "validation": preview_validation,
            "processing_report": preview_report,
        },
        "cancellation": {
            "cancelled": cancelled,
            "final_output_absent": True,
            "staging_files_absent": True,
            "progress_events": progress_events,
        },
        "total_elapsed_seconds": time.perf_counter() - started,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "limitations": [
            (
                "Actual motion is one short pool scene; no broad moving open-water diver corpus."
                if "real_motion" in sources
                else "One licensed still image is the basis; no real underwater motion corpus."
            ),
            "Lossy H.264 encoding changes pixel values; clean outputs contain no QA overlays.",
            "A short optional 4K clip verifies that geometry, not long-duration 4K throughput.",
            "Windows native execution and clean-machine acceptance are reported separately.",
        ],
    }
    manifest = inputs / "manifest.json"
    if manifest.is_file():
        evidence["input_fixture_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    motion_license = inputs / "real_motion.license.json"
    if "real_motion" in sources:
        license_text = motion_license.read_text(encoding="utf-8")
        evidence["real_motion_license"] = json.loads(license_text)
        (directory / "real_motion.license.json").write_text(license_text, encoding="utf-8")
    (directory / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return evidence


def audit_existing(inputs: Path, directory: Path) -> dict:
    """Recheck completed outputs without rendering again or touching any source media."""
    evidence_path = directory / "evidence.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for name, record in evidence["clips"].items():
        source = inspect_video(inputs / f"{name}.mp4")
        output = inspect_video(directory / f"{name}_natural.mp4")
        validation = verify_full_export(source, output)
        if name == "static":
            extra = output["maximum_luminance_step"] - source["maximum_luminance_step"]
            if extra > 0.003:
                raise AssertionError("The static output failed the independent flicker audit")
            validation["extra_static_luminance_step"] = extra
            validation["static_luminance_step_allowance"] = 0.003
        record.update(source=source, output=output, validation=validation)
    preview = evidence["preview"]
    preview_output = inspect_video(directory / "preview_10s.mp4")
    previous_validation = preview["validation"]
    preview.update(
        output=preview_output,
        validation=verify_preview(
            evidence["clips"]["vfr_offset"]["source"],
            preview_output,
            previous_validation["requested_start_seconds"],
            previous_validation["requested_duration_seconds"],
        ),
    )
    for old_path, digest in evidence["source_sha256"].items():
        if sha256_file(inputs / Path(old_path).name) != digest:
            raise AssertionError("A source file changed since the original evidence run")
    evidence["independent_final_audit"] = {
        "utc": datetime.now(UTC).isoformat(),
        "all_outputs_decoded_again": True,
        "authoritative_track_durations_checked": True,
        "original_hashes_checked_again": True,
    }
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-seconds", type=float, default=10.0)
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    evidence = (
        audit_existing(arguments.inputs, arguments.output)
        if arguments.audit_only
        else render(arguments.inputs, arguments.output, preview_seconds=arguments.preview_seconds)
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output.resolve()),
                "clips": list(evidence["clips"]),
                "originals_unchanged": evidence["originals_unchanged"],
                "elapsed_seconds": evidence["total_elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
