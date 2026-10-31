import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from freedive_color.core import masks
from freedive_color.core.skin import SkinPrediction
from tools import evaluate_skin_candidate as evaluator
from tools import verify_skin_photo, verify_skin_video


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    photo = tmp_path / "photo.png"
    video = tmp_path / "video.mp4"
    photo.write_bytes(b"unchanged-photo-fixture")
    video.write_bytes(b"unchanged-video-fixture")
    regions = {"skin_arm": [20, 20, 2], "water": [2, 2, 1]}
    photo_samples, video_samples = tmp_path / "photo.json", tmp_path / "video.json"
    photo_samples.write_text(json.dumps({"points": {"photo": regions}}))
    video_samples.write_text(json.dumps({"points": {"0": regions, "1": regions}}))
    rgb = np.full((40, 60, 3), [0.06, 0.25, 0.40], np.float32)
    person = np.zeros(rgb.shape[:2], np.float32)
    person[5:35, 5:55] = 0.99
    predictor_calls, factory_calls = [], []
    state = {"skin_found": True}

    def predictor(source, hint):
        predictor_calls.append((source.shape, hint.shape))
        skin = np.zeros(source.shape[:2], np.float32)
        if state["skin_found"]:
            skin[12:28, 14:46] = 0.96
        exclusion = ((person > 0) & (skin == 0)).astype(np.float32) * 0.99
        return SkinPrediction(skin, person.copy(), exclusion, model_id="test-candidate")

    def factory(name, model_dir):
        factory_calls.append((name, model_dir))
        return predictor

    monkeypatch.setattr(evaluator, "make_predictor", factory)
    monkeypatch.setattr(masks, "infer_person", lambda _rgb: person.copy())
    monkeypatch.setattr(
        verify_skin_photo,
        "load_photo",
        lambda _source: SimpleNamespace(rgb=rgb.copy(), width=60, height=40),
    )
    frames = [SimpleNamespace(pts=value, time_base=Fraction(1, 30)) for value in (0, 30)]

    class Container:
        streams = SimpleNamespace(video=[object()])

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, _stream):
            yield from frames

    monkeypatch.setattr(verify_skin_video, "open_local_media", lambda _source: Container())
    monkeypatch.setattr(
        verify_skin_video,
        "probe_video",
        lambda _source: SimpleNamespace(video_start=Fraction(0), width=60, height=40),
    )
    monkeypatch.setattr(verify_skin_video, "decode_video_frame", lambda _frame, _info: rgb.copy())
    return SimpleNamespace(
        photo=photo,
        video=video,
        photo_samples=photo_samples,
        video_samples=video_samples,
        predictor=predictor,
        predictor_calls=predictor_calls,
        factory_calls=factory_calls,
        state=state,
    )


def run_candidate(inputs, output, **options):
    return evaluator.evaluate_candidate(
        "fcn-skin",
        inputs.photo,
        inputs.photo_samples,
        inputs.video,
        inputs.video_samples,
        output,
        **options,
    )


@pytest.mark.parametrize("static_only", [False, True])
def test_process_local_candidate_drives_real_photo_builder_and_video_tracker(
    local_inputs, tmp_path, static_only
):
    original = masks.infer_skin
    output = tmp_path / "new-evidence"
    model_dir = tmp_path / "explicit-models"
    summary = run_candidate(local_inputs, output, static_only=static_only, model_dir=model_dir)
    assert masks.infer_skin is original
    assert local_inputs.factory_calls == [("fcn-skin", model_dir)]
    assert len(local_inputs.predictor_calls) >= (3 if static_only else 5)
    assert summary["execution_status"] == "completed"
    assert summary["mode"] == ("static-only" if static_only else "static-and-tracked")
    assert summary["acceptance_passed"] and summary["failed_checks"] == 0
    assert summary["measured_checks"] == (6 if static_only else 10)
    assert not summary["product_default_changed"]
    assert not summary["media_contents_returned"]
    photo = json.loads((output / "photo.json").read_text())
    video = json.loads((output / "video.json").read_text())
    assert photo["metadata"]["skin_source"] == "test-candidate"
    assert photo["source_unchanged"] and video["source_unchanged"]
    assert photo["sample_manifest_sha256"] and video["sample_manifest_sha256"]
    assert ("tracked" in video["frames"]["0"]) is not static_only
    if not static_only:
        assert video["tracker"]["frames"] == 2
        assert video["frames"]["0"]["tracked"]["metadata"]["skin_source"] == "test-candidate"
    assert json.loads((output / "summary.json").read_text())["reports"] == summary["reports"]
    assert local_inputs.photo.read_bytes() == b"unchanged-photo-fixture"
    assert local_inputs.video.read_bytes() == b"unchanged-video-fixture"


def test_candidate_override_restored_after_video_failure_and_no_success_summary_written(
    local_inputs, tmp_path, monkeypatch
):
    original = masks.infer_skin

    def failed_video(*_args, **_kwargs):
        assert masks.infer_skin is local_inputs.predictor
        raise RuntimeError("deliberate verification failure")

    monkeypatch.setattr(evaluator, "evaluate_video", failed_video)
    output = tmp_path / "failed-evidence"
    with pytest.raises(RuntimeError, match="deliberate"):
        run_candidate(local_inputs, output)
    assert masks.infer_skin is original
    assert (output / "photo.json").is_file()
    assert not (output / "summary.json").exists()


def test_existing_output_directory_is_protected_before_candidate_initialization(
    local_inputs, tmp_path
):
    output = tmp_path / "previous-evidence"
    output.mkdir()
    previous = output / "summary.json"
    previous.write_text("previous evidence")
    with pytest.raises(FileExistsError):
        run_candidate(local_inputs, output)
    assert not local_inputs.factory_calls
    assert not local_inputs.predictor_calls
    assert previous.read_text() == "previous evidence"


@pytest.mark.parametrize("require_pass", [False, True])
def test_cli_quality_failure_is_recorded_and_optional_exit_code_is_nonzero(
    local_inputs, tmp_path, capsys, require_pass
):
    local_inputs.state["skin_found"] = False
    output = tmp_path / "quality-failure"
    arguments = [
        "--candidate",
        "fcn-skin",
        "--photo",
        str(local_inputs.photo),
        "--photo-samples",
        str(local_inputs.photo_samples),
        "--video",
        str(local_inputs.video),
        "--video-samples",
        str(local_inputs.video_samples),
        "--output-directory",
        str(output),
        "--static-only",
    ]
    if require_pass:
        arguments.append("--require-pass")
    assert evaluator.main(arguments) == (2 if require_pass else 0)
    summary = json.loads(capsys.readouterr().out)
    assert summary["failed_checks"] == 3
    assert summary["passed_checks"] == 3
    assert not summary["acceptance_passed"]
    assert summary["execution_status"] == "completed"
    assert Path(summary["reports"]["summary"]).is_file()
