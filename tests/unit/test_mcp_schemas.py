"""Public control boundary tests: reject coercion before any project mutation."""

import importlib
import math
import subprocess
import sys
from pathlib import Path

import pytest

from freedive_color.control.errors import ControlError
from freedive_color.control.schemas import (
    LOCAL_DEFAULTS,
    LOCAL_RANGES,
    PARAM_MODELS,
    get_capabilities,
    tool_schemas,
    validate_params,
)


def grade_params(**patch):
    return {
        "session_id": "session-1",
        "media_ids": ["a", "b"],
        "expected_revisions": {"a": 0, "b": 3},
        **patch,
    }


@pytest.mark.parametrize("value", [True, False, "0.1", None, math.inf, -math.inf, math.nan])
def test_grade_rejects_non_numeric_or_non_finite_without_coercion(value):
    with pytest.raises(ControlError) as raised:
        validate_params("set_grade", grade_params(grade={"exposure": value}))
    assert raised.value.code == "INVALID_PARAMS"
    assert "get_capabilities" in raised.value.hint
    assert raised.value.details["fields"][0]["field"] == "grade.exposure"
    assert "input" not in raised.value.details["fields"][0]


@pytest.mark.parametrize("value", [0, 1, "true", "false", None])
def test_local_enabled_requires_boolean(value):
    with pytest.raises(ControlError):
        validate_params("set_grade", grade_params(local={"enabled": value}))


@pytest.mark.parametrize("value", [True, "3", 3.0, -1])
def test_revision_requires_nonnegative_strict_integer(value):
    params = grade_params(preset="natural")
    params["expected_revisions"]["b"] = value
    with pytest.raises(ControlError):
        validate_params("set_grade", params)


def test_partial_patch_preserves_omitted_local_and_mask_state_contract():
    result = validate_params(
        "set_grade",
        grade_params(preset="cinematic", grade={"exposure": 1}, local={"enabled": False}),
    )
    assert result["grade"] == {"exposure": 1.0}
    assert result["local"] == {"enabled": False}
    assert "mask_edits" not in result
    assert result["preset"] == "cinematic"


@pytest.mark.parametrize(
    "params",
    [
        grade_params(grade={"unknown": 0}),
        grade_params(grade={"preset": "vivid"}),
        grade_params(local={"unknown": True}),
        {**grade_params(preset="natural"), "unknown": 0},
        {**grade_params(preset="natural"), "media_ids": ["a", "a"]},
        {**grade_params(preset="natural"), "expected_revisions": {"a": 0}},
        {**grade_params(preset="natural"), "expected_revisions": {"a": 0, "b": 3, "c": 0}},
        grade_params(grade={}),
    ],
)
def test_group_patch_rejects_unknown_fields_incomplete_revision_map_and_noop(params):
    with pytest.raises(ControlError) as raised:
        validate_params("set_grade", params)
    assert raised.value.code == "INVALID_PARAMS"


def test_every_range_is_enforced_and_the_public_schema_matches_core():
    core_local = importlib.import_module("freedive_color.core.local_pipeline")
    core_grade = importlib.import_module("freedive_color.core.presets")
    assert LOCAL_RANGES == core_local.LOCAL_RANGES
    assert core_local.LocalSettings().to_dict() == LOCAL_DEFAULTS
    for group, bounds in (("grade", core_grade.RANGES), ("local", LOCAL_RANGES)):
        for field, (low, high) in bounds.items():
            for invalid in (low - 0.01, high + 0.01):
                with pytest.raises(ControlError):
                    validate_params("set_grade", grade_params(**{group: {field: invalid}}))
            for valid in (low, high):
                result = validate_params("set_grade", grade_params(**{group: {field: valid}}))
                assert result[group] == {field: valid}


@pytest.mark.parametrize("value", [8.0, 16.0, "8", True, 12])
def test_export_bit_depth_rejects_float_string_boolean_and_unsupported_integer(value, tmp_path):
    with pytest.raises(ControlError):
        validate_params(
            "export_media",
            {
                "session_id": "s",
                "media_ids": ["a"],
                "directory": str(tmp_path),
                "options": {"bit_depth": value},
            },
        )


@pytest.mark.parametrize("value", [False, "10", 0, -1, 15.1, math.inf, math.nan])
def test_preview_limits_require_finite_positive_duration(value):
    with pytest.raises(ControlError):
        validate_params(
            "render_preview", {"session_id": "s", "media_id": "a", "duration_seconds": value}
        )


@pytest.mark.parametrize("value", [False, 0, "1", 1.0, 101])
def test_page_limit_requires_bounded_strict_integer(value):
    with pytest.raises(ControlError):
        validate_params("list_media", {"session_id": "s", "limit": value})


def test_source_reference_is_unambiguous_and_cannot_mix_sources(tmp_path):
    valid = validate_params(
        "set_reference", {"session_id": "s", "source": "frame", "media_id": "a", "time_seconds": 3}
    )
    assert valid["time_seconds"] == 3
    for params in (
        {"source": "photo"},
        {"source": "frame", "media_id": "a"},
        {"source": "clear", "path": str(tmp_path / "photo.png")},
        {"source": "frame", "media_id": "a", "time_seconds": 0, "path": str(tmp_path)},
    ):
        with pytest.raises(ControlError):
            validate_params("set_reference", {"session_id": "s", **params})


@pytest.mark.parametrize(
    "path", ["relative.mp4", "https://example.test/a.mp4", "//host/a.mp4", "a\0b"]
)
def test_paths_are_explicit_absolute_and_local(path):
    with pytest.raises(ControlError) as raised:
        validate_params("import_media", {"session_id": "s", "paths": [path]})
    assert raised.value.code == "INVALID_PARAMS"


def test_all_fifteen_schemas_forbid_unknown_top_level_and_nested_patch_fields():
    schemas = tool_schemas()
    assert len(schemas) == len(PARAM_MODELS) == 15
    for schema in schemas.values():
        assert schema["additionalProperties"] is False
        for definition in schema.get("$defs", {}).values():
            assert definition["additionalProperties"] is False
    assert schemas["export_media"]["$defs"]["PhotoExportOptions"]["properties"]["bit_depth"] == {
        "default": 8,
        "enum": [8, 16],
        "title": "Bit Depth",
        "type": "integer",
    }


def test_schema_import_and_static_capabilities_do_not_import_qt_or_media_engine():
    script = """
import json,sys
from freedive_color.control.schemas import get_capabilities
caps=get_capabilities()
assert len(caps['operations'])==15
forbidden=('PySide6','cv2','numpy','av','onnxruntime')
assert not any(name.startswith(forbidden) for name in sys.modules)
assert caps['privacy']['media_contents_in_responses'] is False
print(json.dumps({'ok':True}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert result.stdout.strip() == '{"ok": true}'
    assert get_capabilities()["limits"]["preview_max_edge"] == 1280


def test_unknown_method_has_machine_code_and_russian_recovery_hint():
    with pytest.raises(ControlError) as raised:
        validate_params("not_supported", {})
    assert raised.value.code == "METHOD_NOT_FOUND"
    assert "get_capabilities" in raised.value.hint
