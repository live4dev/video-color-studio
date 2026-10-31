from copy import deepcopy

import pytest

from tools.compare_skin_runs import select_variant


def reports(skin=0.9, excluded=0.01, seconds=1):
    rows = {
        "skin_arm": {"mean_alpha": skin, "expected": "skin", "pixel_count": 9},
        "hair": {"mean_alpha": excluded, "expected": "excluded", "pixel_count": 9},
    }
    common = {
        "source_unchanged": True,
        "source_sha256": "fixed-source",
        "source_sha256_after": "fixed-source",
        "sample_manifest_sha256": "fixed-manifest",
        "thresholds": {"positive_mean_alpha_min": 0.5, "negative_mean_alpha_max": 0.05},
        "performance": {"total_seconds": seconds},
    }
    photo = {**deepcopy(common), "regions": deepcopy(rows)}
    video = {
        **deepcopy(common),
        "mode": "static-and-tracked",
        "missing_samples": [],
        "frames": {
            "sample": {
                "static": {"regions": deepcopy(rows)},
                "tracked": {"regions": deepcopy(rows)},
            }
        },
    }
    return photo, video


def test_selects_only_fully_accepted_candidate_then_worst_error_then_speed():
    baseline = reports(0.1)
    selected = select_variant(
        *baseline,
        {
            "fast_bad": reports(0.99, 0.06, 0.1),
            "lower_error": reports(0.95, 0.01, 5),
            "fast": reports(0.8, 0.01, 1),
        },
    )
    assert selected["selected_variant"] == "lower_error"
    assert not selected["scores"]["fast_bad"]["eligible"]
    tied = select_variant(*baseline, {"slow": reports(seconds=2), "fast": reports(seconds=1)})
    assert tied["selected_variant"] == "fast"


def test_no_go_does_not_authorize_installation_and_lists_regressions():
    result = select_variant(*reports(), {"candidate": reports(0.1, 0.2)})
    assert not result["installation_update_allowed"]
    assert result["selected_variant"] is None
    assert len(result["scores"]["candidate"]["regressed_checks"]) == 6
    assert result["scores"]["candidate"]["negative_failures"] == 3


@pytest.mark.parametrize(
    "change",
    ("source", "threshold", "missing", "annotation", "pixels", "nan", "coordinates", "time"),
)
def test_candidate_cannot_change_inputs_annotations_or_acceptance(change):
    photo, video = reports()
    if change == "source":
        photo["source_sha256"] = photo["source_sha256_after"] = "other"
    elif change == "threshold":
        photo["thresholds"]["negative_mean_alpha_max"] = 0.5
    elif change == "missing":
        video["missing_samples"] = ["missing"]
    elif change == "annotation":
        photo["regions"]["skin_arm"]["expected"] = "excluded"
    elif change == "pixels":
        photo["regions"]["skin_arm"]["pixel_count"] = 1
    elif change == "coordinates":
        photo["sample_manifest_sha256"] = "other-coordinates"
    elif change == "time":
        video["frames"]["sample"]["requested_seconds"] = 10
    else:
        photo["regions"]["skin_arm"]["mean_alpha"] = float("nan")
    with pytest.raises(ValueError):
        select_variant(*reports(), {"candidate": (photo, video)})


def test_acceptance_is_recomputed_instead_of_trusting_report_flags():
    photo, video = reports(0.1)
    photo["acceptance_passed"] = video["acceptance_passed"] = True
    for row in photo["regions"].values():
        row["passed"] = True
    assert select_variant(*reports(), {"candidate": (photo, video)})["selected_variant"] is None
