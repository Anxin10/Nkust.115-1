import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from bxjudge.reid_appearance import (
    describe_crop,
    distances,
    load_policy,
    sample_observations,
)


def policy(tmp_path):
    data = {
        "purpose": "offline_diagnostic",
        "window_s": 0.3,
        "max_samples": 4,
        "min_crop_side_px": 8,
        "max_outside_fraction": 0.1,
        "color_bins": [8, 4, 4],
        "gradient_bins": 8,
    }
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.safe_dump(data))
    return p, load_policy(p)


def test_color_distance_is_separate_from_texture_and_is_not_a_probability(tmp_path):
    _, p = policy(tmp_path)
    a = np.full((32, 32, 3), (0, 0, 255), np.uint8)
    b = np.full_like(a, (255, 0, 0))
    x = describe_crop(a, [0, 0, 32, 32], p)
    y = describe_crop(b, [0, 0, 32, 32], p)
    assert distances(x, x)["color_hellinger"] == pytest.approx(0)
    assert distances(x, y)["color_hellinger"] > 0.9
    assert distances(x, y)["gradient_hellinger"] is None
    assert x["quality"]["visibility"] == "unverified"
    assert "match" not in x and "confidence" not in x


@pytest.mark.parametrize(
    "box", ([2, 2, 2, 10], [0, 0, 4, 4], [float("nan"), 0, 10, 10], [-20, 0, 20, 20])
)
def test_bad_or_severely_truncated_crops_never_produce_descriptors(tmp_path, box):
    _, p = policy(tmp_path)
    assert describe_crop(np.zeros((32, 32, 3), np.uint8), box, p)["descriptor"] is None


def test_sampling_is_directional_bounded_and_does_not_fill_missing_frames(tmp_path):
    _, p = policy(tmp_path)
    track = {
        "cls": "bey",
        "dets": [
            {"frame": f, "cls": "bey", "box": [0, 0, 10, 10], "score": 0.5}
            for f in (0, 1, 3, 6, 9)
        ],
    }
    assert [d["frame"] for d in sample_observations(track, 3, "before", 10, p)] == [
        0,
        1,
        3,
    ]
    assert [d["frame"] for d in sample_observations(track, 3, "after", 10, p)] == [3, 6]
    assert track["dets"][0]["frame"] == 0
    with pytest.raises(ValueError):
        sample_observations(track, 2, "before", 10, p)


def test_policy_is_strict_and_cannot_enable_binding(tmp_path):
    path, _ = policy(tmp_path)
    data = yaml.safe_load(path.read_text())
    for key, value in [
        ("match_threshold", 0.9),
        ("max_samples", True),
        ("window_s", float("nan")),
        ("color_bins", [8, 4, 0]),
    ]:
        bad = dict(data, **{key: value})
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError):
            load_policy(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    from bxjudge.reid_review import LABELS_SCHEMA, MANIFEST_SCHEMA
    from bxjudge.review_store import save_immutable

    source = tmp_path / "synthetic.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:size=64x48:rate=10:duration=1",
            "-c:v",
            "libx264",
            str(source),
        ],
        check=True,
    )
    det = lambda f: {"frame": f, "cls": "bey", "box": [8, 8, 24, 24], "score": 0.8}
    native = {
        "source": str(source),
        "source_sha256": sha(source),
        "fps": 10,
        "tracks": [
            {"track_id": 1, "cls": "bey", "dets": [det(f) for f in range(4)]},
            {"track_id": 2, "cls": "bey", "dets": [det(f) for f in range(5, 9)]},
        ],
    }
    cache = tmp_path / "synthetic-tracks.json"
    cache.write_text(json.dumps(native))
    case = {
        "case_id": "case",
        "video": source.name,
        "source_sha256": sha(source),
        "fps": 10,
        "coordinate_size": [64, 48],
        "source_size": [64, 48],
        "native_ids": [1, 2],
        "before": {"frame": 3, "native_box": det(3)["box"]},
        "after": {"frame": 5, "native_box": det(5)["box"]},
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "cases": [case],
        "source_files_sha256": {str(source): sha(source)},
        "native_files_sha256": {str(cache): sha(cache)},
    }
    m = tmp_path / "manifest.json"
    m.write_text(json.dumps(manifest))
    label = {
        "case_id": "case",
        "reviewed": True,
        "note": "",
        "relation": "same",
        "scope": "same_round",
        **{
            k: "unreviewed"
            for k in (
                "side_before",
                "side_after",
                "visibility_before",
                "visibility_after",
                "collision",
            )
        },
    }
    payload = {
        "schema": LABELS_SCHEMA,
        "manifest_sha256": sha(m),
        "submitted_at": "2026-10-26T00:00:00Z",
        "supersedes": None,
        "cases": [label],
    }
    labels, _ = save_immutable(tmp_path / "labels", payload, prefix="reid")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"labels_sha256": sha(labels), "endpoints": []}))
    cfg, _ = policy(tmp_path)
    return m, labels, annotations, cfg


