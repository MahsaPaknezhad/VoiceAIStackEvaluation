"""
Shared data loading for the reviewer-response analyses.

Builds, for each perceptual dimension (naturalness, noisiness, loudness), an
(n_items x n_raters) matrix aligned on the 76-sample held-out test IDs, with
raters = [original, brian, joe]. Also loads the fine-tuned scorer's per-sample
predictions on the same 76 clips.
"""
from __future__ import annotations

import json
import os

import numpy as np

LABELS_DIR = "/home/ubuntu/VoiceAIStackEvaluation/data/training_dataset/labels"
MODEL_PRED = "/home/ubuntu/VoiceAIStackEvaluation/data/trained_models/best_full/test_correlation.json"
DIMENSIONS = ["naturalness", "noisiness", "loudness"]
# Annotators included in the agreement analysis. All three now live as
# per-labeler dirs under by_labeler/ with identical schema. Anonymised names.
# labeler1 = original annotator (from labels.json); labeler2/3 = additional raters.
RATERS = ["labeler1", "labeler2", "labeler3"]
RATER_DISPLAY = {"labeler1": "labeler1", "labeler2": "labeler2", "labeler3": "labeler3"}


def display_names() -> list[str]:
    """Anonymised rater names, in RATERS order, for tables/figures."""
    return [RATER_DISPLAY.get(r, r) for r in RATERS]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def load_test_ids() -> list[str]:
    return _load_json(os.path.join(LABELS_DIR, "test76_ids.json"))["sample_ids"]


def _dim_map_from_file(path: str, dim: str) -> dict[str, float]:
    """Map sample_id -> rating from a per-dimension label file."""
    return {d["sample_id"]: float(d[dim]) for d in _load_json(path)}


def _original_map(dim: str) -> dict[str, float]:
    """Original annotator ratings come from the combined labels.json."""
    data = _load_json(os.path.join(LABELS_DIR, "labels.json"))
    return {d["sample_id"]: float(d[dim]) for d in data}


def _labeler_map(labeler: str, dim: str) -> dict[str, float]:
    path = os.path.join(LABELS_DIR, "by_labeler", labeler, f"{dim}.json")
    return _dim_map_from_file(path, dim)


def build_rater_matrix(dim: str, test_ids: list[str]) -> np.ndarray:
    """(n_items x n_raters) matrix for RATERS, in order, on the given dimension."""
    maps = [_labeler_map(r, dim) for r in RATERS]
    mat = np.full((len(test_ids), len(RATERS)), np.nan)
    for r, sid in enumerate(test_ids):
        for c, m in enumerate(maps):
            mat[r, c] = m.get(sid, np.nan)
    return mat


def load_model_predictions() -> dict[str, dict[str, float]]:
    """
    Returns {sample_id: {pred_naturalness, pred_noisiness, pred_loudness,
                         human_* (original annotator)}} from the saved scorer run.
    """
    data = _load_json(MODEL_PRED)
    out = {}
    for p in data["pairs"]:
        out[p["sample_id"]] = p
    return out, data.get("metrics", {})
