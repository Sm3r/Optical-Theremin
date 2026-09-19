import argparse
import csv
import glob
import math
import os
import random
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "train":
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
else:
    PROJECT_ROOT = SCRIPT_DIR

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from network import HandNet
from core import (
    assert_not_lfs_pointer,
    build_loader,
    clean_target_array,
    coerce_array_to_float32,
    collect_predictions,
    compute_feature_stats,
    compute_regression_metrics,
    cycle_seq_lens,
    evaluate,
    get_lr,
    load_csv_array,
    load_npy_array,
    make_kfold_end_indices,
    plot_cycle_comparison,
    plot_error_histogram,
    plot_loss_curve,
    plot_lr_curve,
    plot_predictions,
    plot_scatter,
    plot_summary_metric,
    print_target_stats,
    save_predictions,
    save_summary_plots,
    set_seed,
    split_train_val_end_indices,
    train_one_epoch,
    write_csv,
)


DEFAULT_FEATURE_DIR = os.path.join(PROJECT_ROOT, "data", "features")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs_mocap_multi_cv_dual2")

HAND_FPS = 360
TARGET_FPS = 60.0
DEFAULT_MAX_NILS_TO_FILL = -1

AXES = ("X", "Y", "Z")


def load_target_array(path: str, csv_target_column: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return load_npy_array(path)
    if ext == ".csv":
        return load_csv_array(path, csv_target_column=csv_target_column)
    raise RuntimeError(f"Unsupported file extension for {path}. Expected .npy or .csv.")


def safe_float(value: str) -> float:
    value = value.strip()
    if value == "" or value.lower() in {"nil", "none", "nan"}:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def load_mocap_csv_matrix(path: str) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"Empty MOCAP CSV file: {path}") from exc

        header = [col.strip() for col in header]
        rows = []

        for row in reader:
            if not row:
                continue

            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[:len(header)]

            rows.append([safe_float(value) for value in row])

    if not rows:
        raise RuntimeError(f"MOCAP CSV file has no data rows: {path}")

    data = np.asarray(rows, dtype=np.float32)
    return header, data


# -----------------------------------------------------------------------------
# MOCAP column handling
# -----------------------------------------------------------------------------

def category_from_stimulus(stimulus: str) -> Optional[str]:
    name = stimulus.lower()
    is_pitch = "pitch" in name
    is_volume = "volume" in name

    if is_pitch and is_volume:
        raise RuntimeError(f"Stimulus name matches both pitch and volume: {stimulus}")
    if is_pitch:
        return "pitch"
    if is_volume:
        return "volume"
    return None


def strip_known_mocap_suffixes(stem: str) -> List[str]:
    candidates = [stem]
    suffixes = ["_cleaned", "_mocap", "_features"]

    changed = True
    current = stem
    while changed:
        changed = False
        for suffix in suffixes:
            if current.endswith(suffix):
                current = current[:-len(suffix)]
                if current and current not in candidates:
                    candidates.append(current)
                changed = True

    return candidates


def marker_has_xyz(header_set: set, marker: str) -> bool:
    return all(f"{marker}_{axis}" in header_set for axis in AXES)


def available_marker_ids(header: Sequence[str], prefix: str) -> List[str]:
    header_set = set(header)
    marker_ids = []

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{3}})_X$")
    for col in header:
        match = pattern.match(col)
        if not match:
            continue
        marker_id = match.group(1)
        marker_name = f"{prefix}_{marker_id}"
        if marker_has_xyz(header_set, marker_name):
            marker_ids.append(marker_id)

    return sorted(set(marker_ids))


def candidate_marker_prefixes(header: Sequence[str]) -> List[str]:
    header_set = set(header)
    prefixes = set()
    pattern = re.compile(r"^(.+)_(\d{3})_X$")

    for col in header:
        match = pattern.match(col)
        if not match:
            continue
        prefix = match.group(1)
        marker_id = match.group(2)
        marker_name = f"{prefix}_{marker_id}"
        if marker_has_xyz(header_set, marker_name):
            prefixes.add(prefix)

    return sorted(prefixes)


def resolve_hand_markers(
    header: Sequence[str],
    category: str,
    hand_prefixes: Sequence[str],
    pitch_hand_max_markers: int,
    volume_hand_max_markers: int,
) -> Tuple[str, List[str]]:
    for prefix in hand_prefixes:
        marker_ids = available_marker_ids(header, prefix)
        if marker_ids:
            max_markers = pitch_hand_max_markers if category == "pitch" else volume_hand_max_markers
            if max_markers > 0:
                marker_ids = marker_ids[:max_markers]
            markers = [f"{prefix}_{marker_id}" for marker_id in marker_ids]
            return prefix, markers

    raise RuntimeError(
        "Could not find hand marker columns. Tried prefixes: "
        f"{list(hand_prefixes)}"
    )


def resolve_antenna_prefix(
    header: Sequence[str],
    category: str,
    pitch_antenna_prefix: Optional[str],
    volume_antenna_prefix: Optional[str],
) -> str:
    header_set = set(header)

    override = pitch_antenna_prefix if category == "pitch" else volume_antenna_prefix
    if override:
        required = [f"{override}_{idx:03d}" for idx in range(1, 4)]
        if all(marker_has_xyz(header_set, marker) for marker in required):
            return override
        raise RuntimeError(
            f"Requested {category} antenna prefix {override!r}, but the CSV does not "
            "contain markers 001, 002, and 003 with X/Y/Z columns."
        )

    prefixes = candidate_marker_prefixes(header)

    excluded_tokens = ["right", "left", "hand", "webcam", "camera", "zed"]
    valid_prefixes = []
    for prefix in prefixes:
        prefix_l = prefix.lower()
        if any(token in prefix_l for token in excluded_tokens):
            continue
        required = [f"{prefix}_{idx:03d}" for idx in range(1, 4)]
        if all(marker_has_xyz(header_set, marker) for marker in required):
            valid_prefixes.append(prefix)

    exact = [prefix for prefix in valid_prefixes if prefix.lower() == category]
    if exact:
        return exact[0]

    if category == "volume":
        vol_exact = [prefix for prefix in valid_prefixes if prefix.lower() == "vol"]
        if vol_exact:
            return vol_exact[0]

    contains_category = [prefix for prefix in valid_prefixes if category in prefix.lower()]
    if contains_category:
        return contains_category[0]

    raise RuntimeError(
        f"Could not auto-detect {category} antenna markers. "
        "Use --pitch-antenna-prefix or --volume-antenna-prefix."
    )


def extract_marker_xyz(
    data: np.ndarray,
    header_to_index: Dict[str, int],
    marker: str,
) -> np.ndarray:
    cols = [header_to_index[f"{marker}_{axis}"] for axis in AXES]
    return data[:, cols].astype(np.float32)


def extract_frame_column(data: np.ndarray, header_to_index: Dict[str, int]) -> np.ndarray:
    if "Frame" in header_to_index:
        frames = data[:, header_to_index["Frame"]]
        valid = np.isfinite(frames)
        out = np.arange(len(data), dtype=np.int64)
        out[valid] = np.rint(frames[valid]).astype(np.int64)
        return out

    return np.arange(len(data), dtype=np.int64)


def compute_antenna_center_and_scale(
    antenna_xyz: np.ndarray,
    min_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    center = np.nanmean(antenna_xyz, axis=1).astype(np.float32)

    d01 = np.linalg.norm(antenna_xyz[:, 0, :] - antenna_xyz[:, 1, :], axis=1)
    d02 = np.linalg.norm(antenna_xyz[:, 0, :] - antenna_xyz[:, 2, :], axis=1)
    d12 = np.linalg.norm(antenna_xyz[:, 1, :] - antenna_xyz[:, 2, :], axis=1)

    pairwise = np.stack([d01, d02, d12], axis=1)
    scale = np.nanmax(pairwise, axis=1).astype(np.float32)

    valid_scale = np.isfinite(scale) & (scale > min_scale)
    if np.any(valid_scale):
        fallback_scale = float(np.nanmedian(scale[valid_scale]))
    else:
        raise RuntimeError("Could not compute a valid antenna scale from the three antenna markers.")

    scale[~valid_scale] = fallback_scale
    scale[scale < min_scale] = fallback_scale

    return center, scale.astype(np.float32)



def nanmedian_abs_deviation(x: np.ndarray, axis: int = 0) -> np.ndarray:
    med = np.nanmedian(x, axis=axis)
    return np.nanmedian(np.abs(x - np.expand_dims(med, axis=axis)), axis=axis)


def replace_feature_outliers_with_nan(
    x: np.ndarray,
    method: str,
    mad_z: float,
    min_mad: float,
    stimulus: str,
) -> np.ndarray:
    """
    Mark finite but implausible coordinate explosions as NaN before range stats.

    This is deliberately applied after hand-minus-antenna normalization and before
    category-level range normalization. A few huge finite triangulation values can
    otherwise dominate min/max normalization and flatten all useful motion.
    """
    x = np.asarray(x, dtype=np.float32)
    if method == "none":
        return x
    if method != "mad":
        raise RuntimeError(f"Unknown feature outlier cleaning mode: {method}")

    finite = np.isfinite(x)
    if not np.any(finite):
        return x

    med = np.nanmedian(x, axis=0).astype(np.float32)
    mad = nanmedian_abs_deviation(x, axis=0).astype(np.float32)
    robust_scale = (1.4826 * mad).astype(np.float32)

    usable = np.isfinite(robust_scale) & (robust_scale > float(min_mad))
    if not np.any(usable):
        print()
        print("MOCAP robust outlier cleaning")
        print(f"stimulus:             {stimulus}")
        print("status:               skipped; no usable MAD scale")
        return x

    delta = np.abs(x - med.reshape(1, -1))
    threshold = float(mad_z) * robust_scale.reshape(1, -1)
    outlier = finite & usable.reshape(1, -1) & (delta > threshold)

    cleaned = x.copy()
    cleaned[outlier] = np.nan

    n_values = int(np.prod(x.shape))
    n_outliers = int(outlier.sum())
    row_outliers = outlier.sum(axis=1)
    n_rows_touched = int(np.sum(row_outliers > 0))
    n_rows_mostly_bad = int(np.sum(row_outliers > max(1, int(round(0.5 * x.shape[1])))))

    per_feature = outlier.sum(axis=0)
    worst_indices = np.argsort(per_feature)[::-1][: min(8, x.shape[1])]

    print()
    print("MOCAP robust outlier cleaning")
    print(f"stimulus:             {stimulus}")
    print(f"method:               MAD z > {float(mad_z):.3f}")
    print(f"values set to NaN:    {n_outliers} / {n_values} ({100.0 * n_outliers / max(n_values, 1):.3f}%)")
    print(f"rows touched:         {n_rows_touched} / {x.shape[0]}")
    print(f"rows mostly bad:      {n_rows_mostly_bad} / {x.shape[0]}")
    print("worst feature counts: " + ", ".join(f"{int(i)}:{int(per_feature[i])}" for i in worst_indices))

    return cleaned.astype(np.float32)

def load_mocap_features(
    path: str,
    stimulus: str,
    args,
) -> Dict:
    category = category_from_stimulus(stimulus)
    if category is None:
        raise RuntimeError(f"Could not infer category from stimulus name: {stimulus}")

    header, data = load_mocap_csv_matrix(path)
    header_to_index = {name: idx for idx, name in enumerate(header)}

    frames = extract_frame_column(data, header_to_index)

    hand_prefix, hand_markers = resolve_hand_markers(
        header=header,
        category=category,
        hand_prefixes=args.hand_prefixes,
        pitch_hand_max_markers=args.pitch_hand_max_markers,
        volume_hand_max_markers=args.volume_hand_max_markers,
    )

    antenna_prefix = resolve_antenna_prefix(
        header=header,
        category=category,
        pitch_antenna_prefix=args.pitch_antenna_prefix,
        volume_antenna_prefix=args.volume_antenna_prefix,
    )

    antenna_markers = [f"{antenna_prefix}_{idx:03d}" for idx in range(1, 4)]

    hand_xyz = np.stack(
        [extract_marker_xyz(data, header_to_index, marker) for marker in hand_markers],
        axis=1,
    )
    antenna_xyz = np.stack(
        [extract_marker_xyz(data, header_to_index, marker) for marker in antenna_markers],
        axis=1,
    )

    antenna_center, antenna_scale = compute_antenna_center_and_scale(
        antenna_xyz=antenna_xyz,
        min_scale=args.min_antenna_scale,
    )

    relative = hand_xyz - antenna_center[:, None, :]
    relative = relative / antenna_scale[:, None, None]
    x_relative = relative.reshape(relative.shape[0], -1).astype(np.float32)
    x_relative = replace_feature_outliers_with_nan(
        x_relative,
        method=args.feature_outlier_cleaning,
        mad_z=args.feature_outlier_mad_z,
        min_mad=args.feature_outlier_min_mad,
        stimulus=stimulus,
    )

    print()
    print("MOCAP feature extraction")
    print(f"stimulus:        {stimulus}")
    print(f"category:        {category}")
    print(f"mocap path:      {path}")
    print(f"frames:          {len(frames)}")
    print(f"hand prefix:     {hand_prefix}")
    print(f"hand markers:    {', '.join(hand_markers)}")
    print(f"antenna prefix:  {antenna_prefix}")
    print(f"antenna markers: {', '.join(antenna_markers)}")
    print(f"feature shape:   {x_relative.shape}")
    print(f"antenna scale median: {float(np.nanmedian(antenna_scale)):.6f}")

    return {
        "stimulus": stimulus,
        "category": category,
        "mocap_path": path,
        "frames": frames.astype(np.int64),
        "x_relative": x_relative,
        "hand_prefix": hand_prefix,
        "hand_markers": hand_markers,
        "antenna_prefix": antenna_prefix,
        "antenna_markers": antenna_markers,
    }


# -----------------------------------------------------------------------------
# Stimulus discovery
# -----------------------------------------------------------------------------

def resolve_target_path(
    feature_dir: str,
    stimulus_candidates: Sequence[str],
    target_suffix: str,
    target_ext: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    if target_ext == "auto":
        exts = [".npy", ".csv"]
    else:
        ext = target_ext if target_ext.startswith(".") else f".{target_ext}"
        exts = [ext]

    tried_paths = []
    for stimulus in stimulus_candidates:
        for ext in exts:
            candidate = os.path.join(feature_dir, f"{stimulus}_{target_suffix}{ext}")
            tried_paths.append(candidate)
            if os.path.exists(candidate):
                return stimulus, candidate, tried_paths

    return None, None, tried_paths


def discover_mocap_stimuli(
    feature_dir: str,
    mocap_dir: str,
    target_suffix: str,
    target_ext: str,
) -> List[Dict[str, str]]:
    mocap_paths = sorted(glob.glob(os.path.join(mocap_dir, "*_cleaned.csv")))

    if not mocap_paths:
        raise RuntimeError(f"No cleaned MOCAP CSV files found in: {mocap_dir}")

    pairs = []

    for mocap_path in mocap_paths:
        stem = os.path.splitext(os.path.basename(mocap_path))[0]
        stimulus_candidates = strip_known_mocap_suffixes(stem)

        stimulus, target_path, tried_paths = resolve_target_path(
            feature_dir=feature_dir,
            stimulus_candidates=stimulus_candidates,
            target_suffix=target_suffix,
            target_ext=target_ext,
        )

        if stimulus is None or target_path is None:
            print(
                f"Skipping {stem}: could not find target file in {feature_dir}. "
                f"Tried these exact paths: {tried_paths}"
            )
            continue

        category = category_from_stimulus(stimulus)
        if category is None:
            print(f"Skipping {stimulus}: name does not contain pitch or volume.")
            continue

        pairs.append(
            {
                "stimulus": stimulus,
                "category": category,
                "mocap_path": mocap_path,
                "target_path": target_path,
            }
        )

    if not pairs:
        raise RuntimeError(
            f"No valid cleaned MOCAP/audio pairs found. MOCAP dir: {mocap_dir}, "
            f"target dir: {feature_dir}."
        )

    return pairs


# -----------------------------------------------------------------------------
# Category range normalization
# -----------------------------------------------------------------------------

def compute_category_range_stats(
    prepared_rows: Sequence[Dict],
    epsilon: float,
    stat_mode: str,
    percentile_low: float,
    percentile_high: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    stats = {}

    if stat_mode not in {"minmax", "percentile"}:
        raise RuntimeError(f"Unknown range stat mode: {stat_mode}")
    if not (0.0 <= percentile_low < percentile_high <= 100.0):
        raise RuntimeError(
            "Expected 0 <= --range-percentile-low < --range-percentile-high <= 100."
        )

    for category in sorted(set(row["category"] for row in prepared_rows)):
        xs = [row["x_relative"] for row in prepared_rows if row["category"] == category]
        if not xs:
            continue

        feature_dims = sorted(set(x.shape[1] for x in xs))
        if len(feature_dims) != 1:
            raise RuntimeError(
                f"MOCAP files in category {category!r} do not have the same feature dimension: "
                f"{feature_dims}. Check marker counts or use category-specific settings."
            )

        stacked = np.vstack(xs).astype(np.float32)
        finite_counts = np.isfinite(stacked).sum(axis=0)
        if np.any(finite_counts == 0):
            bad = np.where(finite_counts == 0)[0].tolist()
            raise RuntimeError(
                f"Category {category!r} has feature columns with no finite values after cleaning: {bad}"
            )

        if stat_mode == "minmax":
            x_min = np.nanmin(stacked, axis=0).astype(np.float32)
            x_max = np.nanmax(stacked, axis=0).astype(np.float32)
            label = "min/max"
        else:
            x_min = np.nanpercentile(stacked, percentile_low, axis=0).astype(np.float32)
            x_max = np.nanpercentile(stacked, percentile_high, axis=0).astype(np.float32)
            label = f"percentile {percentile_low:g}/{percentile_high:g}"

        x_range = (x_max - x_min).astype(np.float32)
        x_range[~np.isfinite(x_range)] = 1.0
        x_range[x_range < epsilon] = 1.0

        stats[category] = {
            "min": x_min,
            "max": x_max,
            "range": x_range,
        }

        print()
        print("Category range normalization stats")
        print(f"category:      {category}")
        print(f"stat mode:     {label}")
        print(f"n files:       {len(xs)}")
        print(f"feature dim:   {stacked.shape[1]}")
        print(f"rows:          {stacked.shape[0]}")
        print(f"finite values: {int(np.isfinite(stacked).sum())} / {int(np.prod(stacked.shape))}")
        print(f"global finite min: {float(np.nanmin(stacked)):.6f}")
        print(f"global finite max: {float(np.nanmax(stacked)):.6f}")
        print(f"normalization low median:  {float(np.nanmedian(x_min)):.6f}")
        print(f"normalization high median: {float(np.nanmedian(x_max)):.6f}")

    return stats

def apply_range_normalization(
    x: np.ndarray,
    stats: Dict[str, np.ndarray],
    mode: str,
    clip: bool,
) -> np.ndarray:
    if mode == "none":
        return x.astype(np.float32)

    x_min = stats["min"].reshape(1, -1)
    x_range = stats["range"].reshape(1, -1)

    x_norm = (x - x_min) / x_range

    if mode == "zero_one":
        if clip:
            x_norm = np.clip(x_norm, 0.0, 1.0)
        return x_norm.astype(np.float32)

    if mode == "minus_one_one":
        x_norm = 2.0 * x_norm - 1.0
        if clip:
            x_norm = np.clip(x_norm, -1.0, 1.0)
        return x_norm.astype(np.float32)

    raise RuntimeError(f"Unknown range normalization mode: {mode}")

def write_normalization_report(
    path: str,
    stats: Dict[str, Dict[str, np.ndarray]],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []
    for category, category_stats in stats.items():
        x_min = category_stats["min"]
        x_max = category_stats["max"]
        x_range = category_stats["range"]
        for idx in range(len(x_min)):
            rows.append(
                {
                    "category": category,
                    "feature_index": idx,
                    "min": float(x_min[idx]),
                    "max": float(x_max[idx]),
                    "range": float(x_range[idx]),
                }
            )

    write_csv(path, rows, ["category", "feature_index", "min", "max", "range"])


# -----------------------------------------------------------------------------
# Cleaning / FPS matching
# -----------------------------------------------------------------------------

def resolved_max_nils_to_fill(max_nils_to_fill: int, n_features: int) -> int:
    if max_nils_to_fill >= 0:
        return max_nils_to_fill
    return max(1, int(round(0.20 * n_features)))


def clean_feature_array(feature_arr: np.ndarray, max_nils_to_fill: int) -> Tuple[np.ndarray, np.ndarray]:
    if feature_arr.ndim != 2:
        raise RuntimeError(f"Expected feature array [frames, features], got {feature_arr.shape}")

    effective_max_nils = resolved_max_nils_to_fill(max_nils_to_fill, feature_arr.shape[1])

    nil_count = np.isnan(feature_arr).sum(axis=1)
    keep_mask = nil_count <= effective_max_nils

    cleaned = feature_arr[keep_mask].copy()
    cleaned[np.isnan(cleaned)] = 0.0

    print()
    print("MOCAP feature cleaning")
    print(f"Original frames:        {len(feature_arr)}")
    print(f"Feature dimension:      {feature_arr.shape[1]}")
    print(f"Max NaNs filled/frame:  {effective_max_nils}")
    print(f"Kept frames:            {len(cleaned)}")
    print(f"Dropped frames:         {len(feature_arr) - len(cleaned)}")

    return cleaned.astype(np.float32), keep_mask


def resample_mocap_to_target_rate(
    feature_arr: np.ndarray,
    target_arr: np.ndarray,
    source_frames: np.ndarray,
    hand_fps: float,
    target_fps: float,
    mode: str,
    use_source_frames: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert high-rate MOCAP features to one feature row per target/CV frame.

    The previous implementation kept every MOCAP frame and mapped many MOCAP
    rows to the same audio target frame. With 360 Hz MOCAP and 60 Hz targets,
    this made the model see each target value roughly six times. This function
    collapses the MOCAP stream to the target frame rate before cleaning,
    splitting, and training.

    Returns:
        x_resampled:       [target_frames_with_mocap, features]
        y_resampled:       [target_frames_with_mocap]
        target_frames:     target/CV frame indices, typically 0..N-1 at 60 Hz
        source_counts:     number of MOCAP frames contributing to each target frame
    """
    if feature_arr.ndim != 2:
        raise RuntimeError(f"Expected feature_arr [frames, features], got {feature_arr.shape}")
    if target_arr.ndim != 1:
        raise RuntimeError(f"Expected target_arr [frames], got {target_arr.shape}")
    if mode not in {"mean", "nearest"}:
        raise RuntimeError(f"Unknown MOCAP resample mode: {mode}")

    raw_feature_len = len(feature_arr)
    raw_target_len = len(target_arr)
    n_features = feature_arr.shape[1]

    row_indices = np.arange(raw_feature_len, dtype=np.int64)
    if use_source_frames:
        source_frame_numbers = np.asarray(source_frames, dtype=np.float64).reshape(-1)
        if len(source_frame_numbers) != raw_feature_len:
            raise RuntimeError(
                f"source_frames length mismatch: {len(source_frame_numbers)} vs {raw_feature_len}"
            )
    else:
        source_frame_numbers = row_indices.astype(np.float64)

    source_times_s = source_frame_numbers / float(hand_fps)
    target_idx_for_source = np.rint(source_times_s * float(target_fps)).astype(np.int64)

    valid_source = (
        np.isfinite(source_frame_numbers)
        & (target_idx_for_source >= 0)
        & (target_idx_for_source < raw_target_len)
    )
    source_indices = row_indices[valid_source]
    source_frame_numbers = source_frame_numbers[valid_source]
    target_idx_for_source = target_idx_for_source[valid_source]

    if len(source_indices) == 0:
        raise RuntimeError("No overlapping MOCAP frames could be mapped to target frames.")

    order = np.argsort(target_idx_for_source, kind="stable")
    sorted_source_indices = source_indices[order]
    sorted_target_indices = target_idx_for_source[order]

    target_frames, starts, source_counts = np.unique(
        sorted_target_indices,
        return_index=True,
        return_counts=True,
    )

    if mode == "nearest":
        x_resampled = np.full((len(target_frames), n_features), np.nan, dtype=np.float32)
        for out_idx, start, count in zip(range(len(target_frames)), starts, source_counts):
            source_group = sorted_source_indices[start:start + count]
            frame_group = source_frame_numbers[order][start:start + count]
            desired_source_frame = (float(target_frames[out_idx]) / float(target_fps)) * float(hand_fps)
            nearest_local = int(np.argmin(np.abs(frame_group - desired_source_frame)))
            x_resampled[out_idx] = feature_arr[source_group[nearest_local]].astype(np.float32)
    else:
        x_resampled = np.full((len(target_frames), n_features), np.nan, dtype=np.float32)
        for out_idx, start, count in zip(range(len(target_frames)), starts, source_counts):
            source_group = sorted_source_indices[start:start + count]
            rows = feature_arr[source_group]
            finite = np.isfinite(rows)
            finite_counts = finite.sum(axis=0)
            finite_sums = np.where(finite, rows, 0.0).sum(axis=0, dtype=np.float64)

            averaged = np.full(n_features, np.nan, dtype=np.float64)
            np.divide(
                finite_sums,
                finite_counts,
                out=averaged,
                where=finite_counts > 0,
            )
            x_resampled[out_idx] = averaged.astype(np.float32)

    y_resampled = target_arr[target_frames].astype(np.float32)

    return (
        x_resampled.astype(np.float32),
        y_resampled.astype(np.float32),
        target_frames.astype(np.int64),
        source_counts.astype(np.int64),
    )


def build_fps_matched_arrays(
    feature_arr: np.ndarray,
    target_arr: np.ndarray,
    source_frames: np.ndarray,
    max_nils_to_fill: int,
    hand_fps: float,
    target_fps: float,
    mocap_resample_mode: str = "mean",
    use_source_frames: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hand_fps <= 0:
        raise RuntimeError(f"hand_fps must be > 0, got {hand_fps}")
    if target_fps <= 0:
        raise RuntimeError(f"target_fps must be > 0, got {target_fps}")

    raw_feature_len = len(feature_arr)
    raw_target_len = len(target_arr)

    if raw_feature_len == 0:
        raise RuntimeError("MOCAP feature array has 0 frames.")
    if raw_target_len == 0:
        raise RuntimeError("Target array has 0 frames.")
    if len(source_frames) != raw_feature_len:
        raise RuntimeError(
            f"source_frames length mismatch: {len(source_frames)} vs {raw_feature_len}"
        )

    print()
    print("=" * 80)
    print("Raw count check")
    print("=" * 80)
    print(f"Raw MOCAP frames:      {raw_feature_len}")
    print(f"Raw target frames:     {raw_target_len}")
    print(f"MOCAP FPS:             {hand_fps}")
    print(f"Target/CV FPS:         {target_fps}")
    print(f"MOCAP duration:        {raw_feature_len / hand_fps:.6f} s")
    print(f"Target duration:       {raw_target_len / target_fps:.6f} s")
    print(f"MOCAP / target ratio:  {raw_feature_len / max(raw_target_len, 1):.6f}")
    print(f"MOCAP resample mode:   {mocap_resample_mode}")
    print(f"Use CSV Frame column:  {bool(use_source_frames)}")
    if len(source_frames) > 0:
        print(f"Source frame range:    {int(np.nanmin(source_frames))} -> {int(np.nanmax(source_frames))}")
    print("=" * 80)

    target_arr = clean_target_array(target_arr)

    x_target_rate, y_target_rate, target_frames, source_counts = resample_mocap_to_target_rate(
        feature_arr=feature_arr,
        target_arr=target_arr,
        source_frames=source_frames,
        hand_fps=hand_fps,
        target_fps=target_fps,
        mode=mocap_resample_mode,
        use_source_frames=use_source_frames,
    )

    print()
    print("MOCAP resampled to target/CV frame rate")
    print(f"Resampled feature shape:   {x_target_rate.shape}")
    print(f"Resampled target shape:    {y_target_rate.shape}")
    print(f"Target-frame range:        {int(target_frames[0])} -> {int(target_frames[-1])}")
    print(f"MOCAP frames/CV frame:     mean={float(np.mean(source_counts)):.3f}, "
          f"min={int(np.min(source_counts))}, max={int(np.max(source_counts))}")

    x_clean, keep_mask = clean_feature_array(x_target_rate, max_nils_to_fill)
    y_clean = y_target_rate[keep_mask]
    frames_clean = target_frames[keep_mask]

    valid_y = np.isfinite(y_clean)
    x_clean = x_clean[valid_y]
    y_clean = y_clean[valid_y]
    frames_clean = frames_clean[valid_y]

    if len(x_clean) == 0:
        raise RuntimeError("Final arrays have 0 rows after cleaning.")

    print()
    print("Final 60 Hz CV data")
    print(f"x shape:       {x_clean.shape}")
    print(f"y shape:       {y_clean.shape}")
    print(f"frames shape:  {frames_clean.shape}")
    print(f"Final samples: {len(x_clean)}")
    print(f"First CV frame: {int(frames_clean[0])}")
    print(f"Last CV frame:  {int(frames_clean[-1])}")

    if hand_fps == target_fps and raw_feature_len == raw_target_len:
        print("One-to-one frame mapping detected.")

    return x_clean.astype(np.float32), y_clean.astype(np.float32), frames_clean.astype(np.int64)


# -----------------------------------------------------------------------------
# Dataset / splits
# -----------------------------------------------------------------------------

class MocapSequenceDataset(Dataset):
    def __init__(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        frames: np.ndarray,
        end_indices: np.ndarray,
        seq_len: int,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.x = np.asarray(x_arr, dtype=np.float32)
        self.y = np.asarray(y_arr, dtype=np.float32).reshape(-1)
        self.frames = np.asarray(frames, dtype=np.int64).reshape(-1)
        self.end_indices = np.asarray(end_indices, dtype=np.int64).reshape(-1)
        self.seq_len = int(seq_len)

        if self.x.ndim != 2:
            raise RuntimeError(f"Expected x_arr [frames, features], got {self.x.shape}")
        if len(self.x) != len(self.y):
            raise RuntimeError(f"x/y length mismatch: {len(self.x)} vs {len(self.y)}")
        if len(self.x) != len(self.frames):
            raise RuntimeError(f"x/frames length mismatch: {len(self.x)} vs {len(self.frames)}")
        if self.seq_len <= 0:
            raise RuntimeError(f"seq_len must be > 0, got {self.seq_len}")

        self.end_indices = self.end_indices[self.end_indices >= self.seq_len - 1]
        self.end_indices = self.end_indices[self.end_indices < len(self.x)]

        self.x_mean = None
        self.x_std = None
        if x_mean is not None and x_std is not None:
            self.set_feature_stats(x_mean, x_std)

    def set_feature_stats(self, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.x_mean = np.asarray(x_mean, dtype=np.float32).reshape(1, -1)
        self.x_std = np.asarray(x_std, dtype=np.float32).reshape(1, -1)
        self.x_std[self.x_std < 1e-6] = 1.0

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        end = int(self.end_indices[idx])
        start = end - self.seq_len + 1

        x = self.x[start:end + 1].copy()
        if self.x_mean is not None and self.x_std is not None:
            x = (x - self.x_mean) / self.x_std

        y = np.float32(self.y[end])
        frame = int(self.frames[end])

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "frame": torch.tensor(frame, dtype=torch.long),
        }


@torch.no_grad()
# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------

@torch.no_grad()
@torch.no_grad()
# -----------------------------------------------------------------------------
# Plotting / output
# -----------------------------------------------------------------------------

def save_data_overview_plots(x_arr: np.ndarray, y_arr: np.ndarray, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    audit_rows = []
    for feature_idx in range(x_arr.shape[1]):
        values = x_arr[:, feature_idx]
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            audit_rows.append({
                "feature_index": feature_idx,
                "finite_count": 0,
                "min": float("nan"),
                "max": float("nan"),
                "mean": float("nan"),
                "std": float("nan"),
                "p01": float("nan"),
                "p99": float("nan"),
            })
        else:
            audit_rows.append({
                "feature_index": feature_idx,
                "finite_count": int(len(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "p01": float(np.percentile(finite, 1.0)),
                "p99": float(np.percentile(finite, 99.0)),
            })
    write_csv(
        os.path.join(out_dir, "mocap_feature_audit.csv"),
        audit_rows,
        ["feature_index", "finite_count", "min", "max", "mean", "std", "p01", "p99"],
    )

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(np.arange(len(y_arr)), y_arr)
    ax.set_title("Full target signal")
    ax.set_xlabel("Aligned MOCAP-frame index")
    ax.set_ylabel("Target")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "full_target.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_arr, bins=50, alpha=0.8)
    ax.set_title("Target distribution")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_histogram.png"), dpi=150)
    plt.close(fig)

    y_norm = (y_arr - np.mean(y_arr)) / (np.std(y_arr) + 1e-8)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(y_norm, label="target normalized", linewidth=1.8)

    feature_scores = []
    for feature_idx in range(x_arr.shape[1]):
        x_feat = x_arr[:, feature_idx]
        finite = x_feat[np.isfinite(x_feat)]
        if len(finite) < 2:
            score = 0.0
        else:
            score = float(np.percentile(finite, 95.0) - np.percentile(finite, 5.0))
        feature_scores.append((score, feature_idx))
    feature_indices = [idx for _, idx in sorted(feature_scores, reverse=True)[: min(3, x_arr.shape[1])]]
    for feature_idx in feature_indices:
        x_feat = x_arr[:, feature_idx].astype(np.float64)
        med = np.nanmedian(x_feat)
        p05 = np.nanpercentile(x_feat, 5.0)
        p95 = np.nanpercentile(x_feat, 95.0)
        scale = max((p95 - p05) / 3.29, 1e-8)
        x_feat = np.clip((x_feat - med) / scale, -8.0, 8.0)
        ax.plot(x_feat, label=f"mocap feature {feature_idx} robust-normalized", alpha=0.7)

    ax.set_title("Target vs selected MOCAP features")
    ax.set_xlabel("Aligned MOCAP-frame index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_vs_mocap_features.png"), dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Fold / cycle runner
# -----------------------------------------------------------------------------

def make_model(args, device, input_dim: int):
    return HandNet(
        input_dim=input_dim,
        coord_mlp_dim=args.coord_mlp_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)


def train_one_fold(
    cycle_name: str,
    seq_len: int,
    stimulus: str,
    category: str,
    fold_idx: int,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    frames: np.ndarray,
    train_ends: np.ndarray,
    test_ends: np.ndarray,
    args,
    device,
) -> Dict:
    print()
    print("=" * 80)
    print(
        f"Cycle: {cycle_name} | seq_len={seq_len} | Category: {category} | "
        f"Stimulus: {stimulus} | Fold {fold_idx + 1}/{args.n_folds}"
    )
    print("=" * 80)

    train_ends, val_ends = split_train_val_end_indices(
        train_ends=train_ends,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=args.seed + fold_idx,
    )

    print_target_stats("Train", y_arr, train_ends)
    if val_ends is not None:
        print_target_stats("Val", y_arr, val_ends)
    print_target_stats("Test", y_arr, test_ends)

    train_ds = MocapSequenceDataset(x_arr, y_arr, frames, train_ends, seq_len)
    test_ds = MocapSequenceDataset(x_arr, y_arr, frames, test_ends, seq_len)

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset has 0 samples.")
    if len(test_ds) == 0:
        raise RuntimeError("Test dataset has 0 samples.")

    input_dim = int(x_arr.shape[1])

    if args.use_fold_standardization:
        x_mean, x_std = compute_feature_stats(train_ds, batch_size=args.batch_size)
    else:
        x_mean = np.zeros(input_dim, dtype=np.float32)
        x_std = np.ones(input_dim, dtype=np.float32)

    train_ds.set_feature_stats(x_mean, x_std)
    test_ds.set_feature_stats(x_mean, x_std)

    if val_ends is not None:
        val_ds = MocapSequenceDataset(x_arr, y_arr, frames, val_ends, seq_len)
        val_ds.set_feature_stats(x_mean, x_std)
    else:
        val_ds = None

    print()
    print("Dataset sizes")
    print(f"Input dim:     {input_dim}")
    print(f"Train samples: {len(train_ds)}")
    if val_ds is not None:
        print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    train_loader = build_loader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    monitor_loader = build_loader(val_ds if val_ds is not None else train_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_loader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = make_model(args, device, input_dim=input_dim)

    print()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    loss_fn = nn.MSELoss()

    fold_dir = os.path.join(args.output_dir, cycle_name, stimulus, f"fold_{fold_idx + 1:02d}")
    checkpoint_path = os.path.join(fold_dir, "checkpoints", "best_model.pt")
    plot_dir = os.path.join(fold_dir, "plots")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_monitor_loss = float("inf")
    monitor_name = "val" if val_ds is not None else "train"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        monitor_loss = evaluate(model, monitor_loader, loss_fn, device)

        old_lr = get_lr(optimizer)
        scheduler.step(monitor_loss)
        new_lr = get_lr(optimizer)

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "monitor_mse": monitor_loss,
                "monitor_name": monitor_name,
                "lr": new_lr,
            }
        )

        print(
            f"Epoch {epoch:03d} | train MSE: {train_loss:.6f} | "
            f"{monitor_name} MSE: {monitor_loss:.6f} | lr: {new_lr:.8f}"
        )

        if new_lr < old_lr:
            print(f"Scheduler reduced LR: {old_lr:.8f} -> {new_lr:.8f}")

        if monitor_loss < best_monitor_loss:
            best_monitor_loss = monitor_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "cycle": cycle_name,
                "seq_len": seq_len,
                "stimulus": stimulus,
                "category": category,
                "input_dim": input_dim,
                "fold": fold_idx + 1,
                "train_end_indices": train_ends,
                "val_end_indices": val_ends,
                "test_end_indices": test_ends,
                "x_mean": torch.tensor(x_mean, dtype=torch.float32),
                "x_std": torch.tensor(x_std, dtype=torch.float32),
                "best_monitor_loss": float(best_monitor_loss),
                "monitor_name": monitor_name,
            }
            torch.save(checkpoint, checkpoint_path)

    history_csv_path = os.path.join(fold_dir, "training_history.csv")
    write_csv(
        history_csv_path,
        history,
        fieldnames=["epoch", "train_mse", "monitor_mse", "monitor_name", "lr"],
    )

    print()
    print(f"Best {monitor_name} MSE: {best_monitor_loss:.6f}")
    print(f"Saved best model to: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    frames_out, y_true, y_pred = collect_predictions(model, test_loader, device)
    metrics = compute_regression_metrics(y_true, y_pred)

    print()
    print("Test results")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R2:   {metrics['r2']:.6f}")

    prediction_csv_path = os.path.join(fold_dir, "test_predictions.csv")
    save_predictions(frames_out, y_true, y_pred, prediction_csv_path)

    if args.save_plots:
        title_prefix = f"{cycle_name} | {stimulus} | fold {fold_idx + 1:02d}"
        plot_loss_curve(history, os.path.join(plot_dir, "loss_curve.png"))
        plot_lr_curve(history, os.path.join(plot_dir, "lr_curve.png"))
        plot_predictions(
            frames_out,
            y_true,
            y_pred,
            os.path.join(plot_dir, "test_predictions_chunk.png"),
            args.plot_chunk_size,
            title_prefix,
        )
        plot_predictions(
            frames_out,
            y_true,
            y_pred,
            os.path.join(plot_dir, "test_predictions_full.png"),
            len(frames_out),
            title_prefix,
        )
        plot_scatter(y_true, y_pred, os.path.join(plot_dir, "scatter.png"), title_prefix)
        plot_error_histogram(y_true, y_pred, os.path.join(plot_dir, "error_histogram.png"), title_prefix)

    return {
        "cycle": cycle_name,
        "seq_len": seq_len,
        "category": category,
        "stimulus": stimulus,
        "fold": fold_idx + 1,
        "input_dim": input_dim,
        "n_train_samples": len(train_ds),
        "n_val_samples": 0 if val_ds is None else len(val_ds),
        "n_test_samples": len(test_ds),
        "best_monitor_mse": best_monitor_loss,
        "monitor_name": monitor_name,
        "test_mse": metrics["mse"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"],
        "checkpoint_path": checkpoint_path,
        "prediction_csv_path": prediction_csv_path,
        "history_csv_path": history_csv_path,
    }


def summarize_fold_rows(fold_rows: Sequence[Dict]) -> List[Dict]:
    summary_rows = []
    keys = sorted(set((row["cycle"], row["seq_len"], row["category"], row["stimulus"]) for row in fold_rows))
    metric_names = ["test_mse", "test_rmse", "test_mae", "test_r2"]
    count_names = ["n_train_samples", "n_val_samples", "n_test_samples"]

    for cycle, seq_len, category, stimulus in keys:
        rows = [row for row in fold_rows if row["cycle"] == cycle and row["stimulus"] == stimulus]
        out = {
            "cycle": cycle,
            "seq_len": seq_len,
            "category": category,
            "stimulus": stimulus,
            "input_dim": rows[0]["input_dim"] if rows else 0,
            "n_folds": len(rows),
        }

        for key in count_names:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        for key in metric_names:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.nanmean(values))
            out[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0

        summary_rows.append(out)

    return summary_rows


# -----------------------------------------------------------------------------
# Aggregate category stages: all pitch cleaned, all volume cleaned
# -----------------------------------------------------------------------------

def make_group_balanced_kfold_end_indices(
    group_ranges: Sequence[Dict],
    seq_len: int,
    n_folds: int,
    split: str,
    seed: int,
):
    """
    Balanced aggregate folds for concatenated stimuli.

    Each stimulus contributes test samples to every fold. End indices are created
    inside each original stimulus range, so sequence windows never cross files.
    """
    if n_folds < 2:
        raise RuntimeError(f"n_folds must be >= 2, got {n_folds}")
    if seq_len <= 0:
        raise RuntimeError(f"seq_len must be > 0, got {seq_len}")
    if split not in {"chronological", "random"}:
        raise RuntimeError(f"Unknown split mode: {split}")
    if not group_ranges:
        raise RuntimeError("Cannot build aggregate folds without group ranges.")

    rng = np.random.default_rng(seed)
    group_folds = []

    for group in group_ranges:
        start = int(group["start"])
        end = int(group["end"])
        stimulus = str(group["stimulus"])

        valid_ends = np.arange(start + seq_len - 1, end, dtype=np.int64)
        if len(valid_ends) < n_folds:
            raise RuntimeError(
                f"Cannot create {n_folds} aggregate folds for {stimulus!r}: "
                f"only {len(valid_ends)} valid sequence ends for seq_len={seq_len}."
            )

        fold_source = np.array(valid_ends, copy=True)
        if split == "random":
            rng.shuffle(fold_source)

        folds = [np.sort(fold) for fold in np.array_split(fold_source, n_folds)]
        group_folds.append(folds)

    out = []
    for fold_idx in range(n_folds):
        test_parts = [folds[fold_idx] for folds in group_folds]
        train_parts = [
            folds[i]
            for folds in group_folds
            for i in range(n_folds)
            if i != fold_idx
        ]
        test_ends = np.sort(np.concatenate(test_parts))
        train_ends = np.sort(np.concatenate(train_parts))
        out.append((train_ends, test_ends))

    return out


def build_category_aggregate_arrays(rows: Sequence[Dict], category: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    selected = [row for row in rows if row["category"] == category]
    selected = [row for row in selected if "x_arr" in row and "y_arr" in row and "frames_clean" in row]

    if not selected:
        raise RuntimeError(f"No prepared rows available for aggregate category {category!r}.")

    feature_dims = sorted(set(int(row["x_arr"].shape[1]) for row in selected))
    if len(feature_dims) != 1:
        details = [(row["stimulus"], tuple(row["x_arr"].shape)) for row in selected]
        raise RuntimeError(
            f"Cannot aggregate category {category!r}; feature dimensions differ: {details}"
        )

    xs = []
    ys = []
    frames = []
    group_ranges = []
    cursor = 0

    print()
    print("=" * 80)
    print(f"Building aggregate category arrays: {category}")
    print("=" * 80)

    for row in selected:
        x = np.asarray(row["x_arr"], dtype=np.float32)
        y = np.asarray(row["y_arr"], dtype=np.float32).reshape(-1)
        n = len(x)
        if n != len(y):
            raise RuntimeError(f"Aggregate length mismatch for {row['stimulus']}: x={len(x)}, y={len(y)}")
        if n == 0:
            raise RuntimeError(f"Aggregate row has 0 samples for {row['stimulus']}")

        xs.append(x)
        ys.append(y)
        frames.append(np.arange(cursor, cursor + n, dtype=np.int64))
        group_ranges.append(
            {
                "stimulus": row["stimulus"],
                "category": category,
                "start": cursor,
                "end": cursor + n,
                "n_samples": n,
            }
        )
        print(f"- {row['stimulus']}: samples={n}, x_shape={x.shape}")
        cursor += n

    x_all = np.vstack(xs).astype(np.float32)
    y_all = np.concatenate(ys).astype(np.float32)
    frames_all = np.concatenate(frames).astype(np.int64)

    print(f"Aggregate category: {category}")
    print(f"Total samples:      {len(x_all)}")
    print(f"Feature dim:        {x_all.shape[1]}")

    return x_all, y_all, frames_all, group_ranges


def run_category_aggregate_stages(prepared: Sequence[Dict], args, device) -> List[Dict]:
    aggregate_rows = []

    for category in ["pitch", "volume"]:
        selected = [row for row in prepared if row["category"] == category]
        if len(selected) == 0:
            print()
            print(f"Skipping aggregate category {category}: no rows.")
            continue
        if len(selected) < 2:
            print()
            print(f"Skipping aggregate category {category}: only one row available.")
            continue

        x_all, y_all, frames_all, group_ranges = build_category_aggregate_arrays(prepared, category)
        aggregate_stimulus = f"all_{category}_cleaned"

        if args.save_plots:
            save_data_overview_plots(
                x_arr=x_all,
                y_arr=y_all,
                out_dir=os.path.join(args.output_dir, "data_overview", aggregate_stimulus),
            )

        aggregate_cycles = [("aggregate_frame", args.frame_seq_len), ("aggregate_seq5", args.sequence_seq_len)]
        for cycle_name, seq_len in aggregate_cycles:
            print()
            print("#" * 80)
            print(f"Starting aggregate cycle: {cycle_name} | seq_len={seq_len} | category: {category}")
            print("#" * 80)

            fold_indices = make_group_balanced_kfold_end_indices(
                group_ranges=group_ranges,
                seq_len=seq_len,
                n_folds=args.n_folds,
                split=args.split,
                seed=args.seed,
            )

            for fold_idx, (train_ends, test_ends) in enumerate(fold_indices):
                set_seed(args.seed + 900000 + 1000 * (fold_idx + 1) + 17 * seq_len)
                fold_row = train_one_fold(
                    cycle_name=cycle_name,
                    seq_len=seq_len,
                    stimulus=aggregate_stimulus,
                    category=category,
                    fold_idx=fold_idx,
                    x_arr=x_all,
                    y_arr=y_all,
                    frames=frames_all,
                    train_ends=train_ends,
                    test_ends=test_ends,
                    args=args,
                    device=device,
                )
                aggregate_rows.append(fold_row)

    return aggregate_rows


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--mocap-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.set_defaults(run_aggregate_stages=True)
    parser.add_argument("--run-aggregate-stages", dest="run_aggregate_stages", action="store_true")
    parser.add_argument("--no-run-aggregate-stages", dest="run_aggregate_stages", action="store_false")

    parser.add_argument("--target-suffix", type=str, default="audio")
    parser.add_argument("--target-ext", type=str, default="auto", choices=["auto", "npy", ".npy", "csv", ".csv"])
    parser.add_argument("--csv-target-column", type=int, default=-1)

    parser.add_argument("--stimuli", type=str, nargs="*", default=None)
    parser.add_argument("--split", type=str, default="random", choices=["chronological", "random"])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-ratio-within-train", type=float, default=0.0)

    parser.add_argument("--cycles", type=str, nargs="+", default=["frame", "seq5"], choices=["frame", "seq5"])
    parser.add_argument("--frame-seq-len", type=int, default=1)
    parser.add_argument("--sequence-seq-len", type=int, default=5)

    parser.add_argument("--hand-fps", type=float, default=HAND_FPS)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--cv-fps", type=float, default=None, help="Alias for --target-fps.")
    parser.add_argument(
        "--mocap-resample-mode",
        type=str,
        default="mean",
        choices=["mean", "nearest"],
        help=(
            "How to collapse high-rate MOCAP frames to one feature row per target/CV frame. "
            "mean averages all MOCAP frames mapped to the same target frame; nearest uses the nearest MOCAP frame."
        ),
    )
    parser.add_argument("--max-nils-to-fill", type=int, default=DEFAULT_MAX_NILS_TO_FILL)

    parser.add_argument("--hand-prefixes", type=str, nargs="+", default=["right", "left", "hand"])
    parser.add_argument("--pitch-hand-max-markers", type=int, default=6)
    parser.add_argument("--volume-hand-max-markers", type=int, default=3)
    parser.add_argument("--pitch-antenna-prefix", type=str, default="pitch")
    parser.add_argument("--volume-antenna-prefix", type=str, default=None)
    parser.add_argument("--min-antenna-scale", type=float, default=1e-6)

    parser.add_argument("--range-normalization", type=str, default="minus_one_one", choices=["minus_one_one", "zero_one", "none"])
    parser.add_argument("--range-epsilon", type=float, default=1e-8)
    parser.add_argument("--range-stat-mode", type=str, default="percentile", choices=["minmax", "percentile"])
    parser.add_argument("--range-percentile-low", type=float, default=1.0)
    parser.add_argument("--range-percentile-high", type=float, default=99.0)
    parser.set_defaults(clip_normalized_features=True)
    parser.add_argument("--clip-normalized-features", dest="clip_normalized_features", action="store_true")
    parser.add_argument("--no-clip-normalized-features", dest="clip_normalized_features", action="store_false")
    parser.add_argument("--feature-outlier-cleaning", type=str, default="mad", choices=["none", "mad"])
    parser.add_argument("--feature-outlier-mad-z", type=float, default=8.0)
    parser.add_argument("--feature-outlier-min-mad", type=float, default=1e-8)
    parser.set_defaults(use_mocap_frame_column=True)
    parser.add_argument("--use-mocap-frame-column", dest="use_mocap_frame_column", action="store_true")
    parser.add_argument("--no-use-mocap-frame-column", dest="use_mocap_frame_column", action="store_false")
    parser.set_defaults(use_fold_standardization=False)
    parser.add_argument("--use-fold-standardization", dest="use_fold_standardization", action="store_true")
    parser.add_argument("--no-use-fold-standardization", dest="use_fold_standardization", action="store_false")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--coord-mlp-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.75)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.set_defaults(save_plots=True)
    parser.add_argument("--save-plots", dest="save_plots", action="store_true")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false")
    parser.add_argument("--plot-chunk-size", type=int, default=300)

    parser.add_argument("--fold-results-csv", type=str, default=None)
    parser.add_argument("--summary-csv", type=str, default=None)

    args = parser.parse_args()

    if args.mocap_dir is None:
        args.mocap_dir = os.path.join(args.feature_dir, "mocap")

    if args.cv_fps is not None:
        args.target_fps = args.cv_fps

    if args.frame_seq_len != 1:
        print(f"Warning: frame cycle is configured with seq_len={args.frame_seq_len}, not 1.")
    if args.sequence_seq_len <= 1:
        print(f"Warning: seq5 cycle is configured with seq_len={args.sequence_seq_len}.")

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Feature dir: {args.feature_dir}")
    print(f"MOCAP dir:   {args.mocap_dir}")

    cycles = cycle_seq_lens(args)
    print()
    print("Training cycles")
    for cycle_name, seq_len in cycles:
        print(f"- {cycle_name}: seq_len={seq_len}")

    pairs = discover_mocap_stimuli(
        feature_dir=args.feature_dir,
        mocap_dir=args.mocap_dir,
        target_suffix=args.target_suffix,
        target_ext=args.target_ext,
    )

    if args.stimuli:
        selected = set(args.stimuli)
        pairs = [pair for pair in pairs if pair["stimulus"] in selected]
        found = set(pair["stimulus"] for pair in pairs)
        missing = selected - found
        if missing:
            raise RuntimeError(f"Requested stimuli not found: {sorted(missing)}")

    print()
    print("Discovered MOCAP/audio pairs")
    for pair in pairs:
        print(f"- {pair['stimulus']} [{pair['category']}]")
        print(f"  mocap:  {pair['mocap_path']}")
        print(f"  target: {pair['target_path']}")

    prepared = []
    for pair in pairs:
        row = load_mocap_features(
            path=pair["mocap_path"],
            stimulus=pair["stimulus"],
            args=args,
        )
        row["target_path"] = pair["target_path"]
        prepared.append(row)

    range_stats = compute_category_range_stats(
        prepared,
        epsilon=args.range_epsilon,
        stat_mode=args.range_stat_mode,
        percentile_low=args.range_percentile_low,
        percentile_high=args.range_percentile_high,
    )
    write_normalization_report(
        os.path.join(args.output_dir, "mocap_category_range_normalization.csv"),
        range_stats,
    )

    for row in prepared:
        category = row["category"]
        row["x_normalized"] = apply_range_normalization(
            row["x_relative"],
            range_stats[category],
            mode=args.range_normalization,
            clip=args.clip_normalized_features,
        )

    all_fold_rows = []

    for row in prepared:
        stimulus = row["stimulus"]
        category = row["category"]

        print()
        print("#" * 80)
        print(f"Loading target and training stimulus: {stimulus} [{category}]")
        print("#" * 80)

        target_arr = load_target_array(row["target_path"], csv_target_column=args.csv_target_column)

        x_arr, y_arr, frames = build_fps_matched_arrays(
            feature_arr=row["x_normalized"],
            target_arr=target_arr,
            source_frames=row["frames"],
            max_nils_to_fill=args.max_nils_to_fill,
            hand_fps=args.hand_fps,
            target_fps=args.target_fps,
            mocap_resample_mode=args.mocap_resample_mode,
            use_source_frames=args.use_mocap_frame_column,
        )

        # Keep the exact per-stimulus arrays for the later category aggregate stages.
        # This does not affect the normal per-file training path below.
        row["x_arr"] = x_arr
        row["y_arr"] = y_arr
        row["frames_clean"] = frames

        if args.save_plots:
            save_data_overview_plots(
                x_arr=x_arr,
                y_arr=y_arr,
                out_dir=os.path.join(args.output_dir, "data_overview", stimulus),
            )

        for cycle_name, seq_len in cycles:
            print()
            print("#" * 80)
            print(f"Starting cycle: {cycle_name} | seq_len={seq_len} | stimulus: {stimulus}")
            print("#" * 80)

            fold_indices = make_kfold_end_indices(
                n_frames=len(x_arr),
                seq_len=seq_len,
                n_folds=args.n_folds,
                split=args.split,
                seed=args.seed,
            )

            for fold_idx, (train_ends, test_ends) in enumerate(fold_indices):
                set_seed(args.seed + 1000 * (fold_idx + 1) + 17 * seq_len)
                fold_row = train_one_fold(
                    cycle_name=cycle_name,
                    seq_len=seq_len,
                    stimulus=stimulus,
                    category=category,
                    fold_idx=fold_idx,
                    x_arr=x_arr,
                    y_arr=y_arr,
                    frames=frames,
                    train_ends=train_ends,
                    test_ends=test_ends,
                    args=args,
                    device=device,
                )
                all_fold_rows.append(fold_row)

    if args.run_aggregate_stages:
        aggregate_fold_rows = run_category_aggregate_stages(prepared, args, device)
        all_fold_rows.extend(aggregate_fold_rows)

    fold_fieldnames = [
        "cycle",
        "seq_len",
        "category",
        "stimulus",
        "fold",
        "input_dim",
        "n_train_samples",
        "n_val_samples",
        "n_test_samples",
        "best_monitor_mse",
        "monitor_name",
        "test_mse",
        "test_rmse",
        "test_mae",
        "test_r2",
        "checkpoint_path",
        "prediction_csv_path",
        "history_csv_path",
    ]

    if args.fold_results_csv is None:
        args.fold_results_csv = os.path.join(args.output_dir, "mocap_crossval_fold_results.csv")
    write_csv(args.fold_results_csv, all_fold_rows, fold_fieldnames)

    summary_rows = summarize_fold_rows(all_fold_rows)
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else ["cycle", "seq_len", "category", "stimulus", "n_folds"]

    if args.summary_csv is None:
        args.summary_csv = os.path.join(args.output_dir, "mocap_crossval_summary.csv")
    write_csv(args.summary_csv, summary_rows, summary_fieldnames)

    if args.save_plots:
        save_summary_plots(summary_rows, args.output_dir)

    print()
    print("Final summary")
    for row in summary_rows:
        print(
            f"{row['cycle']} | seq_len={row['seq_len']} | {row['category']} | {row['stimulus']} | "
            f"MSE {row['test_mse_mean']:.6f} +/- {row['test_mse_std']:.6f} | "
            f"RMSE {row['test_rmse_mean']:.6f} +/- {row['test_rmse_std']:.6f} | "
            f"MAE {row['test_mae_mean']:.6f} +/- {row['test_mae_std']:.6f} | "
            f"R2 {row['test_r2_mean']:.6f} +/- {row['test_r2_std']:.6f}"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
