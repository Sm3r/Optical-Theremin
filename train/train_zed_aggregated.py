#!/usr/bin/env python3
"""
Run aggregated ThereminMocap cross-validation training.

Instead of training one model per stimulus, this script trains models on two
aggregated stimulus groups:

    1. pitch  -> all discovered stimuli whose name contains "pitch"
    2. volume -> all discovered stimuli whose name contains "volume"

For each aggregated group, it runs both sequence configurations:

    frame : seq_len = 1
    seq5  : seq_len = 5

Expected files in --feature-dir:

    <stimulus>_hand.npy
    <stimulus>_audio.npy

Example:

    fast_sweep_pitch_hand.npy
    fast_sweep_pitch_audio.npy
    slow_volume_hand.npy
    slow_volume_audio.npy

Important:
    Sequence windows are generated inside each original stimulus segment only.
    A seq_len=5 sample will never cross from one stimulus into another after
    aggregation.

The script is standalone for the exported .npy workflow. It does not import
or depend on dataset.py. It uses HandNet from network.py.
"""

import argparse
import csv
import glob
import math
import os
import random
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

from network import HandNet


DEFAULT_FEATURE_DIR = "/home/mmlab/Desktop/Theremin/ThereminMocap/data/features"
DEFAULT_OUTPUT_DIR = "/home/mmlab/Desktop/Theremin/ThereminMocap/runs_aggregated_pitch_volume_cv"

HAND_FPS = 30.0
TARGET_FPS = 60.0
MAX_NILS_TO_FILL = 10


# =============================================================================
# Reproducibility / loading
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def assert_not_lfs_pointer(path: str) -> None:
    with open(path, "rb") as f:
        head = f.read(128)

    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            "This file is a Git LFS pointer, not a real data file:\n"
            f"  {path}\n"
            "Download the real LFS object first, then rerun this script."
        )


def coerce_array_to_float32(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.shape == ():
        arr = arr.item()

    arr = np.asarray(arr)

    if arr.dtype == object:
        arr = np.where(arr == "nil", np.nan, arr)
        arr = np.where(arr == "None", np.nan, arr)
        arr = np.where(arr == "", np.nan, arr)

    return arr.astype(np.float32)


def load_npy_array(path: str) -> np.ndarray:
    assert_not_lfs_pointer(path)

    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load NumPy file: {path}\nOriginal error: {exc}"
        ) from exc

    return coerce_array_to_float32(arr)


def load_csv_array(path: str, csv_target_column: int) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float32)

    if arr.ndim == 2:
        all_nan_rows = np.all(~np.isfinite(arr), axis=1)
        arr = arr[~all_nan_rows]
    elif arr.ndim == 1:
        arr = arr[np.isfinite(arr)]

    if arr.ndim == 2 and arr.shape[1] > 1:
        arr = arr[:, csv_target_column]

    return coerce_array_to_float32(arr)


def load_array(path: str, csv_target_column: int) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        return load_npy_array(path)

    if ext == ".csv":
        return load_csv_array(path, csv_target_column=csv_target_column)

    raise RuntimeError(
        f"Unsupported file extension for {path}. Expected .npy or .csv."
    )


# =============================================================================
# Stimulus discovery / grouping
# =============================================================================

def discover_stimuli(
    feature_dir: str,
    target_suffix: str,
    target_ext: str,
) -> List[Dict[str, str]]:
    target_ext = target_ext if target_ext.startswith(".") else f".{target_ext}"
    hand_paths = sorted(glob.glob(os.path.join(feature_dir, "*_hand.npy")))

    pairs = []

    for hand_path in hand_paths:
        base = os.path.basename(hand_path)
        stimulus = base[: -len("_hand.npy")]
        target_path = os.path.join(
            feature_dir,
            f"{stimulus}_{target_suffix}{target_ext}",
        )

        if os.path.exists(target_path):
            pairs.append(
                {
                    "stimulus": stimulus,
                    "hand_path": hand_path,
                    "target_path": target_path,
                }
            )
        else:
            print(f"Skipping {stimulus}: missing target {target_path}")

    if not pairs:
        raise RuntimeError(
            f"No pairs found in {feature_dir}. Expected files named "
            f"<stimulus>_hand.npy and <stimulus>_{target_suffix}{target_ext}."
        )

    return pairs


def name_matches_keywords(name: str, keywords: Sequence[str]) -> bool:
    name_l = name.lower()
    return any(keyword.lower() in name_l for keyword in keywords)


def group_stimuli(
    pairs: Sequence[Dict[str, str]],
    pitch_keywords: Sequence[str],
    volume_keywords: Sequence[str],
) -> Dict[str, List[Dict[str, str]]]:
    groups = {
        "pitch": [],
        "volume": [],
    }

    for pair in pairs:
        stimulus = pair["stimulus"]

        is_pitch = name_matches_keywords(stimulus, pitch_keywords)
        is_volume = name_matches_keywords(stimulus, volume_keywords)

        if is_pitch and is_volume:
            raise RuntimeError(
                f"Stimulus matched both pitch and volume groups: {stimulus}"
            )

        if is_pitch:
            groups["pitch"].append(pair)
        elif is_volume:
            groups["volume"].append(pair)
        else:
            print(
                f"Skipping {stimulus}: did not match pitch or volume keywords."
            )

    return groups


# =============================================================================
# Cleaning / FPS matching
# =============================================================================

def clean_target_array(target_arr: np.ndarray) -> np.ndarray:
    if target_arr.ndim == 2 and target_arr.shape[1] == 1:
        target_arr = target_arr[:, 0]

    if target_arr.ndim != 1:
        raise RuntimeError(
            f"Expected target array [frames] or [frames, 1], got {target_arr.shape}"
        )

    print()
    print("Target cleaning")
    print(f"Original target frames: {len(target_arr)}")
    print(f"Target NaNs:            {int(np.isnan(target_arr).sum())}")

    return target_arr.astype(np.float32)


def clean_hand_array(
    hand_arr: np.ndarray,
    max_nils_to_fill: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if hand_arr.ndim != 2:
        raise RuntimeError(f"Expected hand array [frames, 63], got {hand_arr.shape}")

    if hand_arr.shape[1] != 63:
        raise RuntimeError(f"Expected 63 hand features, got {hand_arr.shape[1]}")

    nil_count = np.isnan(hand_arr).sum(axis=1)
    keep_mask = nil_count <= max_nils_to_fill

    cleaned = hand_arr[keep_mask].copy()
    cleaned[np.isnan(cleaned)] = 0.0

    print()
    print("Hand cleaning")
    print(f"Original hand frames: {len(hand_arr)}")
    print(f"Kept hand frames:     {len(cleaned)}")
    print(f"Dropped hand frames:  {len(hand_arr) - len(cleaned)}")

    return cleaned.astype(np.float32), keep_mask


def build_fps_matched_arrays(
    hand_arr: np.ndarray,
    target_arr: np.ndarray,
    max_nils_to_fill: int,
    hand_fps: float,
    target_fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hand_fps <= 0:
        raise RuntimeError(f"hand_fps must be > 0, got {hand_fps}")

    if target_fps <= 0:
        raise RuntimeError(f"target_fps must be > 0, got {target_fps}")

    raw_hand_len = len(hand_arr)
    raw_target_len = len(target_arr)

    if raw_hand_len == 0:
        raise RuntimeError("Hand array has 0 frames.")

    if raw_target_len == 0:
        raise RuntimeError("Target array has 0 frames.")

    print()
    print("=" * 80)
    print("Raw count check")
    print("=" * 80)
    print(f"Raw hand frames:     {raw_hand_len}")
    print(f"Raw target frames:   {raw_target_len}")
    print(f"Hand FPS:            {hand_fps}")
    print(f"Target FPS:          {target_fps}")
    print(f"Hand duration:       {raw_hand_len / hand_fps:.6f} s")
    print(f"Target duration:     {raw_target_len / target_fps:.6f} s")
    print(f"Target / hand ratio: {raw_target_len / max(raw_hand_len, 1):.6f}")
    print("=" * 80)

    if hand_arr.ndim != 2:
        raise RuntimeError(f"Expected hand array [frames, 63], got {hand_arr.shape}")

    if hand_arr.shape[1] != 63:
        raise RuntimeError(f"Expected 63 hand features, got {hand_arr.shape[1]}")

    target_arr = clean_target_array(target_arr)

    hand_frame_idx = np.arange(raw_hand_len, dtype=np.float64)
    hand_time_s = hand_frame_idx / float(hand_fps)

    target_idx = np.rint(hand_time_s * float(target_fps)).astype(np.int64)
    target_idx = np.clip(target_idx, 0, raw_target_len - 1)

    y_for_hand = target_arr[target_idx]

    x_clean, keep_mask = clean_hand_array(hand_arr, max_nils_to_fill)
    y_clean = y_for_hand[keep_mask]
    frames_clean = np.arange(raw_hand_len, dtype=np.int64)[keep_mask]

    valid_y = np.isfinite(y_clean)
    x_clean = x_clean[valid_y]
    y_clean = y_clean[valid_y]
    frames_clean = frames_clean[valid_y]

    if len(x_clean) == 0:
        raise RuntimeError("Final arrays have 0 rows after cleaning.")

    print()
    print("FPS-matched data")
    print(f"x shape:       {x_clean.shape}")
    print(f"y shape:       {y_clean.shape}")
    print(f"frames shape:  {frames_clean.shape}")
    print(f"Final frames:  {len(x_clean)}")
    print(f"First target index: {target_idx[0]}")
    print(f"Last target index:  {target_idx[-1]}")

    if hand_fps == target_fps and raw_hand_len == raw_target_len:
        print("One-to-one frame mapping detected.")

    return (
        x_clean.astype(np.float32),
        y_clean.astype(np.float32),
        frames_clean.astype(np.int64),
    )


# =============================================================================
# Aggregation
# =============================================================================

def build_aggregated_group_data(
    group_name: str,
    pairs: Sequence[Dict[str, str]],
    args,
) -> Dict:
    if not pairs:
        raise RuntimeError(f"No stimuli available for group: {group_name}")

    x_parts = []
    y_parts = []
    local_frame_parts = []
    stimulus_id_parts = []
    segments = []
    stimulus_names = []

    global_start = 0

    print()
    print("#" * 80)
    print(f"Building aggregated group: {group_name}")
    print("#" * 80)

    for stimulus_id, pair in enumerate(pairs):
        stimulus = pair["stimulus"]
        stimulus_names.append(stimulus)

        print()
        print("=" * 80)
        print(f"Loading stimulus for group {group_name}: {stimulus}")
        print("=" * 80)
        print(f"hand:   {pair['hand_path']}")
        print(f"target: {pair['target_path']}")

        hand_arr = load_array(
            pair["hand_path"],
            csv_target_column=args.csv_target_column,
        )
        target_arr = load_array(
            pair["target_path"],
            csv_target_column=args.csv_target_column,
        )

        x_arr, y_arr, frames = build_fps_matched_arrays(
            hand_arr=hand_arr,
            target_arr=target_arr,
            max_nils_to_fill=args.max_nils_to_fill,
            hand_fps=args.hand_fps,
            target_fps=args.target_fps,
        )

        n = len(x_arr)
        global_stop = global_start + n

        x_parts.append(x_arr)
        y_parts.append(y_arr)
        local_frame_parts.append(frames)
        stimulus_id_parts.append(
            np.full(n, stimulus_id, dtype=np.int64)
        )

        segments.append(
            {
                "stimulus": stimulus,
                "stimulus_id": stimulus_id,
                "start": global_start,
                "stop": global_stop,
                "n_frames": n,
                "first_local_frame": int(frames[0]),
                "last_local_frame": int(frames[-1]),
            }
        )

        if args.save_plots:
            save_data_overview_plots(
                x_arr=x_arr,
                y_arr=y_arr,
                out_dir=os.path.join(
                    args.output_dir,
                    "data_overview_individual",
                    group_name,
                    stimulus,
                ),
                title_prefix=f"{group_name} | {stimulus}",
            )

        global_start = global_stop

    x_all = np.vstack(x_parts).astype(np.float32)
    y_all = np.concatenate(y_parts).astype(np.float32)
    local_frames_all = np.concatenate(local_frame_parts).astype(np.int64)
    stimulus_ids_all = np.concatenate(stimulus_id_parts).astype(np.int64)

    global_indices = np.arange(len(x_all), dtype=np.int64)

    print()
    print("=" * 80)
    print(f"Aggregated group summary: {group_name}")
    print("=" * 80)
    print(f"Stimuli:        {len(stimulus_names)}")
    print(f"Total frames:   {len(x_all)}")
    print(f"x shape:        {x_all.shape}")
    print(f"y shape:        {y_all.shape}")
    print(f"target min:     {np.min(y_all):.6f}")
    print(f"target max:     {np.max(y_all):.6f}")
    print(f"target mean:    {np.mean(y_all):.6f}")
    print(f"target std:     {np.std(y_all):.6f}")
    print("Stimulus segments:")
    for segment in segments:
        print(
            f"- {segment['stimulus']}: "
            f"global [{segment['start']}, {segment['stop']}) | "
            f"frames={segment['n_frames']}"
        )
    print("=" * 80)

    if args.save_plots:
        save_data_overview_plots(
            x_arr=x_all,
            y_arr=y_all,
            out_dir=os.path.join(
                args.output_dir,
                "data_overview_aggregated",
                group_name,
            ),
            title_prefix=f"aggregated {group_name}",
        )

    return {
        "group": group_name,
        "x": x_all,
        "y": y_all,
        "local_frames": local_frames_all,
        "stimulus_ids": stimulus_ids_all,
        "global_indices": global_indices,
        "segments": segments,
        "stimulus_names": stimulus_names,
    }


# =============================================================================
# Dataset / splits
# =============================================================================

class AggregatedSequenceDataset(Dataset):
    def __init__(
        self,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        local_frames: np.ndarray,
        stimulus_ids: np.ndarray,
        global_indices: np.ndarray,
        end_indices: np.ndarray,
        seq_len: int,
        x_mean: Optional[np.ndarray] = None,
        x_std: Optional[np.ndarray] = None,
    ):
        super().__init__()

        self.x = np.asarray(x_arr, dtype=np.float32)
        self.y = np.asarray(y_arr, dtype=np.float32).reshape(-1)
        self.local_frames = np.asarray(local_frames, dtype=np.int64).reshape(-1)
        self.stimulus_ids = np.asarray(stimulus_ids, dtype=np.int64).reshape(-1)
        self.global_indices = np.asarray(global_indices, dtype=np.int64).reshape(-1)
        self.end_indices = np.asarray(end_indices, dtype=np.int64).reshape(-1)
        self.seq_len = int(seq_len)

        if self.x.ndim != 2:
            raise RuntimeError(f"Expected x_arr [frames, features], got {self.x.shape}")

        n = len(self.x)

        if len(self.y) != n:
            raise RuntimeError(f"x/y length mismatch: {n} vs {len(self.y)}")

        if len(self.local_frames) != n:
            raise RuntimeError(
                f"x/local_frames length mismatch: {n} vs {len(self.local_frames)}"
            )

        if len(self.stimulus_ids) != n:
            raise RuntimeError(
                f"x/stimulus_ids length mismatch: {n} vs {len(self.stimulus_ids)}"
            )

        if len(self.global_indices) != n:
            raise RuntimeError(
                f"x/global_indices length mismatch: {n} vs {len(self.global_indices)}"
            )

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

        return {
            "x": torch.from_numpy(x),
            "y": torch.tensor(y, dtype=torch.float32),
            "local_frame": torch.tensor(int(self.local_frames[end]), dtype=torch.long),
            "stimulus_id": torch.tensor(int(self.stimulus_ids[end]), dtype=torch.long),
            "global_index": torch.tensor(int(self.global_indices[end]), dtype=torch.long),
        }


@torch.no_grad()
def compute_feature_stats(
    dataset: Dataset,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_sum = None
    total_sq_sum = None
    total_count = 0

    for batch in loader:
        x = batch["x"].float()
        x = x.reshape(-1, x.shape[-1])

        if total_sum is None:
            total_sum = x.sum(dim=0)
            total_sq_sum = (x ** 2).sum(dim=0)
        else:
            total_sum += x.sum(dim=0)
            total_sq_sum += (x ** 2).sum(dim=0)

        total_count += x.shape[0]

    if total_count == 0:
        raise RuntimeError("Cannot compute feature stats: dataset has 0 samples.")

    mean = total_sum / total_count
    var = total_sq_sum / total_count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-8))

    return mean.numpy().astype(np.float32), std.numpy().astype(np.float32)


def make_kfold_end_indices_from_segments(
    segments: Sequence[Dict],
    seq_len: int,
    n_folds: int,
    split: str,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_folds < 2:
        raise RuntimeError(f"n_folds must be >= 2, got {n_folds}")

    valid_end_parts = []

    for segment in segments:
        start = int(segment["start"])
        stop = int(segment["stop"])

        if stop - start < seq_len:
            print(
                f"Skipping segment for seq_len={seq_len}: "
                f"{segment['stimulus']} has only {stop - start} frames."
            )
            continue

        segment_valid_ends = np.arange(
            start + seq_len - 1,
            stop,
            dtype=np.int64,
        )
        valid_end_parts.append(segment_valid_ends)

    if not valid_end_parts:
        raise RuntimeError(f"No valid sequence ends for seq_len={seq_len}.")

    valid_ends = np.concatenate(valid_end_parts)

    if len(valid_ends) < n_folds:
        raise RuntimeError(
            f"Cannot create {n_folds} folds from only {len(valid_ends)} samples."
        )

    if split == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(valid_ends)
    elif split == "chronological":
        valid_ends = np.sort(valid_ends)
    else:
        raise RuntimeError(f"Unknown split mode: {split}")

    folds = np.array_split(valid_ends, n_folds)
    out = []

    for fold_idx in range(n_folds):
        test_ends = np.sort(folds[fold_idx])
        train_ends = np.sort(
            np.concatenate(
                [folds[i] for i in range(n_folds) if i != fold_idx]
            )
        )
        out.append((train_ends, test_ends))

    return out


def split_train_val_end_indices(
    train_ends: np.ndarray,
    val_ratio_within_train: float,
    seed: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    train_ends = np.asarray(train_ends, dtype=np.int64)

    if val_ratio_within_train <= 0.0:
        return np.sort(train_ends), None

    if val_ratio_within_train >= 1.0:
        raise RuntimeError("--val-ratio-within-train must be < 1.0")

    rng = np.random.default_rng(seed)
    shuffled = np.array(train_ends, copy=True)
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_ratio_within_train)))
    val_ends = np.sort(shuffled[:n_val])
    clean_train_ends = np.sort(shuffled[n_val:])

    if len(clean_train_ends) == 0:
        raise RuntimeError("Training split has 0 samples after validation split.")

    return clean_train_ends, val_ends


def print_target_stats(name: str, y: np.ndarray, end_indices: np.ndarray) -> None:
    values = y[np.asarray(end_indices, dtype=np.int64)]

    print()
    print(f"{name} stats")
    print(f"samples: {len(values)}")
    print(f"y min:   {np.min(values):.6f}")
    print(f"y max:   {np.max(values):.6f}")
    print(f"y mean:  {np.mean(values):.6f}")
    print(f"y std:   {np.std(values):.6f}")


def build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


# =============================================================================
# Training / evaluation
# =============================================================================

def make_model(args, device):
    return HandNet(
        input_dim=63,
        coord_mlp_dim=args.coord_mlp_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
    model.train()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)

        y_hat = model(x).reshape_as(y)
        loss = loss_fn(y_hat, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> float:
    model.eval()

    total_loss = 0.0
    total_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        y_hat = model(x).reshape_as(y)
        loss = loss_fn(y_hat, y)

        total_loss += loss.item() * x.size(0)
        total_count += x.size(0)

    return total_loss / max(total_count, 1)


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_global_indices = []
    all_local_frames = []
    all_stimulus_ids = []
    all_y_true = []
    all_y_pred = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        y_hat = model(x).reshape_as(y).cpu().numpy()

        all_global_indices.append(batch["global_index"].cpu().numpy())
        all_local_frames.append(batch["local_frame"].cpu().numpy())
        all_stimulus_ids.append(batch["stimulus_id"].cpu().numpy())
        all_y_true.append(y.cpu().numpy())
        all_y_pred.append(y_hat)

    if not all_global_indices:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    global_indices = np.concatenate(all_global_indices)
    local_frames = np.concatenate(all_local_frames)
    stimulus_ids = np.concatenate(all_stimulus_ids)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    order = np.lexsort((local_frames, stimulus_ids))

    return (
        global_indices[order],
        local_frames[order],
        stimulus_ids[order],
        y_true[order],
        y_pred[order],
    )


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "r2": float("nan"),
        }

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mse))

    denom = np.sum((y_true - np.mean(y_true)) ** 2)

    if denom < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# =============================================================================
# Output / plotting
# =============================================================================

def write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print(f"CSV saved to: {path}")


def save_predictions(
    global_indices: np.ndarray,
    local_frames: np.ndarray,
    stimulus_ids: np.ndarray,
    stimulus_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "GlobalIndex",
                "StimulusID",
                "Stimulus",
                "LocalFrame",
                "GroundTruth",
                "Prediction",
                "Error",
            ]
        )

        for global_idx, stimulus_id, local_frame, true_val, pred_val in zip(
            global_indices,
            stimulus_ids,
            local_frames,
            y_true,
            y_pred,
        ):
            stimulus_id_int = int(stimulus_id)
            stimulus_name = stimulus_names[stimulus_id_int]

            writer.writerow(
                [
                    int(global_idx),
                    stimulus_id_int,
                    stimulus_name,
                    int(local_frame),
                    f"{float(true_val):.8f}",
                    f"{float(pred_val):.8f}",
                    f"{float(pred_val - true_val):.8f}",
                ]
            )

    print(f"Prediction CSV saved to: {out_path}")


def save_data_overview_plots(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    out_dir: str,
    title_prefix: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(np.arange(len(y_arr)), y_arr)
    ax.set_title(f"{title_prefix} | target signal")
    ax.set_xlabel("Aligned hand-frame index")
    ax.set_ylabel("Target")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_signal.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_arr, bins=50, alpha=0.8)
    ax.set_title(f"{title_prefix} | target distribution")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_histogram.png"), dpi=150)
    plt.close(fig)

    y_norm = (y_arr - np.mean(y_arr)) / (np.std(y_arr) + 1e-8)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(y_norm, label="target normalized", linewidth=1.8)

    for feature_idx in [0, 1, 2]:
        x_feat = x_arr[:, feature_idx]
        x_feat = (x_feat - np.mean(x_feat)) / (np.std(x_feat) + 1e-8)
        ax.plot(
            x_feat,
            label=f"hand feature {feature_idx} normalized",
            alpha=0.7,
        )

    ax.set_title(f"{title_prefix} | target vs selected hand features")
    ax.set_xlabel("Aligned hand-frame index")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "target_vs_hand_features.png"), dpi=150)
    plt.close(fig)


def plot_loss_curve(history: Sequence[Dict[str, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_mse"] for row in history]
    monitor_loss = [row["monitor_mse"] for row in history]
    monitor_name = history[0].get("monitor_name", "monitor")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, train_loss, label="train MSE", linewidth=2.0)
    ax.plot(epochs, monitor_loss, label=f"{monitor_name} MSE", linewidth=2.0)
    ax.set_title("Training curve")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_lr_curve(history: Sequence[Dict[str, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not history:
        return

    epochs = [row["epoch"] for row in history]
    lr = [row["lr"] for row in history]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, lr, linewidth=2.0)
    ax.set_title("Learning rate schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predictions(
    local_frames: np.ndarray,
    stimulus_ids: np.ndarray,
    stimulus_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    chunk_size: int,
    title_prefix: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if len(y_true) == 0:
        return

    metrics = compute_regression_metrics(y_true, y_pred)
    chunk_size = min(chunk_size, len(y_true))

    x_axis = np.arange(chunk_size)

    labels = [
        stimulus_names[int(stimulus_id)]
        for stimulus_id in stimulus_ids[:chunk_size]
    ]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(
        x_axis,
        y_true[:chunk_size],
        linewidth=2.0,
        label="Ground truth",
    )
    ax.plot(
        x_axis,
        y_pred[:chunk_size],
        linewidth=2.0,
        linestyle="--",
        label="Prediction",
    )

    ax.set_title(
        f"{title_prefix} | MSE={metrics['mse']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | "
        f"R2={metrics['r2']:.4f}"
    )

    ax.set_xlabel("Sorted test sample index")
    ax.set_ylabel("Target")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    unique_labels = []
    unique_positions = []

    last_label = None
    for idx, label in enumerate(labels):
        if label != last_label:
            unique_labels.append(label)
            unique_positions.append(idx)
            last_label = label

    if unique_positions:
        ax2 = ax.secondary_xaxis("top")
        ax2.set_xticks(unique_positions)
        ax2.set_xticklabels(unique_labels, rotation=45, ha="left", fontsize=8)
        ax2.set_xlabel("Stimulus segment")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title_prefix: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if len(y_true) == 0:
        return

    metrics = compute_regression_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18)

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        linewidth=2.0,
    )

    ax.set_title(
        f"{title_prefix}\nMSE={metrics['mse']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | "
        f"R2={metrics['r2']:.4f}"
    )
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_error_histogram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    title_prefix: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if len(y_true) == 0:
        return

    err = y_pred - y_true

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(err, bins=50, alpha=0.85)
    ax.set_title(f"{title_prefix} error distribution")
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_metric(
    summary_rows: Sequence[Dict],
    metric: str,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not summary_rows:
        return

    labels = [
        f"{row['group']}\n{row['cycle']}"
        for row in summary_rows
    ]

    means = np.asarray(
        [row[f"{metric}_mean"] for row in summary_rows],
        dtype=np.float64,
    )
    stds = np.asarray(
        [row[f"{metric}_std"] for row in summary_rows],
        dtype=np.float64,
    )

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(labels)), 6))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_title(f"Aggregated cross-validation summary: {metric}")
    ax.set_xlabel("Group / cycle")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cycle_comparison(
    summary_rows: Sequence[Dict],
    metric: str,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if not summary_rows:
        return

    groups = sorted(set(row["group"] for row in summary_rows))
    cycles = sorted(set(row["cycle"] for row in summary_rows))

    if len(cycles) < 2:
        return

    by_key = {
        (row["group"], row["cycle"]): row
        for row in summary_rows
    }

    x = np.arange(len(groups))
    width = 0.8 / len(cycles)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, cycle in enumerate(cycles):
        values = []
        errors = []

        for group in groups:
            row = by_key.get((group, cycle))

            if row is None:
                values.append(np.nan)
                errors.append(0.0)
            else:
                values.append(row[f"{metric}_mean"])
                errors.append(row[f"{metric}_std"])

        offset = (i - (len(cycles) - 1) / 2.0) * width

        ax.bar(
            x + offset,
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=cycle,
        )

    ax.set_title(f"Frame vs sequence comparison: {metric}")
    ax.set_xlabel("Aggregated group")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_summary_plots(summary_rows: Sequence[Dict], output_dir: str) -> None:
    plot_dir = os.path.join(output_dir, "summary_plots")
    metrics = ["test_mse", "test_rmse", "test_mae", "test_r2"]

    for metric in metrics:
        plot_summary_metric(
            summary_rows,
            metric,
            os.path.join(plot_dir, f"{metric}_summary.png"),
        )
        plot_cycle_comparison(
            summary_rows,
            metric,
            os.path.join(plot_dir, f"{metric}_cycle_comparison.png"),
        )


# =============================================================================
# Fold / cycle runner
# =============================================================================

def train_one_fold(
    group_data: Dict,
    cycle_name: str,
    seq_len: int,
    fold_idx: int,
    train_ends: np.ndarray,
    test_ends: np.ndarray,
    args,
    device,
) -> Dict:
    group_name = group_data["group"]

    print()
    print("=" * 80)
    print(
        f"Group: {group_name} | Cycle: {cycle_name} | "
        f"seq_len={seq_len} | Fold {fold_idx + 1}/{args.n_folds}"
    )
    print("=" * 80)

    train_ends, val_ends = split_train_val_end_indices(
        train_ends=train_ends,
        val_ratio_within_train=args.val_ratio_within_train,
        seed=args.seed + fold_idx,
    )

    x_arr = group_data["x"]
    y_arr = group_data["y"]
    local_frames = group_data["local_frames"]
    stimulus_ids = group_data["stimulus_ids"]
    global_indices = group_data["global_indices"]

    print_target_stats("Train", y_arr, train_ends)

    if val_ends is not None:
        print_target_stats("Val", y_arr, val_ends)

    print_target_stats("Test", y_arr, test_ends)

    train_ds = AggregatedSequenceDataset(
        x_arr=x_arr,
        y_arr=y_arr,
        local_frames=local_frames,
        stimulus_ids=stimulus_ids,
        global_indices=global_indices,
        end_indices=train_ends,
        seq_len=seq_len,
    )

    test_ds = AggregatedSequenceDataset(
        x_arr=x_arr,
        y_arr=y_arr,
        local_frames=local_frames,
        stimulus_ids=stimulus_ids,
        global_indices=global_indices,
        end_indices=test_ends,
        seq_len=seq_len,
    )

    if len(train_ds) == 0:
        raise RuntimeError("Training dataset has 0 samples.")

    if len(test_ds) == 0:
        raise RuntimeError("Test dataset has 0 samples.")

    x_mean, x_std = compute_feature_stats(
        train_ds,
        batch_size=args.batch_size,
    )

    train_ds.set_feature_stats(x_mean, x_std)
    test_ds.set_feature_stats(x_mean, x_std)

    if val_ends is not None:
        val_ds = AggregatedSequenceDataset(
            x_arr=x_arr,
            y_arr=y_arr,
            local_frames=local_frames,
            stimulus_ids=stimulus_ids,
            global_indices=global_indices,
            end_indices=val_ends,
            seq_len=seq_len,
        )
        val_ds.set_feature_stats(x_mean, x_std)
    else:
        val_ds = None

    print()
    print("Dataset sizes")
    print(f"Train samples: {len(train_ds)}")
    if val_ds is not None:
        print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    train_loader = build_loader(
        train_ds,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    monitor_loader = build_loader(
        val_ds if val_ds is not None else train_ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = build_loader(
        test_ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = make_model(args, device)

    print()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    loss_fn = nn.MSELoss()

    fold_dir = os.path.join(
        args.output_dir,
        group_name,
        cycle_name,
        f"fold_{fold_idx + 1:02d}",
    )

    checkpoint_path = os.path.join(
        fold_dir,
        "checkpoints",
        "best_model.pt",
    )

    plot_dir = os.path.join(fold_dir, "plots")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    best_monitor_loss = float("inf")
    monitor_name = "val" if val_ds is not None else "train"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
        )

        monitor_loss = evaluate(
            model,
            monitor_loader,
            loss_fn,
            device,
        )

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
                "group": group_name,
                "cycle": cycle_name,
                "seq_len": seq_len,
                "fold": fold_idx + 1,
                "stimulus_names": group_data["stimulus_names"],
                "segments": group_data["segments"],
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
        fieldnames=[
            "epoch",
            "train_mse",
            "monitor_mse",
            "monitor_name",
            "lr",
        ],
    )

    print()
    print(f"Best {monitor_name} MSE: {best_monitor_loss:.6f}")
    print(f"Saved best model to: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    (
        global_indices_out,
        local_frames_out,
        stimulus_ids_out,
        y_true,
        y_pred,
    ) = collect_predictions(model, test_loader, device)

    metrics = compute_regression_metrics(y_true, y_pred)

    print()
    print("Test results")
    print(f"MSE:  {metrics['mse']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE:  {metrics['mae']:.6f}")
    print(f"R2:   {metrics['r2']:.6f}")

    prediction_csv_path = os.path.join(fold_dir, "test_predictions.csv")

    save_predictions(
        global_indices=global_indices_out,
        local_frames=local_frames_out,
        stimulus_ids=stimulus_ids_out,
        stimulus_names=group_data["stimulus_names"],
        y_true=y_true,
        y_pred=y_pred,
        out_path=prediction_csv_path,
    )

    if args.save_plots:
        title_prefix = f"{group_name} | {cycle_name} | fold {fold_idx + 1:02d}"

        plot_loss_curve(
            history,
            os.path.join(plot_dir, "loss_curve.png"),
        )

        plot_lr_curve(
            history,
            os.path.join(plot_dir, "lr_curve.png"),
        )

        plot_predictions(
            local_frames=local_frames_out,
            stimulus_ids=stimulus_ids_out,
            stimulus_names=group_data["stimulus_names"],
            y_true=y_true,
            y_pred=y_pred,
            out_path=os.path.join(plot_dir, "test_predictions_chunk.png"),
            chunk_size=args.plot_chunk_size,
            title_prefix=title_prefix,
        )

        plot_predictions(
            local_frames=local_frames_out,
            stimulus_ids=stimulus_ids_out,
            stimulus_names=group_data["stimulus_names"],
            y_true=y_true,
            y_pred=y_pred,
            out_path=os.path.join(plot_dir, "test_predictions_full.png"),
            chunk_size=len(y_true),
            title_prefix=title_prefix,
        )

        plot_scatter(
            y_true,
            y_pred,
            os.path.join(plot_dir, "scatter.png"),
            title_prefix,
        )

        plot_error_histogram(
            y_true,
            y_pred,
            os.path.join(plot_dir, "error_histogram.png"),
            title_prefix,
        )

    return {
        "group": group_name,
        "cycle": cycle_name,
        "seq_len": seq_len,
        "fold": fold_idx + 1,
        "n_stimuli": len(group_data["stimulus_names"]),
        "stimuli": ";".join(group_data["stimulus_names"]),
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

    keys = sorted(
        set(
            (row["group"], row["cycle"], row["seq_len"])
            for row in fold_rows
        )
    )

    metric_names = [
        "test_mse",
        "test_rmse",
        "test_mae",
        "test_r2",
    ]

    count_names = [
        "n_train_samples",
        "n_val_samples",
        "n_test_samples",
    ]

    for group, cycle, seq_len in keys:
        rows = [
            row for row in fold_rows
            if row["group"] == group
            and row["cycle"] == cycle
            and row["seq_len"] == seq_len
        ]

        stimuli = rows[0]["stimuli"] if rows else ""

        out = {
            "group": group,
            "cycle": cycle,
            "seq_len": seq_len,
            "n_folds": len(rows),
            "n_stimuli": rows[0]["n_stimuli"] if rows else 0,
            "stimuli": stimuli,
        }

        for key in count_names:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.mean(values))
            out[f"{key}_std"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            )

        for key in metric_names:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            out[f"{key}_mean"] = float(np.nanmean(values))
            out[f"{key}_std"] = (
                float(np.nanstd(values, ddof=1))
                if len(values) > 1
                else 0.0
            )

        summary_rows.append(out)

    return summary_rows


def cycle_seq_lens(args) -> List[Tuple[str, int]]:
    out = []

    for cycle in args.cycles:
        if cycle == "frame":
            out.append(("frame", args.frame_seq_len))
        elif cycle == "seq5":
            out.append(("seq5", args.sequence_seq_len))
        else:
            raise RuntimeError(f"Unsupported cycle: {cycle}")

    return out


# =============================================================================
# CLI / main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--feature-dir", type=str, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--target-suffix", type=str, default="audio")
    parser.add_argument(
        "--target-ext",
        type=str,
        default="npy",
        choices=["npy", ".npy", "csv", ".csv"],
    )
    parser.add_argument("--csv-target-column", type=int, default=-1)

    parser.add_argument("--stimuli", type=str, nargs="*", default=None)

    parser.add_argument(
        "--groups",
        type=str,
        nargs="+",
        default=["pitch", "volume"],
        choices=["pitch", "volume"],
    )

    parser.add_argument(
        "--pitch-keywords",
        type=str,
        nargs="+",
        default=["pitch"],
        help="Stimulus-name keywords used to aggregate pitch data.",
    )

    parser.add_argument(
        "--volume-keywords",
        type=str,
        nargs="+",
        default=["volume"],
        help="Stimulus-name keywords used to aggregate volume data.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="random",
        choices=["chronological", "random"],
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--val-ratio-within-train", type=float, default=0.0)

    parser.add_argument(
        "--cycles",
        type=str,
        nargs="+",
        default=["frame", "seq5"],
        choices=["frame", "seq5"],
    )
    parser.add_argument("--frame-seq-len", type=int, default=1)
    parser.add_argument("--sequence-seq-len", type=int, default=5)

    parser.add_argument("--hand-fps", type=float, default=HAND_FPS)
    parser.add_argument("--target-fps", type=float, default=TARGET_FPS)
    parser.add_argument("--cv-fps", type=float, default=None)
    parser.add_argument("--max-nils-to-fill", type=int, default=MAX_NILS_TO_FILL)

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

    if args.cv_fps is not None:
        args.target_fps = args.cv_fps

    if args.frame_seq_len != 1:
        print(
            f"Warning: frame cycle is configured with "
            f"seq_len={args.frame_seq_len}, not 1."
        )

    if args.sequence_seq_len <= 1:
        print(
            f"Warning: seq5 cycle is configured with "
            f"seq_len={args.sequence_seq_len}."
        )

    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cycles = cycle_seq_lens(args)

    print()
    print("Training cycles")
    for cycle_name, seq_len in cycles:
        print(f"- {cycle_name}: seq_len={seq_len}")

    pairs = discover_stimuli(
        args.feature_dir,
        args.target_suffix,
        args.target_ext,
    )

    if args.stimuli:
        selected = set(args.stimuli)
        pairs = [pair for pair in pairs if pair["stimulus"] in selected]
        found = set(pair["stimulus"] for pair in pairs)
        missing = selected - found

        if missing:
            raise RuntimeError(f"Requested stimuli not found: {sorted(missing)}")

    print()
    print("Discovered stimulus pairs")
    for pair in pairs:
        print(f"- {pair['stimulus']}")
        print(f"  hand:   {pair['hand_path']}")
        print(f"  target: {pair['target_path']}")

    grouped_pairs = group_stimuli(
        pairs=pairs,
        pitch_keywords=args.pitch_keywords,
        volume_keywords=args.volume_keywords,
    )

    print()
    print("Aggregated groups")
    for group_name in ["pitch", "volume"]:
        print(f"- {group_name}: {len(grouped_pairs[group_name])} stimuli")
        for pair in grouped_pairs[group_name]:
            print(f"  - {pair['stimulus']}")

    all_fold_rows = []

    for group_name in args.groups:
        group_pairs = grouped_pairs[group_name]

        if not group_pairs:
            raise RuntimeError(
                f"Requested group {group_name!r}, but no matching stimuli were found."
            )

        group_data = build_aggregated_group_data(
            group_name=group_name,
            pairs=group_pairs,
            args=args,
        )

        for cycle_name, seq_len in cycles:
            print()
            print("#" * 80)
            print(
                f"Starting aggregated training | group={group_name} | "
                f"cycle={cycle_name} | seq_len={seq_len}"
            )
            print("#" * 80)

            fold_indices = make_kfold_end_indices_from_segments(
                segments=group_data["segments"],
                seq_len=seq_len,
                n_folds=args.n_folds,
                split=args.split,
                seed=args.seed,
            )

            for fold_idx, (train_ends, test_ends) in enumerate(fold_indices):
                set_seed(args.seed + 1000 * (fold_idx + 1) + 17 * seq_len)

                row = train_one_fold(
                    group_data=group_data,
                    cycle_name=cycle_name,
                    seq_len=seq_len,
                    fold_idx=fold_idx,
                    train_ends=train_ends,
                    test_ends=test_ends,
                    args=args,
                    device=device,
                )

                all_fold_rows.append(row)

    fold_fieldnames = [
        "group",
        "cycle",
        "seq_len",
        "fold",
        "n_stimuli",
        "stimuli",
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
        args.fold_results_csv = os.path.join(
            args.output_dir,
            "aggregated_crossval_fold_results.csv",
        )

    write_csv(
        args.fold_results_csv,
        all_fold_rows,
        fold_fieldnames,
    )

    summary_rows = summarize_fold_rows(all_fold_rows)

    summary_fieldnames = (
        list(summary_rows[0].keys())
        if summary_rows
        else ["group", "cycle", "seq_len", "n_folds"]
    )

    if args.summary_csv is None:
        args.summary_csv = os.path.join(
            args.output_dir,
            "aggregated_crossval_summary.csv",
        )

    write_csv(
        args.summary_csv,
        summary_rows,
        summary_fieldnames,
    )

    if args.save_plots:
        save_summary_plots(summary_rows, args.output_dir)

    print()
    print("Final aggregated summary")

    for row in summary_rows:
        print(
            f"{row['group']} | {row['cycle']} | seq_len={row['seq_len']} | "
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