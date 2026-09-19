"""Training helpers shared by the per-modality entry points.

These functions are identical across train_zed.py and train_mocap.py:
checkpoint IO, metrics, normalisation and the plotting helpers. Keep them
free of module-level configuration so every entry point can reuse them.
"""

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch

def assert_not_lfs_pointer(path: str) -> None:
    with open(path, "rb") as f:
        head = f.read(128)

    if head.startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            "This file is a Git LFS pointer, not a real data file:\n"
            f"  {path}\n"
            "Download the real LFS object first, then rerun this script."
        )

def build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )

def clean_target_array(target_arr: np.ndarray) -> np.ndarray:
    if target_arr.ndim == 2 and target_arr.shape[1] == 1:
        target_arr = target_arr[:, 0]

    if target_arr.ndim != 1:
        raise RuntimeError(f"Expected target array [frames] or [frames, 1], got {target_arr.shape}")

    print()
    print("Target cleaning")
    print(f"Original target frames: {len(target_arr)}")
    print(f"Target NaNs:            {int(np.isnan(target_arr).sum())}")

    return target_arr.astype(np.float32)

def coerce_array_to_float32(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.shape == ():
        arr = arr.item()

    arr = np.asarray(arr)

    if arr.dtype == object:
        arr = np.where(arr == "nil", np.nan, arr)
        arr = np.where(arr == "None", np.nan, arr)
        arr = np.where(arr == "", np.nan, arr)

    return arr.astype(np.float32)

def collect_predictions(model, loader, device):
    model.eval()

    all_frames = []
    all_y_true = []
    all_y_pred = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        frames = batch["frame"].cpu().numpy()
        y_hat = model(x).reshape_as(batch["y"].to(device)).cpu().numpy()

        all_frames.append(frames)
        all_y_true.append(y)
        all_y_pred.append(y_hat)

    if not all_frames:
        return np.array([]), np.array([]), np.array([])

    frames = np.concatenate(all_frames)
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    order = np.argsort(frames)
    return frames[order], y_true[order], y_pred[order]

def compute_feature_stats(dataset: Dataset, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray]:
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

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mse": float("nan"), "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}

    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(mse))

    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom < 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

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

def get_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])

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

def load_npy_array(path: str) -> np.ndarray:
    assert_not_lfs_pointer(path)
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception as exc:
        raise RuntimeError(f"Could not load NumPy file: {path}\nOriginal error: {exc}") from exc
    return coerce_array_to_float32(arr)

def make_kfold_end_indices(n_frames: int, seq_len: int, n_folds: int, split: str, seed: int):
    if n_folds < 2:
        raise RuntimeError(f"n_folds must be >= 2, got {n_folds}")
    if n_frames < seq_len:
        raise RuntimeError(f"Not enough frames for seq_len={seq_len}: got {n_frames}")

    valid_ends = np.arange(seq_len - 1, n_frames, dtype=np.int64)

    if len(valid_ends) < n_folds:
        raise RuntimeError(f"Cannot create {n_folds} folds from only {len(valid_ends)} samples")

    if split == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(valid_ends)
    elif split == "chronological":
        pass
    else:
        raise RuntimeError(f"Unknown split mode: {split}")

    folds = np.array_split(valid_ends, n_folds)
    out = []

    for fold_idx in range(n_folds):
        test_ends = np.sort(folds[fold_idx])
        train_ends = np.sort(np.concatenate([folds[i] for i in range(n_folds) if i != fold_idx]))
        out.append((train_ends, test_ends))

    return out

def plot_cycle_comparison(summary_rows: Sequence[Dict], metric: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not summary_rows:
        return

    stimuli = sorted(set(row["stimulus"] for row in summary_rows))
    cycles = sorted(set(row["cycle"] for row in summary_rows))
    if len(cycles) < 2:
        return

    by_key = {(row["stimulus"], row["cycle"]): row for row in summary_rows}
    x = np.arange(len(stimuli))
    width = 0.8 / len(cycles)

    fig, ax = plt.subplots(figsize=(max(12, 0.7 * len(stimuli)), 6))
    for i, cycle in enumerate(cycles):
        values = []
        errors = []
        for stimulus in stimuli:
            row = by_key.get((stimulus, cycle))
            if row is None:
                values.append(np.nan)
                errors.append(0.0)
            else:
                values.append(row[f"{metric}_mean"])
                errors.append(row[f"{metric}_std"])
        offset = (i - (len(cycles) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, yerr=errors, capsize=3, label=cycle)

    ax.set_title(f"Frame vs sequence comparison: {metric}")
    ax.set_xlabel("Stimulus")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(stimuli, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_error_histogram(y_true, y_pred, out_path: str, title_prefix: str) -> None:
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

def plot_predictions(frames, y_true, y_pred, out_path: str, chunk_size: int, title_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(frames) == 0:
        return

    metrics = compute_regression_metrics(y_true, y_pred)
    chunk_size = min(chunk_size, len(frames))

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(frames[:chunk_size], y_true[:chunk_size], linewidth=2.0, label="Ground truth")
    ax.plot(frames[:chunk_size], y_pred[:chunk_size], linewidth=2.0, linestyle="--", label="Prediction")
    ax.set_title(
        f"{title_prefix} | MSE={metrics['mse']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | R2={metrics['r2']:.4f}"
    )
    ax.set_xlabel("Frame")
    ax.set_ylabel("Target")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_scatter(y_true, y_pred, out_path: str, title_prefix: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if len(y_true) == 0:
        return

    metrics = compute_regression_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.45, s=18)

    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2.0)

    ax.set_title(
        f"{title_prefix}\nMSE={metrics['mse']:.6f} | "
        f"RMSE={metrics['rmse']:.6f} | MAE={metrics['mae']:.6f} | R2={metrics['r2']:.4f}"
    )
    ax.set_xlabel("Ground truth")
    ax.set_ylabel("Prediction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_summary_metric(summary_rows: Sequence[Dict], metric: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not summary_rows:
        return

    labels = [f"{row['stimulus']}\n{row['cycle']}" for row in summary_rows]
    means = np.asarray([row[f"{metric}_mean"] for row in summary_rows], dtype=np.float64)
    stds = np.asarray([row[f"{metric}_std"] for row in summary_rows], dtype=np.float64)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(12, 0.75 * len(labels)), 6))
    ax.bar(x, means, yerr=stds, capsize=3)
    ax.set_title(f"Cross-validation summary: {metric}")
    ax.set_xlabel("Stimulus / cycle")
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def print_target_stats(name: str, y: np.ndarray, end_indices: np.ndarray) -> None:
    values = y[np.asarray(end_indices, dtype=np.int64)]
    print()
    print(f"{name} stats")
    print(f"samples: {len(values)}")
    print(f"y min:   {np.min(values):.6f}")
    print(f"y max:   {np.max(values):.6f}")
    print(f"y mean:  {np.mean(values):.6f}")
    print(f"y std:   {np.std(values):.6f}")

def save_predictions(frames, y_true, y_pred, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    data = np.column_stack([frames, y_true, y_pred, y_pred - y_true])
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header="Frame,GroundTruth,Prediction,Error",
        comments="",
        fmt=["%d", "%.8f", "%.8f", "%.8f"],
    )
    print(f"Prediction CSV saved to: {out_path}")

def save_summary_plots(summary_rows: Sequence[Dict], output_dir: str) -> None:
    plot_dir = os.path.join(output_dir, "summary_plots")
    metrics = ["test_mse", "test_rmse", "test_mae", "test_r2"]
    for metric in metrics:
        plot_summary_metric(summary_rows, metric, os.path.join(plot_dir, f"{metric}_summary.png"))
        plot_cycle_comparison(summary_rows, metric, os.path.join(plot_dir, f"{metric}_cycle_comparison.png"))

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def write_csv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"CSV saved to: {path}")
