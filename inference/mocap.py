#!/usr/bin/env python3
"""Cluster unlabeled markers into pitch (×6) and volume (×3) groups and run live inference."""
#  py .\inference\mocap.py --no-multicast --osc-port 57121 --no-plot

import argparse
import os
import sys
import threading
import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pythonosc.udp_client import SimpleUDPClient

from train.network import HandNet
from mocap_tools.natnet.NatNetClient import NatNetClient

_DEFAULT_ANTENNA = {
    "pitch": {
        "center": np.array([3569.959, 1294.028, 353.822], dtype=np.float32),
        "scale": 420.340,
    },
    "volume": {
        "center": np.array([3548.409, 1046.094, -170.730], dtype=np.float32),
        "scale": 225.582,
    },
}


def load_checkpoint(path, device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    needed = ["model_state_dict", "input_dim", "seq_len", "x_mean", "x_std"]
    for key in needed:
        if key not in ckpt:
            raise RuntimeError(f"Checkpoint {path} missing key: {key}")

    args = ckpt.get("args", {}) or {}
    model = HandNet(
        input_dim=ckpt["input_dim"],
        coord_mlp_dim=int(args.get("coord_mlp_dim", 128)),
        hidden_dim=int(args.get("hidden_dim", 48)),
        num_layers=int(args.get("num_layers", 1)),
        dropout=float(args.get("dropout", 0.2)),
    ).to(device)

    state_dict = ckpt["model_state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_"):
            continue
        cleaned[k[len("module."):] if k.startswith("module.") else k] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    x_mean = ckpt["x_mean"].float().to(device).reshape(1, 1, -1)
    x_std = ckpt["x_std"].float().to(device).reshape(1, 1, -1)
    x_std = torch.where(x_std < 1e-6, torch.ones_like(x_std), x_std)

    max_markers = ckpt["input_dim"] // 3
    print(f"  input_dim: {ckpt['input_dim']} ({max_markers} markers)")
    print(f"  seq_len:   {ckpt['seq_len']}")
    print(f"  category:  {ckpt.get('category', '?')}")
    print(f"  stimulus:  {ckpt.get('stimulus', '?')}")
    return {"model": model, "seq_len": ckpt["seq_len"],
            "input_dim": ckpt["input_dim"], "max_markers": max_markers,
            "x_mean": x_mean, "x_std": x_std}


def cluster_markers(marker_dict, n):
    ids = list(marker_dict.keys())
    if len(ids) < n:
        return None
    pos = np.array([marker_dict[mid] for mid in ids])

    best_combo = None
    best_cost = float("inf")

    for combo in combinations(range(len(ids)), n):
        pts = pos[list(combo)]
        cost = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                cost += np.linalg.norm(pts[i] - pts[j])
        if cost < best_cost:
            best_cost = cost
            best_combo = combo

    return sorted([ids[i] for i in best_combo])


def get_ordered_positions(marker_dict, id_list, max_markers):
    n = len(id_list)
    positions = np.zeros((max_markers, 3), dtype=np.float32)
    for i, mid in enumerate(id_list):
        if i < max_markers and mid in marker_dict:
            positions[i] = marker_dict[mid]
    return positions


def _detach_state(state):
    if state is None:
        return None
    h, c = state
    return (h.detach(), c.detach())


def main():
    parser = argparse.ArgumentParser(
        description="Cluster unlabeled markers and run live MOCAP inference"
    )
    parser.add_argument("--pitch-checkpoint", default=None)
    parser.add_argument("--volume-checkpoint", default=None)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--client-ip", default="127.0.0.1")
    parser.add_argument("--multicast", action="store_true", default=True)
    parser.add_argument("--no-multicast", dest="multicast", action="store_false")
    parser.add_argument("--max-missed", type=int, default=30)
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=57120)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    if not args.pitch_checkpoint or not args.volume_checkpoint:
        ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")
        fallbacks = [
            ("MOCAP_PITCH.pt", "pitch_checkpoint"),
            ("MOCAP_VOLUME.pt", "volume_checkpoint"),
        ]
        for name, attr in fallbacks:
            if getattr(args, attr) is None:
                path = os.path.join(ckpt_dir, name)
                if os.path.exists(path):
                    setattr(args, attr, path)
        if not args.pitch_checkpoint:
            print("ERROR: Pitch checkpoint not found.")
            return 1
        if not args.volume_checkpoint:
            print("ERROR: Volume checkpoint not found.")
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading pitch checkpoint:")
    pitch_bundle = load_checkpoint(args.pitch_checkpoint, device)
    print("\nLoading volume checkpoint:")
    volume_bundle = load_checkpoint(args.volume_checkpoint, device)

    targets = [
        ("pitch", pitch_bundle, pitch_bundle["max_markers"]),
        ("volume", volume_bundle, volume_bundle["max_markers"]),
    ]

    for target_name, bundle, n_cluster in targets:
        dummy = torch.zeros(1, 1, bundle["input_dim"], device=device)
        with torch.inference_mode():
            bundle["model"](dummy)
    print("  Model warmup done.")

    frame_queue = []
    frame_lock = threading.Lock()
    frame_event = threading.Event()

    client = NatNetClient()
    client.set_client_address(args.client_ip)
    client.set_server_address(args.server_ip)
    client.set_use_multicast(args.multicast)

    def on_mocap_data(mocap_data):
        with frame_lock:
            frame_queue.append(mocap_data)
            frame_event.set()

    client.mocap_data_listener = on_mocap_data

    if not client.run():
        print("ERROR: Could not start streaming client.")
        return 1
    time.sleep(0.5)
    if not client.connected():
        print("ERROR: Could not connect to Motive.")
        client.shutdown()
        return 1

    print(f"\nConnected to {client.get_application_name()}")
    osc = SimpleUDPClient(args.osc_host, args.osc_port)
    print(f"OSC -> {args.osc_host}:{args.osc_port}")
    print("Waiting for labeled markers to cluster...\n")

    marker_counts = []

    lstm_states = {t: None for t, _, _ in targets}
    missed = {t: 0 for t, _, _ in targets}
    frame_count = 0
    latencies = {"frame_get": [], "build": [], "cluster": [], "infer": [], "osc": [], "plot": []}
    fig = None
    ax = None
    plot_init_done = False

    FRAME_INTERVAL = 1.0 / 30.0
    next_tick = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
                continue
            next_tick += FRAME_INTERVAL

            t0 = time.perf_counter()
            with frame_lock:
                if frame_queue:
                    mocap_data = frame_queue[-1]
                    frame_queue.clear()
                    frame_event.clear()
                else:
                    mocap_data = None

            if mocap_data is None:
                continue
            latencies["frame_get"].append(time.perf_counter() - t0)

            fn = mocap_data.prefix_data.frame_number
            lmd = mocap_data.labeled_marker_data
            if lmd is None or not lmd.labeled_marker_list:
                continue

            t0 = time.perf_counter()
            marker_dict = {}
            for lm in lmd.labeled_marker_list:
                marker_dict[lm.id_num] = np.array(lm.pos, dtype=np.float32) * 1000.0
            marker_counts.append(len(marker_dict))
            latencies["build"].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            if len(marker_dict) >= 9:
                pitch_ids = cluster_markers(marker_dict, 6)
                if pitch_ids is not None:
                    remaining = {k: v for k, v in marker_dict.items() if k not in pitch_ids}
                    volume_ids = cluster_markers(remaining, 3)
                else:
                    volume_ids = None
            else:
                pitch_ids = None
                volume_ids = None
            latencies["cluster"].append(time.perf_counter() - t0)

            _t = time.perf_counter()
            if args.plot and not plot_init_done and pitch_ids is not None and volume_ids is not None:
                plot_init_done = True
                plt.ion()
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                pitch_pts = np.array([marker_dict[mid] for mid in pitch_ids])
                volume_pts = np.array([marker_dict[mid] for mid in volume_ids])
                ax.scatter(pitch_pts[:, 0], pitch_pts[:, 2], pitch_pts[:, 1],
                           c='red', marker='o', s=60, label='Pitch')
                ax.scatter(volume_pts[:, 0], volume_pts[:, 2], volume_pts[:, 1],
                           c='blue', marker='o', s=60, label='Volume')
                for name, color in [('pitch', 'orange'), ('volume', 'cyan')]:
                    c = _DEFAULT_ANTENNA[name]['center']
                    ax.scatter(c[0], c[2], c[1], c=color, marker='D', s=120,
                               label=f'{name} antenna')
                ax.set_xlabel('X')
                ax.set_ylabel('Z')
                ax.set_zlabel('Y (height)')
                ax.set_xlim(3000, 4100)
                ax.set_ylim(-700, 900)
                ax.set_zlim(500, 1800)
                ax.legend()
                fig.tight_layout()
                fig.canvas.draw()
                latencies["plot"].append(time.perf_counter() - _t)
            elif args.plot and plot_init_done and frame_count % 10 == 1 \
                    and pitch_ids is not None and volume_ids is not None:
                ax.clear()
                for name, ids, color in [("pitch", pitch_ids, "red"),
                                          ("volume", volume_ids, "blue")]:
                    pts = np.array([marker_dict.get(mid, [np.nan] * 3)
                                    for mid in ids])
                    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1],
                               c=color, marker='o', s=60, label=name.title())
                for name, color in [("pitch", "orange"), ("volume", "cyan")]:
                    c = _DEFAULT_ANTENNA[name]["center"]
                    ax.scatter(c[0], c[2], c[1], c=color, marker="D", s=120,
                               label=f"{name} antenna")
                ax.set_xlabel("X")
                ax.set_ylabel("Z")
                ax.set_zlabel("Y (height)")
                ax.set_xlim(3000, 4100)
                ax.set_ylim(-700, 900)
                ax.set_zlim(500, 1800)
                ax.legend()
                fig.canvas.draw()
                plt.pause(0.001)
                latencies["plot"].append(time.perf_counter() - _t)

            t0 = time.perf_counter()
            outputs = {}
            id_map = {"pitch": pitch_ids, "volume": volume_ids}

            for target_name, bundle, n_cluster in targets:
                ids = id_map[target_name]
                if ids is None:
                    missed[target_name] += 1
                    if missed[target_name] > args.max_missed:
                        lstm_states[target_name] = None
                    outputs[target_name] = None
                    continue

                missed[target_name] = 0
                hand_xyz = get_ordered_positions(marker_dict, ids, n_cluster)
                defaults = _DEFAULT_ANTENNA[target_name]
                relative = (hand_xyz - defaults["center"].reshape(1, 3)) / defaults["scale"]
                features = relative.reshape(-1)

                x = torch.as_tensor(features, dtype=torch.float32, device=device)
                x = x.reshape(1, 1, -1)
                x = (x - bundle["x_mean"]) / bundle["x_std"]

                state = lstm_states[target_name]
                with torch.inference_mode():
                    z = bundle["model"].coord_encoder(x)
                    lstm_out, next_state = bundle["model"].lstm(z, state)
                    value = bundle["model"].regressor(lstm_out[:, -1, :]).item()
                lstm_states[target_name] = _detach_state(next_state)
                outputs[target_name] = value
            latencies["infer"].append(time.perf_counter() - t0)

            frame_count += 1
            t0 = time.perf_counter()
            for t, _, _ in targets:
                v = outputs[t]
                if v is not None:
                    osc.send_message(f"/{t}", [v])
            latencies["osc"].append(time.perf_counter() - t0)

            if frame_count % 30 == 0:
                for t, _, _ in targets:
                    v = outputs[t]
                    ids = id_map[t]
                    if ids is not None:
                        positions = {f"m{mid}": f"({marker_dict[mid][0]:.2f},{marker_dict[mid][1]:.2f},{marker_dict[mid][2]:.2f})"
                                     for mid in ids if mid in marker_dict}
                        ids_str = " ".join(f"m{mid}" for mid in ids)
                        print(f"  {t.upper()} ids={ids_str}")
                        print(f"    positions: {positions}")
                        print(f"    model out: {v:.6f}  OSC /{t}: {v:.6f}" if v is not None else f"    model out: LOST")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if marker_counts:
            print(f"  Avg markers/frame: {np.mean(marker_counts):.1f}  "
                  f"(min={np.min(marker_counts)}, max={np.max(marker_counts)}, "
                  f"{len(marker_counts)} frames)")
        print("\nLatency breakdown (ms):")
        for step, vals in latencies.items():
            if vals:
                print(f"  {step:12s}  avg={np.mean(vals)*1000:.2f}  "
                      f"min={np.min(vals)*1000:.2f}  max={np.max(vals)*1000:.2f}  "
                      f"({len(vals)} samples)")
        client.shutdown()
        if args.plot:
            plt.ioff()
            plt.close("all")

    return 0


if __name__ == "__main__":
    sys.exit(main())
