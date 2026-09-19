#!/usr/bin/env python3
import os
import sys
import threading
import time

import cv2
import numpy as np
import pyzed.sl as sl
import torch
from dotenv import load_dotenv
from pythonosc.udp_client import SimpleUDPClient

# Add project root so that hand_tracking_ZED6D, train, config, etc. resolve.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hand_tracking_ZED6D.tracking import HandTracking
from hand_tracking_ZED6D.zed import Zed
from train.network import HandNet
from config import config


dotenv_path = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(dotenv_path)

SC_IP = "127.0.0.1"
SC_PORT = 57120

FEATURE_DIM = 63
MAX_MISSED_FRAMES = int(os.getenv("MAX_MISSED_FRAMES", "8"))

PITCH_CKPT = os.getenv(
    "PITCH_CHECKPOINT",
    os.path.join(PROJECT_ROOT, "checkpoints", "ZED_PITCH.pt"),
)

VOLUME_CKPT = os.getenv(
    "VOLUME_CHECKPOINT",
    os.path.join(PROJECT_ROOT, "checkpoints", "ZED_VOLUME.pt"),
)

LSTMState = tuple[torch.Tensor, torch.Tensor] | None

def _torch_load_checkpoint(checkpoint_path: str, device: torch.device) -> dict:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a dictionary: {checkpoint_path}")

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}"
        )

    if "x_mean" not in checkpoint or "x_std" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'x_mean' and 'x_std'. "
            "Use checkpoints produced by the training script, because inference "
            "must apply the same feature normalization used during training."
        )

    if "seq_len" not in checkpoint:
        raise RuntimeError(
            "Checkpoint does not contain 'seq_len'. "
            "Use checkpoints produced by the training script so inference can "
            "mirror the trained temporal context."
        )

    return checkpoint


def _to_1d_float_tensor(value, device: torch.device, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float().to(device)
    else:
        tensor = torch.tensor(value, dtype=torch.float32, device=device)

    tensor = tensor.reshape(-1)

    if tensor.numel() != FEATURE_DIM:
        raise RuntimeError(
            f"{name} must contain {FEATURE_DIM} values, got {tensor.numel()}."
        )

    return tensor


def _clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("_"):
            continue

        if key.startswith("module."):
            key = key[len("module.") :]

        cleaned[key] = value

    return cleaned


def _load_model_bundle(checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = _torch_load_checkpoint(checkpoint_path, device)

    args = checkpoint.get("args", {})
    if args is None:
        args = {}

    coord_mlp_dim = int(args.get("coord_mlp_dim", 128))
    hidden_dim = int(args.get("hidden_dim", 48))
    num_layers = int(args.get("num_layers", 1))
    dropout = float(args.get("dropout", 0.0))

    seq_len = int(checkpoint["seq_len"])
    if seq_len <= 0:
        raise RuntimeError(f"Invalid seq_len in checkpoint: {seq_len}")

    x_mean = _to_1d_float_tensor(checkpoint["x_mean"], device, "x_mean")
    x_std = _to_1d_float_tensor(checkpoint["x_std"], device, "x_std")
    x_std = torch.where(x_std < 1e-6, torch.ones_like(x_std), x_std)

    model = HandNet(
        input_dim=FEATURE_DIM,
        coord_mlp_dim=coord_mlp_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    state_dict = _clean_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Shape for broadcasting over [batch, seq_len, features].
    x_mean = x_mean.reshape(1, 1, FEATURE_DIM)
    x_std = x_std.reshape(1, 1, FEATURE_DIM)

    print()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  seq_len:       {seq_len}")
    print(f"  input_dim:     {FEATURE_DIM}")
    print(f"  coord_mlp_dim: {coord_mlp_dim}")
    print(f"  hidden_dim:    {hidden_dim}")
    print(f"  num_layers:    {num_layers}")
    print(f"  dropout:       {dropout}")

    return {
        "model": model,
        "seq_len": seq_len,
        "x_mean": x_mean,
        "x_std": x_std,
        "checkpoint_path": checkpoint_path,
    }


def _flatten_hand(hand_data) -> np.ndarray | None:
    if hand_data is None:
        return None

    arr = np.asarray(hand_data, dtype=np.float32)

    if arr.shape == (21, 3):
        arr = arr.reshape(-1)
    elif arr.shape == (FEATURE_DIM,):
        arr = arr.reshape(-1)
    else:
        return None

    if arr.shape[0] != FEATURE_DIM:
        return None

    if not np.all(np.isfinite(arr)):
        return None

    return arr.astype(np.float32)


def _detect_one_frame(cam: Zed, detector: HandTracking) -> tuple:
    err = cam.zed.grab(cam.runtime_parameters)
    if err != sl.ERROR_CODE.SUCCESS:
        return None, None, None

    cam.get_image()
    img = cam.img.copy()

    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    pcl = cam.point_cloud

    det_img = detector.findHands(img)
    left_data, right_data = detector.findpostion(det_img, pcl, cam.camera_params)

    right_feat = _flatten_hand(right_data)
    left_feat = _flatten_hand(left_data)

    return right_feat, left_feat, det_img


def _detect_one_frame_triangulation(
    cam: Zed,
    detector: HandTracking,
    right_detector: HandTracking | None = None,
) -> tuple:
    if right_detector is None:
        right_detector = detector

    result = detector.detect_stereo(cam, right_detector)
    if not result["success"]:
        return None, None, None

    det_img = detector.findHands(result["img_left"])

    right_feat = _flatten_hand(result["right_data"])
    left_feat = _flatten_hand(result["left_data"])

    return right_feat, left_feat, det_img


def _detach_lstm_state(state: LSTMState) -> LSTMState:
    if state is None:
        return None
    h, c = state
    return h.detach(), c.detach()


def _run_model_on_frame_streaming(
    bundle: dict,
    frame_np: np.ndarray,
    state: LSTMState,
    device: torch.device,
) -> tuple[float, LSTMState]:
    if frame_np.ndim != 1 or frame_np.shape[0] != FEATURE_DIM:
        raise RuntimeError(
            f"Expected frame shape [{FEATURE_DIM}], got {frame_np.shape}"
        )

    x = torch.as_tensor(frame_np, dtype=torch.float32, device=device).reshape(
        1, 1, FEATURE_DIM
    )

    x = (x - bundle["x_mean"]) / bundle["x_std"]

    model = bundle["model"]

    with torch.inference_mode():
        z = model.coord_encoder(x)
        lstm_out, next_state = model.lstm(z, state)
        last = lstm_out[:, -1, :]
        value = model.regressor(last).reshape(-1)[0].item()

    return float(value), _detach_lstm_state(next_state)


def _send_prediction(osc_client: SimpleUDPClient, osc_path: str, value: float) -> None:

    value = max(0.0, min(1.0, value))

    if osc_path == "/pitch":
        freq = value
        osc_client.send_message(osc_path, [freq])
    else:
        amp = value * 2.0
        osc_client.send_message(osc_path, [amp])


def _inference_thread(
    cam: Zed,
    detector: HandTracking,
    bundle: dict,
    hand_label: str,
    window_name: str,
    osc_path: str,
    device: torch.device,
    running: threading.Event,
    detect_fn=_detect_one_frame,
):
    seq_len = int(bundle["seq_len"])
    min_frames_before_output = max(1, seq_len)

    state: LSTMState = None
    frames_seen = 0
    missed_frames = 0
    last_feat: np.ndarray | None = None

    extract = (lambda r, l: r) if hand_label == "right" else (lambda r, l: l)

    osc_client = SimpleUDPClient(SC_IP, SC_PORT)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while running.is_set():
        right_feat, left_feat, det_img = detect_fn(cam, detector)
        hand_feat = extract(right_feat, left_feat)

        if hand_feat is not None:
            last_feat = hand_feat.copy()
            missed_frames = 0
        else:
            missed_frames += 1

            if last_feat is None or missed_frames > MAX_MISSED_FRAMES:
                state = None
                frames_seen = 0
                last_feat = None
                if det_img is not None:
                    cv2.imshow(window_name, det_img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    running.clear()
                    break
                continue

            hand_feat = last_feat.copy()

        value, state = _run_model_on_frame_streaming(
            bundle=bundle,
            frame_np=hand_feat,
            state=state,
            device=device,
        )
        frames_seen += 1

        if frames_seen >= min_frames_before_output:
            _send_prediction(osc_client, osc_path, value)

        if det_img is not None:
            cv2.imshow(window_name, det_img)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            running.clear()
            break


def _auto_detect_hands(cams, detectors, detect_fn, serials):
    cam_hand = [None, None]

    print()
    print("=== AUTO-DETECT HANDS ===")
    print("Show one hand in each camera. Detection runs until both hands are found.")
    print("Press Ctrl+C to abort.")
    print()

    while True:
        any_new = False

        for i, (cam, det) in enumerate(zip(cams, detectors)):
            if cam_hand[i] is not None:
                continue

            right_feat, left_feat, _ = detect_fn(cam, det)

            if right_feat is not None:
                cam_hand[i] = "right"
                print(f"Camera {i} (serial {serials[i]}) -> right hand")
                any_new = True

            elif left_feat is not None:
                cam_hand[i] = "left"
                print(f"Camera {i} (serial {serials[i]}) -> left hand")
                any_new = True

        if all(cam_hand):
            break

        status = []
        for i in range(2):
            if cam_hand[i] is None:
                status.append(f"Camera {i}: waiting")
            else:
                status.append(f"Camera {i}: {cam_hand[i]} hand")

        if not any_new:
            print(f"\r  {' | '.join(status)}", end="", flush=True)

        time.sleep(0.005)

    print()

    if "right" in cam_hand and "left" in cam_hand:
        pitch_cam_idx = cam_hand.index("right")
        volume_cam_idx = cam_hand.index("left")

        print(
            f"Route: Camera {pitch_cam_idx} right hand -> Pitch model, "
            f"Camera {volume_cam_idx} left hand -> Volume model"
        )
    else:
        print(f"Both cameras detected the same hand type: {cam_hand[0]} / {cam_hand[1]}")
        print("Falling back to camera index: Camera 0 -> Pitch, Camera 1 -> Volume")
        pitch_cam_idx = 0
        volume_cam_idx = 1

    return pitch_cam_idx, volume_cam_idx


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print()
    print("Loading pitch model...")
    pitch_bundle = _load_model_bundle(PITCH_CKPT, device)

    print()
    print("Loading volume model...")
    volume_bundle = _load_model_bundle(VOLUME_CKPT, device)

    serial_1 = int(os.getenv("ZED_SERIAL_1"))
    serial_2 = int(os.getenv("ZED_SERIAL_2"))
    serials = [serial_1, serial_2]

    print()
    print(f"Opening ZED camera 1, serial: {serial_1}")
    cam0 = Zed(None, camera_serial=serial_1, fps=config.rates.zed_fps)

    print(f"Opening ZED camera 2, serial: {serial_2}")
    cam1 = Zed(None, camera_serial=serial_2, fps=config.rates.zed_fps)

    cams = [cam0, cam1]

    detectors = [
        HandTracking(
            maxHands=1,
            detectionCon=0.2,
            trackCon=0.8,
            complexity=0,
            draw=False,
        )
        for _ in range(2)
    ]

    use_triangulation = config.depth_mode == "triangulation"

    if use_triangulation:
        left_detectors = [
            HandTracking(
                maxHands=1,
                detectionCon=0.2,
                trackCon=0.8,
                complexity=0,
                draw=False,
            )
            for _ in range(2)
        ]

        right_detectors = [
            HandTracking(
                maxHands=1,
                detectionCon=0.2,
                trackCon=0.8,
                complexity=0,
                draw=False,
            )
            for _ in range(2)
        ]

        detect_fn = _detect_one_frame_triangulation
    else:
        left_detectors = None
        right_detectors = None
        detect_fn = _detect_one_frame

    threads = []

    try:
        pitch_cam_idx, volume_cam_idx = _auto_detect_hands(
            cams=cams,
            detectors=detectors,
            detect_fn=detect_fn,
            serials=serials,
        )

        running = threading.Event()
        running.set()

        if use_triangulation:
            pitch_detect_fn = lambda cam, det: _detect_one_frame_triangulation(
                cam,
                det,
                right_detectors[pitch_cam_idx],
            )

            volume_detect_fn = lambda cam, det: _detect_one_frame_triangulation(
                cam,
                det,
                right_detectors[volume_cam_idx],
            )

            pitch_detector = left_detectors[pitch_cam_idx]
            volume_detector = left_detectors[volume_cam_idx]

        else:
            pitch_detect_fn = detect_fn
            volume_detect_fn = detect_fn
            pitch_detector = detectors[pitch_cam_idx]
            volume_detector = detectors[volume_cam_idx]

        pitch_thread = threading.Thread(
            target=_inference_thread,
            args=(
                cams[pitch_cam_idx],
                pitch_detector,
                pitch_bundle,
                "right",
                "Pitch Camera - right hand - press Q to quit",
                "/pitch",
                device,
                running,
                pitch_detect_fn,
            ),
            daemon=True,
        )

        volume_thread = threading.Thread(
            target=_inference_thread,
            args=(
                cams[volume_cam_idx],
                volume_detector,
                volume_bundle,
                "left",
                "Volume Camera - left hand - press Q to quit",
                "/volume",
                device,
                running,
                volume_detect_fn,
            ),
            daemon=True,
        )

        pitch_thread.start()
        volume_thread.start()
        threads.extend([pitch_thread, volume_thread])

        print()
        print("Real-time prediction running. Press Q in either window or Ctrl+C to stop.")
        print()

        while running.is_set():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print()
        print("Stopping...")

    finally:
        try:
            running.clear()
        except NameError:
            pass

        for thread in threads:
            thread.join(timeout=2)

        for cam in cams:
            cam.zed.close()

        cv2.destroyAllWindows()
        print("Cameras closed. Done.")


if __name__ == "__main__":
    main()
