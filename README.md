# Optical Theremin

Code, data and pretrained models for the paper **Issues and Challenges in Vision-Based Mid-Air Musical Instruments: Control Estimation in an Optical Theremin** (IEEE IS2 2026).

A small LSTM estimates the pitch and volume control voltages (CVs) of a Moog Etherwave Theremin from hand trajectories tracked with a ZED 2i stereo camera or an OptiTrack motion capture system.

## Installation

```bash
pip install -r requirements.txt
```

The ZED scripts also need the ZED SDK and its Python API (`pyzed`).

## Data

`data/features/` contains nine takes (five pitch, four volume) recorded by a single performer:

- `<take>_audio.npy`: normalized CV at 60 Hz (target)
- `<take>_hand.npy`: 21 MediaPipe landmarks of the playing hand in 3D from the ZED, 30 fps
- `mocap/<take>_cleaned.csv`: OptiTrack markers of the hands, antennas and cameras, 360 Hz, in millimeters

## Training

```bash
python train/train_mocap.py
python train/train_zed_aggregated.py --feature-dir data/features --output-dir runs_zed
```

Both scripts train single-frame and five-frame models with 5-fold cross-validation and write checkpoints, predictions and summaries to their output folder. Run them with `--help` for all options.

## Pretrained models

`checkpoints/` holds the five-frame models used in the paper: `ZED_PITCH.pt`, `ZED_VOLUME.pt`, `MOCAP_PITCH.pt` and `MOCAP_VOLUME.pt`.

## Live inference

1. Run `audio_utils/ThereSynth.scd` in SuperCollider, after setting the output device on its first line. The synth listens for OSC messages on `/pitch` and `/volume`, port 57120.
2. Start one pipeline:

```bash
python inference/zed.py      # two ZED cameras, serials set in .env as ZED_SERIAL_1 and ZED_SERIAL_2
python inference/mocap.py    # OptiTrack Motive streaming over NatNet
```

## Citation

```bibtex
@inproceedings{sguario2026optical,
  author    = {Sguario, Mario and Garau, Nicola and Conci, Nicola and Dal R{\`i}, Francesco Ardan},
  title     = {Issues and Challenges in Vision-Based Mid-Air Musical Instruments: Control Estimation in an Optical Theremin},
  booktitle = {IEEE International Symposium on the Internet of Sounds (IS2)},
  year      = {2026}
}
```

## License

MIT, except `mocap_tools/natnet/` (Apache-2.0, NaturalPoint).
