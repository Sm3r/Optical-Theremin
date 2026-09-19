"""Live control estimation from a running sensor.

zed.py    ZED stereo camera with MediaPipe landmarks (63 features per hand).
mocap.py  OptiTrack markers over NatNet, labelled on the fly (18 pitch, 9 volume).

Both load a checkpoint written by the matching train_* script and reuse its
x_mean/x_std, so the features they build match the ones the model was trained
on. Neither module is imported here: each pulls in a sensor SDK that is only
installed on the machine wired to that sensor.
"""
