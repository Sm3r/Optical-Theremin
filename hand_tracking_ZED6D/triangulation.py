import cv2
import numpy as np


def _detect_landmarks_2d(mp_hands, img):
    """Run MediaPipe on an image and return 2D pixel coordinates per hand.

    Parameters
    ----------
    mp_hands : mediapipe.solutions.hands.Hands
    img : ndarray (H, W, 3) BGR

    Returns
    -------
    left_pts : ndarray (21, 2) or empty
    right_pts : ndarray (21, 2) or empty
    """
    h, w, _ = img.shape
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    left_pts = np.array([])
    right_pts = np.array([])

    if results.multi_hand_landmarks:
        for i, lm in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[i].classification[0].index
            pts = np.array([[lm.landmark[j].x * w, lm.landmark[j].y * h]
                            for j in range(21)], dtype=np.float64)
            if handedness == 1:
                left_pts = pts
            else:
                right_pts = pts

    return left_pts, right_pts


def triangulate_hands(left_pts, right_pts, cam_left, cam_right, stereo_T):
    """Triangulate 3D positions from matched stereo 2D landmarks.

    Parameters
    ----------
    left_pts : ndarray (N, 2) — points in the left image
    right_pts : ndarray (N, 2) — corresponding points in the right image
    cam_left : CameraParameters — left camera intrinsics
    cam_right : CameraParameters — right camera intrinsics
    stereo_T : ndarray (4, 4) — left→right transformation matrix

    Returns
    -------
    ndarray (N, 3) — 3D points in the left-camera coordinate frame
    """
    K_left = np.array([[cam_left.fx, 0,           cam_left.cx],
                       [0,           cam_left.fy,  cam_left.cy],
                       [0,           0,            1]], dtype=np.float64)
    K_right = np.array([[cam_right.fx, 0,            cam_right.cx],
                        [0,            cam_right.fy, cam_right.cy],
                        [0,            0,            1]], dtype=np.float64)

    R = stereo_T[:3, :3]
    t = stereo_T[:3, 3]

    P_left = K_left @ np.column_stack([np.eye(3), np.zeros(3)])
    P_right = K_right @ np.column_stack([R, t])

    pts_4d = cv2.triangulatePoints(P_left, P_right,
                                   left_pts.T, right_pts.T)
    return (pts_4d[:3] / pts_4d[3]).T


def stereo_detect(mp_left, mp_right, img_left, img_right,
                  cam_left, cam_right, stereo_T):
    """Detect 2D landmarks in stereo image pair and triangulate 3D hands.

    Parameters
    ----------
    mp_left : mediapipe.solutions.hands.Hands — MediaPipe model for left image
    mp_right : mediapipe.solutions.hands.Hands — MediaPipe model for right image
    img_left : ndarray (H, W, 3) BGR — left camera image
    img_right : ndarray (H, W, 3) BGR — right camera image
    cam_left : CameraParameters — left camera intrinsics
    cam_right : CameraParameters — right camera intrinsics
    stereo_T : ndarray (4, 4) — left→right transformation matrix

    Returns
    -------
    dict with keys:
        left_data : ndarray (21, 3) or empty
        right_data : ndarray (21, 3) or empty
        mp_left_detected : bool
        mp_right_detected : bool
    """
    left_left, left_right = _detect_landmarks_2d(mp_left, img_left)
    right_left, right_right = _detect_landmarks_2d(mp_right, img_right)

    left_data = np.array([])
    right_data = np.array([])

    if left_right.shape == (21, 2) and right_right.shape == (21, 2):
        right_data = triangulate_hands(
            left_right, right_right,
            cam_left, cam_right, stereo_T,
        )

    if left_left.shape == (21, 2) and right_left.shape == (21, 2):
        left_data = triangulate_hands(
            left_left, right_left,
            cam_left, cam_right, stereo_T,
        )

    return {
        "left_data": left_data,
        "right_data": right_data,
        "mp_left_detected": left_left.shape == (21, 2) and right_left.shape == (21, 2),
        "mp_right_detected": left_right.shape == (21, 2) and right_right.shape == (21, 2),
    }
