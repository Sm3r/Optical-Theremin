import cv2
import time
import mediapipe as mp
import numpy as np
import pyzed.sl as sl

from .triangulation import stereo_detect as _stereo_detect


class HandTracking():
    def __init__(self, maxHands=2, detectionCon=0.4, trackCon=0.8, complexity=1, draw=True):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=maxHands,           
            min_detection_confidence=detectionCon,   
            min_tracking_confidence=trackCon,
            model_complexity=complexity
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.time1 = time.time()
        self.wrist = []
        self.image_height = None
        self.image_width = None
        self.detection_str = ""
        self.draw = draw

    def findHands(self, img):
        self.image_height, self.image_width, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_img)

        if self.draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(), 
                    self.mp_styles.get_default_hand_connections_style()
                )

        return img

    
    def findpostion(self, img, pcl, camera_params):
        fx = camera_params.fx
        fy = camera_params.fy
        h, w, _ = img.shape
        left_data = []
        right_data = []
        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:
                handedness = self.results.multi_handedness[self.results.multi_hand_landmarks.index(landmarks)].classification[0].index

                wrist_landmark = landmarks.landmark[0]
                wrist_u = wrist_landmark.x * w
                wrist_v = wrist_landmark.y * h
                X, Y = int(wrist_u), int(wrist_v)

                if self.draw:
                    cv2.circle(img, (X, Y), 10, (0, 0, 255), -1)

                try:
                    err, point_cloud_value = pcl.get_value(X, Y)
                except Exception:
                    continue
                if err != sl.ERROR_CODE.SUCCESS:
                    continue
                wrist_position = [point_cloud_value[0], point_cloud_value[1], point_cloud_value[2]]

                for id, landmark in enumerate(landmarks.landmark):
                    u = landmark.x * w
                    v = landmark.y * h
                    x_3d = wrist_position[0] + (u - wrist_u) * wrist_position[2] / fx
                    y_3d = wrist_position[1] + (v - wrist_v) * wrist_position[2] / fy
                    z_3d = wrist_position[2] + landmark.z * wrist_position[2] * w / fx
                    hand_landmarks_3d = [x_3d, y_3d, z_3d]

                    if handedness == 1:
                        left_data.append(hand_landmarks_3d)
                        if self.draw:
                            cv2.putText(img, "Left", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif handedness == 0:
                        right_data.append(hand_landmarks_3d)
                        if self.draw:
                            cv2.putText(img, "Right", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        left_data = np.array(left_data)
        right_data = np.array(right_data)
        self.stdout_hand_detection(left_data, right_data)

        return left_data, right_data

    def detect_stereo(self, cam, right_tracker=None):
        """Grab a ZED frame, detect 2D landmarks in both views, triangulate 3D hands.

        Parameters
        ----------
        cam : Zed
        right_tracker : HandTracking or None
            Separate tracker for the right image. When None, uses ``self`` for both.

        Returns
        -------
        dict with keys:
            success : bool
            left_data : ndarray (21, 3) or empty
            right_data : ndarray (21, 3) or empty
            mp_left_detected : bool
            mp_right_detected : bool
            img_left : ndarray (H, W, 3) BGR or None
        """
        if right_tracker is None:
            right_tracker = self

        err = cam.zed.grab(cam.runtime_parameters)
        if err != sl.ERROR_CODE.SUCCESS:
            return {"success": False}

        cam.get_image()

        img_left = cam.img.copy()
        if img_left.ndim == 3 and img_left.shape[2] == 4:
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGRA2BGR)

        img_right = cam.img_right.copy()
        if img_right.ndim == 3 and img_right.shape[2] == 4:
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGRA2BGR)

        result = _stereo_detect(
            self.hands, right_tracker.hands,
            img_left, img_right,
            cam.cam_left, cam.cam_right, cam.stereo_transform,
        )
        result["success"] = True
        result["img_left"] = img_left
        return result

    def calculate_orientation(self,hand_landmarks_3d):
        if hand_landmarks_3d.shape != (21,3):
            zero_array = np.zeros((3,))
            return  zero_array
        
        # Get the 3D positions of landmarks 0, 5, and 17
        wrist = hand_landmarks_3d[0]
        index = hand_landmarks_3d[5]
        pinky = hand_landmarks_3d[17]

        # Compute the vectors between the landmarks
        v1 = np.subtract(index, wrist)
        v2 = np.subtract(pinky, wrist)

        # Compute the normal vector to the plane defined by the landmarks
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        # Guard against degenerate cases where the normal is zero
        if norm < 1e-8:
            zero_array = np.zeros((3,))
            return zero_array
        normal = normal / norm

        # Compute the yaw, pitch, and roll angles based on the orientation of the normal vector
        yaw = np.arctan2(normal[1], normal[0])
        pitch = np.arctan2(-normal[2], np.sqrt(normal[0]**2 + normal[1]**2))
        roll = np.arctan2(np.sin(yaw)*v2[0]-np.cos(yaw)*v2[1], np.cos(yaw)*v1[1]-np.sin(yaw)*v1[0])

        self.orientation = np.array([yaw, pitch, roll])

        # Convert angles to degrees and return
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def calculate_centroid(self,hand_landmarks_3d):
        if hand_landmarks_3d.shape != (21,3):
            zero_array = np.zeros((3,))
            return  zero_array
        # Get the 3D positions of landmarks 0, 5, and 17
        
        wrist = hand_landmarks_3d[0]
        index = hand_landmarks_3d[5]
        pinky = hand_landmarks_3d[17]

        # Compute a middle point of three landmarks
        centroid = (wrist + index + pinky)/3



        return centroid

    def findNormalizedPosition(self,img):
        left_data = []
        right_data = []
        h, w, _ = img.shape

        if self.results.multi_hand_landmarks:
            
            for landmarks in self.results.multi_hand_landmarks:
                handedness = self.results.multi_handedness[self.results.multi_hand_landmarks.index(landmarks)].classification[0].index
                for id, landmark in enumerate(landmarks.landmark):
                    # Find the pixel coordinates of the wrist
                    if id == 0:
                        X, Y = int(landmark.x * w), int(landmark.y * h)
                        # circle X, Y
                        if self.draw:
                            cv2.circle(img, (X, Y), 10, (0, 0, 255), -1)
         
                    hand_landmarks_3d = [landmark.x, landmark.y, landmark.z]
                    # append the 3D position of each 3D landmark 
                    if handedness == 1:
                        left_data.append(hand_landmarks_3d)
                        if self.draw:
                            cv2.putText(img, "Left", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif handedness == 0:
                        right_data.append(hand_landmarks_3d)
                        # put text left hand
                        if self.draw:
                            cv2.putText(img, "Right", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        left_data = np.array(left_data)
        right_data = np.array(right_data)
        self.stdout_hand_detection(left_data, right_data)

        return left_data, right_data
    
    def get_fps(self):
        # Set the time for this frame to the current time.
        self.time2 = time.time()

        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (self.time2 - self.time1) > 0:

            # Calculate the number of frames per second.
            fps = 1.0 / (self.time2 - self.time1)
            self.time1 = self.time2
            return fps

        return None
    
    def displayFPS(self, img):
        fps = self.get_fps()
        
        # Only draw if a valid FPS was calculated
        if fps is not None:
            # Write the calculated number of frames per second on the frame
            cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        return img

    def stdout_hand_detection(self, left_data, right_data):
        # Update the current detection string instead of printing directly.
        if left_data.shape == (21,3) and right_data.shape == (21,3):
            self.detection_str = "Left and Right hands all 21 landmarks detected"
        elif left_data.shape == (21,3) and right_data.shape != (21,3):
            self.detection_str = "Left hand all 21 landmarks detected"
        elif left_data.shape != (21,3) and right_data.shape == (21,3):
            self.detection_str = "Right hand all 21 landmarks detected"
        else:
            self.detection_str = "No hand landmarks detected"
    

    def plot(self,ax,plt,data,xlim=(-0.5, 0.1),ylim=(-0.5, 0.1),zlim=(0.2, 1.0)):
        # Create 3D plot

        if data.shape >= (21,3):
          
            # Clear the plot and add new data
            ax.clear()
            
            # auto scale the plot
            # ax.autoscale(enable=True, axis='both', tight=None)

            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
            ax.scatter3D(*zip(*data))
     
            #  C
            edges = [(1,2),(2,3),(3,4),(0,5),(5,6),(5,9),(1,0),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
            edges2 = [(22,23),(23,24),(24,25),(21,26),(26,27),(26,30),(22,21),(27,28),(28,29),(21,30),(30,31),(31,32),(32,33),(30,34),(34,35),(35,36),(36,37),(34,38),(38,39),(39,40),(40,41),(21,38)] 

            if data.shape != (42,3):
                for edge in edges:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')

            else:
                for edge in edges:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')
                for edge in edges2:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='blue')


            # Draw the plot
            plt.draw()
            plt.pause(0.0001)