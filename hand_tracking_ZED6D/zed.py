import numpy as np
import pyzed.sl as sl


class Zed():
    def __init__(self, filename=None, depth_confidence=100, camera_serial=None,
                 fps=None):

        print("Bringing Up ZED CAMERA Information...")
        # Decide if SVO or Live
        if filename is None:
            serial_str = str(camera_serial) if camera_serial is not None else "default"
            print(f"Using Live stream from ZED camera (serial: {serial_str})")
            self.input_type = sl.InputType()
            if camera_serial is not None:
                self.input_type.set_from_serial_number(camera_serial)
            self.svo_mode = False
        else:
            filepath = filename
            print("Reading SVO file: {0}".format(filepath))
            self.input_type = sl.InputType()
            self.input_type.set_from_svo_file(filepath)
            self.svo_mode = True

        # Initialize the ZED camera
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters(input_t=self.input_type)

        if self.svo_mode:
            # Best quality for dataset creation – speed doesn't matter
            self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
            self.init_params.svo_real_time_mode = False
            confidence = 20
        else:
            # Live preview – prioritise FPS
            self.init_params.camera_resolution = sl.RESOLUTION.VGA
            self.init_params.camera_fps = fps if fps is not None else 60
            self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT

            confidence = depth_confidence

        self.init_params.depth_minimum_distance = 0.3
        self.init_params.depth_maximum_distance = 40
        self.init_params.coordinate_units = sl.UNIT.METER

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            msg = repr(err)
            print(msg)
            print("If using SVO, check if the path is correct")
            try:
                self.zed.close()
            except Exception:
                pass
            raise RuntimeError(f"ZED open failed: {msg}")

        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.enable_fill_mode = True
        self.runtime_parameters.confidence_threshold = confidence

        # Get full stereo calibration parameters
        self._load_calibration()

        # declare image, depth, point cloud
        self.image = sl.Mat()
        self.image_right = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()
        # self.confidence_map = sl.Mat()

    def _load_calibration(self):
        calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
        self.cam_left = calib.left_cam
        self.cam_right = calib.right_cam
        # 4x4 transformation from left → right camera coordinate frame
        self.stereo_transform = np.array(calib.stereo_transform.m, dtype=np.float64)

        # Backward-compatible aliases
        self.camera_params = self.cam_left
        self.fx = self.cam_left.fx
        self.fy = self.cam_left.fy
        self.cx = self.cam_left.cx
        self.cy = self.cam_left.cy

    def print_information(self):
        print("Resolution: {0}, {1}.".format(
            self.zed.get_camera_information().camera_configuration.resolution.width,
            self.zed.get_camera_information().camera_configuration.resolution.height
        ))
        #print("Camera FPS: {0}".format(self.zed.get_camera_information().camera_fps))
        cam_info = self.zed.get_camera_information()
        fps = cam_info.camera_configuration.fps

        print("Camera FPS: {0}".format(fps))
        print("Depth mode: {0}.".format(self.init_params.depth_mode))
        #print("Sensing mode: {0}.".format(self.runtime_parameters.sensing_mode))
        if self.svo_mode:
            print("Frame count: {0}.\n".format(self.zed.get_svo_number_of_frames())) 

    def get_image(self):
        # Retrieve left rectified image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        # Retrieve right rectified image (for stereo triangulation)
        self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
        # Retrieve depth map. Depth is aligned on the left image
        self.zed.retrieve_image(self.depth, sl.VIEW.CONFIDENCE)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
        # Retrieve confidence map.
        # self.zed.retrieve_measure(self.confidence_map, sl.MEASURE.CONFIDENCE, sl.MEM.CPU)

        # convert zed image to numpy array
        self.img = self.image.get_data()
        self.img_right = self.image_right.get_data()
        self.depth_img = self.depth.get_data()
