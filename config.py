import glob
import json
import os
from types import SimpleNamespace


_VALID_DEPTH_MODES = ("point_cloud", "triangulation")


class Config:
    default = "take1"

    _DEFAULT_RATES = {
        'mocap_fps': 360,
        'target_fps': 60,
        'there_fps': 60,
        'zed_fps': 30,
    }

    _DEFAULT_WEBCAM = {
        'index': 1,
    }

    _DEFAULT_ZED = {
        'svo_codec': 'H264',
    }

    _DEFAULT_MOCAP = {
        'rigid_bodies': {},
        'markersets': {},
    }

    def __init__(self):
        self.depth_mode = "point_cloud"
        self.crow_port = None
        self.webcam = SimpleNamespace(**self._DEFAULT_WEBCAM)
        self.zed = SimpleNamespace(**self._DEFAULT_ZED)
        self.mocap = SimpleNamespace(**self._DEFAULT_MOCAP)

        self.pitch_take_name = self.default
        self.volume_take_name = self.default

        project_root = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(project_root, "config.json")

        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                self.pitch_take_name = data.get('pitch_take_name', self.default)
                self.volume_take_name = data.get('volume_take_name', self.default)
                self.crow_port = data.get('crow_port', None)
                self.rates = SimpleNamespace(**(data.get('rates', self._DEFAULT_RATES)))

            depth_mode = data.get('depth_mode', 'point_cloud')
            if depth_mode not in _VALID_DEPTH_MODES:
                raise ValueError(
                    f"Invalid depth_mode '{depth_mode}' in config.json. "
                    f"Must be one of {_VALID_DEPTH_MODES}"
                )
            self.depth_mode = depth_mode

            webcam_data = data.get('webcam', self._DEFAULT_WEBCAM)
            self.webcam = SimpleNamespace(**webcam_data)

            zed_data = data.get('zed', self._DEFAULT_ZED)
            self.zed = SimpleNamespace(**zed_data)

            mocap_data = data.get('mocap', {})
            self.mocap = SimpleNamespace(
                rigid_bodies=SimpleNamespace(**(mocap_data.get('rigid_bodies', {}))),
                markersets=SimpleNamespace(**(mocap_data.get('markersets', {}))),
            )
        else:
            self.pitch_take_name = self.default
            self.volume_take_name = self.default
            self.crow_port = None
            self.names = SimpleNamespace()
            self.rates = SimpleNamespace(**self._DEFAULT_RATES)

    def iter_takes(self, *sources, all_takes: bool = True):
        """Yield (take_name, target) for every take found on disk.

        Each source is a (glob pattern, suffix) pair relative to the project
        root; a take is discovered from any of them. With all_takes False, or
        when nothing matches, only the two takes named in config.json are used.
        """
        project_root = os.path.dirname(os.path.abspath(__file__))
        names = set()
        if all_takes:
            for pattern, suffix in sources:
                for path in glob.glob(os.path.join(project_root, pattern)):
                    base = os.path.basename(path)
                    if base.endswith(suffix):
                        names.add(base[: -len(suffix)])
        if not names:
            names = {self.pitch_take_name, self.volume_take_name}
        for name in sorted(names):
            yield name, ("pitch" if "pitch" in name else "volume")

    def get_take_name(self, target: str) -> str:
        if target == "pitch":
            return self.pitch_take_name
        elif target == "volume":
            return self.volume_take_name
        raise ValueError(f"Unknown target '{target}'. Must be 'pitch' or 'volume'.")

    def get_mocap_rigid_body(self, target: str) -> str:
        mapping = {"pitch": "pitch", "volume": "volume"}
        key = mapping.get(target)
        if key is None:
            raise ValueError(f"Unknown target '{target}' for mocap rigid body.")
        return getattr(self.mocap.rigid_bodies, key, key)

    def get_mocap_camera(self, target: str) -> str:
        mapping = {"pitch": "zed_pitch", "volume": "zed_volume"}
        key = mapping.get(target)
        if key is None:
            raise ValueError(f"Unknown target '{target}' for mocap camera.")
        return getattr(self.mocap.rigid_bodies, key, key)

    def get_mocap_webcam(self) -> str:
        return getattr(self.mocap.rigid_bodies, "webcam", "Webcam")

    def get_mocap_markerset(self, target: str) -> str:
        mapping = {"pitch": "right", "volume": "left"}
        key = mapping.get(target)
        if key is None:
            raise ValueError(f"Unknown target '{target}' for mocap markerset.")
        return getattr(self.mocap.markersets, key, key)

    # Alias for convenience
    @property
    def take_name(self):
        return self.pitch_take_name

    def set_take_name(self, name):
        self.pitch_take_name = name
        self.volume_take_name = name
        self._write_config()

    def set_pitch_take_name(self, name):
        self.pitch_take_name = name
        self._write_config()

    def set_volume_take_name(self, name):
        self.volume_take_name = name
        self._write_config()

    def _write_config(self):
        with open(self.config_file, 'w') as f:
            json.dump({
                'pitch_take_name': self.pitch_take_name,
                'volume_take_name': self.volume_take_name,
                'depth_mode': self.depth_mode,
                'crow_port': self.crow_port,
                'webcam': {
                    'index': self.webcam.index,
                },
                'zed': {
                    'svo_codec': self.zed.svo_codec,
                },
                'mocap': {
                    'rigid_bodies': {
                        k: getattr(self.mocap.rigid_bodies, k)
                        for k in ['zed_volume', 'zed_pitch', 'webcam', 'pitch', 'volume']
                        if hasattr(self.mocap.rigid_bodies, k)
                    },
                    'markersets': {
                        k: getattr(self.mocap.markersets, k)
                        for k in ['left', 'right']
                        if hasattr(self.mocap.markersets, k)
                    },
                },
                'rates': {
                    k: getattr(self.rates, k)
                    for k in ('mocap_fps', 'target_fps', 'there_fps', 'zed_fps')
                    if hasattr(self.rates, k)
                },
            }, f, indent=4)

    def check_files_exist(self, take_name=None):
        if take_name is None:
            take_name = self.pitch_take_name
        pattern = f"data/recordings/{take_name}.*"
        existing = glob.glob(pattern)
        return bool(existing)


config = Config()
