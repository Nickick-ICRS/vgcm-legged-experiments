import mujoco
import mujoco.viewer
import onnxruntime as ort

import os

from vgcm.simple_rate import Rate


def load_onnx_model(onnx_path):
    print(f"Loading ONNX model from {onnx_path}...")
    session = ort.InferenceSession(onnx_path)
    print("ONNX model loaded successfully.")
    return session


def prepare_mujoco_xml(model_xml, xml_path):
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    model_xml = model_xml.replace("..", f"{os.path.dirname(xml_dir)}")
    return model_xml


def load_robot_model(xml_path):
    print(f"Loading robot model from {xml_path}...")
    assert os.path.exists(xml_path)
    with open(xml_path) as model_file:
        model_xml = prepare_mujoco_xml(model_file.read(), xml_path)
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    print(f"Robot model loaded successfully.")
    return model, data


class Simulator:
    def __init__(self, xml_path, onnx_model_path, test_duration=20, headless=False):
        self.mujoco_model, self.mujoco_data = load_robot_model(xml_path)
        self.visualise = not headless
        self.onnx_sesh = load_onnx_model(onnx_model_path)

        if self.visualise:
            self.viewer = mujoco.viewer.launch_passive(
                self.mujoco_model, self.mujoco_data, key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
            self.viewer.cam.distance = 10
            self.viewer.cam.elevation = -20

        self.dt = self.mujoco_model.opt.timestep
        self.fps = 1. / self.dt
        self.rate = Rate(self.fps)

        self.steps = test_duration * self.fps

    def key_callback(self, keycode):
        pass

    def run(self):
        print("Starting MuJoCo simulation...")
        if self.visualise:
            while self.viewer.is_running():
                state = self.read_state()
                self.control(state)
                mujoco.mj_step(self.mujoco_model, self.mujoco_data)
                self.viewer.sync()
                self.rate.sleep()
        else:
            for _ in range(self.steps):
                state = self.read_state()
                self.control(state)
                mujoco.mj_step(self.mujoco_model, self.mujoco_data)
        print("Simulation finished.")

    def read_state(self):
        pass

    def control(self, state):
        pass