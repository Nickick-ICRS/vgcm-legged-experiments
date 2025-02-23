import mujoco
import mujoco.viewer
import onnxruntime as ort

import os

import numpy as np

from vgcm.simple_rate import Rate
from vgcm.robot_state import RobotState
from vgcm.onnx_controller import Controller


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
    def __init__(self, xml_path, controller: Controller, test_duration=20, headless=False):
        self.mujoco_model, self.mujoco_data = load_robot_model(xml_path)
        self.visualise = not headless
        self.controller = controller
        self.dt = self.mujoco_model.opt.timestep
        self.fps = 1. / self.dt
        self.steps = test_duration * self.fps
        self.state = RobotState()

        self.commands = np.array([1, 0, 0], dtype=np.float32)

        if self.visualise:
            self.viewer = mujoco.viewer.launch_passive(
                self.mujoco_model, self.mujoco_data, key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
            self.viewer.cam.distance = 10
            self.viewer.cam.elevation = -20

            self.viewer_update_frame = int(self.fps / 60.)

    def key_callback(self, keycode):
        pass

    def run(self):
        print("Starting MuJoCo simulation...")
        if self.visualise:
            rate = Rate(60.)
            frame = 0
            import time
            start = time.time()
            while self.viewer.is_running():
                self.read_state()
                self.control()
                mujoco.mj_step(self.mujoco_model, self.mujoco_data)
                frame = (frame + 1) % self.viewer_update_frame
                if frame == 0:
                    now = time.time()
                    start = now
                    self.viewer.sync()
                    rate.sleep()
        else:
            for _ in range(self.steps):
                self.read_state()
                self.control()
                mujoco.mj_step(self.mujoco_model, self.mujoco_data)
        print("Simulation finished.")

    def read_state(self):
        for i in range(self.state.num_joints):
            # Quaternion + Pos, so 7 variables
            self.state.q[i] = self.mujoco_data.qpos[i + 7]
            self.state.dq[i] = self.mujoco_data.qvel[i + 6]
            self.state.tau[i] = self.mujoco_data.ctrl[i]

        for i in range(self.state.num_compensators):
            # TODO: Process GC states
            pass

        base_id = self.mujoco_model.body(name="base_Link").id

        self.state.base_pos = self.mujoco_data.xpos[base_id]
        self.state.base_quat = self.mujoco_data.xquat[base_id]
        self.state.base_lin_vel = self.mujoco_data.cvel[base_id][:3]
        self.state.base_ang_vel = self.mujoco_data.cvel[base_id][3:]
        self.state.projected_gravity = self.mujoco_data.xmat[base_id].reshape(3, 3) @ self.mujoco_model.opt.gravity
        
        imu_quat_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "quat")
        imu_gyro_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        imu_acc_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SENSOR, "acc")
        for i in range(4):
            self.state.imu_data_quat[i] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_quat_id] + i]
        for i in range(3):
            self.state.imu_data_gyro[i] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_gyro_id] + i]
            self.state.imu_data_acc[i] = self.mujoco_data.sensordata[self.mujoco_model.sensor_adr[imu_acc_id] + i]
        self.state.stamp = self.mujoco_data.time

    def control(self):
        control = self.controller.control(self.state, self.commands)
        for i in range(self.state.num_joints):
            self.mujoco_data.ctrl[i] = control.tau[i]