import mujoco
import mujoco.viewer
import onnxruntime as ort

import os
from typing import List

import numpy as np

from vgcm.simple_rate import Rate
from vgcm.robot_state import RobotState
from vgcm.onnx_controller import Controller
from vgcm.vgcm_ideal_model import VGCMTorsionSpringModel


def prepare_mujoco_xml(model_xml, xml_path):
    xml_dir = os.path.dirname(os.path.abspath(xml_path))
    model_xml = model_xml.replace("..", f"{os.path.dirname(xml_dir)}")
    return model_xml


def load_robot_model(xml_path, n_robots=1):
    print(f"Loading {n_robots} robot models from {xml_path}...")
    assert os.path.exists(xml_path)
    with open(xml_path) as model_file:
        model_xml = prepare_mujoco_xml(model_file.read(), xml_path)
    models = [mujoco.MjModel.from_xml_string(model_xml) for _ in range(n_robots)]
    data = [mujoco.MjData(models[i]) for i in range(n_robots)]
    print(f"{n_robots} robot models loaded successfully.")
    return models, data


class Simulator:
    def __init__(
            self, xml_path, controllers: List[Controller], compensators: List[VGCMTorsionSpringModel],
            test_duration=20, headless=False, callback=None):

        self.num_robots = len(controllers)
        self.mujoco_models, self.mujoco_data_instances = load_robot_model(xml_path, self.num_robots)
        self.visualise = not headless
        self.controllers = controllers
        self.compensators = compensators
        self.dt = self.mujoco_models[0].opt.timestep
        self.fps = 1. / self.dt
        self.steps = int(test_duration * self.fps)
        self.states = [RobotState() for _ in range(self.num_robots)]

        self.callback=callback

        self.commands = [np.array([1., 0, 0], dtype=np.float32) for _ in range(self.num_robots)]
        self.ext_forces = [np.zeros(3, dtype=np.float32) for _ in range(self.num_robots)]

        self.base_id = self.mujoco_models[0].body(name="base_Link").id
        self.base_mass = self.mujoco_models[0].body_mass[self.base_id]

        if self.visualise:
            self.viewer = mujoco.viewer.launch_passive(
                self.mujoco_models[0], self.mujoco_data_instances[0], key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
            self.viewer.cam.distance = 10
            self.viewer.cam.elevation = -20

            self.viewer_update_frame = int(self.fps / 60.)

    def get_full_state_dict(self, robot_idx):
        entry = self.states[robot_idx].to_dict()
        entry["cmd_x"] = self.commands[robot_idx][0]
        entry["cmd_y"] = self.commands[robot_idx][1]
        entry["cmd_yaw"] = self.commands[robot_idx][2]
        entry["ext_f_x"] = self.ext_forces[robot_idx][0]
        entry["ext_f_y"] = self.ext_forces[robot_idx][1]
        entry["ext_f_z"] = self.ext_forces[robot_idx][2]
        for i, act in enumerate(self.controllers[robot_idx]._prev_actions):
            entry[f"action{i}"] = act
        return entry

    def key_callback(self, keycode):
        pass

    def run(self):
        print("Starting MuJoCo simulation...")
        # Step once to fill state etc.
        for robot_idx, data in enumerate(self.mujoco_data_instances):
            mujoco.mj_step(self.mujoco_models[robot_idx], data)
        if self.visualise:
            rate = Rate(60.)
            frame = 0
            import time
            start = time.time()
            while self.viewer.is_running():
                self.step()
                frame = (frame + 1) % self.viewer_update_frame
                if frame == 0:
                    now = time.time()
                    start = now
                    self.viewer.sync()
                    rate.sleep()
        else:
            for _ in range(self.steps):
                self.step()
        print("Simulation finished.")

    def run_custom_callback(self):
        if self.callback is not None:
            self.callback(self)

    def step(self):
        for i, data in enumerate(self.mujoco_data_instances):
            self.read_state(i, data)
        self.run_custom_callback()
        for i, data in enumerate(self.mujoco_data_instances):
            self.control(i)
            self.apply_forces(i)
            mujoco.mj_step(self.mujoco_models[i], data)

    def read_state(self, robot_idx, mujoco_data):
        for i in range(self.states[robot_idx].num_joints):
            # Quaternion + Pos, so 7 variables
            self.states[robot_idx].q[i] = mujoco_data.qpos[i + 7]
            self.states[robot_idx].dq[i] = mujoco_data.qvel[i + 6]
            self.states[robot_idx].tau[i] = mujoco_data.ctrl[i]

        for i in range(self.states[robot_idx].num_compensators):
            # TODO: Process GC states
            pass

        self.states[robot_idx].base_pos = mujoco_data.xpos[self.base_id]
        # Mujoco stores the quat W X Y Z, we want X Y Z W
        quat_wxyz = mujoco_data.xquat[self.base_id]
        self.states[robot_idx].base_quat[:3] = quat_wxyz[1:]
        self.states[robot_idx].base_quat[3] = quat_wxyz[0]
        base_mat = mujoco_data.xmat[self.base_id].reshape(3, 3)
        # In global frame -> convert to local
        self.states[robot_idx].base_lin_vel = base_mat.transpose() @ mujoco_data.cvel[self.base_id][3:]
        self.states[robot_idx].base_ang_vel = base_mat.transpose() @ mujoco_data.cvel[self.base_id][:3]
        self.states[robot_idx].projected_gravity = base_mat.transpose() @ self.mujoco_models[robot_idx].opt.gravity
        
        imu_quat_id = mujoco.mj_name2id(self.mujoco_models[robot_idx], mujoco.mjtObj.mjOBJ_SENSOR, "quat")
        imu_gyro_id = mujoco.mj_name2id(self.mujoco_models[robot_idx], mujoco.mjtObj.mjOBJ_SENSOR, "gyro")
        imu_acc_id = mujoco.mj_name2id(self.mujoco_models[robot_idx], mujoco.mjtObj.mjOBJ_SENSOR, "acc")
        # Mujoco stores the quat W X Y Z, we want X Y Z W
        self.states[robot_idx].imu_data_quat[3] = mujoco_data.sensordata[self.mujoco_models[robot_idx].sensor_adr[imu_quat_id]]
        for i in range(3):
            # Mujoco stores the quat W X Y Z, we want X Y Z W
            self.states[robot_idx].imu_data_quat[i] = mujoco_data.sensordata[self.mujoco_models[robot_idx].sensor_adr[imu_quat_id] + i+1]
            self.states[robot_idx].imu_data_gyro[i] = mujoco_data.sensordata[self.mujoco_models[robot_idx].sensor_adr[imu_gyro_id] + i]
            self.states[robot_idx].imu_data_acc[i] = mujoco_data.sensordata[self.mujoco_models[robot_idx].sensor_adr[imu_acc_id] + i]

        self.states[robot_idx].stamp = mujoco_data.time

    def control(self, robot_idx):
        control = self.controllers[robot_idx].control(self.states[robot_idx], self.commands[robot_idx])
        for i in range(self.states[robot_idx].num_joints):
            self.mujoco_data_instances[robot_idx].ctrl[i] = control.tau[i]

    def apply_forces(self, robot_idx):
        # Gravity compensators
        if self.compensators[robot_idx] is not None:
            for i, comp in enumerate(self.compensators[robot_idx]):
                q = self.states[robot_idx].q[i]
                self.mujoco_data_instances[robot_idx].ctrl[i] += comp.update(q, k_ideal, zero_ideal, dt)
        # External forces
        self.mujoco_data_instances[robot_idx].xfrc_applied[self.base_id, :3] = self.ext_forces[robot_idx]

    def set_payload(self, robot_idx, mass):
        print(f"Setting robot {robot_idx} payload to {mass}.")
        g = self.mujoco_models[robot_idx].opt.gravity
        self.ext_forces[robot_idx] = g * mass
