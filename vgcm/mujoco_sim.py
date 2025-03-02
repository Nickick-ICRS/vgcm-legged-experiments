import mujoco
import mujoco.viewer

import os
from typing import List

import numpy as np

from vgcm.simple_rate import Rate
from vgcm.robot_state import RobotState
from vgcm.onnx_controller import Controller
from vgcm.vgcm_ideal_model import VGCMSpringModel, K_MIN, K_MAX, X_MIN, X_MAX


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

    # Read in control limits and disable - we handle manually
    min_tau = models[0].actuator_ctrlrange[:, 0]
    max_tau = models[0].actuator_ctrlrange[:, 1]
    for model in models:
        model.actuator_ctrllimited[:] = 0

    data = [mujoco.MjData(models[i]) for i in range(n_robots)]
    print(f"{n_robots} robot models loaded successfully.")
    return models, data, min_tau, max_tau


class Simulator:
    def __init__(
            self, xml_path, controllers: List[Controller], compensators: List[VGCMSpringModel],
            test_duration=20, headless=False, callback=None):

        self.num_robots = len(controllers)
        self.mujoco_models, self.mujoco_data_instances, self.min_tau, self.max_tau = load_robot_model(xml_path, self.num_robots)
        self.visualise = not headless
        self.controllers = controllers
        self.compensators = compensators
        self.dt = self.mujoco_models[0].opt.timestep
        self.fps = 1. / self.dt
        self.set_test_duration(test_duration)
        self.states = [RobotState(None not in comp) for comp in self.compensators]

        self.callback = callback

        self.commands = [np.zeros(3, dtype=np.float32) for _ in range(self.num_robots)]
        self.ext_forces = [np.zeros(3, dtype=np.float32) for _ in range(self.num_robots)]

        self.base_id = self.mujoco_models[0].body(name="base_Link").id
        self.base_mass = self.mujoco_models[0].body_mass[self.base_id]

        # 5s moving averages for GC torque calcs
        self.tau_ma_n = 5. / self.dt
        self.tau_mas = [np.zeros(8, dtype=np.float32) for _ in range(self.num_robots)]
        self.prev_tau = [np.zeros(8, dtype=np.float32) for _ in range(self.num_robots)]
        self.compensation_tau = [np.zeros(6, dtype=np.float32) for _ in range(self.num_robots)]
        self.target_k = [None for _ in range(self.num_robots)]
        self.target_x = [None for _ in range(self.num_robots)]

        if self.visualise:
            self.viewer = mujoco.viewer.launch_passive(
                self.mujoco_models[0], self.mujoco_data_instances[0], key_callback=self.key_callback, show_left_ui=True, show_right_ui=True)
            self.viewer.cam.distance = 10
            self.viewer.cam.elevation = -20

            self.viewer_update_frame = int(self.fps / 60.)

        self.paused = False
        self.done = False

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

    def set_test_duration(self, duration):
        self.steps = int(duration * self.fps)

    def set_command(self, idx, command):
        self.commands[idx] = command

    def register_callback(self, callback):
        self.callback = callback

    def signal_shutdown(self):
        self.done = True

    def key_callback(self, keycode):
        # Space key to pause
        if keycode == 32:
            self.paused = not self.paused
        # Esc key to quit
        elif keycode == 256:
            self.done = True
        else:
            print(f"Keycode pressed: {keycode}")

    def run(self):
        print("Starting MuJoCo simulation...")
        # Step once to fill state etc.
        for robot_idx, data in enumerate(self.mujoco_data_instances):
            mujoco.mj_step(self.mujoco_models[robot_idx], data)
            # Set gravity compensators to current position
            idxs = [0, 1, 2, 4, 5, 6]
            for i, comp in enumerate(self.compensators[robot_idx]):
                if comp is not None:
                    x = data.qpos[7+idxs[i]]
                    comp.preload = x
                    comp.tx = x

        if self.visualise:
            rate = Rate(60.)
            frame = 0
            while self.viewer.is_running() and not self.done:
                self.step()
                frame = (frame + 1) % self.viewer_update_frame
                if frame == 0:
                    self.viewer.sync()
                    rate.sleep()
        else:
            # We did one initial step already
            for _ in range(self.steps):
                self.step()
                if self.done:
                    break
        print("Simulation finished.")

    def run_custom_callback(self):
        if self.callback is not None:
            self.callback()

    def step(self):
        if not self.paused:
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
            comp = self.compensators[robot_idx][i]
            self.states[robot_idx].gc_k[i] = comp.k
            self.states[robot_idx].gc_tk[i] = comp.tk
            self.states[robot_idx].gc_x[i] = comp.preload
            self.states[robot_idx].gc_tx[i] = comp.tx
        self.states[robot_idx].gc_tau = self.compensation_tau[robot_idx]

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
        self.prev_tau[robot_idx] = self.mujoco_data_instances[robot_idx].qfrc_actuator[6:]
        control = self.controllers[robot_idx].control(self.states[robot_idx], self.commands[robot_idx])
        for i in range(self.states[robot_idx].num_joints):
            self.mujoco_data_instances[robot_idx].ctrl[i] = control.tau[i]
            if control.controls_compensators:
                self.target_k[robot_idx] = control.tk
                self.target_x[robot_idx] = control.tx

    def apply_forces(self, robot_idx):
        # Gravity compensators
        self.apply_gravity_compensation(robot_idx)
        # External forces
        self.mujoco_data_instances[robot_idx].xfrc_applied[self.base_id, :3] = self.ext_forces[robot_idx]

    def set_payload(self, robot_idx, mass):
        print(f"Setting robot {robot_idx} payload to {mass}.")
        g = self.mujoco_models[robot_idx].opt.gravity
        self.ext_forces[robot_idx] = g * mass

    def apply_gravity_compensation(self, robot_idx):
        # No compensators for wheels
        idxs = [0, 1, 2, 4, 5, 6]
        q = self.states[robot_idx].q[idxs]
        Jp_l, Jp_r, dJp_l, dJp_r = self.calc_jacs(robot_idx)
        # Use forces directly or Moving Average?
        # bias_tau = self.tau_mas[robot_idx]
        bias_tau = self.mujoco_data_instances[robot_idx].qfrc_actuator[6:]

        # ideal stiffness = J_dot.T @ J @ (tau - Mqddot - C)
        if self.target_k[robot_idx] is not None:
            stiffness = self.target_k[robot_idx]
            zero_pos = self.target_x[robot_idx]
        else:
            stiffness = dJp_l.T @ Jp_l @ bias_tau + dJp_r.T @ Jp_r @ bias_tau
            # Stiffness can't be negative
            stiffness = np.clip(np.abs(stiffness[idxs]), K_MIN, K_MAX)
            zero_pos = np.clip(q - bias_tau[idxs] / stiffness, X_MIN, X_MAX)

        for i, (idx, comp) in enumerate(zip(idxs, self.compensators[robot_idx])):
            if comp is None:
                g_comp = 0
            else:
                g_comp = comp.update(q[i], stiffness[i], zero_pos[i], self.dt)
            self.compensation_tau[robot_idx][i] = g_comp
            # Ensure actuator torque doesn't exceed limits
            act_tau = np.clip(
                self.mujoco_data_instances[robot_idx].ctrl[idx] - g_comp,
                self.min_tau[idx], self.max_tau[idx])
            self.mujoco_data_instances[robot_idx].ctrl[idx] = act_tau + g_comp
            n = self.tau_ma_n
            self.tau_mas[robot_idx][idx] = self.tau_mas[robot_idx][idx] * (n-1) / n \
                + self.mujoco_data_instances[robot_idx].ctrl[idx] / n

    def calc_jacs(self, robot_idx):
        model = self.mujoco_models[robot_idx]
        # We're going to step forwards, so make a copy
        data = self.mujoco_data_instances[robot_idx]
        # + 2 accounts for base body and ground plane
        l_wheel_id = 3+2
        r_wheel_id = 7+2
        l_wheel_gp = data.xpos[l_wheel_id]
        r_wheel_gp = data.xpos[r_wheel_id]
        Jp_l = np.zeros((3, model.nv), dtype=np.float64)
        Jp_r = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp=Jp_l, jacr=None, point=l_wheel_gp, body=l_wheel_id)
        mujoco.mj_jac(model, data, jacp=Jp_r, jacr=None, point=r_wheel_gp, body=r_wheel_id)

        # Now integrate by some very small delta
        DELTA = 1e-6
        mujoco.mj_integratePos(model, data.qpos, data.qvel, DELTA)
        mujoco.mj_forward(model, data)

        dJp_l = np.zeros((3, model.nv), dtype=np.float64)
        dJp_r = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(model, data, jacp=dJp_l, jacr=None, point=l_wheel_gp, body=l_wheel_id)
        mujoco.mj_jac(model, data, jacp=dJp_r, jacr=None, point=r_wheel_gp, body=r_wheel_id)
        dJp_l = (dJp_l - Jp_l) / DELTA
        dJp_r = (dJp_r - Jp_r) / DELTA

        mujoco.mj_integratePos(model, data.qpos, -data.qvel, DELTA)
        mujoco.mj_forward(model, data)

        # Ignore the base joint
        return Jp_l[:, 6:], Jp_r[:, 6:], dJp_l[:, 6:], dJp_r[:, 6:]
