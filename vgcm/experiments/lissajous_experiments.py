from .experiment_base import ExperimentBase

import numpy as np
import quaternion
import matplotlib.pyplot as plt

from cycler import cycler
from vgcm.colours import colours
plt.rcParams['axes.prop_cycle'] = cycler(color=colours)


class LissajousExperiment(ExperimentBase):
    def __init__(self, sim, compensation_type: str):
        self.start_time = 0.5

        self.period = 10.
        self.alpha = 2.
        self.offset = -self.period / 4

        self.kp_x = 2
        self.kp_yaw = 2.
        self.kd_x = 0.1
        self.kd_yaw = 0.1

        self.vmax_x = 1.5
        self.vmax_yaw = 2.

        self.compensation_type = compensation_type

        super().__init__(sim, 60. + self.start_time)

    def update(self):
        self.update_history()
        if self.sim.states[0].stamp >= self.start_time:
            for idx, state in enumerate(self.sim.states):
                pos = state.base_pos
                quat = state.base_quat
                quat = quaternion.from_float_array([quat[3], quat[0], quat[1], quat[2]])
                lin_vel = state.base_lin_vel
                ang_vel = state.base_ang_vel
                t = state.stamp + self.sim.dt - self.start_time
                cmd = self.get_velocity_command(pos, quat, lin_vel, ang_vel, t)
                self.sim.set_command(idx, cmd)

        if self.is_done():
            print("Experiment Finished.")

    def finish(self):
        print("Processing Experiment Results")
        self.save_results()
        joint_names = [
            "abad_L_Joint", "hip_L_Joint", "knee_L_Joint",# "wheel_L_Joint",
            "abad_R_Joint", "hip_R_Joint", "knee_R_Joint",# "wheel_R_Joint"
        ]
        joint_idxs = [0, 1, 2, 4, 5, 6]
        fig, axes = plt.subplots(2, 3, figsize=(10, 8))
        axes = axes.flatten()
        df = self.state_histories[0]
        timesteps = df["step"]
        for i, (idx, name) in enumerate(zip(joint_idxs, joint_names)):
            tau_raw = df[f"tau{idx}"]
            tau_ma = df[f"tau{idx}"].rolling(window=5000).mean()
            q = df[f"q{idx}"]
            ax_t = axes[i]
            ax_p = ax_t.twinx()
            line_raw, = ax_t.plot(timesteps, tau_raw, label=f'Torque', color=colours[0])
            line_ma, = ax_t.plot(timesteps, tau_ma, label=f'Torque (5s MA)', color=colours[1])
            if self.compensation_type != 'none':
                gc = df[f"gc{i}_tau"]
                actuator_tau = tau_raw - gc
                line_gc, = ax_t.plot(timesteps, gc, label=f'Compensation', color=colours[2])
                line_act, = ax_t.plot(timesteps, actuator_tau, label=f'Required Actuator Torque', color=colours[3])
            line_pos, = ax_p.plot(timesteps, q, label=f'Joint Position', color=colours[4])
            ax_t.set_ylabel("Torque (Nm)")
            ax_p.set_ylabel("Position (rad)")
            ax_t.set_title(f"{name}")
            ax_t.set_xlabel('Time (s)')
            if self.compensation_type != 'none':
                lines = [line_raw, line_ma, line_gc, line_act, line_pos]
            else:
                lines = [line_raw, line_ma, line_pos]
            labels = [l.get_label() for l in lines]
            ax_t.legend(lines, labels, loc="best")

        cmd_vel = df["cmd_x"]
        base_vel = df["base_lin_vel_x"]
        fig2, ax = plt.subplots()
        line_cmd, = ax.plot(timesteps, cmd_vel, label=f'Command X Vel')
        line_act, = ax.plot(timesteps, base_vel, label=f'Actual X Vel')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        lines = [line_cmd, line_act]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="lower right")

        # Column Titles
        ax.set_title('Base Velocity')

        fig.set_tight_layout(True)
        fig2.set_tight_layout(True)
        plt.show()

    def get_lissajous_trajectory(self, t):
        pos_target = np.array([
            self.alpha * np.cos(2 * np.pi * (t+self.offset) / self.period),
            self.alpha * np.sin(4 * np.pi * (t+self.offset) / self.period) / 2,
            0.9], dtype=np.float32) 
        return pos_target

    def get_velocity_command(self, current_pos, current_quat, current_lin_vel, current_ang_vel, t):
        target_pos = self.get_lissajous_trajectory(t)
        error = target_pos - current_pos
        target_quat = quaternion.from_rotation_vector(
            np.array([0, 0, np.arctan2(error[1], error[0])]))
        err_quat = np.dot(target_quat, np.conjugate(current_quat))
        w, x, y, z = quaternion.as_float_array(err_quat)
        err_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        err_x = np.sqrt(np.power(error[0], 2) + np.power(error[1], 2)) * np.cos(err_yaw)
        return self.calculate_command(err_x, err_yaw, current_lin_vel, current_ang_vel)

    def calculate_command(self, err_x, err_yaw, lin_vel, ang_vel):
        cmd = np.array([
            self.kp_x * err_x - self.kd_x * lin_vel[0],
            0.,
            self.kp_yaw * err_yaw - self.kd_x * ang_vel[2]])
        cmd[0] = np.clip(cmd[0], -self.vmax_x, self.vmax_x)
        cmd[2] = np.clip(cmd[2], -self.vmax_yaw, self.vmax_yaw)
        return cmd

    def prepare_sim(self, sim):
        p0 = self.get_lissajous_trajectory(0)
        p1 = self.get_lissajous_trajectory(sim.dt)
        mdir = p1 - p0
        target_quat = quaternion.from_rotation_vector(
            np.array([0, 0, np.arctan2(mdir[1], mdir[0])]))
        for i in range(self.n_robots):
            sim.set_command(i, np.array([0, 0, 0], dtype=np.float32))
            sim.mujoco_data_instances[i].qpos[3:7] = quaternion.as_float_array(target_quat)

    def experiment_name(self):
        return "lissajous_trajectory_experiment_" + self.compensation_type + "_compensation"
