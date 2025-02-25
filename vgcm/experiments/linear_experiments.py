from .experiment_base import ExperimentBase

import numpy as np
import quaternion
import matplotlib.pyplot as plt

from cycler import cycler
from vgcm.colours import colours
plt.rcParams['axes.prop_cycle'] = cycler(color=colours)


class BasicLinearExperiment(ExperimentBase):
    def __init__(self, sim):
        super().__init__(sim, 20.)

        assert self.n_robots == 1

        self.acc = 0.5
        self.target_speed = 1.

        self.start_acc_time = 2
        self.fin_acc_time = self.start_acc_time + self.target_speed / self.acc

    def update(self):
        self.update_history()

        t = self.sim.states[0].stamp
        command = np.zeros(3, dtype=np.float32)
        if t > self.start_acc_time and t < self.fin_acc_time:
            command[0] = (t-self.start_acc_time) * self.acc
        if t >= self.fin_acc_time:
            command[0] = self.target_speed
        for i in range(self.n_robots):
                self.sim.set_command(i, command)

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
            tau = df[f"tau{idx}"]
            q = df[f"q{idx}"]
            ax_t = axes[i]
            ax_q = ax_t.twinx()
            line_t, = ax_t.plot(timesteps, tau, label=f'Torque', color=colours[0])
            line_q, = ax_q.plot(timesteps, q, label=f'Position', color=colours[1])
            ax_t.set_ylabel("Torque (Nm)")
            ax_q.set_ylabel("Position (rad)")
            ax_t.set_title(f"{name}")
            ax_t.set_xlabel('Time (s)')
            lines = [line_t, line_q]
            labels = [l.get_label() for l in lines]
            ax_t.legend(lines, labels, loc="upper right")

        cmd_vel = df["cmd_x"]
        base_vel = df["base_lin_vel_x"]
        ang_vel = df["base_ang_vel_z"]
        fig2, ax = plt.subplots()
        line_cmd, = ax.plot(timesteps, cmd_vel, label=f'Command X Vel')
        line_act, = ax.plot(timesteps, base_vel, label=f'Actual X Vel')
        ax_ang = ax.twinx()
        line_ang, = ax_ang.plot(timesteps, ang_vel, label=f'Actual Yaw Vel', color=colours[2])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax_ang.set_ylabel('Velocity (rad/s)')
        lines = [line_cmd, line_act, line_ang]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="lower right")

        # Column Titles
        ax.set_title('Base Velocity')

        fig.set_tight_layout(True)
        fig2.set_tight_layout(True)
        plt.show()

    def prepare_sim(self, sim):
        for i in range(self.n_robots):
            sim.set_command(i, np.array([0, 0, 0], dtype=np.float32))
    
    def experiment_name(self):
        return "basic_linear_experiment"


class WeightChangeLinearExperiment(BasicLinearExperiment):
    def __init__(self, sim):
        super().__init__(sim)

        self.target_payload = 10
        self.current_payload = 0

    def update(self):
        super().update()

        t = self.sim.states[0].stamp

        if t > 6 and t < 14 and self.current_payload != self.target_payload:
            self.sim.set_payload(0, self.target_payload)
            self.current_payload = self.target_payload
        elif t > 14 and self.current_payload != 0:
            self.sim.set_payload(0, 0)
            self.current_payload = 0

    def experiment_name(self):
        return "weight_change_linear_experiment"
