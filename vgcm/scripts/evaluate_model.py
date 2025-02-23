import argparse
import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from vgcm.mujoco_sim import Simulator
from vgcm.onnx_controller import Controller

import matplotlib.pyplot as plt
import numpy as np

import copy


class ResultAggregator:
    def __init__(self, n_robots, test_duration):
        self.n_robots = n_robots
        self.command_history = [[] for _ in range(self.n_robots)]
        self.state_history = [[] for _ in range(self.n_robots)]
        self.action_history = [[] for _ in range(self.n_robots)]
        self.test_duration = test_duration
        self.done = False
        self.skip_first_secs = 1.
        self.prev_t = 0

    def aggregate(self, sim):
        if self.done:
            return self.done
        t = sim.states[0].stamp
        if t < self.skip_first_secs:
            return self.done
        for i in range(self.n_robots):
            # self.state_history[i].append(copy.deepcopy(sim.states[i]))
            # self.command_history[i].append(copy.deepcopy(sim.commands[i]))
            # self.action_history[i].append(copy.deepcopy(sim.controllers[i]._last_ctrl.tau))
            if self.prev_t < 4. and t >= 4.:
                sim.set_payload(i, i % 10)
            if self.prev_t < 8. and t >= 8.:
                sim.set_payload(i, 0)
        if t >= self.test_duration + self.skip_first_secs:
            self.process_results()
            self.done = True
        self.prev_t = t
        return self.done

    def process_results(self):
        print("Processing Experiment Results")
        history = len(self.state_history[0])
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        start = self.skip_first_secs
        end = self.skip_first_secs + self.test_duration
        timesteps = np.linspace(start, end, history)
        for idx in range(self.n_robots):
            avg_torque = np.zeros(history)
            max_torque = np.zeros(history)
            for t in range(history):
                avg_torque[t] = np.mean(np.abs(self.state_history[idx][t].tau))
                max_torque[t] = np.max(np.abs(self.state_history[idx][t].tau))
            ax.plot(timesteps, avg_torque, label=f"{idx} kg Payload")
            ax2.plot(timesteps, max_torque, label=f"{idx} kg Payload")
        ax.set_xlabel("Timestep (s)")
        ax2.set_xlabel("Timestep (s)")
        ax.set_ylabel("Average Absolute Torque (Nm)")
        ax2.set_ylabel("Peak Absolute Torque (Nm)")
        ax.set_title("Average Torque for Varying Payloads")
        ax2.set_title("Peak Torque for Varying Payloads")
        ax.legend()
        ax2.legend()
        print("Done")
        plt.show()


def main(args):
    controllers = []
    for model_file in os.listdir(args.onnx_dir):
        if model_file.endswith(".onnx"):
            onnx_model_path = os.path.join(args.onnx_dir, model_file)
            controllers.append(Controller(onnx_model_path))
    controllers = [Controller(onnx_model_path) for _ in range(10)]
    experiment = ResultAggregator(len(controllers), args.test_duration)
    sim = Simulator(
        args.xml_path, controllers, test_duration=args.test_duration+experiment.skip_first_secs,
        headless=args.headless, callback=experiment.aggregate)
    sim.run()
    if not experiment.done:
        experiment.process_results()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX models in MuJoCo simulation.")

    default_onnx_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs/pointfoot_flat/exported/policies/')
    default_xml_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/pointfoot/WF_TRON1A/xml/robot.xml')

    parser.add_argument("--onnx_dir", type=str, default=default_onnx_dir, help=f"Path to directory containing ONNX models (default: {default_onnx_dir})")
    parser.add_argument("--xml_path", type=str, default=default_xml_path, help=f"Path to robot MJCF model (default: {default_xml_path})")
    parser.add_argument("--test_duration", type=float, default=20, help="Simulation duration in seconds (default: 20)")
    parser.add_argument("--headless", action='store_true', help="Run without visualiser")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)