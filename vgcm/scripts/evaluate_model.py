import argparse
import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from vgcm.mujoco_sim import Simulator
from vgcm.onnx_controller import Controller

from vgcm.experiments.linear_experiments import BasicLinearExperiment, WeightChangeLinearExperiment
from vgcm.experiments.lissajous_experiments import LissajousExperiment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_CHOICES = ['basic_linear', 'weights_linear', 'lissajous']


def main(args):
    controllers = []
    for model_file in os.listdir(args.onnx_dir):
        if model_file.endswith(".onnx") and args.ctrl_name in model_file:
            onnx_model_path = os.path.join(args.onnx_dir, model_file)
            controllers.append(Controller(onnx_model_path))
    compensators = [None for _ in controllers]
    sim = Simulator(
        args.xml_path, controllers, compensators,
        headless=args.headless)
    if args.experiment == 'basic_linear':
        experiment = BasicLinearExperiment(sim)
    elif args.experiment == 'weights_linear':
        experiment = WeightChangeLinearExperiment(sim)
    elif args.experiment == 'lissajous':
        experiment = LissajousExperiment(sim)
    sim.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX models in MuJoCo simulation.")

    default_onnx_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs/pointfoot_flat/exported/policies/')
    default_xml_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources/robots/pointfoot/WF_TRON1A/xml/robot.xml')

    parser.add_argument("--onnx_dir", type=str, default=default_onnx_dir, help=f"Path to directory containing ONNX models (default: {default_onnx_dir})")
    parser.add_argument("--xml_path", type=str, default=default_xml_path, help=f"Path to robot MJCF model (default: {default_xml_path})")
    parser.add_argument("--ctrl_name", type=str, default="", help="Only onnx files with <ctrl_name> in their name will be used")
    parser.add_argument("--headless", action='store_true', help="Run without visualiser")
    parser.add_argument("--experiment", type=str, choices=EXPERIMENT_CHOICES, required=True, help=f"Choose the experiment from {EXPERIMENT_CHOICES}")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)