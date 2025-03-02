import argparse
import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from vgcm.mujoco_sim import Simulator
from vgcm.onnx_controller import Controller
from vgcm.vgcm_ideal_model import make_compensator, COMPENSATION_OPTIONS, ALPHA_MUL_K, ALPHA_MUL_X

from vgcm.experiments.linear_experiments import BasicLinearExperiment, WeightChangeLinearExperiment
from vgcm.experiments.lissajous_experiments import LissajousExperiment

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_CHOICES = ['basic_linear', 'weights_linear', 'lissajous']


def main(args):
    controllers = []
    alphas = np.zeros((2,), dtype=float)
    if args.with_compensation != 'none':
        alphas[0] = ALPHA_MUL_K[args.with_compensation]
        alphas[1] = ALPHA_MUL_X[args.with_compensation]
    for model_file in os.listdir(args.onnx_dir):
        if model_file.endswith(".onnx") and args.ctrl_name in model_file:
            onnx_model_path = os.path.join(args.onnx_dir, model_file)
            controllers.append(Controller(onnx_model_path, alphas))
    jnt_compensators = []
    for jnt_id in range(6):
        jnt_compensators.append(make_compensator(args.with_compensation, jnt_id))
    compensators = [jnt_compensators for _ in controllers]
    sim = Simulator(
        args.xml_path, controllers, compensators,
        headless=args.headless)
    if args.experiment == 'basic_linear':
        _experiment = BasicLinearExperiment(sim)
    elif args.experiment == 'weights_linear':
        _experiment = WeightChangeLinearExperiment(sim)
    elif args.experiment == 'lissajous':
        _experiment = LissajousExperiment(sim, args.with_compensation)

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
    parser.add_argument("--with_compensation", type=str, choices=COMPENSATION_OPTIONS, default='none', help=f"Choose which compensator parameters to use. Options are {COMPENSATION_OPTIONS}")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)