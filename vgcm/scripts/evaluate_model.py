import argparse
import os

from legged_gym import LEGGED_GYM_ROOT_DIR
from vgcm.mujoco_sim import Simulator
from vgcm.onnx_controller import Controller


def main(args):
    for model_file in os.listdir(args.onnx_dir):
        if model_file.endswith(".onnx"):
            onnx_model_path = os.path.join(args.onnx_dir, model_file)
            controller = Controller(onnx_model_path)
            sim = Simulator(args.xml_path, controller, test_duration=args.test_duration, headless=args.headless)
            sim.run()


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