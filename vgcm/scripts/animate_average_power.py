import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from cycler import cycler

from vgcm.colours import colours
from vgcm.vgcm_ideal_model import COMPENSATION_OPTIONS
from legged_gym import LEGGED_GYM_ROOT_DIR

plt.rcParams['axes.prop_cycle'] = cycler(color=colours)
print(f"Available writers: {animation.writers.list()}")


main_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'vgcm/experiment_results')
models = ['VGCM-ma', 'VGCM-cv']
models = ['VGCM-cv']

dfs = {}
for model in models:
    dfs[model] = {}
    path_to_files = os.path.join(main_dir, model)
    for file in os.listdir(path_to_files):
        filepath = os.path.join(path_to_files, file)
        for opt in COMPENSATION_OPTIONS:
            if opt in filepath:
                dfs[model][opt] = pd.read_csv(filepath)


WHEELS = [3, 7]
COMPENSATED = [0, 1, 2, 4, 5, 6]


def get_power_array(model, opt):
    df = dfs[model][opt]
    P = np.zeros(len(df))
    v = df["base_lin_vel_x"]
    t = df["step"]
    for i in range(8):
        dq = df[f'dq{i}']
        full_tau = df[f'tau{i}']
        if opt != 'none' and i in COMPENSATED:
            idx = COMPENSATED.index(i)
            gc_tau = df[f'gc{idx}_tau']
            actuator_tau = full_tau - gc_tau
        else:
            actuator_tau = full_tau

        if i in WHEELS:
            Kt = 1. / (2*np.pi)
            R = 0.144
        else:
            Kt = 1. / (2*np.pi)
            R = 0.144

        P += actuator_tau * dq + np.square(actuator_tau) * R / np.square(Kt)
    return P, t, v


# Precompute power data for each model-option pair
data = {}
for opt in COMPENSATION_OPTIONS:
    data[opt] = {}
    if opt == 'none':
        P, t, v = get_power_array('VGCM-cv', opt)
        data[opt]['VGCM-cv'] = (P, t)
    else:
        for model in models:
            P, t, v = get_power_array(model, opt)
            data[opt][model] = (P, t)


for opt in COMPENSATION_OPTIONS:
    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.set_ylim(0, 40_000)
    lines = []
    P, t = data[opt]['VGCM-cv']
    if opt == 'none':
        line, = ax.plot([], [], label="no compensation")
        lines.append(line)
    else:
        for model in data[opt]:
            line, = ax.plot([], [], label=f"{model}-{opt}")
            lines.append(line)
    ax.legend(loc="upper left")

    fps = int(1000/16)
    n_frames = (t.iloc[-1] - t.iloc[0]) * fps

    # Animation update function
    def update(frame):
        current_time = frame / fps
        window = 5.
        if frame % 60 == 0:
            print(f"Frame: {frame} time: {current_time}")
        if opt == 'none':
            P, t = data[opt]['VGCM-cv']
            mask = (t <= current_time) & (current_time-window <= t)  # Select data up to current time
            lines[0].set_data(t[mask], P[mask])
            ax.set_xlim(current_time-window, current_time)
        else:
            for model, line in zip(data[opt], lines):
                P, t = data[opt][model]
                mask = (t <= current_time) & (current_time-window <= t)  # Select data up to current time
                line.set_data(t[mask], P[mask])
                ax.set_xlim(current_time-window, current_time)
        return lines[0]

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, n_frames), interval=16, blit=False)

    # Save as MP4 using FFMpegWriter
    output_filename = f"{model}-{opt}_power_animation.mp4"
    output_path = os.path.join(LEGGED_GYM_ROOT_DIR, "vgcm/experiment_results/plots", output_filename)
    writer = animation.FFMpegWriter(fps=fps)
    print(f"Saving {n_frames} frame animation to {output_path}...")
    ani.save(output_path, writer=writer)

    print(f"Saved animation to {output_path}.")
