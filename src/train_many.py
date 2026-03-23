import itertools
import os
import subprocess
import sys
import time
from typing import Optional
from dataclasses import replace

from rl import Hyperparameters

# Global run settings passed to each child process.
ROBOT = "go2"
USE_VIEWER = False
SAVE_ON_END = True

# Indicates if we should load a previously saved network.
# the time YY_MM_DD_HH_MM_SS when it was run
LOAD_NETWORK_TIME : Optional[str] = None
# whether or not to index the model loaded. If false, we always load the first model.
# Has no effect if LOAD_NETWORK_TIME is None
LOAD_NETWORK_INDEXED = False

value_lrs = [3e-6, 1e-5, 3e-5]
policy_lrs = [3e-6, 1e-5, 3e-5]
discount_factors = [0.95, 0.98]
trace_decays = [0.9, 0.95]

baseline_hyperparams = Hyperparameters(
    policy_learning_rate = 1e-8,
    value_learning_rate = 3e-8,
    policy_trace_decay = 0.95,
    value_trace_decay = 0.95,
    discount_factor = 0.95,
    max_td_error_mag = 2.0,
    max_value_trace = 8.0,
    max_policy_trace = 8.0,
    max_value_weight_update = 0.01,
    max_policy_weight_update = 0.01,
    max_policy_grad_norm = 2.0,
    max_value_grad_norm = 1.0,
    discount_factor_decay = 1,
    value_function_changeout = None
)

def build_hyperparams_grid():
    grid = []
    for (vlr, plr, discount_factor, trace_decay) in itertools.product(value_lrs, policy_lrs, discount_factors,
                                                                  trace_decays):
        h = replace(baseline_hyperparams,
                    policy_learning_rate=plr,
                    value_learning_rate=vlr,
                    discount_factor=discount_factor,
                    value_trace_decay=trace_decay,
                    policy_trace_decay=trace_decay)
        grid.append(h)
    return grid



def build_command(hyperparams, robot, instance, use_viewer, save_on_end, load_network_time, load_network_indexed):
    command = [
        sys.executable, "train_one.py",
        "--robot", str(robot),
        "--instance", str(instance),
        "--use_viewer", str(use_viewer),
        "--save_on_end", str(save_on_end),
        "--start_time", str(int(time.time()))
    ]

    if load_network_time is not None:
        load_index = instance if load_network_indexed else 0
        command.extend(["--load_network_time", load_network_time])
        command.extend(["--load_network_index", str(load_index)])

    for name, value in vars(hyperparams).items():
        if value is not None:
            command.extend([f"--{name}", str(value)])

    return command

def main():
    jobs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i, hyperparams in enumerate(build_hyperparams_grid()):
        instance = i
        cmd = build_command(
            hyperparams=hyperparams,
            robot=ROBOT,
            instance=instance,
            use_viewer=USE_VIEWER,
            save_on_end=SAVE_ON_END,
            load_network_time=LOAD_NETWORK_TIME,
            load_network_indexed=LOAD_NETWORK_INDEXED
        )
        p = subprocess.Popen(cmd, cwd=script_dir)
        jobs.append(p)

    try:
        for p in jobs:
            p.wait()
    except KeyboardInterrupt:
        for p in jobs:
            if p.poll() is None:
                p.terminate()

        for p in jobs:
            if p.poll() is None:
                try:
                    p.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    main()
