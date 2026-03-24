import itertools
import os
import subprocess
import sys
import time
import signal
from typing import Optional
from dataclasses import replace

from io_controller import IOController
from rl import Hyperparameters

START_TIME = str(int(time.time()))

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
CONTINUE_TRAINING = False

# run 1 (not actually 1, but keeping track now)
'''value_lrs = [1e-7, 1e-9, 1e-11]
policy_lrs = [1e-3, 1e-4]
discount_factors = [0.98]
policy_trace_decays = [0.95]
value_trace_decays = [0.9, 0.95]
max_value_grad_norms = [1.0, 5.0]
value_function_changeouts = [200, 400]'''

# run 2
'''value_lrs = [1e-3, 1e-5, 1e-7, 1e-9]
policy_lrs = [1e-3, 1e-4]
discount_factors = [0.98]
policy_trace_decays = [0.95]
value_trace_decays = [0.95]
max_value_grad_norms = [1.0, 2.0, 5.0, 10.0]
value_function_changeouts = [400]'''

value_lrs = [1e-6, 1e-7, 3e-7, 3e-8]
policy_lrs = [3e-4, 1e-4]
discount_factors = [0.98]
policy_trace_decays = [0.95, 0.99]
value_trace_decays = [0.95, 0.99]
unified_trace_decays = [0.95, 0.99]
max_value_grad_norms = [2.0, 5.0, 10.0]
value_function_changeouts = [400]

'''value_lrs = [1e-7]
policy_lrs = [1e-3]
discount_factors = [0.98]
policy_trace_decays = [0.95]
value_trace_decays = [0.9]
max_value_grad_norms = [1.0]
value_function_changeouts = [200]'''

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
    value_function_changeout = None,
    max_episodes=1000
)

def build_hyperparams_grid():
    grid = set()
    for (
            vlr,
            plr,
            discount_factor,
            policy_trace_decay,
            value_trace_decay,
            unified_trace_decay,
            max_value_grad_norm,
            value_function_changeout
        ) in itertools.product(
            value_lrs,
            policy_lrs,
            discount_factors,
            policy_trace_decays,
            value_trace_decays,
            unified_trace_decays,
            max_value_grad_norms,
            value_function_changeouts
    ):
        if unified_trace_decay is not None:
            policy_trace_decay = unified_trace_decay
            value_trace_decay = unified_trace_decay

        h = replace(baseline_hyperparams,
                    policy_learning_rate=plr,
                    value_learning_rate=vlr,
                    discount_factor=discount_factor,
                    value_trace_decay=value_trace_decay,
                    policy_trace_decay=policy_trace_decay,
                    max_value_grad_norm=max_value_grad_norm,
                    value_function_changeout=value_function_changeout)
        grid.add(h)
    return list(grid)



def build_command(hyperparams, robot, instance, use_viewer, save_on_end, load_network_time, load_network_indexed,
                  continue_training):
    command = [
        sys.executable, "train_one.py",
        "--robot", str(robot),
        "--instance", str(instance),
        "--use_viewer", str(use_viewer),
        "--save_on_end", str(save_on_end),
        "--start_time", START_TIME
    ]

    if load_network_time is not None:
        load_index = instance if load_network_indexed else 0
        command.extend(["--load_network_time", load_network_time])
        command.extend(["--load_network_index", str(load_index)])
        command.extend(["--continue_training", str(continue_training)])

    for name, value in vars(hyperparams).items():
        if value is not None:
            command.extend([f"--{name}", str(value)])

    return command

def start_sub_processes():
    jobs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i, hyperparams in enumerate(build_hyperparams_grid()):
        cmd = build_command(
            hyperparams=hyperparams,
            robot=ROBOT,
            instance=i,
            use_viewer=USE_VIEWER,
            save_on_end=SAVE_ON_END,
            load_network_time=LOAD_NETWORK_TIME,
            load_network_indexed=LOAD_NETWORK_INDEXED,
            continue_training=CONTINUE_TRAINING,
        )

        p = subprocess.Popen(
            cmd,
            cwd=script_dir,
            stdin=subprocess.PIPE,
            text=True
        )
        jobs.append(p)
    return jobs

def write_to_jobs(jobs, msg):
    for p in jobs:
        if p.poll() is None and p.stdin is not None:
            try:
                p.stdin.write(f"{msg}\n")
                p.stdin.flush()
            except Exception:
                pass

def handle_input(io_controller, keywords):
    while io_controller.has_message():
        message = io_controller.read()
        if message.lower() in keywords:
            return message.lower()
    return None


def main():
    jobs = start_sub_processes()

    # Example: parent runs until some condition becomes true

    io_controller = IOController()

    io_keywords = ['stop', 'save_networks', 'save_data']
    while True:
        running_processes = [p for p in jobs if p.poll() is None]
        if not running_processes:
            break

        std_input = handle_input(io_controller, io_keywords)
        if std_input == 'stop':
            write_to_jobs(jobs, 'stop')
            break
        if std_input == 'save_networks':
            write_to_jobs(jobs, 'save_networks')
        if std_input == 'save_data':
            write_to_jobs(jobs, 'save_data')

        time.sleep(0.1)

    print('shutting down train many')
    time.sleep(30)

    for p in jobs:
        if p.poll() is None:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    print('shut down train many')


    io_controller.stop()

if __name__ == "__main__":
    main()
