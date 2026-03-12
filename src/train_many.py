import os
import subprocess
import sys
from rl import Hyperparameters

# Global run settings passed to each child process.
ROBOT = "go2"
USE_VIEWER = False
SAVE_OUTPUT = True
LOAD_SUFFIX = None
LOAD_SUFFIX_INDEXED = False

hyperparams_grid = [
    Hyperparameters(
        policy_learning_rate=1e-12,
        value_learning_rate=1e-8,
        policy_trace_decay=0.90,
        value_trace_decay=0.90,
        discount_factor=0.98,
        policy_changeout=0,
        value_changeout=0,
    ),
    Hyperparameters(
        policy_learning_rate=1e-11,
        value_learning_rate=1e-7,
        policy_trace_decay=0.95,
        value_trace_decay=0.95,
        discount_factor=0.95,
        policy_changeout=0,
        value_changeout=0,
    ),
    Hyperparameters(
        policy_learning_rate=1e-11,
        value_learning_rate=1e-7,
        policy_trace_decay=0.90,
        value_trace_decay=0.90,
        discount_factor=0.98,
        policy_changeout=0,
        value_changeout=0,
    ),
    Hyperparameters(
        policy_learning_rate=1e-10,
        value_learning_rate=1e-6,
        policy_trace_decay=0.95,
        value_trace_decay=0.95,
        discount_factor=0.95,
        policy_changeout=0,
        value_changeout=0,
    ),
    Hyperparameters(
        policy_learning_rate=1e-10,
        value_learning_rate=1e-6,
        policy_trace_decay=0.90,
        value_trace_decay=0.90,
        discount_factor=0.98,
        policy_changeout=0,
        value_changeout=0,
    )
]


def build_cmd(hyperparams, robot, instance, use_viewer, save_output, load_suffix):
    cmd = [
        sys.executable, "train_one.py",
        "--robot", str(robot),
        "--instance", str(instance),
        "--use_viewer", str(use_viewer),
        "--save_output", str(save_output),
    ]

    if load_suffix is not None:
        load_suffix = f"_I{instance}{load_suffix}" if LOAD_SUFFIX_INDEXED else str(load_suffix)
        cmd.extend(["--load_suffix", load_suffix])

    for name, value in vars(hyperparams).items():
        if value is not None:
            cmd.extend([f"--{name}", str(value)])

    return cmd


if __name__ == "__main__":
    jobs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for i, hyperparams in enumerate(hyperparams_grid):
        instance = i
        cmd = build_cmd(
            hyperparams=hyperparams,
            robot=ROBOT,
            instance=instance,
            use_viewer=USE_VIEWER,
            save_output=SAVE_OUTPUT,
            load_suffix=LOAD_SUFFIX,
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
