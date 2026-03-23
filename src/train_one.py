import argparse
import time
from dataclasses import fields

import numpy as np
import torch
import random

from rl import Hyperparameters
from trainer import Trainer


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid bool value: {value}")


def hyperparam_arg_type(field_type):
    if field_type in {float, int, str}:
        return field_type
    if isinstance(field_type, str):
        return {"float": float, "int": int, "str": str}.get(field_type, float)
    return float


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def main(args):
    seed_everything(1234)

    hyperparam_values = {field.name: getattr(args, field.name) for field in fields(Hyperparameters)}

    if all(value is None for value in hyperparam_values.values()):
        hyperparams = None
    else:
        hyperparams = Hyperparameters(**hyperparam_values)

    trainer = Trainer(args.robot,
                      use_viewer=args.use_viewer,
                      save_on_end=args.save_on_end,
                      instance=args.instance,
                      start_time=args.start_time,
                      load_network_time=args.load_network_time,
                      load_network_instance=args.load_network_index,
                      hyperparameters=hyperparams)
    trainer.train()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default='go2')
    parser.add_argument("--use_viewer", type=str2bool, default=True)
    parser.add_argument("--save_on_end", type=str2bool, default=False)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--start_time", type=int, default=int(time.time()))
    parser.add_argument("--load_network_time", type=str, default=None)
    parser.add_argument("--load_network_index", type=int, default=0)

    for field in fields(Hyperparameters):
        parser.add_argument(f"--{field.name}", type=hyperparam_arg_type(field.type), default=None)

    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
