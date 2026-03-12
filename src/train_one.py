import argparse
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
                      viewer=args.use_viewer,
                      save_on_end=args.save_output,
                      load_suffix=args.load_suffix,
                      instance=args.instance,
                      hyperparameters=hyperparams)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=str, default='go2')
    parser.add_argument("--use_viewer", type=str2bool, default=True)
    parser.add_argument("--save_output", type=str2bool, default=False)
    parser.add_argument("--load_suffix", type=str, default=None)
    parser.add_argument("--instance", type=int, default=None)

    for field in fields(Hyperparameters):
        parser.add_argument(f"--{field.name}", type=hyperparam_arg_type(field.type), default=None)

    args = parser.parse_args()
    main(args)
