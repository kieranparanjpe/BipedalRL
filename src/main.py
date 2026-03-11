import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import random

from robot import Robot
from rl import ActorCritic, MujocoEnvironment, BetaPolicy, NeuralNetwork, Reward1


def load_scene():
    model = mujoco.MjModel.from_xml_path("../robots/g1/scene_29dof.xml", None)
    data = mujoco.MjData(model)
    return model, data

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_networks(value_function, policy_network, new_name = False):
    suffix = f"_{int(time.time())}" if new_name else ""
    torch.save(value_function.state_dict(), f"../saved_networks/value/value_function{suffix}.pth")
    torch.save(policy_network.state_dict(), f"../saved_networks/policy/policy{suffix}.pth")


def main():
    seed_everything(1234)
    model, data = load_scene()
    robot = Robot(model, data, 'pelvis')
    policy = BetaPolicy(NeuralNetwork(layer_dimensions=(107, 256, 256, 58)))
    value_function = NeuralNetwork(layer_dimensions=(107, 128, 64, 1))

    environment = MujocoEnvironment(model, data)

    policy.neural_network.load_state_dict(torch.load("../saved_networks/policy/policy.pth"))
    value_function.load_state_dict(torch.load("../saved_networks/value/value_function.pth"))

    reward = Reward1(robot, np.array([10, 0, 0]), "torso_link")
    actor_critic = ActorCritic(environment, policy, value_function, reward, robot)

    def on_key(keycode: int):
        if keycode in (ord('l'), ord('L')):
            save_networks(value_function, policy.neural_network)
        if keycode in (ord('j'), ord('J')):
            save_networks(value_function, policy.neural_network, new_name=True)
        if keycode in (ord('p'), ord('P')):
            actor_critic.plot_stats()
    environment.set_on_key(on_key)

    actor_critic.train()

    actor_critic.plot_stats()


if __name__ == '__main__':
    main()
