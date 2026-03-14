import time
import csv
import os
import mujoco
import numpy as np
import torch

from robot import Robot
from rl import ActorCritic, MujocoEnvironment, BetaPolicy, NeuralNetwork, RewardG1, RewardGo2, Hyperparameters

class Trainer:

    def __init__(self, robot_type, viewer=False, load_suffix=None, save_on_end=True, instance = None,
                 hyperparameters : Hyperparameters | None = None):
        robot_type_to_scene_file = {'g1' : '../robots/g1/scene_29dof.xml', 'go2': '../robots/go2/scene.xml'}

        self.reward = None
        self.value_function = None
        self.policy = None
        self.robot = None

        self.instanceSuffix = f"_I{instance}" if instance is not None else ''
        self.instance = instance

        self.model = mujoco.MjModel.from_xml_path(robot_type_to_scene_file[robot_type], None)
        self.data = mujoco.MjData(self.model)

        self.viewer = viewer
        self.load_suffix = load_suffix
        self.save_on_end = save_on_end

        self.robot_type = robot_type
        if robot_type == 'g1':
            self.init_g1()
        elif robot_type == 'go2':
            self.init_go2()
        else:
            raise ValueError(f"invalid robot selection {robot_type}")

        if self.policy is None or self.value_function is None or self.robot is None or self.reward is None:
            return

        if load_suffix is not None:
            self.policy.neural_network.load_state_dict(torch.load(
                f"../saved_networks/{robot_type}/policy/policy{load_suffix}.pth"))
            self.value_function.load_state_dict(torch.load(
                f"../saved_networks/{robot_type}/value/value_function{load_suffix}.pth"))

        self.environment = MujocoEnvironment(self.model, self.data, use_viewer=viewer, on_key=self.on_key)
        if hyperparameters is None:
            self.actor_critic = ActorCritic(
                self.environment,
                self.policy,
                self.value_function,
                self.reward,
                self.robot
            )
        else:
            self.actor_critic = ActorCritic(
                self.environment,
                self.policy,
                self.value_function,
                self.reward,
                self.robot,
                hyperparams=hyperparameters
            )


    def init_g1(self):
        self.robot = Robot(self.model, self.data, 'pelvis', 'g1')
        self.policy = BetaPolicy(NeuralNetwork(layer_dimensions=(107, 256, 256, 58)))
        self.value_function = NeuralNetwork(layer_dimensions=(107, 128, 64, 1))
        self.reward = RewardG1(self.robot, np.array([10, 0, 0]), "torso_link")

    def init_go2(self):
        self.robot = Robot(self.model, self.data, 'base_link', 'go2')
        self.policy = BetaPolicy(NeuralNetwork(layer_dimensions=(56, 256, 256, 24)))
        self.value_function = NeuralNetwork(layer_dimensions=(56, 128, 64, 1))
        self.reward = RewardGo2(self.robot, np.array([10, 0, 0]),
                                ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot'])

    def train(self):

        self.actor_critic.train()

        if self.save_on_end:
            timeSuffix = f"_{int(time.time())}"
            self._append_train_info_row(timeSuffix)
            self.actor_critic.plot_stats(save_directory=f"../saved_plots/{self.robot.robot_type}",
                                         suffix=self.instanceSuffix + timeSuffix)
            self.save_networks(suffix=self.instanceSuffix + timeSuffix)

    def _append_train_info_row(self, timeSuffix: str):
        csv_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "train_information", "train_info.csv")
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        hyperparams = self.actor_critic.hyperparams
        hyperparam_values = vars(hyperparams)
        fieldnames = ["full_suffix", "instance", "timeSuffix", "robot_type", *hyperparam_values.keys()]

        row = {
            "full_suffix": self.instanceSuffix + timeSuffix,
            "instance": self.instance,
            "timeSuffix": timeSuffix,
            "robot_type": self.robot_type,
            **hyperparam_values,
        }

        needs_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


    def on_key(self, keycode: int):
        timeSuffix = f"_{int(time.time())}"
        if keycode == (ord('S')):
            self.save_networks(self.instanceSuffix + timeSuffix)
        if keycode == (ord('P')):
            self.actor_critic.plot_stats()
        if keycode == (ord('O')):
            self.actor_critic.plot_stats(save_directory=f"../saved_plots/{self.robot_type}",
                                         suffix=self.instanceSuffix + timeSuffix)

    def save_networks(self, suffix=''):
        torch.save(self.value_function.state_dict(),
                   f"../saved_networks/{self.robot_type}/value/value_function{suffix}.pth")
        torch.save(self.policy.neural_network.state_dict(),
                   f"../saved_networks/{self.robot_type}/policy/policy{suffix}.pth")
