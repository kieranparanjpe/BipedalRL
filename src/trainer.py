import shutil
import time
from datetime import datetime, timezone
import os
import mujoco
import numpy as np
import torch

from robot import Robot
from rl import ActorCritic, MujocoEnvironment, BetaPolicy, NeuralNetwork, RewardG1, RewardGo2, Policy, Reward, Hyperparameters


class Trainer:
    def __init__(self, robot_type : str, use_viewer=False, save_on_end=True, instance = 0, hyperparameters :
        Hyperparameters | None = None, start_time = 0, load_network_time=None, load_network_instance=0):

        robot_type_to_scene_file = {'g1' : '../robots/g1/scene_29dof.xml', 'go2': '../robots/go2/scene.xml'}

        self.robot_type = robot_type
        self.start_time_string = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%y_%m_%d_%H_%M_%S")
        self.instance = instance
        self.use_viewer = use_viewer
        self.load_network_time = load_network_time
        self.load_network_instance = load_network_instance
        self.save_on_end = save_on_end

        self.model = mujoco.MjModel.from_xml_path(robot_type_to_scene_file[robot_type], None)
        self.data = mujoco.MjData(self.model)

        if robot_type == 'g1':
            self.robot, self.policy, self.value_function, self.reward = self.init_g1()
        elif robot_type == 'go2':
            self.robot, self.policy, self.value_function, self.reward = self.init_go2()
        else:
            raise ValueError(f"invalid robot selection {robot_type}")

        if load_network_time is not None:
            self.load_networks()

        self.environment = MujocoEnvironment(self.model, self.data, use_viewer=use_viewer, on_key=self.on_key)
        self.actor_critic = ActorCritic(
            self.environment,
            self.policy,
            self.value_function,
            self.reward,
            self.robot,
            hyperparams=hyperparameters
        )

        if self.save_on_end:
            self.create_output_folder_structure()
            self.save_rl_folder()

    def init_g1(self) -> tuple[Robot, Policy, NeuralNetwork, Reward]:
        robot = Robot(self.model, self.data, 'pelvis', 'g1')
        policy = BetaPolicy(NeuralNetwork(layer_dimensions=(107, 256, 256, 58)))
        value_function = NeuralNetwork(layer_dimensions=(107, 128, 64, 1))
        reward = RewardG1(robot, np.array([10, 0, 0]), "torso_link")
        return robot, policy, value_function, reward

    def init_go2(self) -> tuple[Robot, Policy, NeuralNetwork, Reward]:
        robot = Robot(self.model, self.data, 'base_link', 'go2')
        policy = BetaPolicy(NeuralNetwork(layer_dimensions=(56, 256, 256, 24)))
        value_function = NeuralNetwork(layer_dimensions=(56, 128, 64, 1))
        reward = RewardGo2(robot, np.array([10, 0, 0]),
                                ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot'])
        return robot, policy, value_function, reward

    def train(self):
        self.actor_critic.train()

        if self.save_on_end:
            self.actor_critic.save_stats(
                os.path.join(self.root_out_path(self.start_time_string), "raw_data"), self.instance
            )
            self.save_networks()

    def on_key(self, keycode: int):
        timeSuffix = f"_{int(time.time())}"
        if keycode == (ord('S')):
            self.save_networks()
        if keycode == (ord('P')):
            self.actor_critic.save_stats(
                os.path.join(self.root_out_path(self.start_time_string), "raw_data"), self.instance
            )

    def load_networks(self):
        policy_path, value_path = self.network_paths(self.load_network_time, self.load_network_time)
        self.policy.neural_network.load_state_dict(torch.load(policy_path))
        self.value_function.load_state_dict(torch.load(value_path))

    def save_networks(self):
        policy_path, value_path = self.network_paths(self.start_time_string, self.instance)

        torch.save(self.policy.neural_network.state_dict(),policy_path)
        torch.save(self.value_function.state_dict(),value_path)

    def save_rl_folder(self):
        destination = os.path.join(self.root_out_path(self.start_time_string), "rl")
        rl_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "rl"))
        if not os.path.exists(destination):
            shutil.copytree(rl_folder, destination)

    def network_paths(self, time_, instance):
        saved_networks_path = os.path.join(self.root_out_path(time_), "saved_networks")

        policy_path = os.path.join(saved_networks_path, "policy", f"policy_{instance}.pth")
        value_path = os.path.join(saved_networks_path, "value", f"value_{instance}.pth")
        return policy_path, value_path

    def root_out_path(self, time_):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "train_information",
             self.robot_type, f"instance_{time_}")
        )

    def create_output_folder_structure(self):
        output_root = self.root_out_path(self.start_time_string)
        raw_data_path = os.path.join(output_root, "raw_data")
        policy_path = os.path.join(output_root, "saved_networks", "policy")
        value_function = os.path.join(output_root, "saved_networks", "value")

        os.makedirs(raw_data_path, exist_ok=True)
        os.makedirs(policy_path, exist_ok=True)
        os.makedirs(value_function, exist_ok=True)

        print(f"Writing training output for instance [{self.instance}] to {output_root}")
