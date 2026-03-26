import shutil
import time
from datetime import datetime, timezone
import os
from typing import Optional

import mujoco
import numpy as np
import torch

from io_controller import IOController
from robot import Robot
from rl import ActorCritic, MujocoEnvironment, BetaPolicy, NeuralNetwork, RewardG1, RewardGo2, Policy, Reward, \
    Hyperparameters, RewardCube


class Trainer:
    def __init__(self, robot_type : str, io_controller : IOController,
                 use_viewer=False, save_on_end=True, instance = 0,
                 hyperparameters : Optional[Hyperparameters] = None, start_time = 0,
                 load_network_time=None, load_network_instance=0, continue_training=False):

        robot_type_to_scene_file = {'g1' : '../robots/g1/scene_29dof.xml', 'go2': '../robots/go2/scene.xml',
                                    'cube' : '../robots/cube/scene.xml'}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.robot_type = robot_type
        self.start_time_string = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%y_%m_%d_%H_%M_%S")
        self.instance = instance
        self.use_viewer = use_viewer
        self.load_network_time = load_network_time
        self.load_network_instance = load_network_instance
        self.continue_training = continue_training and self.load_network_time is not None
        self.save_on_end = save_on_end
        self.io_controller = io_controller

        self.hyperparams = hyperparameters if hyperparameters is not None else Hyperparameters()

        self.model = mujoco.MjModel.from_xml_path(robot_type_to_scene_file[robot_type], None)
        self.data = mujoco.MjData(self.model)


        if robot_type == 'g1':
            self.robot, self.policy, self.value_function, self.reward = self.init_g1()
        elif robot_type == 'go2':
            self.robot, self.policy, self.value_function, self.reward = self.init_go2()
        elif robot_type == 'cube':
            self.robot, self.policy, self.value_function, self.reward = self.init_cube()
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
            hyperparams=self.hyperparams,
            timestep_callback=self.timestep_callback,
            device=self.device
        )

        if self.load_network_time is not None:
            self.actor_critic.load_stats(
                self.raw_data_path(self.load_network_time), self.load_network_instance, self.continue_training
            )

        if self.save_on_end:
            self.create_output_folder_structure()
            self.save_rl_folder()

    def init_g1(self) -> tuple[Robot, Policy, NeuralNetwork, Reward]:
        robot = Robot(self.model, self.data, 'pelvis', 'g1')
        policy = BetaPolicy(NeuralNetwork(layer_dimensions=(107, 256, 256, 58)).to(self.device))
        value_function = NeuralNetwork(layer_dimensions=(107, 128, 64, 1)).to(self.device)
        reward = RewardG1(robot, np.array([1, 0, 0]), "torso_link")
        return robot, policy, value_function, reward

    def init_go2(self) -> tuple[Robot, Policy, NeuralNetwork, Reward]:
        robot = Robot(self.model, self.data, 'base_link', 'go2')
        policy = BetaPolicy(NeuralNetwork(layer_dimensions=(56, 256, 256, 24)).to(self.device))
        value_function = NeuralNetwork(layer_dimensions=(56, 128, 64, 1)).to(self.device)
        reward = RewardGo2(robot, np.array([1, 0, 0]),
                                ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot'])
        return robot, policy, value_function, reward

    def init_cube(self) -> tuple[Robot, Policy, NeuralNetwork, Reward]:
        robot = Robot(self.model, self.data, 'cube', 'cube')
        policy = BetaPolicy(NeuralNetwork(layer_dimensions=(4, 16, 32, 4)).to(self.device))
        value_function = NeuralNetwork(layer_dimensions=(4, 16, 32, 1)).to(self.device)
        reward = RewardCube(robot, np.array([1, 0, 0]),
                            max_steps_without_improvement=self.hyperparams.episodes_without_improvement)
        return robot, policy, value_function, reward

    def timestep_callback(self, timestep : int):
        while self.io_controller.has_message():
            message = self.io_controller.read()
            if message.lower() == "stop":
                self.environment.end_environment()
            if message.lower() == "save_networks" and self.save_on_end:
                self.save_networks()
            if message.lower() == "save_data" and self.save_on_end:
                self.actor_critic.save_stats(
                    self.raw_data_path(self.start_time_string), self.instance
                )

    def train(self):
        self.actor_critic.train(self.instance)
        print(f"Shutting down instance {self.instance}")
        if self.save_on_end:
            self.save_networks()
            self.actor_critic.save_stats(
                self.raw_data_path(self.start_time_string), self.instance
            )
        print(f"Shut down instance {self.instance}")

    def on_key(self, keycode: int):
        timeSuffix = f"_{int(time.time())}"
        if keycode == (ord('S')):
            self.save_networks()
        if keycode == (ord('P')):
            self.actor_critic.save_stats(
                self.raw_data_path(self.start_time_string), self.instance
            )

    def load_networks(self):
        policy_path, value_path = self.network_paths(self.load_network_time, self.load_network_instance)
        self.policy.neural_network.load_state_dict(torch.load(policy_path, map_location=self.device))
        self.value_function.load_state_dict(torch.load(value_path, map_location=self.device))

    def save_networks(self):
        print(f"Saving networks for instance: {self.instance}")
        policy_path, value_path = self.network_paths(self.start_time_string, self.instance)

        torch.save(self.policy.neural_network.state_dict(),policy_path)
        torch.save(self.value_function.state_dict(),value_path)

    def save_rl_folder(self):
        destination = os.path.join(self.root_out_path(self.start_time_string), "rl")
        rl_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "rl"))
        if not os.path.exists(destination):
            try:
                shutil.copytree(rl_folder, destination)
            except Exception:
                pass

    def network_paths(self, time_, instance):
        saved_networks_path = os.path.join(self.root_out_path(time_), "saved_networks")

        policy_path = os.path.join(saved_networks_path, "policy", f"policy_{instance}.pth")
        value_path = os.path.join(saved_networks_path, "value", f"value_{instance}.pth")
        return policy_path, value_path

    def raw_data_path(self, time_):
        return os.path.join(self.root_out_path(time_), "raw_data")

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

        # print(f"Writing training output for instance [{self.instance}] to {output_root}")
