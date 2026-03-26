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

def main():
    robot_type_to_scene_file = {'g1': '../robots/g1/scene_29dof.xml', 'go2': '../robots/go2/scene.xml',
                                'cube': '../robots/cube/scene.xml'}

    model = mujoco.MjModel.from_xml_path(robot_type_to_scene_file['cube'], None)
    data = mujoco.MjData(model)

    robot = Robot(model, data, 'cube', 'cube')
    reward = RewardCube(robot, np.array([1, 0, 0]),
                        max_steps_without_improvement=300)

    environment = MujocoEnvironment(model, data, use_viewer=True)

    with environment as environment:
        while environment.is_running():
            robot.set_ctrls(np.array([1, 0]))
            environment.step()
            r = reward.reward()

if __name__ == '__main__':
    main()