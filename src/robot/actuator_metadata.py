from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .joint_metadata import JointMetadata


class ActuatorMetadata:

    def __init__(self, name, localId, actuator_id, joint: JointMetadata):
        self.name = name
        self.localId = localId # local id is how I should refer to this actuator from within this robot. If the robot
        # has 5 actuators, its localIds will be 0 1 2 3 4.
        self.actuator_id = actuator_id # the actual mujoco actuator id.
        self.joint = joint
