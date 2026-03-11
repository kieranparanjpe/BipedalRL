from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .body_metadata import BodyMetadata

if TYPE_CHECKING:
    from .actuator_metadata import ActuatorMetadata


class JointMetadata:

    def __init__(self, name, localId, joint_id, qpos_slice, qvel_slice, body: BodyMetadata, actuator: Optional[
        ActuatorMetadata] = None):
        self.name = name
        self.localId = localId # local id is how I should refer to this joint from within this robot. If the robot
        # has 5 joints, its localIds will be 0 1 2 3 4.
        self.joint_id = joint_id # joint_id is the jid in the mujoco.model. This may not begin at 0. for example,
        # it may be for a robot with 5 joints, its 4, 5, 6, 7, 8. It depends on what else is in the scene.
        self.body = body # mujo body id
        self.qpos_slice = qpos_slice
        self.qvel_slice = qvel_slice
        self.actuator = actuator
