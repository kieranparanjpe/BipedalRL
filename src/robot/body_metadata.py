from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .joint_metadata import JointMetadata


class BodyMetadata:

    def __init__(self, name, localId, body_id, joints: List[JointMetadata]):
        self.name = name
        self.localId = localId # local id is how I should refer to this body from within this robot. If the robot
        # has 5 bodies, its localIds will be 0 1 2 3 4.
        self.body_id = body_id # body_id is the bid in the mujoco.model. This may not begin at 0. for example,
        # it may be for a robot with 5 bodies, its 4, 5, 6, 7, 8. It depends on what else is in the scene.
        self.joints = joints
