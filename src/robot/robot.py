from typing import Optional, List, Dict

import mujoco
import numpy as np

from .actuator_metadata import ActuatorMetadata
from .joint_metadata import JointMetadata
from .body_metadata import BodyMetadata


class Robot:

    def __init__(self, model : mujoco.MjModel, data : mujoco.MjData, root_name : str):
        self.model : mujoco.MjModel= model
        self.data : mujoco.MjData  = data

        self.root_name = root_name

        def direct_joints_of_body(bid: int) -> list[int]:
            nj = model.body_jntnum[bid]
            adr = model.body_jntadr[bid]
            if nj <= 0 or adr < 0:
                return []
            return list(range(adr, adr + nj))

        def actuator_for_joint(jid : int) -> Optional[int]:
            for aid in range(model.nu):
                joint_target = model.actuator_trnid[aid, 0]
                if joint_target == jid:
                    return aid
            return None

        def get_bodies_in_subtree() -> List[int]:
            root = model.body(root_name)
            bodies_in_subtree = []
            for bid in range(model.nbody):
                furthest_parent_id = model.body_parentid[bid]
                while furthest_parent_id != -1:
                    if bid == root.id or furthest_parent_id == root.id:
                        bodies_in_subtree.append(bid)
                        break
                    if model.body_parentid[furthest_parent_id] == furthest_parent_id:
                        break
                    furthest_parent_id = model.body_parentid[furthest_parent_id]
            return bodies_in_subtree

        def qpos_slice(jid : int) -> slice:
            start = model.jnt_qposadr[jid]
            if jid + 1 < model.njnt:
                end = model.jnt_qposadr[jid + 1]
            else:
                end = model.nq
            return slice(start, end)

        def qvel_slice(jid : int) -> slice:
            start = model.jnt_dofadr[jid]
            if jid + 1 < model.njnt:
                end = model.jnt_dofadr[jid + 1]
            else:
                end = model.nv
            return slice(start, end)

        self.joints : List[JointMetadata] = []
        self.actuators : List[ActuatorMetadata] = []
        self.bodies : List[BodyMetadata] = []

        self.qpos_all_indices = []
        self.qvel_all_indices = []
        self.ctrl_all_indices = []
        self.revolute_joint_indices = []

        self.jointNameToLocalId : Dict[str, int] = {}
        self.actuatorNameToLocalId : Dict[str, int] = {}
        self.bodyNameToLocalId : Dict[str, int] = {}

        for body_id in get_bodies_in_subtree():
            body = model.body(body_id)
            bodyMetadata = BodyMetadata(body.name, len(self.bodies), body.id, [])
            self.bodies.append(bodyMetadata)
            self.bodyNameToLocalId[bodyMetadata.name] = bodyMetadata.localId

            joint_ids = direct_joints_of_body(body_id)

            for joint_id in joint_ids:
                joint = model.joint(joint_id)
                qpos_slice_ = qpos_slice(joint.id)
                qvel_slice_ = qvel_slice(joint.id)

                self.qpos_all_indices.extend(range(qpos_slice_.start, qpos_slice_.stop))
                self.qvel_all_indices.extend(range(qvel_slice_.start, qvel_slice_.stop))

                if joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                    self.revolute_joint_indices.extend(range(qpos_slice_.start, qpos_slice_.stop))

                jointMetadata = JointMetadata(joint.name, len(self.joints), joint.id, qpos_slice_, qvel_slice_, bodyMetadata)
                self.jointNameToLocalId[joint.name] = jointMetadata.localId
                bodyMetadata.joints.append(jointMetadata)

                actuator_id = actuator_for_joint(joint_id)
                if actuator_id is not None:
                    actuator = model.actuator(actuator_id)
                    actuatorMetadata = ActuatorMetadata(actuator.name, len(self.actuators), actuator.id, jointMetadata)
                    self.actuatorNameToLocalId[actuatorMetadata.name] = actuatorMetadata.localId
                    jointMetadata.actuator = actuatorMetadata
                    self.ctrl_all_indices.append(actuator.id)
                    self.actuators.append(actuatorMetadata)

                self.joints.append(jointMetadata)

        self.qpos_all_indices = np.array(self.qpos_all_indices)
        self.qvel_all_indices = np.array(self.qvel_all_indices)
        self.ctrl_all_indices = np.array(self.ctrl_all_indices)
        self.revolute_joint_indices = np.array(self.revolute_joint_indices)

        self.ctrl_range = self.model.actuator_ctrlrange[self.ctrl_all_indices][:,1] / 2

    def resolve_joint_name_local_id(self, idOrName : str | int) -> int:
        if isinstance(idOrName, int):
            return idOrName

        return self.jointNameToLocalId[idOrName]

    def resolve_actuator_name_local_id(self, idOrName : str | int) -> int:
        if isinstance(idOrName, int):
            return idOrName

        return self.actuatorNameToLocalId[idOrName]

    def resolve_body_name_local_id(self, idOrName : str | int) -> int:
        if isinstance(idOrName, int):
            return idOrName

        return self.bodyNameToLocalId[idOrName]

    def get_positions(self):
        return self.data.qpos[self.qpos_all_indices]

    def get_position(self, localIdOrName : str | int):
        id_ = self.resolve_joint_name_local_id(localIdOrName)
        return self.data.qpos[self.joints[id_].qpos_slice].copy()

    def get_position_sin_cos(self, localIdOrName : str | int):
        position = self.get_position(localIdOrName)
        return np.concatenate((np.sin(position), np.cos(position)))

    def get_positions_sin_cos(self):
        position = self.get_positions()
        sin_pos = position.copy()
        sin_pos[self.revolute_joint_indices] = np.sin(sin_pos[self.revolute_joint_indices])

        cos_pos = position.copy()
        cos_pos[self.revolute_joint_indices] = np.cos(cos_pos[self.revolute_joint_indices])

        return np.concatenate((sin_pos, cos_pos))

    def get_velocities(self):
        return self.data.qvel[self.qvel_all_indices]

    def get_velocity(self, localIdOrName : str | int):
        id_ = self.resolve_joint_name_local_id(localIdOrName)
        return self.data.qvel[self.joints[id_].qvel_slice].copy()

    def get_accelerations(self):
        return self.data.qacc[self.qvel_all_indices]

    def get_acceleration(self, localIdOrName : str | int):
        id_ = self.resolve_joint_name_local_id(localIdOrName)
        return self.data.qacc[self.joints[id_].qvel_slice].copy()

    def get_state(self):
        return np.concatenate((self.get_positions(), self.get_velocities(), self.get_accelerations()))

    def get_state_sin_cos(self):
        return np.concatenate((self.get_positions_sin_cos(), self.get_velocities(), self.get_accelerations()))

    def get_state_sin_cos_no_accel(self):
        return np.concatenate((self.get_positions_sin_cos(), self.get_velocities()))


    # expects -1 <= ctrl <= 1
    def set_ctrl(self, localIdOrName : str | int, ctrl):
        id_ = self.resolve_actuator_name_local_id(localIdOrName)
        self.data.ctrl[self.actuators[id_].actuator_id] = ctrl * self.ctrl_range[id_]

    # expects -1 <= ctrls <= 1 for all elements
    def set_ctrls(self, ctrls):
        self.data.ctrl[self.ctrl_all_indices] = ctrls * self.ctrl_range

    def compute_forward_kinematics(self):
        mujoco.mj_kinematics(self.model, self.data)

    # returns body position
    # assume we have already called compute_forward_kinematics() above
    def get_world_position(self, localIdOrName : str | int):
        id_ = self.resolve_body_name_local_id(localIdOrName)
        return self.data.xpos[self.bodies[id_].body_id]

    # returns  body rotation
    # assume we have already called compute_forward_kinematics() above
    def get_world_rotation(self, localIdOrName : str | int):
        id_ = self.resolve_body_name_local_id(localIdOrName)
        return self.data.xmat[self.bodies[id_].body_id]




