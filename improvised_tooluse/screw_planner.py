"""Local screw-planning corrections for the project's custom MPLib wheel."""

import numpy as np
from scipy.spatial.transform import Rotation
from mplib.sapien_utils.conversion import SapienPlanner


def pose_error(goal, current):
    """Position error in metres and shortest rotation error in radians."""
    rotation = Rotation.from_matrix(goal.to_transformation_matrix()[:3, :3])
    actual = Rotation.from_matrix(current.to_transformation_matrix()[:3, :3])
    return np.linalg.norm(goal.p - current.p), (rotation * actual.inv()).magnitude()


def spatial_twist(relative_pose):
    """SE(3) logarithm, ordered [translation, rotation] for MPLib's Jacobian."""
    transform = relative_pose.to_transformation_matrix()
    rotation = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    angle = np.linalg.norm(rotation)
    x, y, z = rotation
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    # The series avoids cancellation near zero; rotvec also handles pi rotations.
    coefficient = (1 / 12 + angle**2 / 720 if angle < 1e-3 else
                   (1 - 0.5 * angle / np.tan(0.5 * angle)) / angle**2)
    translation = (np.eye(3) - 0.5 * skew + coefficient * skew @ skew) @ transform[:3, 3]
    return np.concatenate((translation, rotation))


class CheckedSapienPlanner(SapienPlanner):
    """Use the installed wheel's world, IK, RRT and TOPP with corrected screw IK."""

    POSITION_TOLERANCE = 1e-4
    ROTATION_TOLERANCE = 1e-4

    def plan_screw(self, goal_pose, current_qpos, *, qpos_step=0.1,
                   time_step=0.1, wrt_world=True, verbose=False):
        if not np.isfinite(qpos_step) or qpos_step <= 0:
            raise ValueError("qpos_step must be positive and finite")
        if not np.isfinite(time_step) or time_step <= 0:
            raise ValueError("time_step must be positive and finite")
        qpos = np.asarray(self.pad_move_group_qpos(current_qpos.copy()), dtype=float)
        if not np.isfinite(qpos).all():
            return {"status": "screw plan failed: nonfinite joint positions"}
        self.robot.set_qpos(qpos, True)
        goal = self._transform_goal_to_wrt_base(goal_pose) if wrt_world else goal_pose
        indices = self.move_group_joint_indices
        ee_index = self.link_name_2_idx[self.move_group]
        model = self.pinocchio_model
        path = [qpos[indices].copy()]
        # Limit linearization error without requiring thousands of 0.001-rad steps.
        step_limit = min(qpos_step, 0.02)

        def current_pose(q):
            model.compute_forward_kinematics(q)
            return model.get_link_pose(ee_index)

        for _ in range(2000):
            actual = current_pose(qpos)
            position_error, rotation_error = pose_error(goal, actual)
            if (position_error <= self.POSITION_TOLERANCE and
                    rotation_error <= self.ROTATION_TOLERANCE):
                break
            # Recompute from actual FK. Subtracting J*dq accumulates integration error.
            twist = spatial_twist(goal * actual.inv())
            model.compute_full_jacobian(qpos)
            jacobian = model.get_link_jacobian(ee_index, local=False)[:, indices]
            delta = np.linalg.pinv(jacobian) @ twist
            norm = np.linalg.norm(delta)
            if not np.isfinite(norm) or norm < 1e-10:
                return {"status": "screw plan failed: no kinematic progress"}
            delta *= min(1., step_limit / norm)
            # Backtrack only the numerical step; collision failures remain failures.
            for _ in range(12):
                candidate = qpos.copy()
                candidate[indices] += delta
                candidate_pose = current_pose(candidate)
                if np.linalg.norm(spatial_twist(goal * candidate_pose.inv())) < np.linalg.norm(twist):
                    break
                delta *= 0.5
            else:
                return {"status": "screw plan failed: no kinematic progress"}
            if (np.any(candidate < self.joint_limits[:, 0] - 1e-3) or
                    np.any(candidate > self.joint_limits[:, 1] + 1e-3)):
                return {"status": "screw plan failed: joint limit"}
            self.planning_world.set_qpos_all(candidate[indices])
            if self.planning_world.is_state_colliding():
                return {"status": "screw plan failed: collision"}
            qpos = candidate
            path.append(qpos[indices].copy())
        else:
            return {"status": "screw plan failed: iteration limit"}

        if len(path) == 1:
            # Already at the goal: do not normalize a zero step or fit a constant spline.
            if self.planning_world.is_state_colliding():
                return {"status": "screw plan failed: collision"}
            position = np.asarray(path)
            return {"status": "Success", "time": np.array([0.]),
                    "position": position, "velocity": np.zeros_like(position),
                    "acceleration": np.zeros_like(position), "duration": 0.}

        times, position, velocity, acceleration, duration = self.TOPP(
            np.vstack(path), time_step, verbose=verbose)
        if len(position) == 0 or not np.isfinite(position).all():
            return {"status": "screw plan failed: invalid parameterized path"}
        # Validate the actual returned trajectory, not just the IK path or its status.
        qpos[indices] = position[-1]
        position_error, rotation_error = pose_error(goal, current_pose(qpos))
        if (position_error > self.POSITION_TOLERANCE or
                rotation_error > self.ROTATION_TOLERANCE):
            return {"status": "screw plan failed: endpoint outside tolerance"}
        return {"status": "Success", "time": times, "position": position,
                "velocity": velocity, "acceleration": acceleration, "duration": duration}
