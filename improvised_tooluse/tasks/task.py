import numpy as np
import open3d as o3d
from abc import ABC, abstractmethod
from improvised_tooluse.simu_env import simu_env
from improvised_tooluse.planner import PandaPlanner
from improvised_tooluse.affordance import Perception_module, DOWNWARD_GRASP_THRESHOLD
import sapien
from sapien import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat

def quat_distance_wxyz(q1, q2, *, eps: float = 1e-12) -> float:
    """
    Angular distance between two quaternions in (w, x, y, z) format.
    Returns the relative rotation angle in radians in [0, pi].

    Args:
        q1, q2: array-like, shape (4,), unit or non-unit quaternions (w, x, y, z)
        eps: small number to avoid division by zero

    Returns:
        float: angular distance (radians)
    """
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q2 = np.asarray(q2, dtype=np.float64).reshape(4)

    n1 = np.linalg.norm(q1)
    n2 = np.linalg.norm(q2)
    if n1 < eps or n2 < eps:
        raise ValueError("Quaternion norm is zero (or too small).")

    q1 /= n1
    q2 /= n2

    # Handle double cover: q and -q represent the same rotation
    dot = float(np.dot(q1, q2))
    dot = abs(dot)

    # Numerical safety
    dot = max(0.0, min(1.0, dot))

    return 2.0 * float(np.arccos(dot))

class Task(ABC):
    """
    Base class for manipulation / motion planning tasks.
    """
    def __init__(
        self,
        simu_env: simu_env,
        motion_planner: PandaPlanner,
        detector: Perception_module,
        tool: sapien.Entity,
        target: sapien.Entity,
        verbose: bool = False,
        is_soft_gripper: bool = False,
    ):
        self.tool_aff_pcd = detector.tool_aff_pcd
        self.tool_pcd = getattr(detector, "tool_pcd", None)
        self.target_aff_pcd = detector.target_aff_pcd
        self.grasping_poses = detector.pred_grasps_world
        self.tool = tool
        self.target = target
        self.tool_initial_pose = tool.pose
        self.target_initial_pose = target.pose
        detector_tool_center = getattr(detector, "tool_center", None)
        self.tool_center = (
            np.asarray(detector_tool_center, dtype=float)
            if detector_tool_center is not None
            else np.asarray(self.tool_initial_pose.p, dtype=float)
        )
        self.simu_env = simu_env
        self.planner = motion_planner
        self.verbose = verbose
        self.grasping_threshold = DOWNWARD_GRASP_THRESHOLD
        self.is_soft_gripper = is_soft_gripper

    @abstractmethod
    def execute_task(self):
        """Generate a motion trajectory for the task."""
        pass

    def _measure_gripper_pose(self) -> Pose:
        """World gripper pose from robot encoders and FK, without object state.

        Use a separate kinematic model so measuring the robot does not mutate
        the motion planner's model or its candidate-search state.
        """
        robot = self.simu_env.robot
        if not hasattr(self, "_gripper_fk_model"):
            self._gripper_fk_model = robot.create_pinocchio_model()
            self._gripper_fk_index = [link.name for link in robot.get_links()].index(
                self.simu_env.end_effector.name)
        self._gripper_fk_model.compute_forward_kinematics(robot.get_qpos())
        pose = robot.get_root_pose() * self._gripper_fk_model.get_link_pose(self._gripper_fk_index)
        return Pose(pose.p.copy(), pose.q.copy())

    def plan_grasp_sequence(
        self,
        num_attempts: int = 10,
    ) -> tuple[Pose, dict, dict, dict]:
        """Plan and simulate a complete grasp, returning its three trajectories."""
        num_tried = 0
        checkpoint = self.simu_env.pack()
        print("Grasping the tool...")
        self.grasp_pose_at_attachment = None
        while num_tried < num_attempts:
            self.grasp_pose_at_attachment = None
            grasp_mat = self.sample_grasping_pose()
            if grasp_mat[:3, 2].dot(np.array([0, 0, -1])) < self.grasping_threshold:
                continue
            if self.is_soft_gripper:
                T_local = np.eye(4)
                T_local[:3, 3] = np.array([0, 0, -0.135])
                grasp_mat = grasp_mat @ T_local
            num_tried += 1
            grasp_pose_1 = Pose(p=grasp_mat[:3, 3], q=mat2quat(grasp_mat[:3, :3])) * Pose(q=euler2quat(0, 0, np.pi/2))
            grasp_pose_2 = Pose(p=grasp_mat[:3, 3], q=mat2quat(grasp_mat[:3, :3])) * Pose(q=euler2quat(0, 0, -np.pi/2))

            current_ee_pose = self.simu_env.end_effector.pose
            
            q_distance_1 = quat_distance_wxyz(grasp_pose_1.q, current_ee_pose.q)
            q_distance_2 = quat_distance_wxyz(grasp_pose_2.q, current_ee_pose.q)
            if q_distance_1 < q_distance_2:
                grasp_pose = grasp_pose_1
            else:
                grasp_pose = grasp_pose_2
            
            pre_grasp_pose = grasp_pose * Pose([0, 0, -0.1])
            print(pre_grasp_pose)
        
            status, pre_grasp_traj = self.planner.move_to_pose(pre_grasp_pose, return_traj=True)
            if status != 0:
                print("Failed to move to pre-grasp pose, retrying grasp selection...")
                self.retrieve(checkpoint)
                continue

            status, grasp_traj = self.planner.move_to_pose(
                grasp_pose,
                use_screw=True,
                ignore_collisions=True,
                return_traj=True,
            )
            if status != 0:
                print("Failed to move to grasp pose, retrying grasp selection...")
                self.retrieve(checkpoint)
                continue

            self.grasp_pose = grasp_pose
            self.planner.update_from_simulation()
            self.simu_env.attach_object(self.tool)
            # Capture before closing/retreating: perceived tool geometry is
            # expressed in its pre-grasp world frame. Keep the requested pose
            # separately for the existing retreat and other task conventions.
            self.grasp_pose_at_attachment = self._measure_gripper_pose()
            self.simu_env.close_gripper()
            self.planner.update_from_simulation()

            status, post_grasp_traj = self.planner.move_to_pose(
                grasp_pose * Pose([0, 0, -0.2]),
                use_screw=True,
                ignore_collisions=True,
                return_traj=True,
            )
            if status != 0:
                print("Failed to move to post-grasp pose, retrying grasp selection...")
                self.grasp_pose_at_attachment = None
                self.retrieve(checkpoint)
                continue

            self.planner.update_from_simulation()
            self.planner.attach_object(self.tool)
            ee_pose_rel_tool = self.tool.pose.inv() * self.simu_env.end_effector.pose
            return ee_pose_rel_tool, pre_grasp_traj, grasp_traj, post_grasp_traj

        raise RuntimeError(
            f"Failed to plan a complete grasp sequence after {num_attempts} attempts."
        )

    def sample_grasping_pose(self) -> np.ndarray:
        """Sample a grasping pose (can be overridden)."""
        return self._default_sample_grasping_pose()

    def _default_sample_grasping_pose(self) -> np.ndarray:
        idx = np.random.randint(len(self.grasping_poses))
        return self.grasping_poses[idx]

    @abstractmethod
    def evaluate(self):
        """Evaluate task success in simulation."""
        pass
    
    def retrieve(self, checkpoint, detach_tool: bool = True):
        if detach_tool: 
            if self.simu_env.drive is not None:
                self.simu_env.detach_object()
            if self.planner.attached_object is not None:
                self.planner.detach_object()
        self.simu_env.reset_recording()
        self.simu_env.unpack(checkpoint)
        self.planner.update_from_simulation()


class AffordanceRegion(ABC):
    """
    Abstract representation of an affordance region derived from perception.
    """
    @classmethod
    @abstractmethod
    def from_perception(cls, pcd_aff: o3d.geometry.PointCloud, *args, **kwargs):
        """Construct an affordance region from perception output."""
        pass

    @staticmethod
    def make_vector(
        start: np.ndarray,
        direction: np.ndarray,
        length: float = 0.1,
    ) -> o3d.geometry.LineSet:
        """
        Create a 3D vector (LineSet) for visualization.
        """
        start = np.asarray(start, dtype=float)
        direction = np.asarray(direction, dtype=float)

        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            raise ValueError("Direction vector must be non-zero.")

        direction = direction / norm
        end = start + direction * length

        points = np.vstack([start, end])
        lines = [[0, 1]]
        colors = [[0.0, 0.0, 1.0]]  # blue

        vector = o3d.geometry.LineSet()
        vector.points = o3d.utility.Vector3dVector(points)
        vector.lines = o3d.utility.Vector2iVector(lines)
        vector.colors = o3d.utility.Vector3dVector(colors)
        return vector
    
    @staticmethod
    def make_circle(center, radius, num_points=100):
        """For visualization."""
        center = np.asarray(center)
        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        xs = center[0] + radius * np.cos(theta)
        ys = center[1] + radius * np.sin(theta)
        zs = np.full_like(xs, center[2])
        points = np.stack([xs, ys, zs], axis=1)
        lines = [[i, (i + 1) % num_points] for i in range(num_points)]
        colors = [[0, 1, 0] for _ in lines]  # green circle
        circle = o3d.geometry.LineSet()
        circle.points = o3d.utility.Vector3dVector(points)
        circle.lines = o3d.utility.Vector2iVector(lines)
        circle.colors = o3d.utility.Vector3dVector(colors)
        return circle
