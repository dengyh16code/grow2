import trimesh
from improvised_tooluse.sapien_utils import get_actor_pcd
from improvised_tooluse.tasks.task import Task, AffordanceRegion
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from transforms3d.quaternions import mat2quat
from sapien import Pose
from improvised_tooluse.geom_utils import quaternion_from_vectors
from improvised_tooluse.vfm_client import fp_tracking_request
from real_world_client import RobotClient

T_opencv_to_opengl = np.diag([1, -1, -1, 1])

class PoundTask(Task):
    def execute_task(
        self, 
        client: RobotClient = None,
        num_attempts: int = 50, 
    ):
        pounded_region = PoundEDRegion.from_perception(self.target_aff_pcd, self.verbose)
        pound_region = PoundRegion.from_perception(self.tool_aff_pcd, self.verbose)
        self.pound_region = pound_region
        num_trials = 0
        checkpoint = self.simu_env.pack()
        planning_success = False
        while num_trials < num_attempts:
            num_trials += 1
            # grasp
            ee_pose_rel_tool, pre_grasp_traj, grasp_traj, post_grasp_traj = self.plan_grasp_sequence()
            
            offset = 0.2
            pound_direction = np.array([0., 0., -1.]) #top down
            pre_pound_position = pounded_region.center + np.array([0, 0, offset])
            pre_pound_quat = quaternion_from_vectors(pound_region.normal_direction, pound_direction, random_choose=True)
        
            pre_pound_tool_pose = (Pose(pound_region.center) *
                                Pose(p=pre_pound_position-pound_region.center, q=pre_pound_quat) *
                                Pose(p=pound_region.center).inv() *
                                self.tool_initial_pose)
            
            pound_tool_pose = Pose([0, 0, -offset-0.01]) * pre_pound_tool_pose
            pre_pound_pose = pre_pound_tool_pose * ee_pose_rel_tool
            pound_pose = pound_tool_pose * ee_pose_rel_tool
            
            # move to pre pound position
            print("Moving to pre-pound position...")
            status, pre_pound_traj = self.planner.move_to_pose(pre_pound_pose, use_screw=False, return_traj=True) #move to pre-pound pose
            if status != 0:
                print("Failed to move to pre-pound pose, retrying...")
                self.retrieve(checkpoint)
                continue

            # move to pound position
            print("Pounding down...")
            status, pound_traj = self.planner.move_to_pose(pound_pose, use_screw=True, ignore_collisions=True, speed_up_factor=5, return_traj=True)  #pound down
            if status != 0:
                print("Failed to move to pound pose, retrying...")
                self.retrieve(checkpoint)
                continue
            print("Pound task executed successfully.")
            
            evaluation_success = self.evaluate()
            if not evaluation_success:
                print("Pound task evaluation failed, retrying...")
                self.retrieve(checkpoint)
                continue
            
            print("Pound task evaluation succeeded, now visualizing ...")
            self.retrieve(checkpoint)
            self.simu_env.start_recording()
            self.simu_env.create_viewer()
            self.simu_env.follow_path(pre_grasp_traj)
            self.simu_env.follow_path(grasp_traj)
            self.simu_env.attach_object(self.tool)
            self.simu_env.close_gripper()
            self.simu_env.follow_path(post_grasp_traj)
            self.simu_env.follow_path(pre_pound_traj)
            self.simu_env.follow_path(pound_traj)
            
            if client is None:
                return 0
            command = input('Now executing the planned trajectory. Continue? y/n: ')
            if command == 'y' or command == 'Y':
                planning_success = True
                break
            else:
                print('execution rejected, resampling')
                self.retrieve(checkpoint)
                continue

        if not planning_success:
            return -1        
        client.execute_trajectory(pre_grasp_traj)
        input('next')
        client.execute_trajectory(grasp_traj)
        input('next')
        client.close_gripper()
        input('next')
        client.execute_trajectory(post_grasp_traj)
        input('next')
        client.execute_trajectory(pre_pound_traj)
        input('next')
        client.execute_trajectory(pound_traj)
        input('next')
        
        return 0
            
    
    def sample_grasping_pose(self):        
        # random sample from top 100 grasps
        idx = np.random.randint(0, min(100, len(self.grasping_poses)))
        return self.grasping_poses[idx]

    def evaluate(self) -> bool:
        print("Evaluating pound task...")
        pounded_region = PoundEDRegion.from_perception(self.target_aff_pcd)
        pounded_bounding_box = pounded_region.pcd.get_axis_aligned_bounding_box()
        upper_center = pounded_bounding_box.get_center()
        upper_center[2] += pounded_bounding_box.get_extent()[2] / 2
        
        tool_pcd: o3d.geometry.PointCloud = get_actor_pcd(self.tool)
        points = np.asarray(tool_pcd.points)
        mask = points[:, 2] < upper_center[2] + 0.01
        xy_bounds = [
            upper_center[0] - 0.01,
            upper_center[0] + 0.01,
            upper_center[1] - 0.01,
            upper_center[1] + 0.01,
        ]
        xy_mask = (points[:, 0] >= xy_bounds[0]) & (points[:, 0] <= xy_bounds[1]) & \
                  (points[:, 1] >= xy_bounds[2]) & (points[:, 1] <= xy_bounds[3])
        final_mask = mask & xy_mask
        selected_points = points[final_mask]
        # get the bounding box of the selected points
        if len(selected_points) == 0:
            return False
        selected_pcd = o3d.geometry.PointCloud()
        selected_pcd.points = o3d.utility.Vector3dVector(selected_points)
        selected_bounding_box = selected_pcd.get_axis_aligned_bounding_box()
        x_extent = selected_bounding_box.get_extent()[0]
        y_extent = selected_bounding_box.get_extent()[1]
        area = x_extent * y_extent
        print(f"Pounded area: {10000 * area:.6f} cm^2")
        return area >= 0.0001  # threshold = 1cm x 1cm 

@dataclass
class PoundEDRegion(AffordanceRegion):  
    center: np.ndarray  # (3,)
    pcd: o3d.geometry.PointCloud

    @classmethod
    def from_perception(cls, pcd_aff, verbose: bool = False):
        colors = np.asarray(pcd_aff.colors)
        mask = colors[:, 2] > 0

        indices = np.where(mask)[0]
        pcd_hited = pcd_aff.select_by_index(list(indices))
        z_max = np.asarray(pcd_hited.points)[:, 2].max()
        z_threshold = z_max - 0.01  # top 1cm points
        mask = np.asarray(pcd_hited.points)[:, 2] >= z_threshold
        indices = np.where(mask)[0]
        pcd_hited = pcd_hited.select_by_index(list(indices))

        obb = pcd_hited.get_oriented_bounding_box()
        center = np.asarray(obb.center)

        if verbose:
            obb.color = (1, 0, 0)
            normal_vector = cls.make_vector(center, np.array([0, 0, 1]))
            o3d.visualization.draw_geometries([pcd_aff, obb, normal_vector])

        return cls(center=center, pcd=pcd_hited)

@dataclass
class PoundRegion(AffordanceRegion):  #pouring region on tool
    center: np.ndarray  # (3,)
    pcd: o3d.geometry.PointCloud
    normal_direction: np.ndarray

    @classmethod
    def from_perception(cls, pcd_aff, verbose: bool = False):
        colors = np.asarray(pcd_aff.colors)
        mask = colors[:, 1] > 0

        indices = np.where(mask)[0]
        pcd_pound = pcd_aff.select_by_index(list(indices))
        if len(pcd_pound.points) > 10:
            z_min = np.asarray(pcd_pound.points)[:, 2].min()
            z_threshold = z_min + 0.01  # bottom 1cm points
            mask = np.asarray(pcd_pound.points)[:, 2] <= z_threshold
            indices = np.where(mask)[0]
            pcd_pound = pcd_pound.select_by_index(list(indices))

        obb = pcd_pound.get_oriented_bounding_box()
        center = np.asarray(obb.center)
        normal_direction = np.asarray(obb.R)[:, 2]  # assuming z-axis is the normal direction

        if normal_direction [2] > 0:
            normal_direction = -normal_direction #todo

        if verbose:
            obb.color = (1, 0, 0)
            normal_vector = cls.make_vector(obb.get_center(), normal_direction)
            o3d.visualization.draw_geometries([pcd_aff, obb, normal_vector])

        return cls(center=center, pcd=pcd_pound, normal_direction=normal_direction)
    
    def sample_pound_point(self):
        points = np.asarray(self.pcd.points)
        idx = np.random.randint(len(points))
        return points[idx]
