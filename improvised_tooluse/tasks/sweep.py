from improvised_tooluse.tasks.task import Task, AffordanceRegion, quat_distance_wxyz
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from sapien import Pose
from transforms3d.quaternions import mat2quat

from real_world_client import RobotClient

class SweepTask(Task):
    def execute_task(
            self, 
            client: RobotClient = None,
            num_attempts: int = 50
        ):
        sweeped_region = SweepEDRegion.from_perception(self.target_aff_pcd, self.verbose)
        sweep_region = SweepRegion.from_perception(self.tool_aff_pcd, self.verbose)
        self.sweep_region = sweep_region
        checkpoint = self.simu_env.pack()
        offset = 0.05
        z_offset = 0.05

        debug_z_offset = 0.02

        push_distance = 0.1
        num_trials = 0
        planning_success = False
        while num_trials < num_attempts:
            num_trials += 1
            # grasp
            ee_pose_rel_tool, pre_grasp_traj, grasp_traj, post_grasp_traj = self.plan_grasp_sequence()
            # move to pre sweep position
            axis_1_sweep = np.array([0, 1, 0]) # sweep along +Y, require sampling
            axis_2_sweep = np.array([1, 0, 0]) # aligning with the table
            axis_3_sweep = np.cross(axis_1_sweep, axis_2_sweep)
            R_sweep = np.column_stack((axis_1_sweep, axis_2_sweep, axis_3_sweep))
            point_pre_sweep = sweeped_region.center - axis_1_sweep * offset - sweeped_region.bb.get_extent()[1]/2 * np.array([0,1,0])
            point_pre_sweep[2] = self.simu_env.table_height + z_offset + debug_z_offset
            
            
            T_f_pre_sweep = np.eye(4)
            T_f_pre_sweep[:3, :3] = R_sweep
            T_f_pre_sweep[:3, 3] = point_pre_sweep
            
            # sub-retry loop for pre-sweep positioning
            status = -1
            sub_num_attempts = 10
            for sub_num_trials in range(sub_num_attempts):
                obb = self.sweep_region.oriented_bounding_box
                extent = np.asarray(obb.extent, dtype=float)
                shortest_edge_idx = np.argmin(extent)
                longest_edge_idx = np.argmax(extent)
                middle_edge_idx = 3 - shortest_edge_idx - longest_edge_idx
                
                idx_2 = np.random.choice([middle_edge_idx, longest_edge_idx])
                # idx_2 = longest_edge_idx
                # if sub_num_trials < 5:
                #     idx_2 = longest_edge_idx
                # else:
                #     idx_2 = middle_edge_idx
                idx_3 = 3 - shortest_edge_idx - idx_2
                
                axis_1_init = obb.R[:, shortest_edge_idx] * np.random.choice([1, -1])
                axis_2_init = obb.R[:, idx_2] * np.random.choice([1, -1])
                axis_3_init = np.cross(axis_1_init, axis_2_init)
                R_init = np.column_stack((axis_1_init, axis_2_init, axis_3_init))
                point_init = self.sweep_region.center + axis_3_init * extent[idx_3]/2
                
                T_f_init = np.eye(4)
                T_f_init[:3, :3] = R_init
                T_f_init[:3, 3] = point_init
                
                T_pre_sweep_tool = T_f_pre_sweep @ np.linalg.inv(T_f_init) @ self.tool_initial_pose.to_transformation_matrix()
                pre_sweep_tool_pose = Pose(p=T_pre_sweep_tool[:3, 3], q=mat2quat(T_pre_sweep_tool[:3, :3]))
                pre_sweep_pose = pre_sweep_tool_pose * ee_pose_rel_tool
                
                print("Moving to pre-sweep position...")
                self.planner.update_from_simulation()
                status, pre_sweep_traj = self.planner.move_to_pose(pre_sweep_pose, ignore_collisions=False, tolerance=1e-3, return_traj=True, verbose=True)
                
                if status != 0:
                    print("Failed to move to pre-sweep pose, sub-retrying...")
                    continue
                else:
                    self.simu_env.simulate(1)
                    ee_pose = self.simu_env.end_effector.pose
                    tool_pose = self.tool.pose
                    print('current ee pose:', ee_pose)
                    print('target ee pose:', pre_sweep_pose)
                    print('used ee_pose_rel_tool:', ee_pose_rel_tool)
                    print('current ee_pose_rel_tool:', tool_pose.inv() * ee_pose)
                    
                    if np.linalg.norm(ee_pose.p - pre_sweep_pose.p) > 0.01 or quat_distance_wxyz(ee_pose.q, pre_sweep_pose.q) > (10.0 * np.pi / 180.0):
                        print("Large deviation after moving to pre-sweep pose, sub-retrying...")
                        status = -1
                        break
                    break
            
            if status != 0:
                self.retrieve(checkpoint)
                continue
            
            if sub_num_trials >= sub_num_attempts:
                print("Failed to move to pre-sweep pose after multiple attempts, retrying...")
                self.retrieve(checkpoint)
                continue
            
            print("Sweeping...")
            sweep_pose = Pose([0, 0, -z_offset]) * pre_sweep_pose
            
            self.planner.update_from_simulation()
            status, sweep_traj = self.planner.move_to_pose(sweep_pose, use_screw=True, ignore_collisions=True, return_traj=True, qpos_step=0.01)
            if status != 0:
                print("Failed to move to sweep pose, retrying...")
                self.retrieve(checkpoint)
                continue
            
            post_sweep_pose = Pose(axis_1_sweep *  (offset + push_distance)) * sweep_pose
            self.planner.update_from_simulation()
            status, post_sweep_traj = self.planner.move_to_pose(post_sweep_pose, use_screw=True, ignore_collisions=True, return_traj=True, qpos_step=0.01)
            if status != 0:
                print("Failed to move to post-sweep pose, retrying...")
                self.retrieve(checkpoint)
                continue
            
            ee_pose = self.simu_env.end_effector.pose
            tool_pose = self.tool.pose
            
            print("Sweep task executed successfully.")
            
            evaluation_success = self.evaluate()
            if not evaluation_success:
                print("Sweep task evaluation failed, retrying...")
                self.retrieve(checkpoint)
                continue
            
            print("Sweep task evaluation succeeded, now visualizing ...")
            self.retrieve(checkpoint)
            self.simu_env.start_recording()
            self.simu_env.create_viewer()
            self.simu_env.follow_path(pre_grasp_traj)
            self.simu_env.follow_path(grasp_traj)
            self.simu_env.attach_object(self.tool)
            self.simu_env.close_gripper()
            self.simu_env.follow_path(post_grasp_traj)
            self.simu_env.follow_path(pre_sweep_traj)
            self.simu_env.follow_path(sweep_traj)
            self.simu_env.follow_path(post_sweep_traj)

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
        client.execute_trajectory(pre_sweep_traj)
        input('next')
        client.execute_trajectory(sweep_traj)
        input('next')
        client.execute_trajectory(post_sweep_traj)
        return 0
    
    def sample_grasping_pose(self):
        idx = np.random.randint(len(self.grasping_poses))
        grasp_mat = self.grasping_poses[idx]
        return grasp_mat

    def evaluate(self):
        print("Evaluating sweep task...")
        target_position = self.target_initial_pose.p + np.array([0, 0.1, 0])
        current_position = self.target.pose.p
        error = np.linalg.norm((target_position - current_position)[:2])
        print(f"Sweep task position error: {error:.4f} m")
        success = error < 0.05
        return success

@dataclass
class SweepEDRegion(AffordanceRegion):  #bowl
    center: np.ndarray  # (3,)
    pcd: o3d.geometry.PointCloud
    bb: o3d.geometry.AxisAlignedBoundingBox

    @classmethod
    def from_perception(cls, pcd_aff, verbose: bool = False):
        colors = np.asarray(pcd_aff.colors)
        mask = colors[:, 2] > 0

        indices = np.where(mask)[0]
        pcd_sweeped = pcd_aff.select_by_index(list(indices))
        bb = pcd_sweeped.get_axis_aligned_bounding_box()
        center = np.asarray(bb.get_center())

        if verbose:
            bb.color = (1, 0, 0)
            o3d.visualization.draw_geometries([pcd_aff, bb])

        return cls(center=center, pcd=pcd_sweeped, bb=bb)

@dataclass
class SweepRegion(AffordanceRegion):  #pouring region on tool
    pcd: o3d.geometry.PointCloud
    oriented_bounding_box: o3d.geometry.OrientedBoundingBox
    center: np.ndarray

    @classmethod
    def from_perception(cls, pcd_aff, verbose: bool = False):
        colors = np.asarray(pcd_aff.colors)
        mask = colors[:, 1] > 0

        # heuristic: select the bottom points only
        indices = np.where(mask)[0]
        pcd_sweep = pcd_aff.select_by_index(list(indices))

        obb = pcd_sweep.get_oriented_bounding_box()
        extent = obb.extent  
        edge_lengths = extent * 2.0  
        idx = np.argmin(edge_lengths)

        pts = np.asarray(pcd_sweep.points)
        centroid = pts.mean(axis=0)

        if verbose:
            obb.color = (1, 0, 0)
            normal_direction = np.asarray(obb.R)[:, idx]
            normal_vector = cls.make_vector(centroid, normal_direction)
            o3d.visualization.draw_geometries([pcd_aff, obb, normal_vector])

        return cls(pcd=pcd_sweep, oriented_bounding_box=obb, center=centroid)
