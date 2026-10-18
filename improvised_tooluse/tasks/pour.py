from improvised_tooluse.tasks.task import Task, AffordanceRegion, quat_distance_wxyz
import numpy as np
import open3d as o3d
import sapien
from dataclasses import dataclass
from sapien import Pose
from transforms3d.euler import euler2quat
from transforms3d.quaternions import mat2quat, quat2mat
from improvised_tooluse.geom_utils import quaternion_from_vectors
from real_world_client import RobotClient


class PourTask(Task):
    # An 8 mm ball rolls out of the tool more reliably than a box-shaped marker.
    verification_ball_radius = 0.004

    def execute_task(self, num_attempts: int = 50):
        poured_region = PourEDRegion.from_perception(self.target_aff_pcd, self.verbose)
        pour_region = PourRegion.from_perception(
            self.tool_aff_pcd,
            tool_pcd=self.tool_pcd,
            verbose=self.verbose,
        )
        self.poured_region = poured_region
        self.verification_ball = None
        
        checkpoint = self.simu_env.pack()
        for num_trials in range(num_attempts):            
            # grasp
            ee_pose_rel_tool, pre_grasp_traj, grasp_traj, post_grasp_traj = self.plan_grasp_sequence()
            
            # move to pre pour position
            offset = 0.1
            approaching_direction = np.array([0, 1, 0]) # along +Y
            tilt_angle_pre_pour = np.pi / 6  # 30 degree
            axis_1_pre_pour = np.array([0.,  np.cos(tilt_angle_pre_pour), -np.sin(tilt_angle_pre_pour)])
            axis_3_pre_pour = np.array([0.,  np.sin(tilt_angle_pre_pour),  np.cos(tilt_angle_pre_pour)])
            axis_2_pre_pour = np.cross(axis_3_pre_pour, axis_1_pre_pour)
            R_pre_pour = np.column_stack((axis_1_pre_pour, axis_2_pre_pour, axis_3_pre_pour))
            point_pre_pour = poured_region.center - approaching_direction * (poured_region.radius - 0.03) + np.array([0, 0, offset])
            
            # Pour past horizontal so gravity has a component toward the tool
            # opening. At exactly 90 degrees a verification ball can remain on
            # the lower wall indefinitely, regardless of the settling delay.
            tilt_angle_pour = np.deg2rad(100)
            axis_1_pour = np.array([0.,  np.cos(tilt_angle_pour), -np.sin(tilt_angle_pour)])
            axis_3_pour = np.array([0.,  np.sin(tilt_angle_pour),  np.cos(tilt_angle_pour)])
            axis_2_pour = np.cross(axis_3_pour, axis_1_pour)
            R_pour = np.column_stack((axis_1_pour, axis_2_pour, axis_3_pour))
            point_pour = poured_region.center - approaching_direction * (poured_region.radius - 0.03) + np.array([0, 0, offset])
            
            T_f_pre_pour = np.eye(4)
            T_f_pre_pour[:3, :3] = R_pre_pour
            T_f_pre_pour[:3, 3] = point_pre_pour
            
            T_f_pour = np.eye(4)
            T_f_pour[:3, :3] = R_pour
            T_f_pour[:3, 3] = point_pour

            sub_num_attempts = 20
            success_pre_pour = False
            
            # loop to reach pre-pour pose
            for sub_num_trials in range(sub_num_attempts): 
                grasp_pos = self.grasp_pose.p
                grasp_quat = quat2mat(self.grasp_pose.q)
                grasp_mat = np.eye(4)
                grasp_mat[:3, :3] = grasp_quat
                grasp_mat[:3, 3] = np.array(grasp_pos)
                local_points = np.array([[0.0, 0.0, 0.1]])
                pts_h = np.hstack([local_points, np.ones((local_points.shape[0], 1))])
                world_pts = (grasp_mat @ pts_h.T).T[:, :3]
                grasp_point = np.array(world_pts[0])

                pour_point = pour_region.sample_pour_point(grasp_point)
                R_init = pour_region.frame_at(pour_point)
                point_init = pour_point
                
                T_f_init = np.eye(4)
                T_f_init[:3, :3] = R_init
                T_f_init[:3, 3] = point_init
                
                T_pre_pour_tool = T_f_pre_pour @ np.linalg.inv(T_f_init) @ self.tool_initial_pose.to_transformation_matrix()
                pre_pour_tool_pose = Pose(p=T_pre_pour_tool[:3, 3], q=mat2quat(T_pre_pour_tool[:3, :3]))
                pre_pour_pose = pre_pour_tool_pose * ee_pose_rel_tool
                
                T_pour_tool = T_f_pour @ np.linalg.inv(T_f_init) @ self.tool_initial_pose.to_transformation_matrix()
                pour_tool_pose = Pose(p=T_pour_tool[:3, 3], q=mat2quat(T_pour_tool[:3, :3]))
                pour_pose = pour_tool_pose * ee_pose_rel_tool
                
                print("Moving to pre-pour position...")
                self.planner.update_from_simulation()
                status, pre_pour_traj = self.planner.move_to_pose(pre_pour_pose, verbose=self.verbose, return_traj=True)
                
                if status != 0:
                     print("Failed to move to pre-pour pose, sub-retrying...")
                     continue
                else:
                    self.simu_env.simulate(1)
                    ee_pose = self.simu_env.end_effector.pose
                    
                    if np.linalg.norm(ee_pose.p - pre_pour_pose.p) > 0.01 or quat_distance_wxyz(ee_pose.q, pre_pour_pose.q) > (5.0 * np.pi / 180.0):
                        print("Large deviation after moving to pre-pour pose, sub-retrying...")
                        status = -1
                        break
                        
                    success_pre_pour = True
                    break
            
            if not success_pre_pour:
                print("Failed to move to pre-pour pose after multiple attempts or large deviation, retrying...")
                self.retrieve(checkpoint)
                continue

            # pour motion
            print("Pouring...")
            status, pour_traj = self.planner.move_to_pose(pour_pose, verbose=True, ignore_collisions=True, use_screw=True, return_traj=True)
            if status != 0:
                print("Failed to move to pour pose, retrying...")
                self.retrieve(checkpoint)
                continue

            self.retrieve(checkpoint)
            self.simu_env.start_recording()
            self.simu_env.create_viewer()
            self.simu_env.follow_path(pre_grasp_traj)
            self.simu_env.follow_path(grasp_traj)
            self.simu_env.attach_object(self.tool)
            self.simu_env.close_gripper()
            self.simu_env.follow_path(post_grasp_traj)
            self.simu_env.follow_path(pre_pour_traj)
            self._create_verification_ball(pour_region)
            # Evaluate pouring from the pre-pour pose, after transport is over.
            self.simu_env.simulate(time=0.5)
            self.simu_env.follow_path(pour_traj)
            # Give the ball enough time to leave the tool and settle in the target.
            self.simu_env.simulate(time=1.0)
            
            evaluate_success = self.evaluate()
            self.simu_env.close_viewer()
            if evaluate_success:
                print("Pour task executed successfully.")
                return 0
            else:
                print("Pour evaluation fail, retrying...")
                self._remove_verification_ball()
                self.retrieve(checkpoint)
                continue

        print("Failed to execute pour task after all attempts.")
        return -1
        
    def execute_task_real_world(self, client: RobotClient, num_attempts: int = 50):
        poured_region = PourEDRegion.from_perception(self.target_aff_pcd, self.verbose)
        pour_region = PourRegion.from_perception(
            self.tool_aff_pcd,
            tool_pcd=self.tool_pcd,
            verbose=self.verbose,
        )
        
        checkpoint = self.simu_env.pack()
        planning_success = False
        for num_trials in range(num_attempts):            
            # grasp
            ee_pose_rel_tool, pre_grasp_traj, grasp_traj, post_grasp_traj = self.plan_grasp_sequence()
            
            # move to pre pour position
            offset = 0.2
            approaching_direction = np.array([0, 1, 0]) # along +Y
            tilt_angle_pre_pour = np.pi / 6  # 30 degree
            axis_1_pre_pour = np.array([0.,  np.cos(tilt_angle_pre_pour), -np.sin(tilt_angle_pre_pour)])
            axis_3_pre_pour = np.array([0.,  np.sin(tilt_angle_pre_pour),  np.cos(tilt_angle_pre_pour)])
            axis_2_pre_pour = np.cross(axis_3_pre_pour, axis_1_pre_pour)
            R_pre_pour = np.column_stack((axis_1_pre_pour, axis_2_pre_pour, axis_3_pre_pour))
            point_pre_pour = poured_region.center - approaching_direction * (poured_region.radius - 0.03) + np.array([0, 0, offset])
            
            tilt_angle_pour = np.pi / 2  # 90 degree
            axis_1_pour = np.array([0.,  np.cos(tilt_angle_pour), -np.sin(tilt_angle_pour)])
            axis_3_pour = np.array([0.,  np.sin(tilt_angle_pour),  np.cos(tilt_angle_pour)])
            axis_2_pour = np.cross(axis_3_pour, axis_1_pour)
            R_pour = np.column_stack((axis_1_pour, axis_2_pour, axis_3_pour))
            point_pour = poured_region.center - approaching_direction * (poured_region.radius - 0.03) + np.array([0, 0, offset])
            
            T_f_pre_pour = np.eye(4)
            T_f_pre_pour[:3, :3] = R_pre_pour
            T_f_pre_pour[:3, 3] = point_pre_pour
            
            T_f_pour = np.eye(4)
            T_f_pour[:3, :3] = R_pour
            T_f_pour[:3, 3] = point_pour

            sub_num_attempts = 20
            success_pre_pour = False
            
            # loop to reach pre-pour pose
            for sub_num_trials in range(sub_num_attempts): 
                grasp_pos = self.grasp_pose.p
                grasp_quat = quat2mat(self.grasp_pose.q)
                grasp_mat = np.eye(4)
                grasp_mat[:3, :3] = grasp_quat
                grasp_mat[:3, 3] = np.array(grasp_pos)
                local_points = np.array([[0.0, 0.0, 0.1]])
                pts_h = np.hstack([local_points, np.ones((local_points.shape[0], 1))])
                world_pts = (grasp_mat @ pts_h.T).T[:, :3]
                grasp_point = np.array(world_pts[0])

                pour_point = pour_region.sample_pour_point(grasp_point)
                R_init = pour_region.frame_at(pour_point)
                point_init = pour_point
                
                T_f_init = np.eye(4)
                T_f_init[:3, :3] = R_init
                T_f_init[:3, 3] = point_init
                
                T_pre_pour_tool = T_f_pre_pour @ np.linalg.inv(T_f_init) @ self.tool_initial_pose.to_transformation_matrix()
                pre_pour_tool_pose = Pose(p=T_pre_pour_tool[:3, 3], q=mat2quat(T_pre_pour_tool[:3, :3]))
                pre_pour_pose = pre_pour_tool_pose * ee_pose_rel_tool
                
                T_pour_tool = T_f_pour @ np.linalg.inv(T_f_init) @ self.tool_initial_pose.to_transformation_matrix()
                pour_tool_pose = Pose(p=T_pour_tool[:3, 3], q=mat2quat(T_pour_tool[:3, :3]))
                pour_pose = pour_tool_pose * ee_pose_rel_tool
                
                print("Moving to pre-pour position...")
                self.planner.update_from_simulation()
                status, pre_pour_traj = self.planner.move_to_pose(pre_pour_pose, verbose=self.verbose, return_traj=True)
                
                if status != 0:
                     print("Failed to move to pre-pour pose, sub-retrying...")
                     continue
                else:
                    self.simu_env.simulate(1)
                    ee_pose = self.simu_env.end_effector.pose
                    
                    if np.linalg.norm(ee_pose.p - pre_pour_pose.p) > 0.01 or quat_distance_wxyz(ee_pose.q, pre_pour_pose.q) > (5.0 * np.pi / 180.0):
                        print("Large deviation after moving to pre-pour pose, sub-retrying...")
                        status = -1
                        break
                        
                    success_pre_pour = True
                    break
            
            if not success_pre_pour:
                print("Failed to move to pre-pour pose after multiple attempts or large deviation, retrying...")
                self.retrieve(checkpoint)
                continue

            # pour motion
            print("Pouring...")
            status, pour_traj = self.planner.move_to_pose(pour_pose, verbose=True, ignore_collisions=True, use_screw=True, return_traj=True)
            if status != 0:
                print("Failed to move to pour pose, retrying...")
                self.retrieve(checkpoint)
                continue
                
            print("Pouring executed successfully")
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
        client.execute_trajectory(post_grasp_traj)
        input('next')
        client.execute_trajectory(pre_pour_traj)
        input('next')
        client.execute_trajectory(pour_traj)
        return 0
        
    def sample_grasping_pose(self):
        idx = np.random.randint(0, len(self.grasping_poses))
        return self.grasping_poses[idx]

    def _create_verification_ball(self, pour_region):
        """Place a ball clear of the tool surface behind the pouring plane."""
        self._remove_verification_ball()

        # Perception points are expressed in the world frame at the tool's initial
        # pose. Carry the pouring surface and its outward normal with the tool.
        T_initial = self.tool_initial_pose.to_transformation_matrix()
        T_current = self.tool.pose.to_transformation_matrix()
        opening_local = np.linalg.inv(T_initial) @ np.append(pour_region.center, 1.0)
        direction_local = T_initial[:3, :3].T @ pour_region.direction
        opening_world = (T_current @ opening_local)[:3]
        direction_world = T_current[:3, :3] @ direction_local
        direction_world /= np.linalg.norm(direction_world)

        # The first full-tool surface behind the marked plane is a vessel's
        # bottom or an open tool's backing surface. Start six ball radii toward
        # the opening from that surface, leaving room for reconstruction error.
        support_world = opening_world - direction_world * pour_region.support_depth
        ball_position = support_world + direction_world * (
            6.0 * self.verification_ball_radius
        )

        builder = self.simu_env.scene.create_actor_builder()
        ball_material = sapien.physx.PhysxMaterial(
            static_friction=0.05,
            dynamic_friction=0.05,
            restitution=0.0,
        )
        builder.add_sphere_collision(
            radius=self.verification_ball_radius,
            material=ball_material,
            density=1000,
        )
        builder.add_sphere_visual(
            radius=self.verification_ball_radius,
            material=[1.0, 0.2, 0.2],
        )
        self.verification_ball = builder.build(name="pour_verification_ball")
        self.verification_ball.set_pose(Pose(p=ball_position))

    def _remove_verification_ball(self):
        if getattr(self, "verification_ball", None) is not None:
            self.simu_env.scene.remove_actor(self.verification_ball)
            self.verification_ball = None

    def evaluate(self):
        print("Evaluating pour task...")
        if self.verification_ball is None:
            print("Pour verification failed: no verification ball exists.")
            return False

        ball_position = np.asarray(self.verification_ball.pose.p, dtype=float)
        target_center = np.asarray(self.poured_region.center, dtype=float)

        # The perceived target circle is its top opening. The ball is in the
        # target when it is horizontally inside the opening and below the rim.
        allowed_radius = max(
            0.0, self.poured_region.radius - self.verification_ball_radius
        )
        radial_distance = np.linalg.norm(ball_position[:2] - target_center[:2])
        below_rim = ball_position[2] <= (
            target_center[2] + self.verification_ball_radius
        )
        above_table = ball_position[2] >= self.simu_env.table_height
        success = radial_distance <= allowed_radius and below_rim and above_table

        print(
            "Verification ball: radial distance "
            f"{radial_distance:.4f} m (limit {allowed_radius:.4f} m), "
            f"height {ball_position[2]:.4f} m (rim {target_center[2]:.4f} m)"
        )
        return success

@dataclass
class PourEDRegion(AffordanceRegion):  #bowl
    center: np.ndarray  # (3,)
    radius: float

    @classmethod
    def from_perception(cls, pcd_aff, verbose: bool = False):
        colors = np.asarray(pcd_aff.colors)
        mask = colors[:, 2] > 0
        indices = np.where(mask)[0]
        pcd_bowl = pcd_aff.select_by_index(list(indices))
        aabb = pcd_bowl.get_axis_aligned_bounding_box()
        extent = aabb.get_extent()
        radius = max(extent[0], extent[1]) / 2
        center = aabb.get_center().copy()
        center[2] = aabb.get_max_bound()[2]  # use the top z as the rim center z

        if verbose:
            aabb.color = (1, 0, 0)
            circle = cls.make_circle(center, radius)# red
            o3d.visualization.draw_geometries([pcd_aff, aabb, circle])

        return cls(center=center, radius=radius)

@dataclass
class PourRegion(AffordanceRegion):  #pouring region on tool
    center: np.ndarray  # (3,)
    pcd: o3d.geometry.PointCloud
    direction: np.ndarray
    support_depth: float = 0.0

    @classmethod
    def from_perception(
        cls, pcd_aff, tool_pcd, verbose: bool = False, *,
        ray_radius: float = 0.003, ray_start_distance: float = 0.001,
    ):
        """Orient the opening normal toward the farther of two point-cloud hits.

        A ray hits a point when its perpendicular distance is within ray_radius;
        hit distance is measured along the ray. Both distances are in metres.
        Points within ray_start_distance of the opening plane are skipped to
        avoid self-hits. A ray with no hit has infinite clearance.
        """
        if not np.isfinite(ray_radius) or ray_radius <= 0:
            raise ValueError("Ray radius must be positive and finite")
        if not np.isfinite(ray_start_distance) or ray_start_distance < 0:
            raise ValueError("Ray start distance must be nonnegative and finite")
        colors = np.asarray(pcd_aff.colors)
        mask = colors[:, 1] > 0

        indices = np.where(mask)[0]
        pcd = pcd_aff.select_by_index(list(indices))
        points = np.asarray(pcd.points, dtype=float)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) < 3:
            raise ValueError("At least three pouring-affordance points are required")

        # Fit the opening plane, including exactly planar rims (for which a 3D
        # OBB can fail). The full object determines which side faces outward.
        center = points.mean(axis=0)
        _, singular_values, axes = np.linalg.svd(points - center, full_matrices=False)
        if singular_values[1] <= 1e-8:
            raise ValueError("Pouring-affordance points do not define an opening plane")
        direction = axes[-1].copy()
        direction /= np.linalg.norm(direction)

        if tool_pcd is None:
            raise ValueError("Pouring requires the full tool point cloud; regenerate the perception/reconstruction cache")
        tool_points = np.asarray(tool_pcd.points, dtype=float)
        tool_points = tool_points[np.all(np.isfinite(tool_points), axis=1)]
        if len(tool_points) < 3:
            raise ValueError("At least three full-tool points are required")
        # Cast from the opening center along +normal and -normal. Restrict hits
        # to a thin tube around each ray so off-ray handles/walls cannot decide
        # the sign. The tolerance accounts for gaps between point samples.
        offsets = tool_points - center
        depths = offsets @ direction
        radial_distances = np.linalg.norm(offsets - depths[:, None] * direction, axis=1)
        on_ray = radial_distances <= ray_radius
        positive_hits = depths[on_ray & (depths > ray_start_distance)]
        negative_hits = -depths[on_ray & (depths < -ray_start_distance)]
        positive_distance = float(np.min(positive_hits)) if len(positive_hits) else np.inf
        negative_distance = float(np.min(negative_hits)) if len(negative_hits) else np.inf
        if np.isinf(positive_distance) and np.isinf(negative_distance):
            raise ValueError(
                "Neither opening-normal ray hits the tool point cloud; outward direction is ambiguous"
            )
        if np.isclose(positive_distance, negative_distance, rtol=0, atol=1e-4):
            raise ValueError("Opening-normal rays have equal hit distances; outward direction is ambiguous")
        if positive_distance < negative_distance:
            direction = -direction
            support_depth = positive_distance
        else:
            support_depth = negative_distance

        if verbose:
            print(f"Opening-normal ray hits: +normal={positive_distance:.4f} m, -normal={negative_distance:.4f} m")
            normal_vector = cls.make_vector(center, direction)
            o3d.visualization.draw_geometries(
                [tool_pcd, pcd, normal_vector]
            )

        return cls(center=center, pcd=pcd, direction=direction,
                   support_depth=support_depth)

    def frame_at(self, pour_point):
        """Build an orthonormal feature frame at a candidate pouring point."""
        z_axis = self.direction / np.linalg.norm(self.direction)
        x_axis = np.asarray(pour_point, dtype=float) - self.center
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
        x_norm = np.linalg.norm(x_axis)
        if x_norm <= 1e-8:
            raise ValueError("Pouring point does not define an in-plane direction")
        x_axis /= x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        return np.column_stack((x_axis, y_axis, z_axis))
    
    def sample_pour_point(self, guide_point=None):
        points = np.asarray(self.pcd.points)

        if guide_point is not None:
            guide_point = np.asarray(guide_point)

            # Compute distances from all points to the guide point
            dists = np.linalg.norm(points - guide_point, axis=1)

            # Determine the 40% closest points
            num_remove = int(0.40 * len(points))
            remove_idx = np.argsort(dists)[:num_remove]

            mask = np.ones(len(points), dtype=bool)
            mask[remove_idx] = False
            points = points[mask]

        # Sample randomly from the remaining points
        idx = np.random.randint(len(points))
        return points[idx]
