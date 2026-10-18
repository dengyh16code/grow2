from improvised_tooluse.planner import PandaPlanner
from improvised_tooluse.sapien_utils import get_actor_mesh
from improvised_tooluse.simu_env import simu_env
from improvised_tooluse.tasks.task import Task, AffordanceRegion
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from sapien import Pose
from improvised_tooluse.geom_utils import quaternion_from_vectors
from sapien.pysapien.physx import PhysxRigidDynamicComponent

from real_world_client import RobotClient

import trimesh
from scipy.spatial import QhullError
from math import prod

# Three dense boolean grids use at most 24 MB, excluding voxelization temporaries.
MAX_CUT_GRID_VOXELS = 8_000_000

def mesh_to_shared_occupancy(mesh: trimesh.Trimesh, pitch: float, origin: np.ndarray, shape):
    """
    Convert a mesh into a boolean occupancy grid in a shared voxel coordinate system.
    - pitch: voxel size
    - origin: world-space min corner of voxel grid
    - shape: (nx, ny, nz) of the shared occupancy array
    """
    # voxelize and (optionally) fill interior so you get a solid volume
    vg = mesh.voxelized(pitch)
    vg = vg.fill()  # important for volume intersection (requires closed-ish meshes)

    # centers of filled voxels in world coordinates
    pts = vg.points  # (N, 3)

    # map world voxel centers -> shared voxel indices
    ijk = np.floor((pts - origin) / pitch).astype(np.int64)

    # keep only indices inside the shared grid
    m = np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)
    ijk = ijk[m]

    occ = np.zeros(shape, dtype=bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    return occ

def voxel_intersection(mesh_a: trimesh.Trimesh, mesh_b: trimesh.Trimesh, pitch: float):
    if not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("Voxel pitch must be finite and positive")
    for mesh in (mesh_a, mesh_b):
        if mesh is None or mesh.is_empty or not np.isfinite(mesh.vertices).all():
            raise ValueError("Cut evaluation mesh is empty or has nonfinite vertices")
        if mesh.bounds.shape != (2, 3) or not np.isfinite(mesh.bounds).all():
            raise ValueError("Cut evaluation mesh has invalid bounds")

    # shared bounds -> shared grid
    bounds_min = np.minimum(mesh_a.bounds[0], mesh_b.bounds[0])
    bounds_max = np.maximum(mesh_a.bounds[1], mesh_b.bounds[1])

    # a tiny padding so boundary voxels aren't clipped
    pad = pitch * 2
    origin = bounds_min - pad
    extent = (bounds_max - bounds_min) + 2 * pad

    # Check in floating point before integer conversion or any voxelization.
    dimensions = np.ceil(extent / pitch)
    if (not np.isfinite(dimensions).all() or np.any(dimensions < 1)
            or np.any(dimensions > MAX_CUT_GRID_VOXELS)):
        raise ValueError(f"Unsafe cut voxel grid dimensions: {dimensions.tolist()}")
    shape = tuple(int(n) for n in dimensions)
    if prod(shape) > MAX_CUT_GRID_VOXELS:
        raise ValueError(
            f"Cut voxel grid {shape} exceeds {MAX_CUT_GRID_VOXELS} voxels"
        )

    if np.any(np.maximum(mesh_a.bounds[0], mesh_b.bounds[0]) >
              np.minimum(mesh_a.bounds[1], mesh_b.bounds[1])):
        return trimesh.voxel.VoxelGrid(np.zeros((1, 1, 1), dtype=bool))

    occ_a = mesh_to_shared_occupancy(mesh_a, pitch, origin, shape)
    occ_b = mesh_to_shared_occupancy(mesh_b, pitch, origin, shape)

    occ_i = occ_a & occ_b

    # Build a VoxelGrid from the intersection occupancy
    # Transform maps voxel indices -> world voxel centers
    transform = np.eye(4)
    transform[0, 0] = pitch
    transform[1, 1] = pitch
    transform[2, 2] = pitch
    transform[:3, 3] = origin + pitch * 0.5

    vg_i = trimesh.voxel.VoxelGrid(
        encoding=trimesh.voxel.encoding.DenseEncoding(occ_i),
        transform=transform
    )
    return vg_i

class CutTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_collision_component = self.target.find_component_by_type(PhysxRigidDynamicComponent)
        self.target_collision_removed = False
        
    def remove_target_collision(self):
        assert not self.target_collision_removed, "Target collision already removed"
        self.target.remove_component(self.target_collision_component)
        self.target_collision_removed = True
        
    def add_target_collision(self):
        assert self.target_collision_removed, "Target collision not removed yet"
        self.target.add_component(self.target_collision_component)
        self.target_collision_removed = False
    
    def retrieve(self, checkpoint):
        if self.target_collision_removed:
            self.add_target_collision()
        super().retrieve(checkpoint)
    
    def execute_task(self, client: RobotClient = None, num_attempts: int = 50):
        cutted_region = CutEDRegion.from_perception(self.target_aff_pcd, self.verbose)
        cut_region = CutRegion.from_perception(self.tool_aff_pcd, self.verbose)
        checkpoint = self.simu_env.pack()
        
        planning_success = False 
        for num_trials in range(num_attempts):
            # grasp
            ee_pose_rel_tool, pre_grasp_traj, grasp_traj, post_grasp_traj = self.plan_grasp_sequence()
            offset = 0.1
            cut_direction = np.array([0, 0, -1], dtype=float)  # cut downwards
            
            sub_num_attempts = 20
            status = -1
            for sub_num_trials in range(sub_num_attempts):
                cut_point = cut_region.sample_cut_point()
                pre_cut_position = cutted_region.center - cut_direction * offset
                pre_cut_quat = quaternion_from_vectors(cut_region.normal_direction, cut_direction, random_choose=True)
                pre_cut_tool_pose = (Pose(cut_point) *
                                    Pose(p=pre_cut_position - cut_point, q=pre_cut_quat) *
                                    Pose(p=cut_point).inv() *
                                    self.tool_initial_pose)
                pre_cut_pose = pre_cut_tool_pose * ee_pose_rel_tool
                cut_pose = Pose(cut_direction * (offset+0.05)) * pre_cut_pose
                self.planner.update_from_simulation()
                status, pre_cut_traj = self.planner.move_to_pose(pre_cut_pose, return_traj=True)
                if status != 0:
                    print("Failed to move to pre-cut pose, subretrying")
                    continue
                break
            
            if status != 0:
                print("Failed to move to pre-cut pose after several attempts, retrying whole process")
                self.retrieve(checkpoint)
                continue
            
            self.remove_target_collision()            
            self.planner.update_from_simulation()
            status, cut_traj = self.planner.move_to_pose(cut_pose, ignore_collisions=True, return_traj=True)  #cut down
            if status != 0:
                print("Failed to move to cut pose, retrying whole process")
                self.retrieve(checkpoint)
                continue
            
            evaluation_success = self.evaluate()
            if not evaluation_success:
                print("Sweep task evaluation failed, retrying...")
                self.retrieve(checkpoint)
                continue
            
            print("Cutting executed successfully")
            self.retrieve(checkpoint)
            self.simu_env.start_recording()
            self.simu_env.create_viewer()
            self.simu_env.follow_path(pre_grasp_traj)
            self.simu_env.follow_path(grasp_traj)
            self.simu_env.attach_object(self.tool)
            self.simu_env.close_gripper()
            self.simu_env.follow_path(post_grasp_traj)
            self.simu_env.follow_path(pre_cut_traj)
            self.remove_target_collision()
            self.simu_env.follow_path(cut_traj)
            
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
        client.execute_trajectory(pre_cut_traj)
        input('next')
        client.execute_trajectory(cut_traj)
        return 0

    def sample_grasping_pose(self):
        idx = np.random.randint(len(self.grasping_poses))
        grasp_mat = self.grasping_poses[idx]
        return grasp_mat

    def evaluate(self):
        print("Evaluating cut task...")
        if self.target_collision_removed:
            self.add_target_collision()
            
        tool_mesh = get_actor_mesh(self.tool)
        target_mesh = get_actor_mesh(self.target)
        try:
            vg_i = voxel_intersection(tool_mesh, target_mesh, pitch=0.005)
        except (ValueError, MemoryError) as error:
            print(f"Cut task evaluation rejected geometry: {error}")
            for label, actor, mesh in (
                ("tool", self.tool, tool_mesh), ("target", self.target, target_mesh)
            ):
                print(f"Cut {label} pose: {actor.pose}; bounds: "
                      f"{None if mesh is None else mesh.bounds}")
            return False
        points = vg_i.points
        if len(points) == 0:
            print("Cut task voxel intersection is empty!")
            return False
        if len(points) < 4 or not np.isfinite(points).all():
            print(f"Cut task intersection has insufficient or invalid points ({len(points)}).")
            return False
        try:
            if np.linalg.matrix_rank(points - points.mean(axis=0)) < 3:
                print("Cut task intersection is degenerate (no 3D volume).")
                return False
            to_origin, extents = trimesh.bounds.oriented_bounds(points)
        except (QhullError, ValueError, IndexError, np.linalg.LinAlgError) as error:
            print(f"Cut task intersection bounds failed: {type(error).__name__}: {error}")
            return False
        if (not np.isfinite(to_origin).all() or not np.isfinite(extents).all()
                or np.any(extents <= 0)):
            print("Cut task intersection has invalid oriented bounds.")
            return False
        
        # get the shortest side length and the direction of this side
        min_side_length = min(extents)
        min_side_idx = np.argmin(extents)
        
        R = to_origin[:3, :3]
        min_side_direction = R[min_side_idx, :]
        
        print(f"Cut task min side length: {min_side_length:.4f} m")
        print(f"Cut task min side direction: {min_side_direction}")
        success = (
            min_side_length < 0.04 and
            np.abs(np.dot(min_side_direction, np.array([0, 0, 1]))) < 0.5
        )
        return success        

@dataclass
class CutEDRegion(AffordanceRegion):  #bowl
    center: np.ndarray  # (3,)
    pcd: o3d.geometry.PointCloud

    @classmethod
    def from_perception(cls, pcd_aff, verbose: bool = False):
        colors = np.asarray(pcd_aff.colors)
        mask = colors[:, 2] > 0

        indices = np.where(mask)[0]
        pcd_cutted = pcd_aff.select_by_index(list(indices))
        obb = pcd_cutted.get_oriented_bounding_box()
        center = np.asarray(obb.center)

        if verbose:
            obb.color = (1, 0, 0)
            o3d.visualization.draw_geometries([pcd_aff, obb])

        return cls(center=center, pcd=pcd_cutted)
    
@dataclass
class CutRegion(AffordanceRegion):  #pouring region on tool
    oriented_bounding_box: o3d.geometry.OrientedBoundingBox
    pcd: o3d.geometry.PointCloud
    normal_direction: np.ndarray

    @classmethod
    def from_perception(cls, pcd_aff, verbose: bool = False):
        colors = np.asarray(pcd_aff.colors)
        mask = colors[:, 1] > 0

        indices = np.where(mask)[0]
        pcd_cut = pcd_aff.select_by_index(list(indices))

        obb = pcd_cut.get_oriented_bounding_box()
        
        R = np.asarray(obb.R, dtype=float)
        z_direction = np.array([0, 0, 1], dtype=float)
        cosine_values = z_direction @ R
        max_idx = np.argmax(np.abs(cosine_values))
        normal_direction = R[:, max_idx]
        
        if normal_direction [2] > 0: #cut from top to bottom
            normal_direction = -normal_direction #todo

        if verbose:
            obb.color = (1, 0, 0)
            normal_vector = cls.make_vector(obb.get_center(), normal_direction)
            o3d.visualization.draw_geometries([pcd_aff, obb, normal_vector])

        return cls(oriented_bounding_box=obb, pcd=pcd_cut, normal_direction=normal_direction)
    
    def sample_cut_point(self):
        z_min = np.asarray(self.pcd.points)[:, 2].min() 
        z_threshold = z_min + 0.01  # bottom 1cm points, heuristic
        mask = np.asarray(self.pcd.points)[:, 2] <= z_threshold
        indices = np.where(mask)[0]
        pcd_cut = self.pcd.select_by_index(list(indices))

        points = np.asarray(pcd_cut.points)
        idx = np.random.randint(len(points))
        return points[idx]
