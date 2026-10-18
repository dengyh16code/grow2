import os
import time
import numpy as np
from multiprocessing.util import debug
from transforms3d.quaternions import mat2quat
import argparse
from concurrent.futures import ThreadPoolExecutor

import sapien
from improvised_tooluse import tasks
from improvised_tooluse.config import load_config, default_config_path
from improvised_tooluse.affordance import Perception_module
from improvised_tooluse.geom_utils import dilate_bool_mask
from improvised_tooluse.planner import PandaPlanner
from improvised_tooluse.simu_env import simu_env
from improvised_tooluse.tasks.pound import PoundTask
from improvised_tooluse.vfm_client import fp_registration_request, fp_tracking_request, sam3d_request, sam3_request_concept
from improvised_tooluse.utils import plot_best
import trimesh 
from PIL import Image
import cv2
from scipy.spatial import cKDTree
import numpy as np
import json

from improvised_tooluse.geom_utils import visualize_perception_results
from real_world_client import RobotClient

R_opencv_to_p3d = np.diag([-1, -1, 1])
T_opencv_to_p3d = np.diag([-1, -1, 1, 1])
T_opencv_to_opengl = np.diag([1, -1, -1, 1])

rot_robot_camera = [[0.012291455017040709, 0.664614924782309, -0.7470849495808158], 
                    [0.9998155511856613, -0.019195582228370336, -0.0006270807173765007], 
                    [-0.014757497785068109, -0.7469394429132256, -0.664728278982927]]
trans_robot_camera = [1.135345733846695, 0.0069297780339926274, 0.5267756751022766]


T_camera_to_robot = np.eye(4)
T_camera_to_robot[:3, :3] = np.array(rot_robot_camera)
T_camera_to_robot[:3, 3] = np.array(trans_robot_camera)

T_robot_to_world = np.eye(4)
T_robot_to_world[:3, 3] = np.array([-0.3, 0, 0.525 - 0.0036])

T_camera_to_world = T_robot_to_world @ T_camera_to_robot

def sam3_seg(image, text=None, bbox=None):
    scores, bboxes, masks = sam3_request_concept(image, text, bbox, 0.3)
    if len(scores) == 0:
        return None, None, None
    else:
        max_score_index = np.argmax(scores)
        best_score = scores[max_score_index]
        best_mask = masks[max_score_index]
        best_bbox = bboxes[max_score_index]
        return best_score, best_bbox, best_mask
    
def depth2pointmap_cam(depth, K):
    """Returns pointmap in camera convention of pytorch3d, i.e., x-left, y-up, z-forward"""
    invalid_mask = depth < 0.01
    depth[invalid_mask] = np.nan
    
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    print(K)
    H, W = depth.shape
    ugrid, vgrid = np.meshgrid(np.arange(W), np.arange(H))
    x = (ugrid - cx) * depth / fx
    y = (vgrid - cy) * depth / fy
    z = depth
    pointmap = np.stack([x,y,z], axis=-1)
    R_opencv_to_p3d = np.diag([-1, -1, 1])
    pointmap_p3d = pointmap @ R_opencv_to_p3d.T
    return pointmap_p3d

def pointmap2pcd(pointmap, color):
    pointcloud = pointmap.reshape(-1,3)
    pcd_color = color.reshape(-1,3)
    invalid_mask = np.isnan(pointcloud).any(axis=-1)
    pointcloud = pointcloud[~invalid_mask]
    pcd_color = pcd_color[~invalid_mask]
    return trimesh.points.PointCloud(pointcloud, colors=pcd_color)

def depth2pcd(depth, K, color=None):
    """return trimesh pointcloud in the camera coordinate system, pytorch3d convention"""
    pointmap = depth2pointmap_cam(depth, K) # (H,W,3)
    if color is None:
        color = np.zeros_like(pointmap, dtype=np.uint8)
    pcd = pointmap2pcd(pointmap, color) # trimesh pointcloud
    return pcd

def get_mask_and_model(color, depth, K, text, verbose=False):
    pointmap_cam_p3d = depth2pointmap_cam(depth, K) # (H,W,3)
    image = Image.fromarray(color)
    score, bbox, mask = sam3_seg(image, text=text)
    if verbose:
        plot_best(image, mask, bbox, score, show=True)
    glb_trimesh, _, scale, _, _ = sam3d_request(image, mask, pointmap_cam_p3d)
    glb_trimesh: trimesh.Trimesh = next(iter(glb_trimesh.geometry.values()))
    mesh = glb_trimesh.apply_scale(scale)
    target_faces = 10000
    if len(mesh.faces) > target_faces:
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            src_v = mesh.vertices.copy()
            src_c = mesh.visual.vertex_colors.copy()
        else:
            src_v, src_c = None, None
        mesh_ds = mesh.simplify_quadric_decimation(target_faces)
        if src_v is not None and src_c is not None and len(src_c) == len(src_v):
            tree = cKDTree(src_v)
            _, nn = tree.query(mesh_ds.vertices, k=1)
            mesh_ds.visual = trimesh.visual.ColorVisuals(mesh_ds, vertex_colors=src_c[nn])
        mesh = mesh_ds
    return mask, mesh

def read_color_and_depth(dir, index=0):
    color = cv2.cvtColor(cv2.imread(f'{dir}/color_image_{index}.png'), cv2.COLOR_BGR2RGB)
    depth = np.load(f'{dir}/depth_image_{index}.npy')
    return color, depth


def make_observations(K, color, depth, ):
    observations = {}
    observations["front_intrinsic"] = K
    observations["front_color"] = Image.fromarray(color)
    observations["front_depth"] = depth
    pointmap_cam_p3d = depth2pointmap_cam(depth, K)
    observations["front_pointmap_cam"] = pointmap_cam_p3d
    
    pointmap_cam_p3d_aug = np.concatenate(
        [pointmap_cam_p3d, np.ones((*pointmap_cam_p3d.shape[:2], 1))],
        axis=-1,
    )  # (H,W,4)
    pointmap_world = (T_camera_to_world @ T_opencv_to_p3d.T @ pointmap_cam_p3d_aug.reshape(-1, 4).T).T[:, :3]
    pointmap_world = pointmap_world.reshape(pointmap_cam_p3d.shape)
    
    observations["front_pointmap"] = pointmap_world
    observations["front_extrinsic"] = T_camera_to_world @ T_opencv_to_opengl.T
    return observations


def make_scene_pcd(pointmap, remove_masks: list =[], crop_x_range: tuple = None, crop_y_range: tuple = None, crop_z_range: tuple = None, dilate: int = 0):
    mask = np.ones(pointmap.shape[:2], dtype=bool)
    for remove_mask in remove_masks:
        remove_mask = np.array(remove_mask, dtype=bool)
        if dilate > 0:
            remove_mask = dilate_bool_mask(remove_mask, dilate)
        mask = np.logical_and(mask, np.logical_not(remove_mask))
    pointcloud = pointmap.reshape(-1,3)
    pointcloud = pointcloud[mask.reshape(-1)]
    if crop_x_range is not None:
        x_mask = np.logical_and(pointcloud[:,0] >= crop_x_range[0], pointcloud[:,0] <= crop_x_range[1])
        pointcloud = pointcloud[x_mask]
    if crop_y_range is not None:
        y_mask = np.logical_and(pointcloud[:,1] >= crop_y_range[0], pointcloud[:,1] <= crop_y_range[1])
        pointcloud = pointcloud[y_mask]
    if crop_z_range is not None:
        z_mask = np.logical_and(pointcloud[:,2] >= crop_z_range[0], pointcloud[:,2] <= crop_z_range[1])
        pointcloud = pointcloud[z_mask]
    return pointcloud


def main(verbose=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None,
                        help="config.yaml to run with (default: {})".format(default_config_path()))
    parser.add_argument('--load_cache', action='store_true')
    parser.add_argument('--instruction', required=True,
                        help='task instruction, e.g. "pour into the bowl"')
    parser.add_argument('--task_type', required=True, choices=tasks.names,
                        help='type of manipulation task to execute')
    args = parser.parse_args()
    load_config(args.config)
    load_cache = args.load_cache

    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)

    task_type = args.task_type
    instruction = args.instruction
    print(f'whether to load cache: {load_cache}')
    detector = Perception_module()
    cache_file = f'{result_dir}/perception_results.pkl'
    reconstruction_cache_file = f'{result_dir}/reconstruction_results.pkl'
    if load_cache and os.path.exists(cache_file):
        detector.load_perception_results(cache_file, reconstruction_cache_file)
        scene_pcd = np.load(f'{result_dir}/scene_pcd.npy')
        T_tool_mesh_world = np.load(f'{result_dir}/T_tool_mesh_world.npy')
        T_target_mesh_world = np.load(f'{result_dir}/T_target_mesh_world.npy')
    else:
        from real_world_camera_client import CameraClient

        camera_client = CameraClient()
        K, color, depth = camera_client.get_observation()
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir, "color_image.png"), color_bgr)
        detector.update_observations(make_observations(K, color, depth))
        selected_tool_dic = detector.object_and_affordance_selection(instruction)
        detector.affordance_map_extraction(
            selected_tool_dic, target_affordance=None,
            reconstruction_save_dir=reconstruction_cache_file,
        )
        reconstruction_results = detector.reconstruction_results_processed
        detector.affordance_postprocessing()
        detector.grasping_pose_extraction(selected_tool_dic["tool_object"])
        detector.save_perception_results(cache_file)
        
        scene_pcd = make_scene_pcd(
            detector.obs["front_pointmap"], 
            remove_masks=[reconstruction_results[selected_tool_dic["tool_object"]]['mask'], reconstruction_results[selected_tool_dic["target_object"]]['mask']],
            crop_x_range=(0.0, 0.75),
            crop_y_range=(-0.7, 0.7),
            crop_z_range=(0.3, 1.2),
            dilate=5,
        )
        np.save(f'{result_dir}/scene_pcd.npy', scene_pcd)
        
        tool_mesh = reconstruction_results[selected_tool_dic["tool_object"]]['mesh']
        target_mesh = reconstruction_results[selected_tool_dic["target_object"]]['mesh']
        T_tool_mesh_world = reconstruction_results[selected_tool_dic["tool_object"]]['T_mesh_world']
        T_target_mesh_world = reconstruction_results[selected_tool_dic["target_object"]]['T_mesh_world']
        pose_last_opencv = reconstruction_results[selected_tool_dic["tool_object"]].get('pose_last_opencv')

        tool_mesh.export(f'{result_dir}/tool_mesh.glb')
        target_mesh.export(f'{result_dir}/target_mesh.glb')
        np.save(f'{result_dir}/T_tool_mesh_world.npy', T_tool_mesh_world)
        np.save(f'{result_dir}/T_target_mesh_world.npy', T_target_mesh_world)
        np.save(f'{result_dir}/pose_last_opencv.npy', pose_last_opencv)

        vhacd_tasks = [
            (f"{result_dir}/tool_mesh.glb", "_vhacd.obj"),
            (f"{result_dir}/target_mesh.glb", "_vhacd.obj"),
        ]
        from convex_obj import vhacd_decompose

        with ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(vhacd_decompose, path, suffix=suffix) for path, suffix in vhacd_tasks]
            for f in futures:
                f.result()
        
    if verbose:
        visualize_perception_results(detector.obs["front_pointmap"], 
                                np.array(detector.obs["front_color"]),
                                detector.tool_aff_pcd,
                                detector.target_aff_pcd,
                                detector.pred_grasps_world)
    
    simu_environment = simu_env(visualize=False)
    tool = simu_environment._build_actor(
        name="tool",
        collision_model_path=f'{result_dir}/tool_mesh_vhacd.obj',
        visual_model_path=f'{result_dir}/tool_mesh.glb',
        default_pose=sapien.Pose(p=T_tool_mesh_world[:3, 3], q=mat2quat(T_tool_mesh_world[:3, :3])),
        adjust_height=False
    )

    tool_comp = tool.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    tool_comp.set_disable_gravity(True)

    target = simu_environment._build_actor(
        name="target",
        collision_model_path=f'{result_dir}/target_mesh_vhacd.obj',
        visual_model_path=f'{result_dir}/target_mesh.glb',
        default_pose=sapien.Pose(p=T_target_mesh_world[:3, 3], q=mat2quat(T_target_mesh_world[:3, :3])),
        adjust_height=False
    )

    target_comp = target.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
    target_comp.set_disable_gravity(True)
    
    if tool is None:
        print("No valid grasping tool found. Exiting.")
    else:
        motion_planner = PandaPlanner(simu_environment)
        motion_planner.add_pcd_collision(scene_pcd)

        task = tasks.names[task_type](simu_environment, motion_planner, detector, tool, target, verbose=True, is_soft_gripper=True)
        # task.execute_task()
        client = RobotClient()
        client.go_home()
        client.open_gripper()
        execute_task = getattr(task, "execute_task_real_world", task.execute_task)
        execute_task(client=client)
    
    simu_environment.idle()

if __name__ == "__main__":
    main(verbose=True)
