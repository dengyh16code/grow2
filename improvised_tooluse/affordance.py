from doctest import FAIL_FAST
from PIL import Image
import os

import trimesh
from improvised_tooluse.geom_utils import merge_views, o3d_to_numpy, numpy_to_o3d, register_with_icp_fallback, render_views, filter_outliers
from improvised_tooluse.utils import apply_affordance_color, crop_image_and_apply_mask, project_mask_to_original, xyxy_to_xywh, plot_best
from improvised_tooluse.utils import categories_from_prompt, prepare_object_views, decode_object_list
from improvised_tooluse.vfm_client import fp_registration_request, sam3_request_concept, sam3d_request
from improvised_tooluse.vfm_client import grasp_request, grasping_pose_transform
from improvised_tooluse.vlm_client import load_prompts, query_model, query_bounding_box, parse_selection_results
import numpy as np
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import copy
import requests
import open3d as o3d
from scipy.spatial import cKDTree

from improvised_tooluse import config

# The VLM meta-prompt .txt files ship with the package, next to this file.
PROMPT_DIR = Path(__file__).with_name("prompt")
DOWNWARD_GRASP_THRESHOLD = 0.8

def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        return None
    return v / n


def convert_pcd_to_top_down_grasp(
    pcd: o3d.geometry.PointCloud,
    num_samples: int = 20,
    pregrasp: float = 0.1034,
    margin_ratio: float = 0.12,
    world_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=float),):
    """
    Assumption: object is almost upright (one OBB axis ~ world Z).
    Generates top-down grasps:
      - approach (gripper +Z) = world [0,0,-1]
      - closing (gripper +Y) = short footprint axis
      - sliding (gripper +X) = long footprint axis
    Produces multiple poses by translating along long axis.

    Returns:
      poses: list of (4,4)
    """
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    

    grasp_aff_pcd = copy.deepcopy(pcd)
    colors = np.asarray(grasp_aff_pcd.colors)
    mask = colors[:, 0] > 0
    indices = np.where(mask)[0]
    pcd_grasp = grasp_aff_pcd.select_by_index(list(indices))

    obb = pcd_grasp.get_oriented_bounding_box()
    center = np.asarray(obb.center)
    R_obb = np.asarray(obb.R)       # columns are OBB axes in world
    ext = np.asarray(obb.extent)    # lengths along those axes

    up = _normalize(world_up)
    if up is None:
        raise ValueError("world_up is degenerate")

    # 1) find vertical axis: most aligned with world_up
    dots = np.abs(R_obb.T @ up)  # (3,) abs dot of each axis with world_up
    idx_vert = int(np.argmax(dots))

    # 2) footprint axes are the other two
    idxs = [0, 1, 2]
    idxs.remove(idx_vert)
    i0, i1 = idxs[0], idxs[1]

    # choose long/short by extent
    if ext[i0] >= ext[i1]:
        idx_long, idx_short = i0, i1
    else:
        idx_long, idx_short = i1, i0

    long_axis = R_obb[:, idx_long].copy()
    short_axis = R_obb[:, idx_short].copy()

    # Project to XY just in case there's a tiny tilt
    long_xy = _normalize(long_axis - np.dot(long_axis, up) * up)
    short_xy = _normalize(short_axis - np.dot(short_axis, up) * up)
    if long_xy is None or short_xy is None:
        raise ValueError("Footprint axes degenerate after projection. Object may not be upright enough.")

    # Top-down approach direction
    z = _normalize(-up)  # approach downwards
    y = _normalize(short_xy)  # closing direction = thin axis in footprint
    x = _normalize(np.cross(y, z))
    if x is None:
        raise ValueError("Degenerate x axis from cross(y, z).")

    # Re-orthogonalize y
    y = _normalize(np.cross(z, x))
    Rg = np.stack([y, -x, z], axis=1)

    # Slide along long axis within the OBB length (minus margin)
    long_len = float(ext[idx_long])
    m = margin_ratio * (long_len / 2.0)
    half = max(0.0, long_len / 2.0 - m)

    if num_samples == 1:
        offsets = np.array([0.0], dtype=float)
    else:
        offsets = np.linspace(-half, half, num_samples)

    poses = []
    for s in offsets:
        t = center + s * x - pregrasp * z
        T = np.eye(4)
        T[:3, :3] = Rg
        T[:3, 3] = t
        poses.append(T)

    return poses


def convert_pcd_to_grasp_map(pcd, intrinsic, extrinsic_opengl, W, H):
    grasp_aff_pcd = copy.deepcopy(pcd)
    colors = np.asarray(grasp_aff_pcd.colors)
    mask = colors[:, 0] > 0
    indices = np.where(mask)[0]
    pcd_grasp = grasp_aff_pcd.select_by_index(list(indices))
    points = np.asarray(pcd_grasp.points) 

    extrinsc_open3d = extrinsic_opengl @ np.diag([1, -1, -1, 1])
    T_cw = np.linalg.inv(extrinsc_open3d)

    pts_h = np.hstack([points, np.ones((points.shape[0], 1))]).T

    # Transform to camera frame
    Pc = T_cw @ pts_h
    X, Y, Z = Pc[0], Pc[1], Pc[2]

    # Keep only points in front of camera
    mask = Z > 0
    X, Y, Z = X[mask], Y[mask], Z[mask]

    # 6. Project to pixels
    uv = intrinsic @ np.vstack([X / Z, Y / Z, np.ones_like(Z)])
    u = np.round(uv[0]).astype(int)
    v = np.round(uv[1]).astype(int)

    # 7. Build depth image with z-buffer
    depth = np.full((H, W), np.inf, dtype=np.float32)

    for ui, vi, zi in zip(u, v, Z):
        if 0 <= ui < W and 0 <= vi < H:
            if zi < depth[vi, ui]:
                depth[vi, ui] = zi

    depth[depth == np.inf] = 0.0  # or np.nan

    return depth, pcd_grasp

class Perception_module:
    """Perception module for object detection, affordance detection and grasping pose extraction."""
    def __init__(self):
        self.prompts = load_prompts(PROMPT_DIR)
        # Older saved batch configurations default to the simulation vocabulary.
        self.experiment_setting = config.get_config().get("experiment_setting", "simu")
        if self.experiment_setting not in ("simu", "real"):
            raise ValueError("experiment_setting must be 'simu' or 'real'")
        prompt_name = (
            "parse_object_simu" if self.experiment_setting == "simu"
            else "parse_objects_real"
        )
        self.object_parsing_prompt = self.prompts[prompt_name]
        self.object_categories = (
            categories_from_prompt(self.object_parsing_prompt)
            if self.experiment_setting == "simu" else None
        )
        self.sam3_confidence_threshold = 0.3
        self._object_segmentation_cache = {}
        self.reconstruction_results_processed = {}
        self.tool_object = None
        self.tool_pcd = None
        self.tool_center = None
    
    def update_observations(self, obs):
        self.obs = obs
        self._object_segmentation_cache.clear()

    @staticmethod
    def _affordance_fusion_config():
        """Return the shared settings for mask fusion and 3D postprocessing."""
        settings = config.get("affordance_fusion")
        return {
            "voxel_size": float(settings["voxel_size"]),
            "strong_min_positive_views": int(settings["strong_min_positive_views"]),
            "strong_min_positive_ratio": float(settings["strong_min_positive_ratio"]),
            "weak_min_positive_views": int(settings["weak_min_positive_views"]),
            "weak_min_positive_ratio": float(settings["weak_min_positive_ratio"]),
            "weak_neighbor_voxels": int(settings["weak_neighbor_voxels"]),
            "mask_erosion_pixels": int(settings["mask_erosion_pixels"]),
            "max_erosion_loss_ratio": float(settings["max_erosion_loss_ratio"]),
            "min_cluster_ratio": float(settings["min_cluster_ratio"]),
            "dbscan_eps_multiplier": float(settings["dbscan_eps_multiplier"]),
            "dbscan_min_points": int(settings["dbscan_min_points"]),
        }

    @classmethod
    def _merge_affordance_views(cls, render_results):
        settings = cls._affordance_fusion_config()
        return merge_views(
            render_results,
            affordance_voxel_size=settings["voxel_size"],
            strong_min_positive_views=settings["strong_min_positive_views"],
            strong_min_positive_ratio=settings["strong_min_positive_ratio"],
            weak_min_positive_views=settings["weak_min_positive_views"],
            weak_min_positive_ratio=settings["weak_min_positive_ratio"],
            weak_neighbor_voxels=settings["weak_neighbor_voxels"],
            mask_erosion_pixels=settings["mask_erosion_pixels"],
            max_erosion_loss_ratio=settings["max_erosion_loss_ratio"],
        )

    @staticmethod
    def _robust_pointcloud_center(pcd):
        """Estimate a geometry center without letting a few extreme points dominate."""
        points = np.asarray(pcd.points, dtype=np.float64)
        points = points[np.all(np.isfinite(points), axis=1)]
        if len(points) == 0:
            raise ValueError("Cannot estimate the tool center from an empty point cloud")
        lower, upper = np.quantile(points, [0.01, 0.99], axis=0)
        return (lower + upper) / 2.0
    
    def save_perception_results(self, filepath):
        perception_results = {
        "pointmap": self.obs["front_pointmap"],
        "pointcam": self.obs["front_pointmap_cam"],
        "rgb_image": np.array(self.obs["front_color"]),
        "depth_image": np.array(self.obs["front_depth"]),
        "extrinsic": self.obs["front_extrinsic"],
        "intrinsic": self.obs["front_intrinsic"],
        "pcd_tool": o3d_to_numpy(self.tool_aff_pcd),  
        "pcd_target": o3d_to_numpy(self.target_aff_pcd),
        "tool_object": self.tool_object,
        "grasping_poses": self.pred_grasps_world,
        "grasping_scores": self.grasp_scores,
        "grasp_image": np.array(self.grasp_image)
        }

        #channel 0: grasp, channel 1: function on tool, channel 2: function on target

        # Create folder only if filepath includes a directory
        folder = os.path.dirname(filepath)
        if folder != "":
            os.makedirs(folder, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(perception_results, f)

        print(f"Saved perception results to: {filepath}")
    
    def _reconstruction_cache_data(self):
        # Keep geometry and registration metadata, excluding raw service output.
        keys = ("mesh", "mesh_world", "T_mesh_world", "mask", "pose_last_opencv", "registration_diagnostics")
        return {
            name: {key: result[key] for key in keys if key in result}
            for name, result in self.reconstruction_results_processed.items()
        }

    def save_reconstruction_results(self, filepath):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self._reconstruction_cache_data(), f)
        print(f"Saved reconstruction results to: {filepath}")

    def load_perception_results(self, filepath, reconstruction_filepath=None):
        with open(filepath, "rb") as f:
            perception_results = pickle.load(f)

        self._object_segmentation_cache.clear()
        
        self.obs = {
        "front_pointmap": perception_results["pointmap"],
        "front_pointmap_cam": perception_results["pointcam"],
        "front_color": Image.fromarray(perception_results["rgb_image"]),
        "front_depth": perception_results["depth_image"],
        "front_extrinsic": perception_results["extrinsic"],
        "front_intrinsic": perception_results["intrinsic"],
        }

        self.tool_aff_pcd = numpy_to_o3d(perception_results["pcd_tool"])
        self.target_aff_pcd = numpy_to_o3d(perception_results["pcd_target"])
        self.tool_center = None
        self.tool_object = perception_results.get("tool_object")
        self.tool_pcd = None
        self.reconstruction_results_processed = {}
        # Reconstruction geometry is stored separately from perception results.
        if reconstruction_filepath is None:
            path = Path(filepath)
            reconstruction_filepath = path.with_name(
                path.name.replace("perception_results", "reconstruction_results", 1)
            ) if path.name.startswith("perception_results") else None
        if reconstruction_filepath is not None and os.path.exists(reconstruction_filepath):
            with open(reconstruction_filepath, "rb") as f:
                self.reconstruction_results_processed = pickle.load(f)
        if self.reconstruction_results_processed:
            self._build_tool_pointcloud()
        self.pred_grasps_world = perception_results["grasping_poses"]
        self.grasp_scores = perception_results["grasping_scores"]
        gi = perception_results.get("grasp_image", None)
        
        if gi is None:
            self.grasp_image = None
        elif gi.dtype == object:
            assert gi.item() is None
            self.grasp_image = None
        else:
            self.grasp_image = Image.fromarray(gi)

        self._ensure_downward_grasps()
        
        print(f"Loaded perception results from: {filepath}")

    def _build_tool_pointcloud(self):
        """Sample the full tool surface from its reconstructed world-space mesh."""
        reconstructions = self.reconstruction_results_processed
        if self.tool_object not in reconstructions:
            # Legacy caches have no selected-object name. Match their world-space
            # tool affordance points to the closest reconstructed object surface.
            points = np.asarray(self.tool_aff_pcd.points)
            points = points[np.all(np.isfinite(points), axis=1)]
            if len(points) == 0:
                raise ValueError("Cannot match an empty tool affordance to a reconstruction")
            self.tool_object = min(
                reconstructions,
                key=lambda name: np.median(cKDTree(
                    np.asarray(reconstructions[name]["mesh_world"].vertices)
                ).query(points)[0]),
            )
        mesh = reconstructions[self.tool_object]["mesh_world"]
        # Vertices alone can leave large gaps across flat faces, causing a ray
        # through a cup's bottom to miss. Generate surface samples at roughly
        # 1 mm spacing; a fixed seed keeps fresh and cached runs consistent.
        sample_count = max(20000, int(np.ceil(mesh.area / 0.001**2)))
        surface_points, _ = trimesh.sample.sample_surface(mesh, sample_count, seed=0)
        points = np.vstack([np.asarray(mesh.vertices), surface_points])
        self.tool_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        self.tool_center = self._robust_pointcloud_center(self.tool_pcd)
    
    def parse_objects(self, image_list, *, use_crops=True):
        """Detect and normalize object categories in one VLM request.

        Use native camera images here. Overlapping crops expose small or
        partially hidden objects while the full image retains scene context.
        Category validation and deduplication run locally after the response.
        """
        images, views, context = prepare_object_views(image_list, use_crops)
        self.object_detection_report = {"views": views}
        response = query_model(
            messages=[context],
            images=images,
            meta_prompt=self.object_parsing_prompt,
        )
        self.object_detection_report["parse_response"] = response
        objects = decode_object_list(response, self.object_categories)
        self.object_detection_report["objects"] = objects
        return objects
    
    def object_and_affordance_selection(self, task, object_list=None):
        raw_image = self.obs["front_color"]

        if object_list is None:
            object_list = self.parse_objects([raw_image])
            print(f"Detected objects: {object_list}")
        else:
            print(f"Using provided object list: {object_list}")
        
        object_parts_dic = self.parse_object_parts([raw_image], object_list=object_list)
        print(f"Object parts dictionary: {object_parts_dic}")
        if not object_parts_dic:
            raise RuntimeError(
                "No candidate objects could be segmented; tool selection "
                "cannot continue."
            )

        dict_text = json.dumps(object_parts_dic)
        selectable_objects = list(object_parts_dic)
        response = query_model(
            messages=[task, dict_text],
            images=[raw_image],
            meta_prompt=self.prompts["tool_selection"]
        )
        print(response)
        selected_obj_affordance = parse_selection_results(response)
        
        selected_tool = selected_obj_affordance.get("tool_object")
        if selected_tool not in object_parts_dic:
            raise ValueError(
                "Tool selection returned a tool without valid segmentation: "
                f"{selected_tool!r}. "
                f"Selectable objects were: {selectable_objects}."
            )
        return selected_obj_affordance
    
    def affordance_map_extraction(self, selected_obj_affordance, target_affordance=None, reconstruction_save_dir=None):
        """
        returns:
        {
            grasp_part: points on the world frame,
            tool_function_part: points on the world frame ,
            target_function_part: points on the world frame
        }
        """
        tool_object = selected_obj_affordance["tool_object"]
        self.tool_object = tool_object
        grasp_part = selected_obj_affordance["tool_object_info"]["tool_grasp_part"]
        tool_function_part = selected_obj_affordance["tool_object_info"]["tool_function_part"]
        target_object = selected_obj_affordance["target_object"]

        target_function_part = selected_obj_affordance["target_object_info"]["target_function_part"]
        specified_target_object_list = ["screw", "donut", "moth", "cake", "pizza"]
        if target_object in specified_target_object_list:
            target_function_part = target_object #no need to segment part

        tool_affordance_dic = {
            "tool_grasp_part": grasp_part,
            "tool_function_part": tool_function_part
        }
        target_affordance_dic = {
            "target_function_part": target_function_part
        }
        
        if target_affordance is not None:
            print("Using provided target affordance map.")
            obj_name_list = [tool_object]
        else:
            obj_name_list = [tool_object, target_object]

        require_reconstruction = False
        if reconstruction_save_dir and os.path.exists(reconstruction_save_dir):
            with open(reconstruction_save_dir, "rb") as f:
                reconstruction_cache_results = pickle.load(f)
                for obj_name in obj_name_list:
                    if obj_name not in reconstruction_cache_results.keys():
                        require_reconstruction = True
                        break
        else:
            require_reconstruction = True


        if require_reconstruction:
            reconstruction_results = self.run_reconstruction_parallel(obj_name_list, max_workers=2)
            self.reconstruction_results_processed = self.process_reconstruction_results(reconstruction_results, self.obs["front_extrinsic"], self.obs["front_intrinsic"])
            
            print("Reconstruction completed for:", obj_name_list)
            if reconstruction_save_dir:
                self.save_reconstruction_results(reconstruction_save_dir)
        
        else:
            self.reconstruction_results_processed = reconstruction_cache_results
            print("Reconstruction results loaded from cache.")

        if target_affordance is None:
            target_render_results = self.multi_view_rendering(self.reconstruction_results_processed[target_object]['mesh_world'])
            target_render_results = self.run_affordance_parallel(target_object, target_render_results,target_affordance_dic, max_workers=8)

        tool_render_results = self.multi_view_rendering(self.reconstruction_results_processed[tool_object]['mesh_world'])
        tool_render_results = self.run_affordance_parallel(tool_object, tool_render_results, tool_affordance_dic, max_workers=8)

        print("Affordance detection completed.")

        if target_affordance is None:
            _, self.target_aff_pcd = self._merge_affordance_views(target_render_results)
        else:
            points = np.asarray(target_affordance.points)
            z_max = points[:, 2].max()
            threshold = z_max - 0.05
            mask = points[:, 2] >= threshold
            idxs = np.where(mask)[0]
            self.target_aff_pcd = target_affordance.select_by_index(idxs) # for visualization

        _, self.tool_aff_pcd = self._merge_affordance_views(tool_render_results)
        self._build_tool_pointcloud()
        print("Affordance fusion completed.")

    def affordance_postprocessing(self):
        """perform clustering to remove outlier and noise"""
        settings = self._affordance_fusion_config()
        voxel_size = settings["voxel_size"]
        filter_kwargs = {
            "voxel_size": voxel_size,
            "dbscan_eps": voxel_size * settings["dbscan_eps_multiplier"],
            "min_points": settings["dbscan_min_points"],
            "min_cluster_ratio": settings["min_cluster_ratio"],
        }

        colors = np.asarray(self.tool_aff_pcd.colors)
        mask_action = colors[:, 1] > 0
        mask_grasp = colors[:, 0] > 0

        indices_action = np.where(mask_action)[0]
        pcd_action = self.tool_aff_pcd.select_by_index(list(indices_action))

        indices_grasp = np.where(mask_grasp)[0]
        pcd_grasp = self.tool_aff_pcd.select_by_index(list(indices_grasp))

        if len(pcd_action.points) > 0:
            pcd_action = filter_outliers(pcd_action, **filter_kwargs)

        if len(pcd_grasp.points) > 0:
            pcd_grasp = filter_outliers(pcd_grasp, **filter_kwargs)
        
        colors_target = np.asarray(self.target_aff_pcd.colors)
        mask_target = colors_target[:, 2] > 0
        indices_target = np.where(mask_target)[0]
        pcd_target = self.target_aff_pcd.select_by_index(list(indices_target))
        
        if len(pcd_target.points) > 0:
            pcd_target = filter_outliers(pcd_target, **filter_kwargs)
        
        self.tool_aff_pcd = pcd_action + pcd_grasp
        self.target_aff_pcd = pcd_target

        print("post processing completed")

    @staticmethod
    def process_mesh(glb_trimesh, scale) -> trimesh.Trimesh:
        if not isinstance(glb_trimesh, trimesh.Trimesh):
            assert len(glb_trimesh.geometry.values()) == 1, "Expected glb with a single geometry."
            glb_trimesh: trimesh.Trimesh = next(iter(glb_trimesh.geometry.values()))
        mesh = glb_trimesh.apply_scale(scale)
        target_faces = 10000
        
        # import ipdb; ipdb.set_trace()
        if len(mesh.faces) > target_faces:
            if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
                src_v = mesh.vertices.copy()
                src_c = mesh.visual.vertex_colors.copy()
            else:
                src_v, src_c = None, None
            mesh_ds = mesh.simplify_quadratic_decimation(target_faces)
            if src_v is not None and src_c is not None and len(src_c) == len(src_v):
                tree = cKDTree(src_v)
                _, nn = tree.query(mesh_ds.vertices, k=1)
                mesh_ds.visual = trimesh.visual.ColorVisuals(mesh_ds, vertex_colors=src_c[nn])
            mesh = mesh_ds
        return mesh

    def process_reconstruction_results(self, reconstruction_results_dict, T_cam2world, K):
        T_opencv_to_opengl = np.diag([1, -1, -1, 1])
        registration_way = str(config.get("registration_way")).strip().lower()
        valid_registration_ways = {"foundation_pose", "icp"}
        if registration_way not in valid_registration_ways:
            raise ValueError(
                f"Unsupported registration_way {registration_way!r}; expected one of "
                f"{sorted(valid_registration_ways)}."
            )

        for object_name in reconstruction_results_dict.keys():
            #tranform to world frame
            glb_raw = reconstruction_results_dict[object_name]['glb_raw']
            if glb_raw is None:
                raise RuntimeError("object resonstruction failed")
            mask = reconstruction_results_dict[object_name]['mask']
            color = np.array(self.obs["front_color"])
            scale = reconstruction_results_dict[object_name]['scale']
            depth = self.obs["front_depth"]
            
            mesh = self.process_mesh(glb_raw, scale)
            if registration_way == "foundation_pose":
                pose_opencv, pose_last_opencv = fp_registration_request(
                    mesh, K, color, depth, mask=(mask > 0)
                )
                T_mesh_world = T_cam2world @ T_opencv_to_opengl @ pose_opencv
            else:
                # SAM3D already applies scale and pose to glb_processed, yielding
                # vertices in the OpenGL camera frame. Do not scale it again.
                glb_processed = reconstruction_results_dict[object_name].get('glb_processed')
                if glb_processed is None:
                    raise RuntimeError(
                        f"Missing glb_processed for ICP registration of {object_name!r}."
                    )
                mesh = copy.deepcopy(glb_processed)
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)

                pointmap_world = np.asarray(self.obs["front_pointmap"])
                mask_bool = np.asarray(mask, dtype=bool)
                if pointmap_world.shape[:2] != mask_bool.shape:
                    raise ValueError(
                        f"Point-map shape {pointmap_world.shape[:2]} does not match "
                        f"the {object_name!r} mask shape {mask_bool.shape}."
                    )

                observed_points = pointmap_world[mask_bool]
                observed_points = observed_points[
                    np.all(np.isfinite(observed_points), axis=1)
                ]
                if len(observed_points) < 3:
                    raise RuntimeError(
                        f"Not enough valid observed points to register {object_name!r} "
                        "with ICP."
                    )

                T_mesh_world, icp_fitness, registration_diagnostics = register_with_icp_fallback(
                    observed_points,
                    np.asarray(mesh.vertices),
                    init_T=T_cam2world,
                )
                reconstruction_results_dict[object_name]['registration_diagnostics'] = registration_diagnostics
                pose_last_opencv = None
                print(
                    f"ICP registration completed for {object_name}: "
                    f"selected {registration_diagnostics['method']}, "
                    f"median error {registration_diagnostics['selected_median_error_m'] * 1000:.2f} mm, "
                    f"fitness {icp_fitness:.3f}."
                )
                if registration_diagnostics['fallback_attempted']:
                    print(f"Registration candidates for {object_name}: {registration_diagnostics}")

            mesh_world = copy.deepcopy(mesh)     
            mesh_world = mesh_world.apply_transform(T_mesh_world)
            
            reconstruction_results_dict[object_name]['mesh'] = mesh
            reconstruction_results_dict[object_name]['mesh_world'] = mesh_world
            reconstruction_results_dict[object_name]['T_mesh_world'] = T_mesh_world
            reconstruction_results_dict[object_name]['pose_last_opencv'] = pose_last_opencv
        return reconstruction_results_dict    
    
    def multi_view_rendering(self, mesh_world):
        render_results = render_views(
            glb=mesh_world,
            save=True
        )
        return render_results
        
    def run_reconstruction_parallel(self, obj_name_list, max_workers=2):
        def reconstruction_job(object_name):
            glb_raw, glb_processed, obj_mask, scale = self.reconstruction_3d(object_name)
            return object_name, glb_raw, glb_processed, obj_mask, scale

        results = dict()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for obj_name in obj_name_list:
                futures.append(executor.submit(reconstruction_job, obj_name))

            for f in as_completed(futures):
                object_name, glb_raw, glb_processed, obj_mask, scale = f.result()
                if object_name not in results:
                    results[object_name] = {}
                results[object_name]["glb_raw"] = glb_raw
                results[object_name]["glb_processed"] = glb_processed
                results[object_name]["mask"] = obj_mask
                results[object_name]["scale"] = scale
        return results
    
    def reconstruction_3d(self, object_name):
        image = self.obs["front_color"]
        point_map_cam = copy.deepcopy(self.obs["front_pointmap_cam"])
        _, _, obj_mask = self.get_object_segmentation(image, object_name)
        
        if obj_mask is None:
            print(f"No {object_name} detected for 3d reconstruction.")
            return None, None, None, None
        
        R_gl_to_p3d = np.diag([-1, 1, -1])   # reflection matrix
        pointmap_pytorch3d =point_map_cam @ R_gl_to_p3d.T
        glb_raw, glb_processed, scale, _, _ = sam3d_request(
            image=image,
            mask=obj_mask,
            point_map=pointmap_pytorch3d,
        )
        
        return glb_raw, glb_processed, obj_mask, scale

    def run_affordance_parallel(self, obj_name, render_results, affordance_dic, max_workers=8):
        print(affordance_dic)
        def affordance_job(idx, item):
            image = item["color"]
            image_pil = Image.fromarray(image)
            aff_map = self.affordance_detector(obj_name, image_pil, affordance_dic)
            return idx, aff_map

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(affordance_job, i, render_results[i])
                for i in range(len(render_results))
            ]

            for f in as_completed(futures):
                idx, aff_map = f.result()
                render_results[idx]["affordance_map"] = aff_map

        return render_results
    
    def affordance_detector(self, obj_name, image, affordance_dic, use_qwen=True, visualize=False, save_path=None):
        """
        first perform object detection to crop the object from the image
        then perform affordance detection on the cropped image for each affordance in affordance_list
        """
        width, height = image.size
        affordance_feature_map = np.zeros((height, width, 3), dtype=np.uint8)
           
        for affordance_type in affordance_dic.keys():
            affordance_part_name = affordance_dic[affordance_type]
            if affordance_part_name == obj_name:
                print("no need for grounding")
                image_np = np.array(image)
                not_black = np.any(image_np != 0, axis=-1)
                not_white = np.any(image_np != 255, axis=-1)
                aff_mask = (not_black & not_white).astype(np.int8)
            else:
                aff_score, aff_box, aff_mask = self.sam3_seg(image, text=affordance_part_name)
                if aff_mask is None:
                    if use_qwen:
                        print("use_bbox from qwen3vl")
                        affordance_bbox_raw = query_bounding_box(image, affordance_part_name)
                        affordance_bbox = xyxy_to_xywh(affordance_bbox_raw)
                        aff_score, aff_box, aff_mask = self.sam3_seg(image, text=affordance_part_name,  bbox=affordance_bbox)    
                if aff_mask is None: 
                    query_name = affordance_part_name + " of " +   obj_name   
                    aff_score, aff_box, aff_mask = self.sam3_seg(image, text=query_name)
                
                if aff_mask is None:
                    print(f"No mask detected for {affordance_part_name}.")
                    continue
                if visualize:
                    save = True if save_path is not None else False
                    plot_best(image, aff_mask, aff_box, aff_score, save=save, save_path=save_path)
                
            affordance_feature_map = apply_affordance_color(affordance_feature_map, aff_mask, affordance_type)
    
        return affordance_feature_map
    
    def get_part_segmentation(self, image, obj_name, part_name, visualize=False, save_path=None):
        """
        first perform object detection to crop the object from the image
        then perform affordance detection on the cropped image for each affordance in affordance_list
        """
        width, height = image.size
        _, obj_bbox, obj_mask = self.get_object_segmentation(image, obj_name)
        if obj_mask is None:
            return np.zeros((height, width), dtype=np.uint8), image

        cropped_image, crop_bbox = crop_image_and_apply_mask(image, obj_bbox, obj_mask)
        
        part_score, part_bbox, part_mask = self.sam3_seg(cropped_image, text=part_name)
        if part_mask is None:
            part_score, part_bbox, part_mask = self.sam3_seg(cropped_image, text=f"{part_name} of the {obj_name}")
            # if part_mask is None:
            #     print("use part bbox from qwen3vl")
            #     part_bbox_raw = query_bounding_box(image, f"{part_name} of the {obj_name}")
            #     part_bbox = xyxy_to_xywh(part_bbox_raw)
            #     part_score, part_bbox, part_mask = self.sam3_seg(cropped_image, text=part_name, bbox=part_bbox)
            if part_mask is None:
                return obj_mask, cropped_image
                
        part_mask_full = project_mask_to_original(part_mask, crop_bbox, (width, height))
        save = True if save_path is not None else False
        plot_best(image, part_mask_full, None, part_score, save=save, save_path=save_path, show=visualize)
        
        return part_mask_full, cropped_image
    
    def sam3_seg(self, image, text=None, bbox=None):
        scores, bboxes, masks = sam3_request_concept(image, text, bbox, self.sam3_confidence_threshold)
        if len(scores) == 0:
            return None, None, None
        else:
            max_score_index = np.argmax(scores)
            best_score = scores[max_score_index]
            best_mask = masks[max_score_index]
            best_bbox = bboxes[max_score_index]
            return best_score, best_bbox, best_mask

    def get_object_segmentation(self, image, object_name):
        """Segment an object once per image/observation, with Qwen bbox fallback."""
        cache_key = (id(image), object_name.strip().lower())
        cache_entry = self._object_segmentation_cache.setdefault(cache_key, {})

        if cache_entry.get("mask") is not None:
            print(f"Using cached segmentation for {object_name}.")
            return (
                cache_entry["score"],
                cache_entry["bbox"],
                cache_entry["mask"],
            )

        score, bbox, mask = self.sam3_seg(image, text=object_name)
        if mask is None:
            if "qwen_bbox" not in cache_entry:
                print(f"SAM3 text segmentation failed for {object_name}; using Qwen bbox guidance.")
                cache_entry["qwen_bbox"] = query_bounding_box(image, object_name)
            else:
                print(f"Reusing cached Qwen bbox for {object_name}.")

            score, bbox, mask = self.sam3_seg(
                image,
                text=object_name,
                bbox=xyxy_to_xywh(cache_entry["qwen_bbox"]),
            )

        if mask is not None:
            cache_entry.update(score=score, bbox=bbox, mask=mask)

        return score, bbox, mask
    
    def parse_object_parts(self, image_list, object_list):
        object_parts = dict()
        for image in image_list:
            for obj in object_list:
                _, obj_bbox, obj_mask = self.get_object_segmentation(image, obj)

                if obj_mask is not None:
                    cropped_img, _ = crop_image_and_apply_mask(image, obj_bbox, obj_mask)
                else:
                    print(
                        f"No segmentation found for {obj}; skipping it for "
                        "part parsing and tool selection."
                    )
                    continue

                parts_response = query_model(
                    messages=[obj],
                    images = [cropped_img],
                    meta_prompt=self.prompts["parse_parts"]
                )

                parts_data = json.loads(parts_response)
                for k, v in parts_data.items():
                    v = set(v) if not isinstance(v, set) else v
                    object_parts.setdefault(k, set()).update(v)

        for k in object_parts.keys():
            if isinstance(object_parts[k], set):
                object_parts[k] = list(object_parts[k])

        return object_parts
    
    def grasping_pose_extraction(self, tool_object):
        self._extract_grasping_poses(tool_object)
        self._ensure_downward_grasps()

    def _ensure_downward_grasps(self):
        """Use the local top-down predictor when too few grasps can be tried."""
        downward_count = sum(
            pose[:3, 2].dot(np.array([0., 0., -1.])) >= DOWNWARD_GRASP_THRESHOLD
            for pose in self.pred_grasps_world
        )
        if downward_count < 10:
            print(
                f"Only {downward_count} of {len(self.pred_grasps_world)} grasp poses "
                "pass the downward-orientation filter (< 10); using top-down grasp prediction."
            )
            self.pred_grasps_world = convert_pcd_to_top_down_grasp(self.tool_aff_pcd)
            self.grasp_scores = [1.0] * len(self.pred_grasps_world)

    def _extract_grasping_poses(self, tool_object):
        rgb_image = self.obs["front_color"]
        depth_np = self.obs["front_depth"]
        intrinsic = self.obs["front_intrinsic"]
        extrinsic_opengl = self.obs["front_extrinsic"]

        _, _, obj_mask = self.get_object_segmentation(rgb_image, tool_object)
        grasp_part, pcd_grasp = convert_pcd_to_grasp_map(
            self.tool_aff_pcd,
            intrinsic,
            extrinsic_opengl,
            rgb_image.width,
            rgb_image.height,
        )
        grasp_part_binary = (grasp_part > 0).astype(np.uint8)

        use_obj_mask = False
        if np.sum(grasp_part_binary) > 0: 
            print("using object part for grasping extraction")
            self.grasp_image = Image.fromarray(grasp_part_binary) 
        else:
            print("no sufficient grasp point")
            if obj_mask is not None:
                grasp_map_binary = (obj_mask > 0).astype(np.uint8) 
                self.grasp_image = Image.fromarray(grasp_map_binary) 
                use_obj_mask = True
            else:
                raise RuntimeError("No grasp region or object mask is available.")

        initial_uses_obj_mask = use_obj_mask
        response = grasp_request(rgb_image, depth_np, self.grasp_image, intrinsic)
        pred_grasps_world, grasp_scores, _ = grasping_pose_transform(response, extrinsic_opengl)
        # Keep these candidates if the optional object-mask request fails.
        self.pred_grasps_world = pred_grasps_world
        self.grasp_scores = grasp_scores
        
        if not use_obj_mask:
            if len(pred_grasps_world) < 10: 
                print("no sufficient grasp pose")
                use_obj_mask = True
            else:
                self.pred_grasps_world = pred_grasps_world
                self.grasp_scores = grasp_scores
        
        if use_obj_mask:
            if obj_mask is not None:
                if not initial_uses_obj_mask:
                    previous_grasp_image = self.grasp_image
                    grasp_map_binary = (obj_mask > 0).astype(np.uint8)
                    print("try again with object mask")
                    self.grasp_image = Image.fromarray(grasp_map_binary)
                    try:
                        response = grasp_request(rgb_image, depth_np, self.grasp_image, intrinsic)
                        fallback_poses, fallback_scores, _ = grasping_pose_transform(response, extrinsic_opengl)
                    except requests.Timeout:
                        # The server may still be working; propagate so the batch
                        # stops before starting another service request.
                        raise
                    except requests.RequestException as error:
                        self.grasp_image = previous_grasp_image
                        if not pred_grasps_world:
                            raise
                        print(
                            f"Object-mask grasp request failed ({type(error).__name__}); "
                            f"keeping {len(pred_grasps_world)} candidates from the first request."
                        )
                        return
                    if not fallback_poses and pred_grasps_world:
                        self.grasp_image = previous_grasp_image
                        print("Object-mask request returned no grasps; keeping the first candidates.")
                        return
                    pred_grasps_world, grasp_scores = fallback_poses, fallback_scores

                if not pred_grasps_world:
                    self.pred_grasps_world = []
                    self.grasp_scores = []
                    return  # The local top-down predictor handles zero candidates too.

                if (len(pcd_grasp.points)<10):
                    self.pred_grasps_world = pred_grasps_world
                    self.grasp_scores = grasp_scores
                else:
                    print("using object part for grasping filtering")
                    obb = pcd_grasp.get_oriented_bounding_box()
                    # o3d.visualization.draw_geometries([pcd_grasp, obb])
                    # grasp_part_points = pointmap[grasp_part_binary>0]
                    # grasp_point_min = grasp_part_points.min(axis=0)
                    # grasp_point_max = grasp_part_points.max(axis=0)

                    local_points = np.array([[0, 0, 0.1]])
                    filter_grasp_pose = []
                    filter_grasp_score = []
                    for i, grasp_mat_world in enumerate(pred_grasps_world):
                        pts_h = np.hstack([local_points, np.ones((local_points.shape[0], 1))])
                        world_pts = (grasp_mat_world @ pts_h.T).T[:, :3]
                        world_pt = np.array(world_pts[0])
                        world_pt = world_pt.reshape(1, 3)

                        inside = obb.get_point_indices_within_bounding_box(
                            o3d.utility.Vector3dVector(world_pt)
                        )
                        is_inside = (len(inside) > 0)
                        if is_inside:
                            filter_grasp_pose.append(grasp_mat_world)
                            filter_grasp_score.append(grasp_scores[i])

                        # if np.all((world_pt >= grasp_point_min) & (world_pt <= grasp_point_max)):
                        #     filter_grasp_pose.append(grasp_mat_world)
                        #     filter_grasp_score.append(grasp_scores[i])
                    
                    if len(filter_grasp_pose) > 10:
                        print("object part filtered sucessful")
                        self.pred_grasps_world = filter_grasp_pose
                        self.grasp_scores = filter_grasp_score
                    else:
                        print("object part filtered failed")
                        self.pred_grasps_world = pred_grasps_world
                        self.grasp_scores = grasp_scores
