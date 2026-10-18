from __future__ import annotations

import cv2
import numpy as np
import trimesh
import open3d as o3d
from typing import List
import math
import random
import time
import copy
from scipy.spatial import cKDTree

from improvised_tooluse import config

seed = int(time.time() * 1000) & 0xFFFFFFFF
random.seed(seed)
np.random.seed(seed)

# ==================================================
# 3D conversion utilities
# ==================================================
def mesh_to_bytes(mesh: trimesh.Trimesh, file_type="ply") -> bytes:
    data = mesh.export(file_type=file_type)
    # trimesh may return str for some formats; ensure bytes
    if isinstance(data, str):
        data = data.encode("utf-8")
    return data

def mesh_from_bytes(b: bytes, file_type="ply") -> trimesh.Trimesh:
    import io
    return trimesh.load(io.BytesIO(b), file_type=file_type, force="mesh")

def trimesh_to_open3d_mesh(trimesh_mesh: trimesh.Trimesh):
    """Convert a Trimesh mesh to an Open3D mesh.

    Args:
        trimesh_mesh: the Trimesh mesh to convert.

    Returns:
        An Open3D mesh.
    """

    # Check if the input is a Trimesh mesh
    if not isinstance(trimesh_mesh, trimesh.Trimesh):
        raise TypeError("Input must be a Trimesh mesh")

    open3d_mesh = o3d.geometry.TriangleMesh()
    open3d_mesh.vertices = o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)

    # Check and transfer vertex normals, if they exist
    if trimesh_mesh.vertex_normals.size > 0:
        vertex_normals = np.array(trimesh_mesh.vertex_normals, copy=True)
        open3d_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)

    # Check and transfer vertex colors, if they exist
    # Trimesh stores colors in the 'visual' attribute
    if hasattr(trimesh_mesh.visual, 'vertex_colors') and trimesh_mesh.visual.vertex_colors.size > 0:
        # Open3D expects colors in the range [0, 1], but Trimesh uses [0, 255]
        vertex_colors_normalized = trimesh_mesh.visual.vertex_colors[:, :3] / 255.0
        open3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_normalized)

    return open3d_mesh

def o3d_to_numpy(pcd):
    """
    Convert Open3D PointCloud → single concatenated NumPy array.
    Columns: [x y z r g b nx ny nz]
    Missing colors / normals are filled with zeros.
    """

    if hasattr(pcd, "to_legacy"):
        pcd = pcd.to_legacy()

    points = np.asarray(pcd.points, dtype=np.float64)

    N = len(points)

    colors = (
        np.asarray(pcd.colors, dtype=np.float64)
        if pcd.has_colors()
        else np.zeros((N, 3), dtype=np.float64)
    )

    normals = (
        np.asarray(pcd.normals, dtype=np.float64)
        if pcd.has_normals()
        else np.zeros((N, 3), dtype=np.float64)
    )

    return np.concatenate([points, colors, normals], axis=1)

def numpy_to_o3d(arr):
    """
    Convert concatenated NumPy array → Open3D PointCloud.
    Expects columns: [x y z r g b nx ny nz]
    """

    pcd = o3d.geometry.PointCloud()

    points  = arr[:, 0:3]
    colors  = arr[:, 3:6]
    normals = arr[:, 6:9]

    pcd.points  = o3d.utility.Vector3dVector(points)

    if not np.allclose(colors, 0):
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if not np.allclose(normals, 0):
        pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd

def merge_meshes(meshes: List[trimesh.Trimesh]):
    n, vs, fs = 0, [], []
    for mesh in meshes:
        v, f = mesh.vertices, mesh.faces
        vs.append(v)
        fs.append(f + n)
        n = n + v.shape[0]
    if n:
        return trimesh.Trimesh(np.vstack(vs), np.vstack(fs))
    else:
        return None
    
def depth_to_pointmap(depth, K):
    """
    Convert depth image to point map (H, W, 3)

    Args:
        depth (np.ndarray): (H, W) depth image (float32 or float64)
                            depth in meters (recommended)
        K (np.ndarray): (3, 3) camera intrinsic matrix

    Returns:
        pointmap (np.ndarray): (H, W, 3) XYZ point map in camera coordinates
    """
    assert depth.ndim == 2
    assert K.shape == (3, 3)

    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Create pixel coordinate grid
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy

    pointmap = np.stack((X, Y, Z), axis=-1)  # (H, W, 3)
    return pointmap
    
# ==================================================
# Rotation utilities
# ==================================================

def find_orthogonal_vector(v: np.ndarray) -> np.ndarray:
    """Find an orthogonal vector to the given vector.

    The returned vector is guaranteed to be normalized.
    """
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)

    mask = np.abs(v[..., 0]) < 0.5
    return np.cross(v, np.array([1, 0, 0])) * mask + np.cross(v, np.array([0, 1, 0])) * (1 - mask)


def _quat_mul(q1, q2):
    """Hamilton product. Both are [..., 4] in [w, x, y, z]."""
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def _quat_from_axis_angle(axis, angle):
    """Create quaternion from axis [...,3] and angle [...] (radians)."""
    axis = np.asarray(axis)
    angle = np.asarray(angle)

    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-12)
    half = angle * 0.5
    w = np.cos(half)
    s = np.sin(half)
    xyz = axis * np.expand_dims(s, axis=-1)
    return np.concatenate([np.expand_dims(w, axis=-1), xyz], axis=-1)

def quaternion_from_vectors(vec1, vec2, random_choose=False, rng=None):
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    assert vec1.shape == vec2.shape, f'Vector shapes do not match: {vec1.shape} != {vec2.shape}'
    assert vec1.shape[-1] == 3
    assert vec2.shape[-1] == 3

    vec1 = vec1 / (np.linalg.norm(vec1, axis=-1, keepdims=True) + 1e-12)
    vec2 = vec2 / (np.linalg.norm(vec2, axis=-1, keepdims=True) + 1e-12)

    u = np.cross(vec1, vec2)
    s = np.sum(vec1 * vec2, axis=-1)

    eps = 1e-6
    u_norm = np.linalg.norm(u, axis=-1)

    same_dir = (u_norm < eps) & (s > 1 - eps)
    opp_dir  = (u_norm < eps) & (s < -1 + eps)

    if vec1.ndim == 1:
        if same_dir:
            q0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=vec1.dtype)
        else:
            if opp_dir:
                u = find_orthogonal_vector(vec1)  # 1D (3,)
                s = -1.0
            q0 = np.concatenate([np.array([s + 1.0], dtype=vec1.dtype), u], axis=-1)  # (4,)
            q0 = q0 / (np.linalg.norm(q0) + 1e-12)

        if not random_choose:
            return q0

        if rng is None:
            rng = np.random.default_rng()
        theta = rng.uniform(0.0, 2.0 * np.pi)

        q_twist = _quat_from_axis_angle(vec2, theta)   # (4,)
        q = _quat_mul(q_twist, q0)                     # (4,)
        return q / (np.linalg.norm(q) + 1e-12)
    else:
        q0 = np.concatenate([np.expand_dims(s + 1.0, axis=-1), u], axis=-1)
        q0 = q0 / (np.linalg.norm(q0, axis=-1, keepdims=True) + 1e-12)
        if np.any(same_dir):
            q0 = np.where(
                np.expand_dims(same_dir, -1),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=q0.dtype),
                q0
            )

    if not random_choose:
        return q0

    # Randomly twist around vec2, which keeps vec2 fixed, so q * vec1 is still vec2.
    if rng is None:
        rng = np.random.default_rng()

    if vec1.ndim == 1:
        theta = rng.uniform(0.0, 2.0 * np.pi)
        q_twist = _quat_from_axis_angle(vec2, theta)
        q = _quat_mul(q_twist, q0)
        return q / (np.linalg.norm(q) + 1e-12)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=s.shape)  # shape [...]
    q_twist = _quat_from_axis_angle(vec2, theta)          # shape [...,4]
    q = _quat_mul(q_twist, q0)
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)


# ==================================================
# Point cloud registration utilities
# ==================================================
def estimate_normals(pcd, voxel):
    # Normal radius should scale with voxel size
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(50)

def multiscale_icp_point2plane(source, target, init_T=np.eye(4), base_voxel=0.005):
    T = init_T.copy()

    for s in [8, 4, 2, 1, 0.5]:
        v = base_voxel * s
        src = source.voxel_down_sample(v)
        tgt = target.voxel_down_sample(v)
        
        if s <= 2:
            estimate_normals(src, v)
            estimate_normals(tgt, v)
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
            

        threshold = v * 3.0  # typical: 2~4 * voxel
        result = o3d.pipelines.registration.registration_icp(
            src, tgt, threshold, T,
            estimation_method,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        T = result.transformation
    return T, result.fitness

def registration_with_scale(original_pcd_np: np.ndarray, reconstructed_pcd: np.ndarray, max_trail=5, visualize:bool=False, init_T=None):
    """
    The returned transformation T is from reconstructed_pcd to original_pcd, a 4x4 numpy array.
    init_T, if supplied, maps reconstructed_pcd to original_pcd and preserves
    the predicted alignment instead of initializing by centroids. No scaling
    is estimated or applied by this function.
    """
    original_pcd_o3d = o3d.geometry.PointCloud()
    original_pcd_o3d.points = o3d.utility.Vector3dVector(original_pcd_np)
    reconstructed_pcd_o3d = o3d.geometry.PointCloud()
    reconstructed_pcd_o3d.points = o3d.utility.Vector3dVector(reconstructed_pcd)

    # perform dbscan and keep largest cluster for original pcd
    labels = np.array(original_pcd_o3d.cluster_dbscan(eps=0.01, min_points=10))
    if labels.max() < 0:
        print("No clusters found in point cloud.")
    else:
        largest_cluster_label = np.bincount(labels[labels >= 0]).argmax()
        indices = np.where(labels == largest_cluster_label)[0]
        original_pcd_o3d = original_pcd_o3d.select_by_index(indices)
    
    if init_T is None:
        s_cent = original_pcd_o3d.get_center()
        t_cent = reconstructed_pcd_o3d.get_center()
        init_translation = t_cent - s_cent
        init_T = np.eye(4)
        init_T[:3, 3] = init_translation
    else:
        # Internally ICP maps observed points to the reconstructed mesh;
        # the public API accepts and returns the opposite direction.
        init_T = np.linalg.inv(np.asarray(init_T, dtype=float))
    
    best_T = init_T
    best_fitness = 0
    for _ in range(max_trail):
        T, fitness = multiscale_icp_point2plane(original_pcd_o3d, reconstructed_pcd_o3d, init_T=init_T)
        if fitness > best_fitness:
            best_T = T
            best_fitness = fitness
        
        if best_fitness>0.9:
            break
    
    if visualize:
        output_pcd_trimesh = trimesh.points.PointCloud(
            np.asarray(reconstructed_pcd_o3d.points), 
            colors=[[0, 0, 255] for _ in range(len(reconstructed_pcd_o3d.points))]
        )
        partial_pcd_trimesh = trimesh.points.PointCloud(
            np.asarray(original_pcd_o3d.points), 
            colors=[[255, 0, 0] for _ in range(len(original_pcd_o3d.points))]
        )
        output_pcd_transformed_trimesh = output_pcd_trimesh.copy()
        output_pcd_transformed_trimesh.apply_transform(np.linalg.inv(best_T))
        scene = trimesh.Scene()
        scene.add_geometry(output_pcd_transformed_trimesh, node_name='source_transformed')
        scene.add_geometry(partial_pcd_trimesh, node_name='target')
        scene.show()
    
    return np.linalg.inv(best_T), best_fitness


def global_init_fpfh_ransac(original_pcd_np: np.ndarray, reconstructed_pcd: np.ndarray):
    voxel = 0.01  # meters (1 cm)
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(original_pcd_np)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(reconstructed_pcd)

    src_down = src.voxel_down_sample(voxel)
    tgt_down = tgt.voxel_down_sample(voxel)

    src_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    tgt_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )

    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down,
        src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=0.015,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.015),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            80000, 0.999
        )
    )

    return result.transformation, result.fitness

def register_with_fpfh_then_icp(original_pcd_np,
                                reconstructed_pcd_np,
                                visualize=False):
    """
    Returns:
        T_final : reconstructed -> original
    """

    # 1. Global init (original -> reconstructed)
    T_global, fit_ransac = global_init_fpfh_ransac(
        original_pcd_np,
        reconstructed_pcd_np
    )

    # 2. Pull reconstructed back using inverse (IMPORTANT)
    Tg_inv = np.linalg.inv(T_global)
    reconstructed_prealigned = (
        Tg_inv[:3, :3] @ reconstructed_pcd_np.T
    ).T + Tg_inv[:3, 3]

    # 3. ICP refinement (unchanged logic)
    T_refine, fit_icp = registration_with_scale(
        original_pcd_np,
        reconstructed_prealigned,
        visualize=visualize
    )

    # 4. Compose transforms
    T_final = T_refine @ Tg_inv

    return T_final, fit_icp, fit_ransac


def register_with_icp_fallback(original_pcd_np, reconstructed_pcd_np, init_T,
                               fallback_threshold=0.005):
    """Refine the predicted pose, trying FPFH only when local alignment is poor.

    All transforms map reconstructed points to observed points. Error is the
    median observed-point-to-nearest-mesh-vertex distance in metres, evaluated
    on the same full observation for both candidates. Try FPFH above 5 mm
    (by default), or when local ICP has no correspondences. Keep the primary
    pose on ties or if the fallback fails. Fitness is diagnostic, not the
    candidate-selection score; it can be zero when ICP returns an initial pose.
    """
    observed = np.asarray(original_pcd_np, dtype=float)
    reconstructed = np.asarray(reconstructed_pcd_np, dtype=float)
    init_T = np.asarray(init_T, dtype=float)

    def alignment_error(transform):
        if not np.all(np.isfinite(transform)):
            return float('inf')
        aligned = reconstructed @ transform[:3, :3].T + transform[:3, 3]
        distances = cKDTree(aligned).query(observed)[0]
        return float(np.median(distances))

    primary_T, primary_fitness = registration_with_scale(
        observed, reconstructed, init_T=init_T,
    )
    primary_error = alignment_error(primary_T)
    best_T, best_fitness = primary_T, primary_fitness
    diagnostics = {
        'method': 'icp',
        'primary_median_error_m': primary_error,
        'primary_fitness': float(primary_fitness),
        'selected_median_error_m': primary_error,
        'fallback_attempted': False,
    }
    if primary_error > fallback_threshold or primary_fitness <= 0:
        diagnostics['fallback_attempted'] = True
        # glb_processed is already scaled and posed in the camera frame.
        # Move it to the observed frame once; FPFH returns a correction there.
        prealigned = reconstructed @ init_T[:3, :3].T + init_T[:3, 3]
        try:
            correction, fallback_fitness, ransac_fitness = register_with_fpfh_then_icp(
                observed, prealigned,
            )
            fallback_T = correction @ init_T
            fallback_error = alignment_error(fallback_T)
            diagnostics.update(
                fallback_median_error_m=fallback_error,
                fallback_fitness=float(fallback_fitness),
                ransac_fitness=float(ransac_fitness),
            )
            if fallback_error < primary_error:
                best_T, best_fitness = fallback_T, fallback_fitness
                diagnostics.update(method='fpfh_icp', selected_median_error_m=fallback_error)
        except (RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            diagnostics['fallback_error'] = f'{type(error).__name__}: {error}'

    if not np.isfinite(diagnostics['selected_median_error_m']):
        raise RuntimeError('Neither ICP registration candidate has a finite alignment error')
    return best_T, best_fitness, diagnostics


# ==================================================
# Rendering utilities
# ==================================================
def uniform_views_on_sphere(radius, elev_deg_top=45, elev_deg_bottom=-45, view_num=None):
    """`view_num` viewpoints split evenly over a top and a bottom elevation ring.

    The azimuths follow from view_num: each ring gets view_num // 2 of them,
    evenly spaced and offset by half a step, so view_num=8 keeps the original
    45, 135, 225, 315 on each ring rather than sitting on the world axes.
    view_num defaults to `rendering.view_num` in config.yaml.
    """
    if view_num is None:
        view_num = config.get("rendering", "view_num")
    assert view_num % 2 == 0, f"view_num must be even to split over two rings, got {view_num}"

    poses = []

    elev_top = math.radians(elev_deg_top)
    elev_bottom = math.radians(elev_deg_bottom)

    per_ring = view_num // 2
    step = 2 * math.pi / per_ring
    azimuths = [i * step + step / 2 for i in range(per_ring)]

    def sph_to_cart(az, el, r):
        x = r * math.cos(az) * math.cos(el)
        y = r * math.sin(az) * math.cos(el)
        z = r * math.sin(el)
        return np.array([x, y, z], dtype=float)

    # top ring
    for az in azimuths:
        poses.append(sph_to_cart(az, elev_top, radius))

    # bottom ring
    for az in azimuths:
        poses.append(sph_to_cart(az, elev_bottom, radius))

    return poses

def random_point_on_sphere(radius, elev_min, elev_max):
    elevation = random.uniform(elev_min, elev_max)
    azimuth = random.uniform(0, 2 * math.pi)

    x = radius * math.cos(azimuth) * math.cos(elevation)
    y = radius * math.sin(azimuth) * math.cos(elevation)
    z = radius * math.sin(elevation)

    return np.array([x, y, z])

def look_at_c2w(camera_position, target, up=np.array([0.,0.,1.])):
    """Camera-to-World pose matrix (C2W)."""
    camera_position = np.asarray(camera_position, float)
    target = np.asarray(target, float)

    forward = (target - camera_position)
    forward /= np.linalg.norm(forward)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    real_up = np.cross(right, forward)

    T_c2w = np.eye(4, dtype=float)
    T_c2w[:3, 0] = -right
    T_c2w[:3, 1] = -real_up
    T_c2w[:3, 2] = forward
    T_c2w[:3, 3] = camera_position
    return T_c2w

def render_views(
    glb: trimesh.Trimesh,
    density=None,
    elevation_min_deg=-60,
    elevation_max_deg=60,
    img_size=(512, 512),
    camera_distribution=None,
    sphere_angle=None,
    save=False
):
    """density/camera_distribution/sphere_angle default to `rendering` in config.yaml."""
    if density is None:
        density = config.get("rendering", "view_num")
    if camera_distribution is None:
        camera_distribution = config.get("rendering", "camera_distribution")
    if sphere_angle is None:
        sphere_angle = config.get("rendering", "sphere_angle")
    assert camera_distribution in ["sphere", "random"]
    # ---- build Open3D mesh + bbox info ----
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(glb.vertices),
        o3d.utility.Vector3iVector(glb.faces)
    )

    if glb.visual.vertex_colors is not None:
        colors = np.asarray(glb.visual.vertex_colors[:, :3]) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()

    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())
    extent = np.asarray(bbox.get_extent())
    obj_radius = np.linalg.norm(extent) / 2.0

    ratio = 2 # for zoom in
    k = 2.5
    radius = obj_radius * k
    radius = radius * ratio

    # ---- intrinsics ----
    fx = img_size[0] * ratio
    fy = img_size[1] * ratio
    intr = o3d.camera.PinholeCameraIntrinsic(
        img_size[0], img_size[1],
        fx=fx, fy=fy,
        cx=img_size[0] // 2, cy=img_size[1] // 2,
    )
    K = intr.intrinsic_matrix.copy()

    elev_min = math.radians(elevation_min_deg)
    elev_max = math.radians(elevation_max_deg)

    # ---- sample camera poses (C2W) ----
    c2w_list = []

    if camera_distribution == 'sphere':
        points_set = uniform_views_on_sphere(radius, sphere_angle, -sphere_angle, density)

    for i in range(density):
        if camera_distribution == 'random':
            offset = random_point_on_sphere(radius, elev_min, elev_max)
        else:
            offset = points_set[i]

        cam_pos = offset + center
        c2w = look_at_c2w(cam_pos, target=center)
        c2w_list.append(c2w)

    # ---- call multi-view renderer once ----
    render_results = camera_render(
        glb=glb,
        c2w_list=c2w_list,
        K=K,
        img_size=img_size,
        save=save
    )

    return render_results


def depth_to_world_points(depth: np.ndarray, K: np.ndarray, c2w: np.ndarray):
    """
    depth: (H, W) float
    K: (3,3)
    c2w: (4,4)
    return: (N,3) world points
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Pixel grid (u: x axis, v: y axis)
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # (H,W)

    z = depth.astype(np.float64)

    # Valid depth mask 
    valid = np.isfinite(z) & (z > 0)

    uu = uu[valid]
    vv = vv[valid]
    z = z[valid]

    # Unproject to camera frame
    x = (uu - cx) / fx * z
    y = (vv - cy) / fy * z

    pts_cam = np.stack([x, y, z], axis=1)  # (N,3)

    # Transform to world frame
    pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float64)], axis=1)  # (N,4)
    pts_world_h = (c2w @ pts_cam_h.T).T  # (N,4)
    pts_world = pts_world_h[:, :3]

    return pts_cam, pts_world, valid


def camera_render(
    glb: trimesh.Trimesh,
    c2w_list,
    K: np.ndarray,
    img_size=(512, 512),
    save=False
):
    """
    Render multiple views given a list of camera-to-world matrices.

    Args:
        glb: trimesh.Trimesh mesh object
        c2w_list: list of (4,4) camera-to-world matrices
        K: (3,3) intrinsic matrix
        img_size: (W,H)
        save: if True, saves each rendered image

    Returns:
        List of dicts, each with:
        {intrinsic, extrinsic_c2w, extrinsic_w2c, color, depth}
    """

    # ---- build Open3D mesh once ----
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(glb.vertices),
        o3d.utility.Vector3iVector(glb.faces)
    )

    if glb.visual.vertex_colors is not None:
        colors = np.asarray(glb.visual.vertex_colors[:, :3]) / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    mesh.compute_vertex_normals()

    # ---- renderer + scene ----
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        img_size[0], img_size[1]
    )
    scene = renderer.scene
    scene.add_geometry("mesh", mesh, o3d.visualization.rendering.MaterialRecord())

    # ---- intrinsics object ----
    intr = o3d.camera.PinholeCameraIntrinsic(
        img_size[0], img_size[1],
        fx=K[0, 0], fy=K[1, 1],
        cx=K[0, 2], cy=K[1, 2]
    )

    results = []

    # ---- loop over camera poses ----
    for i, c2w in enumerate(c2w_list):
        c2w = np.asarray(c2w, dtype=np.float64)
        w2c = np.linalg.inv(c2w)

        renderer.setup_camera(intr, w2c)

        color_o3d = renderer.render_to_image()
        depth_o3d = renderer.render_to_depth_image(z_in_view_space=True)

        if save:
            o3d.io.write_image(f"view_{i:03d}.png", color_o3d)
        
        pcd_cams, pcd_world, _, = depth_to_world_points(np.asarray(depth_o3d), K, c2w)

        results.append({
            "intrinsic": K,
            "extrinsic_c2w": c2w,
            "extrinsic_w2c": w2c,
            "color": np.asarray(color_o3d),
            "depth": np.asarray(depth_o3d),
            "pcd_world": pcd_world,
            "pcd_cams": pcd_cams
        })

    return results

###############################################
# 3D FUSION utilities
###############################################
def filter_outliers(
    pcd_input,
    voxel_size=None,
    dbscan_eps=None,
    min_points=3,
    min_cluster_ratio=0.02,
):
    """Remove sparse points and very small connected components.

    The returned cloud stays at ``voxel_size`` resolution.  In particular, this
    does not expand retained clusters back into the input with a radius search;
    doing so can re-introduce the nearby outliers that clustering removed.
    Multiple components are retained when they are at least
    ``min_cluster_ratio`` as large as the biggest component.
    """
    if pcd_input is None or len(pcd_input.points) == 0:
        return copy.deepcopy(pcd_input)
    if voxel_size is None:
        voxel_size = float(config.get("affordance_fusion", "voxel_size"))
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    if not 0.0 <= min_cluster_ratio <= 1.0:
        raise ValueError("min_cluster_ratio must be in [0, 1]")

    pcd = copy.deepcopy(pcd_input)
    pts = np.asarray(pcd.points)

    # Affordance clouds use black/white as background sentinels.  Preserve
    # uncoloured clouds so this helper remains useful for generic geometry.
    if pcd.has_colors():
        cols = np.asarray(pcd.colors)
        if len(cols) and cols.max() > 1.0:
            cols = cols / 255.0
        keep = ~(
            np.all(np.isclose(cols, [0, 0, 0]), axis=1)
            | np.all(np.isclose(cols, [1, 1, 1]), axis=1)
        )
        pts = pts[keep]
        cols = cols[keep]
    else:
        cols = None

    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(pts)
    if cols is not None:
        pcd_filtered.colors = o3d.utility.Vector3dVector(cols)
    if len(pcd_filtered.points) == 0:
        return pcd_filtered

    pcd_down = pcd_filtered.voxel_down_sample(float(voxel_size))
    eps = float(dbscan_eps) if dbscan_eps is not None else float(voxel_size) * 2.0
    labels = np.asarray(
        pcd_down.cluster_dbscan(eps=eps, min_points=int(min_points)),
        dtype=np.int64,
    )

    clustered = labels >= 0
    if not np.any(clustered):
        # Avoid deleting a valid but very small/thin affordance merely because
        # DBSCAN could not form a component.
        return pcd_down

    counts = np.bincount(labels[clustered])
    minimum_size = max(int(min_points), int(math.ceil(counts.max() * min_cluster_ratio)))
    retained_labels = np.flatnonzero(counts >= minimum_size)
    keep = clustered & np.isin(labels, retained_labels)
    return pcd_down.select_by_index(np.flatnonzero(keep).tolist())

def crop_pointcloud(pcd, x_range=None, y_range=None, z_range=None, save_path=None, filter=False):
    """
    Crop an Open3D point cloud using axis-aligned constraints in world frame.
    
    Args:
        pcd (o3d.geometry.PointCloud): input point cloud
        x_range (tuple): (xmin, xmax) or None
        y_range (tuple): (ymin, ymax) or None
        z_range (tuple): (zmin, zmax) or None
        
    Returns:
        o3d.geometry.PointCloud: cropped point cloud
    """
    pts = np.asarray(pcd.points)
    # print(np.min(pts[:, 0], axis=0), np.max(pts[:, 0], axis=0))
    # print(np.min(pts[:, 1], axis=0), np.max(pts[:, 1], axis=0))
    # print(np.min(pts[:, 2], axis=0), np.max(pts[:, 2], axis=0))

    mask = np.ones(len(pts), dtype=bool)

    if x_range is not None:
        mask &= (pts[:, 0] >= x_range[0]) & (pts[:, 0] <= x_range[1])
    if y_range is not None:
        mask &= (pts[:, 1] >= y_range[0]) & (pts[:, 1] <= y_range[1])
    if z_range is not None:
        mask &= (pts[:, 2] >= z_range[0]) & (pts[:, 2] <= z_range[1])

    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(pts[mask])

    # Preserve colors if present
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        cropped_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    # Preserve normals if present
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        cropped_pcd.normals = o3d.utility.Vector3dVector(normals[mask])
    
    if filter:
        cropped_pcd = filter_outliers(
            cropped_pcd,
            dbscan_eps=0.03,
            min_points=5,
        )
    
    if save_path is not None:
        o3d.io.write_point_cloud(save_path, cropped_pcd)
        # print("Cropped points:", len(cropped_pcd.points))
        # print(f"Saved cropped point cloud to {save_path}")

    return cropped_pcd

def _prepare_affordance_map(
    affordance_map,
    valid_depth,
    erosion_pixels=1,
    max_erosion_loss_ratio=0.2,
):
    """Gate an affordance map by rendered geometry and conservatively erode it."""
    if erosion_pixels < 0:
        raise ValueError("erosion_pixels must be non-negative")
    if not 0.0 <= max_erosion_loss_ratio <= 1.0:
        raise ValueError("max_erosion_loss_ratio must be in [0, 1]")

    affordance_map = np.asarray(affordance_map)
    if affordance_map.ndim != 3 or affordance_map.shape[2] != 3:
        raise ValueError(
            f"affordance_map must have shape (H, W, 3), got {affordance_map.shape}"
        )
    if valid_depth.shape != affordance_map.shape[:2]:
        raise ValueError(
            "depth and affordance map dimensions differ: "
            f"{valid_depth.shape} vs {affordance_map.shape[:2]}"
        )

    prepared = np.zeros_like(affordance_map, dtype=np.uint8)
    erosion_pixels = int(erosion_pixels)
    if erosion_pixels > 0:
        width = 2 * erosion_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))

    for channel in range(3):
        mask = (affordance_map[..., channel] > 0) & valid_depth
        if not mask.any():
            continue

        selected = mask
        if erosion_pixels > 0:
            # Decide per component so a large region cannot hide the loss of a
            # separate thin handle or rim.
            component_count, labels = cv2.connectedComponents(mask.astype(np.uint8))
            selected = np.zeros_like(mask)
            for label in range(1, component_count):
                component = labels == label
                component_area = int(component.sum())
                eroded = cv2.erode(component.astype(np.uint8), kernel, iterations=1) > 0
                retained_fraction = float(eroded.sum()) / component_area
                if eroded.any() and retained_fraction >= 1.0 - max_erosion_loss_ratio:
                    selected |= eroded
                else:
                    selected |= component

        prepared[..., channel] = selected.astype(np.uint8) * 255

    return prepared


def _fuse_affordance_votes(
    render_results,
    voxel_size,
    strong_min_positive_views,
    strong_min_positive_ratio,
    weak_min_positive_views,
    weak_min_positive_ratio,
    weak_neighbor_voxels,
    mask_erosion_pixels,
    max_erosion_loss_ratio,
):
    """Fuse per-view masks by distinct-view votes on a shared voxel grid."""
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be positive, got {voxel_size}")
    if weak_min_positive_views < 1:
        raise ValueError("weak_min_positive_views must be at least 1")
    if strong_min_positive_views < weak_min_positive_views:
        raise ValueError("strong_min_positive_views must be >= weak_min_positive_views")
    if not 0.0 <= weak_min_positive_ratio <= 1.0:
        raise ValueError("weak_min_positive_ratio must be in [0, 1]")
    if not weak_min_positive_ratio <= strong_min_positive_ratio <= 1.0:
        raise ValueError(
            "strong_min_positive_ratio must be in [weak_min_positive_ratio, 1]"
        )
    if weak_neighbor_voxels < 1:
        raise ValueError("weak_neighbor_voxels must be at least 1")

    # key -> [number of visible views, sum of per-view mean positions,
    #         positive view counts for R/G/B affordances]
    voxel_stats = {}

    for view in render_results:
        affordance_map = view.get("affordance_map")
        if affordance_map is None:
            # A view without a segmentation result provides no negative
            # evidence and therefore must not increase the visibility count.
            continue
        depth = np.asarray(view["depth"])
        valid_depth = np.isfinite(depth) & (depth > 0)
        _, points_world, valid = depth_to_world_points(
            depth,
            view["intrinsic"],
            view["extrinsic_c2w"],
        )
        if len(points_world) == 0:
            continue

        voxel_keys = np.floor(points_world / float(voxel_size)).astype(np.int64)
        unique_keys, inverse = np.unique(voxel_keys, axis=0, return_inverse=True)

        # Use the actual surface samples for output coordinates. Averaging once
        # per view prevents a grazing view with many pixels from dominating.
        point_sums = np.zeros((len(unique_keys), 3), dtype=np.float64)
        np.add.at(point_sums, inverse, points_world)
        point_counts = np.bincount(inverse, minlength=len(unique_keys))
        view_means = point_sums / point_counts[:, None]

        for key_array, mean_point in zip(unique_keys, view_means):
            key = tuple(key_array.tolist())
            stat = voxel_stats.get(key)
            if stat is None:
                stat = [0, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.int32)]
                voxel_stats[key] = stat
            stat[0] += 1
            stat[1] += mean_point

        prepared = _prepare_affordance_map(
            affordance_map,
            valid_depth,
            erosion_pixels=mask_erosion_pixels,
            max_erosion_loss_ratio=max_erosion_loss_ratio,
        )
        positive_at_points = prepared[valid]

        for channel in range(3):
            positive = positive_at_points[:, channel] > 0
            if not np.any(positive):
                continue
            # np.unique makes this a distinct-view vote rather than a pixel or
            # point-density vote.
            positive_keys = np.unique(voxel_keys[positive], axis=0)
            for key_array in positive_keys:
                voxel_stats[tuple(key_array.tolist())][2][channel] += 1

    strong_keys = [set(), set(), set()]
    weak_keys = [set(), set(), set()]
    for key, (visible_views, _, positive_views) in voxel_stats.items():
        positive_ratio = positive_views.astype(np.float64) / visible_views
        strong = (
            (positive_views >= int(strong_min_positive_views))
            & (positive_ratio >= float(strong_min_positive_ratio))
        )
        weak = (
            (positive_views >= int(weak_min_positive_views))
            & (positive_ratio >= float(weak_min_positive_ratio))
        )
        for channel in range(3):
            if strong[channel]:
                strong_keys[channel].add(key)
            elif weak[channel]:
                weak_keys[channel].add(key)

    # Weak voxels are admitted only one hop from a strong core. This is not a
    # recursive region grow, so marginal support cannot bridge into far noise.
    accepted_keys = [keys.copy() for keys in strong_keys]
    neighbor_range = range(-int(weak_neighbor_voxels), int(weak_neighbor_voxels) + 1)
    neighbor_offsets = [
        (dx, dy, dz)
        for dx in neighbor_range
        for dy in neighbor_range
        for dz in neighbor_range
        if (dx, dy, dz) != (0, 0, 0)
    ]
    for channel in range(3):
        strong_channel = strong_keys[channel]
        for key in weak_keys[channel]:
            if any(
                (key[0] + dx, key[1] + dy, key[2] + dz) in strong_channel
                for dx, dy, dz in neighbor_offsets
            ):
                accepted_keys[channel].add(key)

    fused_points = []
    fused_colors = []
    for key, (visible_views, point_sum, _) in voxel_stats.items():
        accepted = np.array(
            [key in accepted_keys[channel] for channel in range(3)],
            dtype=bool,
        )
        if np.any(accepted):
            fused_points.append(point_sum / visible_views)
            fused_colors.append(accepted.astype(np.float64))

    pcd_aff = o3d.geometry.PointCloud()
    if fused_points:
        pcd_aff.points = o3d.utility.Vector3dVector(np.asarray(fused_points))
        pcd_aff.colors = o3d.utility.Vector3dVector(np.asarray(fused_colors))
    return pcd_aff


def merge_views(
    render_results,
    affordance_voxel_size=None,
    strong_min_positive_views=3,
    strong_min_positive_ratio=0.6,
    weak_min_positive_views=2,
    weak_min_positive_ratio=0.5,
    weak_neighbor_voxels=1,
    mask_erosion_pixels=1,
    max_erosion_loss_ratio=0.2,
):
    """
    Merge RGB samples directly and affordances through distinct-view voxel voting.
    """
    global_pts, global_cols = None, None
    has_affordance_maps = False

    for view in render_results:
        o3d_color = o3d.geometry.Image(view["color"])
        o3d_depth = o3d.geometry.Image(view["depth"])
        intrinsic_matrix = view["intrinsic"]
        extrinsic_matrix = view["extrinsic_c2w"]

        # merge RGB points
        global_pts, global_cols = integrate_view(
            global_pts,
            global_cols,
            o3d_color,
            o3d_depth,
            intrinsic_matrix,
            extrinsic_matrix,
        )

        has_affordance_maps |= "affordance_map" in view

    # build Open3D clouds
    pcd = o3d.geometry.PointCloud()
    if global_pts is not None:
        pcd.points = o3d.utility.Vector3dVector(global_pts)
        pcd.colors = o3d.utility.Vector3dVector(global_cols)

    pcd_aff = None
    if has_affordance_maps:
        if affordance_voxel_size is None:
            affordance_voxel_size = config.get("affordance_fusion", "voxel_size")
        pcd_aff = _fuse_affordance_votes(
            render_results,
            voxel_size=float(affordance_voxel_size),
            strong_min_positive_views=int(strong_min_positive_views),
            strong_min_positive_ratio=float(strong_min_positive_ratio),
            weak_min_positive_views=int(weak_min_positive_views),
            weak_min_positive_ratio=float(weak_min_positive_ratio),
            weak_neighbor_voxels=int(weak_neighbor_voxels),
            mask_erosion_pixels=int(mask_erosion_pixels),
            max_erosion_loss_ratio=float(max_erosion_loss_ratio),
        )

    return pcd, pcd_aff

def integrate_view(global_points,
                   global_colors,
                   o3d_color,
                   o3d_depth,
                   intrinsic_matrix,
                   extrinsic_matrix,
                   ):

    # --- Convert Open3D images to numpy ---
    depth = np.asarray(o3d_depth)
    color = np.asarray(o3d_color, dtype=np.float32) / 255.0 
    # Camera-to-world pose
    c2w = extrinsic_matrix

    # --- Back-project depth to world points ---
    _, pts_world, valid = depth_to_world_points(depth, intrinsic_matrix, c2w)

    color_valid = color[valid]

    colors_flat = color_valid.reshape(-1, 3)
    pts_flat = pts_world.reshape(-1, 3)

    if global_points is None:
        global_points = pts_world
        global_colors = colors_flat
    else:
        global_points = np.vstack([global_points, pts_flat])
        global_colors = np.vstack([global_colors, colors_flat])

    return global_points, global_colors


def blend_to_affcolor_by_density(
    pcd,
    aff_rgb,
    back_rgb,
    k=20,
    log_scale=True,
    clip_percentile=(2, 98),
    gamma=1.0,        # >1 makes only very-dense points turn colored
    w_min=0.0, w_max=1.0
):
    """
    Density-based blending: low density -> back_rgb, high density -> aff_rgb.
    """
    if pcd is None or len(pcd.points) == 0:
        return pcd

    aff_rgb = np.asarray(aff_rgb, dtype=np.float64).reshape(1, 3)
    back_rgb = np.asarray(back_rgb, dtype=np.float64).reshape(1, 3)

    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < 4:
        # too few points, just use aff color
        pcd.colors = o3d.utility.Vector3dVector(np.tile(aff_rgb, (n, 1)))
        return pcd

    k = int(min(max(k, 3), n - 1))
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    inv_mean_dist = np.zeros(n, dtype=np.float64)
    for i in range(n):
        _, idx, dist2 = kdtree.search_knn_vector_3d(pts[i], k)
        dist2 = np.asarray(dist2, dtype=np.float64)
        if dist2.size <= 1:
            inv_mean_dist[i] = 0.0
            continue
        d = np.sqrt(dist2[1:])  # skip self
        inv_mean_dist[i] = 1.0 / (np.mean(d) + 1e-12)

    v = inv_mean_dist
    if log_scale:
        v = np.log1p(v)

    lo, hi = np.percentile(v, clip_percentile)
    v = np.clip(v, lo, hi)
    w = (v - lo) / (hi - lo + 1e-12)  # [0,1]

    # shape the curve
    w = np.clip(w, 0.0, 1.0)
    if gamma != 1.0:
        w = w ** float(gamma)
    w = w_min + (w_max - w_min) * w

    colors = (1.0 - w[:, None]) * back_rgb + (w[:, None]) * aff_rgb
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def downsample_pcd(pcd, voxel=0.006, max_points=None):
    """Voxel downsample + optional cap to max_points (uniform)."""
    if pcd is None or len(pcd.points) == 0:
        return pcd
    if voxel is not None and voxel > 0:
        pcd = pcd.voxel_down_sample(float(voxel))
    if max_points is not None and len(pcd.points) > max_points:
        idx = np.random.choice(len(pcd.points), size=max_points, replace=False)
        pcd = pcd.select_by_index(idx.tolist())
    return pcd

def remove_base_near_affordance(pcd_base, aff_pcds, r=0.005):
    """Remove points in pcd_base that are within radius r of any affordance point."""
    # merge affordance point clouds
    aff_all = o3d.geometry.PointCloud()
    for p in aff_pcds:
        if p is not None and len(p.points) > 0:
            aff_all += p
    if len(aff_all.points) == 0 or len(pcd_base.points) == 0:
        return pcd_base

    tree = o3d.geometry.KDTreeFlann(aff_all)
    base_pts = np.asarray(pcd_base.points)

    keep = []
    for i, pt in enumerate(base_pts):
        # find nearest affordance point distance
        _, idx, dist2 = tree.search_knn_vector_3d(pt, 1)
        if len(dist2) == 0:
            keep.append(i)
        else:
            if dist2[0] > (r * r):   # keep if farther than r
                keep.append(i)

    return pcd_base.select_by_index(keep)

def _select_by_mask(pcd: o3d.geometry.PointCloud, keep_mask: np.ndarray) -> o3d.geometry.PointCloud:
    """Return a new pcd containing only points where keep_mask==True."""
    idx = np.nonzero(keep_mask)[0].astype(np.int64)
    return pcd.select_by_index(idx)

def resolve_grasp_action_conflict(
    pcd_grasp: o3d.geometry.PointCloud,
    pcd_action: o3d.geometry.PointCloud,
    eps: float,
    seed = 0,
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    If grasp/action points are within eps (mutual NN), randomly remove ONE point from the pair.
    """
    g_pts = np.asarray(pcd_grasp.points)
    a_pts = np.asarray(pcd_action.points)

    if len(g_pts) == 0 or len(a_pts) == 0:
        return pcd_grasp, pcd_action

    rng = np.random.default_rng(seed)

    # KD-trees
    a_kd = o3d.geometry.KDTreeFlann(pcd_action)
    g_kd = o3d.geometry.KDTreeFlann(pcd_grasp)

    # g -> nearest a
    g2a = np.full(len(g_pts), -1, dtype=np.int64)
    g2a_d = np.full(len(g_pts), np.inf, dtype=np.float64)
    for i in range(len(g_pts)):
        k, idx, dist2 = a_kd.search_knn_vector_3d(g_pts[i], 1)
        if k > 0:
            g2a[i] = idx[0]
            g2a_d[i] = np.sqrt(dist2[0])

    # a -> nearest g
    a2g = np.full(len(a_pts), -1, dtype=np.int64)
    a2g_d = np.full(len(a_pts), np.inf, dtype=np.float64)
    for j in range(len(a_pts)):
        k, idx, dist2 = g_kd.search_knn_vector_3d(a_pts[j], 1)
        if k > 0:
            a2g[j] = idx[0]
            a2g_d[j] = np.sqrt(dist2[0])

    # Mutual nearest neighbors within eps (g_i <-> a_j)
    pairs = []
    for gi, aj in enumerate(g2a):
        if aj < 0 or g2a_d[gi] > eps:
            continue
        if a2g[aj] == gi and a2g_d[aj] <= eps:
            pairs.append((gi, aj))

    if not pairs:
        return pcd_grasp, pcd_action

    remove_g = set()
    remove_a = set()

    # For each conflicting pair, randomly remove ONE of the two points
    for gi, aj in pairs:
        if gi in remove_g or aj in remove_a:
            continue  # already removed by another decision
        if rng.random() < 0.5:
            remove_g.add(gi)
        else:
            remove_a.add(aj)

    keep_g = np.ones(len(g_pts), dtype=bool)
    keep_a = np.ones(len(a_pts), dtype=bool)
    if remove_g:
        keep_g[list(remove_g)] = False
    if remove_a:
        keep_a[list(remove_a)] = False

    return _select_by_mask(pcd_grasp, keep_g), _select_by_mask(pcd_action, keep_a)

def visualize_perception_results(
    pointmap_np=None,
    rgb_np=None,
    tool_aff_pcd=None,
    target_aff_pcd=None,
    pred_grasps_world=None,
    top_grasp_idx=None,
    crop_range=None,
    apply_density=True,
    alpha_base=0.5,
):
    if crop_range is None:
        crop_range = [[0.0, 0.75], [-0.7, 0.7], [0.3, 1.2]]

    voxel_all, max_points_base, max_points_aff, point_size = 0.008, 40000, 5000, 10.0
    back = np.array([190, 190, 190], dtype=np.float64) / 255.0  # background color

    aff_colors = {
        "function": np.array([0.80, 0.16, 0.16], dtype=np.float64),  # deep red
        "grasping": np.array([0.18, 0.65, 0.25], dtype=np.float64),  # deep green
        "target":   np.array([0.80, 0.16, 0.16], dtype=np.float64),  # deep blue
    }

    pcd_base = None
    pcd_action = None
    pcd_grasp = None
    pcd_target = None
    aff_pcds_for_removal = []

    # --------- Base point cloud (optional) ----------
    if pointmap_np is not None:
        xyz = np.asarray(pointmap_np).reshape(-1, 3)
        if xyz.size > 0:
            pcd_base = o3d.geometry.PointCloud()
            pcd_base.points = o3d.utility.Vector3dVector(xyz)

            if rgb_np is None:
                colors = np.tile(back[None, :], (xyz.shape[0], 1))
            else:
                rgb = np.asarray(rgb_np).reshape(-1, 3).astype(np.float64)
                if rgb.size == 0:
                    colors = np.tile(back[None, :], (xyz.shape[0], 1))
                else:
                    if rgb.max() > 1.0:
                        rgb = rgb / 255.0
                    rgb = np.clip(rgb, 0.0, 1.0)
                    # If rgb length mismatches xyz, fall back to background
                    colors = rgb if rgb.shape[0] == xyz.shape[0] else np.tile(back[None, :], (xyz.shape[0], 1))

            pcd_base.colors = o3d.utility.Vector3dVector(colors)

            pcd_base = crop_pointcloud(
                pcd_base,
                x_range=crop_range[0],
                y_range=crop_range[1],
                z_range=crop_range[2],
            )
            if len(pcd_base.points) > 0:
                pcd_base, _ = pcd_base.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
                pcd_base = downsample_pcd(pcd_base, voxel=voxel_all, max_points=max_points_base)

    # ---------- Tool affordances (optional) ----------

    if tool_aff_pcd is not None and len(tool_aff_pcd.points) > 0 and tool_aff_pcd.has_colors():
        tc = np.asarray(tool_aff_pcd.colors)

        idx_action = np.where(tc[:, 1] > 0)[0]
        if idx_action.size > 0:
            pcd_action = tool_aff_pcd.select_by_index(idx_action.tolist())
            pcd_action = downsample_pcd(pcd_action, voxel=voxel_all, max_points=max_points_aff)

        idx_grasp = np.where(tc[:, 0] > 0)[0]
        if idx_grasp.size > 0:
            pcd_grasp = tool_aff_pcd.select_by_index(idx_grasp.tolist())
            pcd_grasp = downsample_pcd(pcd_grasp, voxel=voxel_all, max_points=max_points_aff)

        # pcd_grasp = None
        # Resolve overlap only if both exist
        if (pcd_grasp is not None and len(pcd_grasp.points) > 0) and (pcd_action is not None and len(pcd_action.points) > 0):
            pcd_grasp, pcd_action = resolve_grasp_action_conflict(
                pcd_grasp, pcd_action,
                eps=voxel_all * 0.75,
                seed=None
            )

        # Color + add if present
        if pcd_action is not None and len(pcd_action.points) > 0:
            if apply_density:
                pcd_action = blend_to_affcolor_by_density(
                    pcd_action, aff_colors["function"], back_rgb=back, k=50, gamma=0.8
                )
            else:
                pcd_action.paint_uniform_color(aff_colors["function"])
            aff_pcds_for_removal.append(pcd_action)

        if pcd_grasp is not None and len(pcd_grasp.points) > 0:
            if apply_density:
                pcd_grasp = blend_to_affcolor_by_density(
                    pcd_grasp, aff_colors["grasping"], back_rgb=back, k=50, gamma=0.8
                )
            else:
                pcd_grasp.paint_uniform_color(aff_colors["grasping"])
            aff_pcds_for_removal.append(pcd_grasp)

    # ---------- Target affordance (optional) ----------
    if target_aff_pcd is not None and len(target_aff_pcd.points) > 0 and target_aff_pcd.has_colors():
        tc2 = np.asarray(target_aff_pcd.colors)
        idx_target = np.where(tc2[:, 2] > 0)[0]
        if idx_target.size > 0:
            pcd_target = target_aff_pcd.select_by_index(idx_target.tolist())
            pcd_target = downsample_pcd(pcd_target, voxel=voxel_all, max_points=max_points_aff)

            if apply_density:
                pcd_target = blend_to_affcolor_by_density(
                    pcd_target, aff_colors["target"], back_rgb=back, k=50, gamma=0.8
                )
            else:
                pcd_target.paint_uniform_color(aff_colors["target"])

            aff_pcds_for_removal.append(pcd_target)

    # Remove base points near whichever affordances exist (only if base exists)
    if (pcd_base is not None) and (len(aff_pcds_for_removal) > 0) and (len(pcd_base.points) > 0):
        pcd_base = remove_base_near_affordance(pcd_base, aff_pcds_for_removal, r=voxel_all * 1)

    # Combine for return/visualization + alpha
    pcd_all = o3d.geometry.PointCloud()
    alpha_list = []

    if pcd_base is not None and len(pcd_base.points) > 0:
        pcd_all += pcd_base
        points_num = np.asarray(pcd_base.points).shape[0]
        alpha_list += [alpha_base] * points_num
    if pcd_action is not None and len(pcd_action.points) > 0:
        pcd_all += pcd_action
        points_num = np.asarray(pcd_action.points).shape[0]
        alpha_list += [1.0] * points_num
    if pcd_grasp is not None and len(pcd_grasp.points) > 0:
        pcd_all += pcd_grasp
        points_num = np.asarray(pcd_grasp.points).shape[0]
        alpha_list += [1.0] * points_num
    if pcd_target is not None and len(pcd_target.points) > 0:
        pcd_all += pcd_target
        points_num = np.asarray(pcd_target.points).shape[0]
        alpha_list += [1.0] * points_num

    alpha_array = np.array(alpha_list, dtype=np.float64)

    # Visualize (only if something to show)
    if len(pcd_all.points) > 0:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        opt = vis.get_render_option()
        opt.background_color = np.array([1, 1, 1])
        opt.point_size = float(point_size)

        vis.add_geometry(pcd_all)

        if pred_grasps_world is not None:
            for g in add_grasp_lines_open3d(pred_grasps_world, top_grasp_idx):
                vis.add_geometry(g)

        vis.run()
        vis.destroy_window()

    return pcd_all, pcd_action, pcd_grasp, pcd_target, alpha_array

def add_grasp_lines_open3d(pred_grasps_world, top_grasp_idx=None, color=(1.0, 0.0, 0.0)):
    """
    Build Open3D LineSet(s) representing grasps.
    Returns: list[o3d.geometry.LineSet]
    """
    width = 0.12
    depth = 0.10
    local_points = np.array([
        [0,        0, 0],
        [ width/2, 0, 0],
        [-width/2, 0, 0],
        [0,        0, -0.06],
        [ width/2, 0, depth],
        [-width/2, 0, depth],
    ], dtype=np.float64)

    lines = np.array([
        [0, 1], [0, 2], [0, 3],
        [1, 4], [2, 5],
    ], dtype=np.int64)
    geoms = []
    if pred_grasps_world is None or len(pred_grasps_world) == 0:
        return geoms

    pts_h = np.hstack([local_points, np.ones((local_points.shape[0], 1), dtype=np.float64)])

    for i, grasp_mat_world in enumerate(pred_grasps_world):
        grasp_mat_world = np.asarray(grasp_mat_world, dtype=np.float64)
        if grasp_mat_world.shape != (4, 4):
            raise ValueError(f"Each grasp pose must be 4x4, got {grasp_mat_world.shape} at i={i}")

        world_pts = (grasp_mat_world @ pts_h.T).T[:, :3]

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(world_pts)
        ls.lines  = o3d.utility.Vector2iVector(lines)

        # Per-line colors
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(color, dtype=np.float64), (lines.shape[0], 1))
        )

        geoms.append(ls)

        # same stopping rule you provided
        if top_grasp_idx is not None and i >= top_grasp_idx:
            break

    return geoms


def dilate_bool_mask(mask: np.ndarray, pixels: int, shape: str = "ellipse") -> np.ndarray:
    """
    Expand True regions in a boolean mask by ~'pixels' pixels using morphological dilation.
    Note: Dilation with a (2*pixels+1) kernel expands in a Chebyshev-like sense (grid-based).
    """
    if mask.dtype != np.bool_:
        raise TypeError("mask must be dtype=bool")
    if pixels <= 0:
        return mask.copy()

    # Convert bool -> uint8 (0 or 255) for OpenCV
    m = (mask.astype(np.uint8) * 255)

    ksize = 2 * pixels + 1
    if shape == "ellipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    elif shape == "rect":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    elif shape == "cross":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (ksize, ksize))
    else:
        raise ValueError("shape must be one of: 'ellipse', 'rect', 'cross'")

    dilated = cv2.dilate(m, kernel, iterations=1)
    return dilated > 0
