import os
import requests
import numpy as np
import base64
import open3d as o3d
import trimesh
from io import BytesIO
from PIL import Image

from improvised_tooluse import config

# --------------------------------------------------------------------------- #
#  Server endpoints -- keep every host/port in config.yaml, never in this file
# --------------------------------------------------------------------------- #

def _server(name, host=None, port=None):
    """Fill in whatever the caller left as None from `servers.<name>`."""
    entry = config.get("servers", name)
    return (host if host is not None else entry["host"],
            port if port is not None else entry["port"])


def convert_pil_image_to_base64(image: Image.Image) -> str:
    """PNG-encode a PIL image and return its base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def decode_array(field):
    """Reconstruct a numpy array from a server's {data, shape, dtype} dict."""
    raw = base64.b64decode(field["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(field["dtype"]))
    return arr.reshape(field["shape"])

def decode_base64(base64_image):
    image_data = base64.b64decode(base64_image)
    image_data = BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image

# --------------------------------------------------------------------------- #
#  SAM3  -- single image segmentation
# --------------------------------------------------------------------------- #
def sam3_request_concept(image, text, bbox, confidence_threshold, host=None, port=None):
    """Text- (and optionally bbox-) prompted segmentation (`/sam3_query`).

    image: PIL.image
    text:  prompt string
    bbox:  FLAT pixel xywh box [x, y, w, h] (top-left + size) or None. The server
           normalizes it internally; note this is xywh, NOT xyxy, and must be flat
           (a nested [[...]] unpacks wrong server-side and yields no masks).
    host/port: default to the `sam3` entry of config.yaml.
    Returns (scores, bboxes, decode_masks).
    """
    host, port = _server("sam3", host, port)
    url = f"http://{host}:{port}/sam3_query"
    payload = {
        "rgb_image": convert_pil_image_to_base64(image),
        "text": text,
        "bbox": bbox,
        "confidence_threshold": confidence_threshold,
    }
    # Convert any numpy arrays to lists for JSON serialization.
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()

    response = requests.post(url, json=payload).json()
    return process_return_json(response, image.size)


def sam3_request_prompt(rgb_image, input_point=None, input_label=None,
                        confidence_threshold=0.6, host=None, port=None):
    """Point-prompted segmentation (`/sam3_query_prompt`).

    rgb_image:   PIL.image
    input_point: np.array([[x_1, y_1]])
    input_label: np.array([1])
    host/port:   default to the `sam3` entry of config.yaml.

    Returns (scores, bboxes, decode_masks); empty lists if nothing is found.
    """
    host, port = _server("sam3", host, port)
    payload = {
        "rgb_image": convert_pil_image_to_base64(rgb_image),
        "point": input_point.tolist(),
        "point_label": input_label.tolist(),
        "confidence_threshold": confidence_threshold,
    }

    url = f"http://{host}:{port}/sam3_query_prompt"
    response = requests.post(url, json=payload).json()
    return process_return_json(response, rgb_image.size)


def process_return_json(response, image_size):
    """Decode a SAM3 response into (scores, bboxes, decode_masks).

    image_size: (W, H) as returned by PIL Image.size. The point endpoint may
    omit bboxes, in which case ``bboxes`` is a list of None.
    """
    scores = response.get("scores", [])
    masks = response.get("masks", [])
    bboxes = response.get("bboxes") or [None] * len(scores)

    decode_masks = []
    for i in range(len(scores)):
        mask_byte = base64.b64decode(masks[i])
        mask = np.frombuffer(mask_byte, dtype=np.uint8).reshape(
            image_size[1], image_size[0]
        )
        decode_masks.append(mask)

    return scores, bboxes, decode_masks


# --------------------------------------------------------------------------- #
#  FoundationPose  -- 6-DoF object pose
# --------------------------------------------------------------------------- #

def fp_registration_request(mesh, K, color, depth, mask, host=None, port=None):
    """host/port default to the `foundationpose` entry of config.yaml."""
    host, port = _server("foundationpose", host, port)
    server_url = f"http://{host}:{port}/fp_registration_query"
    mesh_bytes = mesh.export(file_type="glb")
    mesh_base64 = base64.b64encode(mesh_bytes).decode("utf-8")
    
    image_base64 = convert_pil_image_to_base64(Image.fromarray(color))
    
    depth_bytes = depth.astype(np.float32).tobytes()
    depth_base64 = base64.b64encode(depth_bytes).decode('utf-8')
    
    mask_bytes = mask.astype(np.uint8).tobytes()
    mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
    
    payload = {
        'mesh_file': mesh_base64,
        'intrinsics': K.tolist(),
        'rgb_image': image_base64,
        'depth': depth_base64,
        'mask': mask_base64
    }
    
    response = requests.post(server_url, json=payload)
    response_json = response.json()
    
    if response.status_code != 200:
        print("Error in FP registration request:", response_json.get("error", "Unknown error"))

    return (np.array(response_json['pose']).astype(np.float32).reshape(4, 4), 
            np.array(response_json['pose_last']).astype(np.float32).reshape(4, 4))
    
def fp_tracking_request(pose_last, mesh, K, color, depth, host=None, port=None):
    """host/port default to the `foundationpose` entry of config.yaml."""
    host, port = _server("foundationpose", host, port)
    server_url = f"http://{host}:{port}/fp_tracking_query"
    pose_last_list = pose_last.tolist()
    
    mesh_bytes = mesh.export(file_type="glb")
    mesh_base64 = base64.b64encode(mesh_bytes).decode("utf-8")
    
    image_base64 = convert_pil_image_to_base64(Image.fromarray(color))
    
    depth_bytes = depth.astype(np.float32).tobytes()
    depth_base64 = base64.b64encode(depth_bytes).decode('utf-8')
    
    payload = {
        'pose_last': pose_last_list,
        'mesh_file': mesh_base64,
        'intrinsics': K.tolist(),
        'rgb_image': image_base64,
        'depth': depth_base64,
    }
    
    response = requests.post(server_url, json=payload)
    response_json = response.json()
    
    if response.status_code != 200:
        print("Error in FP registration request:", response_json.get("error", "Unknown error"))

    return (np.array(response_json['pose']).astype(np.float32).reshape(4, 4), 
            np.array(response_json['pose_last']).astype(np.float32).reshape(4, 4))


def sam3d_request(image, mask, point_map=None, host=None, port=None):
    """host/port default to the `sam3d` entry of config.yaml."""
    host, port = _server("sam3d", host, port)
    server_url = f"http://{host}:{port}/sam3d_query"
    image_base64 = convert_pil_image_to_base64(image)
    mask_bytes = mask.tobytes()
    mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
    payload = {
        'rgb_image': image_base64,  
        'mask': mask_base64,
    }
    if point_map is not None:
        point_map_bytes = point_map.astype(np.float32).tobytes()
        point_map_base64 = base64.b64encode(point_map_bytes).decode('utf-8')
        payload['point_map'] = point_map_base64

    # import time
    # start_time = time.time()
    response = requests.post(server_url, json=payload)
    # end_time = time.time()

    # print("sam3d time:", end_time-start_time)
    response_json = response.json()
    
    if response.status_code != 200:
        print("Error in SAM3D request:", response_json.get("error", "Unknown error"))

    glb_bytes = base64.b64decode(response_json["glb"])
    glb_trimesh = trimesh.load_mesh(BytesIO(glb_bytes), file_type="glb", process=False)
    
    scale = np.array(response_json["scale"])
    translation = np.array(response_json["translation"])
    quat = np.array(response_json["quat"])

    glb_bytes_processed = base64.b64decode(response_json["glb_processed"])
    glb_trimesh_processed = trimesh.load_mesh(BytesIO(glb_bytes_processed), file_type="glb", process=False)

    return glb_trimesh, glb_trimesh_processed, scale, translation, quat


def grasp_request(image, depth, segmap, intrinsic_matrix, seg_id=1, host=None, port=None,
                  timeout=None):
    """host/port default to the `grasp` entry of config.yaml."""
    host, port = _server("grasp", host, port)
    server_url = f"http://{host}:{port}/sample_grasp"
    # Convert image to base64
    image_base64 = convert_pil_image_to_base64(image)
    depth_image = (depth * 1000.0).astype(np.uint16)
    depth_image_pil = Image.fromarray(depth_image)
    depth_base64 = convert_pil_image_to_base64(depth_image_pil)
    segmap_base64 = convert_pil_image_to_base64(segmap)
    
    # Prepare payload
    payload = {
        'image_rgb': image_base64,
        'image_depth': depth_base64,
        'segmap': segmap_base64,
        'K': intrinsic_matrix,
        'segmap_id': seg_id
    }

    # Convert numpy arrays to lists
    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
    
    # Send request to server
    response = requests.post(server_url, json=payload, timeout=timeout)
    
    response.raise_for_status()
    return response.json()

def grasping_pose_transform(response, T_world_cam):
    # tranform grasps from camera to world frame
    pred_grasps_cam = response["pred_grasps_cam"]
    scores = response["scores"]
    contact_pts = response["contact_pts"]
    pred_grasps_world = []
    opengl_to_opencv = np.diag([1, -1, -1, 1]) #opengl and opencv have different convention
    T_world_cam_opencv = T_world_cam @ opengl_to_opencv

    for i, grasp in enumerate(pred_grasps_cam):
        # grasp pose in camera frame
        grasp_mat_cam = np.array(grasp).reshape(4, 4)
        # convert grasp pose to world frame
        grasp_mat_world = T_world_cam_opencv @ grasp_mat_cam 
        pred_grasps_world.append(grasp_mat_world)
    
    sorted_indices = np.argsort(scores)[::-1]
    pred_grasps_world_sorted = [pred_grasps_world[i] for i in sorted_indices]
    scores_sorted = [scores[i] for i in sorted_indices]
    contact_pts_sorted = [contact_pts[i] for i in sorted_indices]

    return pred_grasps_world_sorted, scores_sorted, contact_pts_sorted

def visualize_grasp(pred_grasps_world, tsdf_pd, top_grasp_idx=None):
    gripper_geometries = []    
    for i, grasp_mat_world in enumerate(pred_grasps_world):
        width = 0.12
        depth = 0.1
        points = np.array([
            [0, 0, 0],
            [width/2, 0, 0],
            [-width/2, 0, 0],
            [0, 0, -0.06],
            [width/2, 0, depth],
            [-width/2, 0, depth],
        ])

        # homogeneous
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])

        # transform into world frame
        transformed_points = (grasp_mat_world @ points_h.T).T[:, :3]

        # Create lines
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(transformed_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # # color top grasp
        # if top_grasp_idx is not None and i == top_grasp_idx:
        #     color = [0, 0, 1]   # blue
        # else:
        #     color = [1, 0, 0]   # red
        color = [1, 0, 0]   # red

        line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

        gripper_geometries.append(line_set)
        if top_grasp_idx is not None and i >= top_grasp_idx:
            break
    
    # Create visualization window with custom view
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    vis.add_geometry(tsdf_pd)
    for gripper in gripper_geometries:
        vis.add_geometry(gripper)

    # Run visualization
    vis.run()
    vis.destroy_window()
