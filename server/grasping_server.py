import argparse
import base64
from io import BytesIO
import numpy as np
import os
from flask import Flask, request, jsonify
from PIL import Image
import json
import sys
import time
import config_utils
import tensorflow.compat.v1 as tf
from contact_grasp_estimator import GraspEstimator

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

class GraspSampler():
    def __init__(self) -> None:
        """
        Predict 6-DoF grasp distribution for given model and input data

        :param global_config: config.yaml from checkpoint directory
        :param checkpoint_dir: checkpoint directory
        :param K: Camera Matrix with intrinsics to convert depth to point cloud
        :param local_regions: Crop 3D local regions around given segments. 
        :param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
        :param filter_grasps: Filter and assign grasp contacts according to segmap.
        :param segmap_id: only return grasps from specified segmap_id.
        :param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
        :param forward_passes: Number of forward passes to run on each point cloud. Default: 1
        """
        self.ckpt_dir = "/data/home/anxing/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"
        self.z_range = [0.22, 1.6]
        self.local_regions = True
        self.filter_grasps = True
        self.skip_border_objects = False
        self.visulize = True
        self.forward_passes = 5
        self.arg_configs = []

        # global_config = config_utils.load_config(self.ckpt_dir, batch_size=self.forward_passes, arg_configs=self.arg_configs)
        global_config = config_utils.load_config(self.ckpt_dir, batch_size=self.forward_passes, arg_configs=self.arg_configs)
        global_config['TEST']['first_thres'] = 0.04
        global_config['TEST']['second_thres'] = 0.02
        global_config['TEST']['num_samples'] = 800
        print(str(global_config))
        # Build the model
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, self.ckpt_dir, mode='test')   
        self.tmp_img = ()
        self.tmp_grasp = ()

    def inference(self, depth, segmap, cam_K, segmap_id, rgb=None):
        begin = time.time()
        print('Converting depth to point cloud(s)...')
        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds( 
            depth, cam_K, segmap=segmap, rgb=rgb,
            skip_border_objects=self.skip_border_objects, z_range=self.z_range)
        
        print('Generating Grasps...')
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
            self.sess, pc_full, pc_segments=pc_segments,
            local_regions=self.local_regions, filter_grasps=self.filter_grasps, forward_passes=self.forward_passes)

        # reponse = GraspGroupResponse()
        pred_grasps_cam_list = []
        scores_list = []
        contact_pts_list = []
        
        target_id = segmap_id if (self.local_regions or self.filter_grasps) else -1
        for k in pred_grasps_cam:
            # if k != target_id:
            #     continue
            for pose_cam, score, contact_pt in zip(pred_grasps_cam[k], scores[k], contact_pts[k]):
                pred_grasps_cam_list.append(pose_cam.flatten().tolist())
                scores_list.append(score.tolist())
                contact_pts_list.append(contact_pt.tolist())

        # Visualize results

        print("inference time: ", time.time() - begin)

        return pred_grasps_cam_list, scores_list, contact_pts_list
    
    def inference_from_pointcloud(self, pc_segments, segmap_id=0):
        """
        Directly predict grasps from point cloud segments using predict_grasps
        
        :param pc_segments: dict[int, np.ndarray] or np.ndarray - Point cloud segments (dict) or single point cloud (Nx3)
        :param segmap_id: int - Segment ID to use if pc_segments is a dict
        :returns: pred_grasps_cam_list, scores_list, contact_pts_list
        """
        begin = time.time()
        print('Generating Grasps from point cloud...')
        
        pred_grasps_cam_list = []
        scores_list = []
        contact_pts_list = []
        
        # Handle both dict and single point cloud
        if isinstance(pc_segments, dict):
            # If it's a dict, use the specified segmap_id or all segments
            segments_to_process = {segmap_id: pc_segments[segmap_id]} if segmap_id in pc_segments else pc_segments
            for seg_id, pc_segment in segments_to_process.items():
                if pc_segment.shape[0] == 0:
                    continue
                pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_grasps(
                    self.sess, pc_segment, convert_cam_coords=True, forward_passes=self.forward_passes)
                
                for pose_cam, score, contact_pt in zip(pred_grasps_cam, scores, contact_pts):
                    pred_grasps_cam_list.append(pose_cam.flatten().tolist())
                    scores_list.append(score.tolist())
                    contact_pts_list.append(contact_pt.tolist())
        else:
            # Single point cloud array
            if pc_segments.shape[0] > 0:
                pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_grasps(
                    self.sess, pc_segments, convert_cam_coords=True, forward_passes=self.forward_passes)
                
                for pose_cam, score, contact_pt in zip(pred_grasps_cam, scores, contact_pts):
                    pred_grasps_cam_list.append(pose_cam.flatten().tolist())
                    scores_list.append(score.tolist())
                    contact_pts_list.append(contact_pt.tolist())
        
        print("inference time: ", time.time() - begin)
        
        return pred_grasps_cam_list, scores_list, contact_pts_list

    def inference_with_context(self, pc_full, pc_segment):
        """
        Predict grasps for a specific point cloud segment while using the full point cloud for context.
        
        :param pc_full: Nx3 - Full scene point cloud
        :param pc_segment: Mx3 - Point cloud of the target object
        :returns: pred_grasps_cam_list, scores_list, contact_pts_list
        """
        begin = time.time()
        print('Generating Grasps from point cloud with context...')
        
        # Construct segments dict for internal API
        pc_segments = {0: pc_segment}
        
        # We rely on self.local_regions=True to extract context around the segment
        # and self.filter_grasps=True to ensure grasps are on the segment
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(
            self.sess, pc_full, pc_segments=pc_segments,
            local_regions=self.local_regions, filter_grasps=self.filter_grasps, forward_passes=self.forward_passes)
        
        pred_grasps_cam_list = []
        scores_list = []
        contact_pts_list = []
        
        # Results are keyed by segment ID (0 in our case)
        if 0 in pred_grasps_cam:
            for pose_cam, score, contact_pt in zip(pred_grasps_cam[0], scores[0], contact_pts[0]):
                pred_grasps_cam_list.append(pose_cam.flatten().tolist())
                scores_list.append(score.tolist())
                contact_pts_list.append(contact_pt.tolist())

        print("inference time: ", time.time() - begin)
        
        return pred_grasps_cam_list, scores_list, contact_pts_list


# Flask app
app = Flask(__name__)

def read_image_from_base64(base64_str):
    """Decode base64 string to image."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image = np.array(image)
    return image

@app.route('/sample_grasp', methods=['POST'])
def grasp_prediction():
    # Parse JSON data
    query_data = request.get_json()
    if "image_rgb" in query_data:
        image_rgb_base64 = query_data["image_rgb"]
        image_rgb = read_image_from_base64(image_rgb_base64)
    else:
        image_rgb = None

    image_depth_base64 = query_data["image_depth"]
    segmap_base64 = query_data["segmap"]
    K = query_data["K"]
    segmap_id = query_data["segmap_id"]
    
    image_depth = read_image_from_base64(image_depth_base64)
    segmap = read_image_from_base64(segmap_base64)

    if len(image_depth.shape) == 3:
        image_depth = image_depth[:,:,0]
    if len(segmap.shape) == 3:
        segmap = segmap[:,:,0]
    image_depth[np.isnan(image_depth)]=0
    image_depth = image_depth.astype(np.float32)/1000

    # print(image_depth.shape)
    print(np.max(image_depth))
    print(np.min(image_depth))

    cam_K = np.array(K).reshape(3, 3)

    pred_grasps_cam_list, scores_list, contact_pts_list = grasp_sampler.inference(image_depth, segmap, cam_K, segmap_id, rgb=image_rgb)
    if len(pred_grasps_cam_list) > 0:
        print("grasping pose candidates: ", len(pred_grasps_cam_list))
    else:
        print("no grasping pose candidates")

    return jsonify({'pred_grasps_cam': pred_grasps_cam_list, 'scores': scores_list, 'contact_pts': contact_pts_list})


@app.route('/sample_grasp_from_pointcloud', methods=['POST'])
def grasp_prediction_from_pointcloud():
    # Parse JSON data
    query_data = request.get_json()
    object_pts = query_data["object_pts"]
    segmap_id = query_data.get("segmap_id", 0)  # Optional, default to 0
    
    # Convert object_pts to numpy array
    # Handle both list of points and dict of segments
    if isinstance(object_pts, dict):
        # If it's a dict, convert each segment
        pc_segments = {}
        for key, value in object_pts.items():
            pc_segments[int(key)] = np.array(value, dtype=np.float32)
    elif isinstance(object_pts, list):
        # If it's a list, assume it's a single point cloud (Nx3)
        pc_segments = np.array(object_pts, dtype=np.float32)
    else:
        return jsonify({'error': 'object_pts must be a list or dict'}), 400
    
    # Validate point cloud shape
    if isinstance(pc_segments, np.ndarray):
        if len(pc_segments.shape) != 2 or pc_segments.shape[1] != 3:
            return jsonify({'error': 'Point cloud must be Nx3 array'}), 400
    elif isinstance(pc_segments, dict):
        for key, pc in pc_segments.items():
            if len(pc.shape) != 2 or pc.shape[1] != 3:
                return jsonify({'error': f'Point cloud segment {key} must be Nx3 array'}), 400
    
    pred_grasps_cam_list, scores_list, contact_pts_list = grasp_sampler.inference_from_pointcloud(pc_segments, segmap_id=segmap_id)
    
    if len(pred_grasps_cam_list) > 0:
        print("grasping pose candidates: ", len(pred_grasps_cam_list))
    else:
        print("no grasping pose candidates")
    
    return jsonify({'pred_grasps_cam': pred_grasps_cam_list, 'scores': scores_list, 'contact_pts': contact_pts_list})



@app.route('/sample_grasp_with_context', methods=['POST'])
def grasp_prediction_with_context():
    # Parse JSON data
    query_data = request.get_json()
    
    if "full_pc" not in query_data or "segment_pc" not in query_data:
        return jsonify({'error': 'Missing full_pc or segment_pc in request'}), 400

    full_pc_list = query_data["full_pc"]
    segment_pc_list = query_data["segment_pc"]
    
    full_pc = np.array(full_pc_list, dtype=np.float32)
    segment_pc = np.array(segment_pc_list, dtype=np.float32)
    
    # Validate point cloud shapes
    if len(full_pc.shape) != 2 or full_pc.shape[1] != 3:
        return jsonify({'error': 'full_pc must be Nx3 array'}), 400
    if len(segment_pc.shape) != 2 or segment_pc.shape[1] != 3:
        return jsonify({'error': 'segment_pc must be Nx3 array'}), 400
    
    pred_grasps_cam_list, scores_list, contact_pts_list = grasp_sampler.inference_with_context(full_pc, segment_pc)
    
    if len(pred_grasps_cam_list) > 0:
        print("grasping pose candidates: ", len(pred_grasps_cam_list))
    else:
        print("no grasping pose candidates")
    
    return jsonify({'pred_grasps_cam': pred_grasps_cam_list, 'scores': scores_list, 'contact_pts': contact_pts_list})


if __name__ == '__main__':
    import mesh_utils
    gripper = mesh_utils.create_gripper('panda')
    gripper_control_points = gripper.get_control_point_tensor(1, False, convex_hull=False).squeeze()
    print(gripper_control_points)
    grasp_sampler = GraspSampler()
    parser = argparse.ArgumentParser(description='Grasping Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=4003, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)