# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import base64
from io import BytesIO
from estimater import *
from datareader import *
import argparse
import os
from flask import Flask, request, jsonify


code_dir = os.path.dirname(os.path.realpath(__file__))


est_refine_iter = 5
track_refine_iter = 2
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()


def fp_registration(mesh, K, color, depth, mask):
    mask = mask.astype(bool)
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=f'{code_dir}/debug',
        debug=0,
        glctx=glctx
    )
    pose, pose_last = est.register(K=K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)
    return pose, pose_last


def fp_tracking(pose_last, mesh, K, color, depth):
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=f'{code_dir}/debug',
        debug=0,
        glctx=glctx
    )
    est.pose_last = torch.from_numpy(pose_last).float().cuda() 
    pose, pose_last = est.track_one(rgb=color, depth=depth, K=K, iteration=track_refine_iter)
    return pose, pose_last


def decode_base64(base64_image):
    image_data = base64.b64decode(base64_image)
    image_data = io.BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image


app = Flask(__name__)

@app.route('/fp_registration_query', methods=['POST']) 
def fp_registration_query():
    try:
        data = request.get_json()
        if 'mesh_file' not in data:
            return jsonify({'error': 'No mesh file provided'}), 400
        if 'intrinsics' not in data:
            return jsonify({'error': 'No intrinsics provided'}), 400
        if 'rgb_image' not in data:
            return jsonify({'error': 'No rgb image provided'}), 400
        if 'depth' not in data:
            return jsonify({'error': 'No depth provided'}), 400
        if 'mask' not in data:
            return jsonify({'error': 'No mask provided'}), 400
        
        image_pil = decode_base64(data['rgb_image'])
        height, width = image_pil.size
        color = np.array(image_pil)
        
        depth_byte64 = data['depth']
        depth_byte = base64.b64decode(depth_byte64)
        depth = np.frombuffer(depth_byte, dtype=np.float32).reshape((width, height))
        
        mask_byte64 = data['mask']
        mask_byte = base64.b64decode(mask_byte64)
        mask = np.frombuffer(mask_byte, dtype=np.uint8).reshape((width, height))
        
        K = np.array(data['intrinsics']).reshape(3,3)
        
        glb_bytes = base64.b64decode(data["mesh_file"])
        mesh_scene = trimesh.load_mesh(BytesIO(glb_bytes), file_type="glb", process=False)
        if isinstance(mesh_scene, trimesh.Trimesh):
            mesh = mesh_scene
        else:
            geoms = list(mesh_scene.geometry.values())
            if len(geoms) != 1:
                raise ValueError(f"Expected exactly 1 geometry, but got {len(geoms)}: "
                                f"{list(mesh_scene.geometry.keys())}")
            mesh = geoms[0]
        
        pose, pose_last = fp_registration(mesh, K, color, depth, mask)
        
        response = {
            'pose': pose.tolist(),
            'pose_last': pose_last.tolist()
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/fp_tracking_query', methods=['POST']) 
def fp_tracking_query():
    try:
        data = request.get_json()
        if 'pose_last' not in data:
            return jsonify({'error': 'No pose_last provided'}), 400
        if 'mesh_file' not in data:
            return jsonify({'error': 'No mesh file provided'}), 400
        if 'intrinsics' not in data:
            return jsonify({'error': 'No intrinsics provided'}), 400
        if 'rgb_image' not in data:
            return jsonify({'error': 'No rgb image provided'}), 400
        if 'depth' not in data:
            return jsonify({'error': 'No depth provided'}), 400
        
        pose_last = np.array(data['pose_last']).reshape(4,4)
        
        image_pil = decode_base64(data['rgb_image'])
        height, width = image_pil.size
        color = np.array(image_pil)
        
        depth_byte64 = data['depth']
        depth_byte = base64.b64decode(depth_byte64)
        depth = np.frombuffer(depth_byte, dtype=np.float32).reshape((width, height))
        
        K = np.array(data['intrinsics']).reshape(3,3)
        
        glb_bytes = base64.b64decode(data["mesh_file"])
        mesh_scene = trimesh.load_mesh(BytesIO(glb_bytes), file_type="glb", process=False)
        if isinstance(mesh_scene, trimesh.Trimesh):
            mesh = mesh_scene
        else:
            geoms = list(mesh_scene.geometry.values())
            if len(geoms) != 1:
                raise ValueError(f"Expected exactly 1 geometry, but got {len(geoms)}: "
                                f"{list(mesh_scene.geometry.keys())}")
            mesh = geoms[0]
        
        pose, pose_last = fp_tracking(pose_last, mesh, K, color, depth)
        
        response = {
            'pose': pose.tolist(),
            'pose_last': pose_last.tolist()
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Foundation Pose Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=4052, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)