import sys
import base64
import io
import traceback
from PIL import Image
# import inference code
sys.path.append("notebook")
import numpy as np
from inference import Inference, load_image, load_single_mask
import argparse
from flask import Flask, request, jsonify
import torch
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
import copy

# load model
tag = "hf"
config_path = f"checkpoints/{tag}/pipeline.yaml"
inference = Inference(config_path, compile=False)
print("Model loaded successfully.")


def decode_base64(base64_image):
    image_data = base64.b64decode(base64_image)
    image_data = io.BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image

_R_ZUP_TO_YUP = np.array([
    [1, 0, 0], 
    [0, 0, -1], 
    [0, 1, 0]
], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

R_pytorch3d_to_cam = np.diag([-1, 1, -1, 1])

def compose_transform(
    scale: torch.Tensor, rotation: torch.Tensor, translation: torch.Tensor
) -> Transform3d:
    """
    Args:
        scale: (3, ) tensor of scale factors
        rotation: (3, 3) tensor of rotation matrices
        translation: (3, ) tensor of translation vectors
    """
    tfm = Transform3d(dtype=scale.dtype, device=scale.device)
    return tfm.scale(scale).rotate(rotation).translate(translation)


app = Flask(__name__)
@app.route('/sam3d_query', methods=['POST']) 
def sam3d_query():
    try:
        data = request.get_json()
        if 'rgb_image' not in data:
            return jsonify({'error': 'No rgb image provided'}), 400
        if 'mask' not in data:
            return jsonify({'error': 'No mask provided'}), 400

        
        image_byte64 = data['rgb_image']
        image_pil = decode_base64(image_byte64)
        height, width = image_pil.size
        image_np = np.array(image_pil)

        mask_byte64 = data['mask']
        mask_byte = base64.b64decode(mask_byte64)
        mask_np = np.frombuffer(mask_byte, dtype=np.uint8).reshape((width, height))

        print(image_np.shape, mask_np.shape)

        if "point_map" in data:
            point_map_byte64 = data['point_map']
            point_map_byte = base64.b64decode(point_map_byte64)
            point_map_np = np.frombuffer(point_map_byte, dtype=np.float32).reshape((width, height,3))
            point_map_tensor = torch.tensor(point_map_np, dtype=torch.float32)
            output = inference(image_np, mask_np, seed=42, pointmap=point_map_tensor)
        else:
            output = inference(image_np, mask_np, seed=42)

        # scale = output["scale"].tolist()[0]
        # translation = output["translation"].tolist()[0]
        # rotation = output["rotation"].tolist()[0]

        scale_tensor = output["scale"]
        translation_tensor = output["translation"]
        rotation_tensor = output["rotation"]


        # rotation_matrix = output["6drotation_normalized"].tolist()[0]
        glb_trimesh = output["glb"]
        glb_bytes = glb_trimesh.export(file_type="glb")
        glb_b64 = base64.b64encode(glb_bytes).decode("utf-8")

        glb_trimesh_process = copy.deepcopy(glb_trimesh)
        vertices = glb_trimesh_process.vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(rotation_tensor.device)
        R_l2c = quaternion_to_matrix(rotation_tensor)

        l2c_transform = compose_transform(
            scale=scale_tensor,
            rotation=R_l2c,
            translation=translation_tensor,
        )

        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        glb_trimesh_process.vertices = vertices.squeeze(0).cpu().numpy()
        glb_trimesh_process.apply_transform(R_pytorch3d_to_cam)
        glb_bytes_process = glb_trimesh_process.export(file_type="glb")
        glb_b64_process = base64.b64encode(glb_bytes_process).decode("utf-8")
    
        scale = scale_tensor.tolist()[0]
        translation = translation_tensor.tolist()[0]
        rotation = rotation_tensor.tolist()[0]

        response = {
            "scale": scale,
            "translation": translation,
            "quat": rotation,
            "glb": glb_b64,
            "glb_processed": glb_b64_process
        }

        return jsonify(response), 200
    
    except Exception as e:
        trace = traceback.format_exc()
        print(trace)  # or use app.logger.error(trace)

        return jsonify({
            "error": str(e),
            "traceback": trace
        }), 500
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM3 Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=4055, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)