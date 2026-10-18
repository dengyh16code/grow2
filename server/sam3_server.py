"""SAM3 image server: /sam3_query and /sam3_query_prompt.

"""

import os

import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox
import base64
import io
import argparse

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
print("SAM3 root directory:", sam3_root)
print(os.environ["HF_HOME"])

import torch
# turn on tfloat32 for Ampere GPUs
# https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
print("Image model built")


def decode_base64(base64_image):
    image_data = base64.b64decode(base64_image)
    image_data = io.BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image


def process_results_for_response(results):
    nb_objects = len(results["scores"])
    score_list = []
    mask_list = []
    mask_logits_list = []
    bbox_list = []
    for i in range(nb_objects):
        score = results["scores"][i].item()
        score_list.append(score)
        mask = results["masks"][i].squeeze(0).cpu().numpy()
        bbox = results["boxes"][i].cpu().numpy()
        bbox_list.append(bbox.tolist())
        mask_np = mask.astype(np.uint8)  # Convert boolean mask to uint8
        mask_bytes = mask_np.tobytes()
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
        mask_list.append(mask_base64)
        logit = results["masks_logits"][i].squeeze(0).float().cpu().numpy().astype(np.float32)
        logit_bytes = logit.tobytes()
        logit_base64 = base64.b64encode(logit_bytes).decode('utf-8')
        mask_logits_list.append(logit_base64)
    return score_list, mask_list, mask_logits_list, bbox_list


def concept_segmentation_query(image, text, bbox, confidence_threshold):
    width, height = image.size
    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)
    if text is not None:
        inference_state = processor.set_text_prompt(state=inference_state, prompt=text)
    if bbox is not None:
        # Normalize bbox
        x, y, w, h = bbox
        box_input_xywh = torch.tensor([x, y, w, h]).view(-1, 4)
        box_input_cxcywh = box_xywh_to_cxcywh(box_input_xywh)
        norm_box_cxcywh = normalize_bbox(box_input_cxcywh, width, height).flatten().tolist()
        print("Normalized box input:", norm_box_cxcywh)
        inference_state = processor.add_geometric_prompt(state=inference_state, box=norm_box_cxcywh, label=True)

    score_list, mask_list, mask_logits_list, bbox_list = process_results_for_response(inference_state)
    return score_list, mask_list, mask_logits_list, bbox_list


def process_results_for_response_prompt(masks, scores):
    """
    masks: np.array
    scores: np.array
    """

    nb_objects = len(scores)
    mask_list = []
    for i in range(nb_objects):
        mask = masks[i]
        mask_np = mask.astype(np.uint8)  # Convert boolean mask to uint8
        mask_bytes = mask_np.tobytes()
        mask_base64 = base64.b64encode(mask_bytes).decode('utf-8')
        mask_list.append(mask_base64)

    score_list = scores.tolist()
    return score_list, mask_list


def segmentation_with_prompt(image, input_box=None, input_label=None, input_point=None, confidence_threshold=0.5):
    """
    input_point: [[575, 750]]
    input_label: np.array([0])
    input_box: np.array([425, 600, 700, 875])
    """

    processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
    inference_state = processor.set_image(image)

    masks, scores, _ = model.predict_inst(
        inference_state,
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]

    score_list, mask_list = process_results_for_response_prompt(masks, scores)

    return score_list, mask_list


from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('/sam3_query', methods=['POST'])
def sam3_query():
    try:
        data = request.get_json()
        if 'rgb_image' not in data:
            return jsonify({'error': 'No rgb image provided'}), 400
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        if 'bbox' not in data:
            return jsonify({'error': 'No bbox provided'}), 400
        if 'confidence_threshold' not in data:
            return jsonify({'error': 'No confidence threshold provided'}), 400

        image = decode_base64(data['rgb_image'])
        text = data['text']
        bbox = data['bbox']  # Here the box is in  (x,y,w,h) format, where (x,y) is the top left corner.
        confidence = float(data['confidence_threshold'])
        scores, masks, mask_logits, bboxes = concept_segmentation_query(image, text, bbox, confidence)
        response = {'scores': scores, 'masks': masks, 'mask_logits': mask_logits, 'bboxes': bboxes}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/sam3_query_prompt', methods=['POST'])
def sam3_query_prompt():
    try:
        data = request.get_json()
        if 'rgb_image' not in data:
            return jsonify({'error': 'No rgb image provided'}), 400
        if ('point' not in data) and ("bbox" not in data):
            return jsonify({'error': 'point or bbox need to be provided'}), 400
        if 'confidence_threshold' not in data:
            return jsonify({'error': 'No confidence threshold provided'}), 400

        image = decode_base64(data['rgb_image'])
        if 'point' not in data:
            point = None
            point_label = None
        else:
            point = data['point']
            point_label = data['point_label']
        if "bbox" not in data:
            bbox = None
        else:
            bbox = data['bbox']

        confidence = float(data['confidence_threshold'])

        score_list, mask_list = segmentation_with_prompt(image, input_box=bbox, input_label=point_label, input_point=point, confidence_threshold=confidence)
        response = {'scores': score_list, 'masks': mask_list}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Cheap readiness probe -- the model is built by the time routes serve."""
    return jsonify({'status': 'ok', 'role': 'image'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM3 Image Server')
    parser.add_argument('--ip', default='0.0.0.0', type=str, help='IP address to run the app on. Use "0.0.0.0" for your machine\'s IP address')
    parser.add_argument('--port', default=4054, type=int, help='Port number to run the app on')
    args = parser.parse_args()

    app.run(host=args.ip, port=args.port, debug=True, use_reloader=False)