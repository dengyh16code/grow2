from openai import APITimeoutError, OpenAI
import re
import time
import numpy as np
import os
import json
import base64
from io import BytesIO

from improvised_tooluse import config


QWEN_BBOX_TIMEOUT_SECONDS = 20.0


def convert_pil_image_to_base64(image) -> str:
    """PNG-encode a PIL image and return its base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

model_dict = {
    "qwen3.7-plus": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": (
            f"https://{os.getenv('DASHSCOPE_WORKSPACE_ID')}"
            ".cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        ),
        "model_name": "qwen3.7-plus",
    },
    "gpt-5.5": {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-5.5",
    },
    "kimi-v3": {
        "api_key": os.getenv("KIMI_API_KEY"),
        "base_url": "https://api.moonshot.cn/v1",
        "model_name": "kimi-v3",
    },
}

# Which entry of model_dict to use -- keep the choice in config.yaml, not here.
def selected_model(role):
    """The model_dict key configured for `role` (`vlm` or `bounding_box`)."""
    return config.get("models", role)


def load_prompts(prompt_dir):
    """Load prompts from files.
    """
    prompts = dict()
    for filename in os.listdir(prompt_dir):
        path = os.path.join(prompt_dir, filename)
        if os.path.isfile(path) and path[-4:] == '.txt':
            with open(path, 'r') as f:
                value = f.read()
            key = filename[:-4]
            prompts[key] = value
    return prompts

def parse_selection_results(text):
    match = re.search(r"\*\*\*(.*?)\*\*\*", text, re.S)
    if not match:
        raise ValueError("Selection results block not found")

    json_str = match.group(1).strip()
    return json.loads(json_str)

def prepare_inputs(
        messages,
        images=None,
        meta_prompt=None,
        example_images=None,
        example_responses=None,
):
    if not isinstance(messages, list):
        messages = [messages]
    
    user_content = []
    system_content = []
    
    for message in messages:
        content = {
            'type': 'text',
            'text': message
        }
        user_content.append(content)
    
    if images is not None:
        if not isinstance(images, list):
            images = [images]

        for img in images:
            b64 = convert_pil_image_to_base64(img)

            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })
    
    if meta_prompt is not None:
        system_content = [{
            "type": "text",
            "text": meta_prompt
        }]
    else:
        system_content = []
    
    if example_images is not None and example_responses is not None:
        if not isinstance(example_images, list):
            example_images = [example_images]
        if not isinstance(example_responses, list):
            example_responses = [example_responses]
        
        for example_image, example_response in zip(
                example_images, example_responses):
            b64 = convert_pil_image_to_base64(example_image)

            content = {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{b64}'
                }
            }
            user_content.append(content)

            content = {
                'type': 'text',
                'text': example_response,
            }
            user_content.append(content)
    
    messages =[
        {
            "role": "system",
            "content": system_content
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    return messages

def _query(selected_model,
           messages,
           images=None,
           meta_prompt=None,
           example_images=None,
           example_responses=None,
           timeout=None,
        ):
    """Send one chat completion to `selected_model`, a key of model_dict."""
    # import time

    # start_time = time.time()
    client_kwargs = {
        "api_key": model_dict[selected_model]["api_key"],
        "base_url": model_dict[selected_model]["base_url"],
        # A timed-out client does not cancel server work. Let the caller stop
        # the case instead of silently issuing another remote request.
        "max_retries": 0,
    }
    if timeout is not None:
        client_kwargs.update(timeout=timeout)

    client = OpenAI(
        **client_kwargs
    )

    model_name = model_dict[selected_model]["model_name"]

    prepared_messages = prepare_inputs(
        messages,
        images,
        meta_prompt,
        example_images,
        example_responses
    )

    # print(prepared_messages)

    response = client.chat.completions.create(
        model=model_name,
        messages=prepared_messages,
    )
    # end_time = time.time()
    # print("time for vlm:", end_time-start_time)

    return response.choices[0].message.content

def query_model(messages,
                images=None,
                meta_prompt=None,
                example_images=None,
                example_responses=None
        ):
    """Query the main VLM -- `models.vlm` in config.yaml."""
    return _query(selected_model("vlm"), messages, images, meta_prompt,
                  example_images, example_responses)

def query_bounding_box(image, query_text):
    start_time = time.monotonic()
    image_width, image_height = image.size
    query_text = "give me the bounding box of {} in the picture, response in format of [x1,y1,x2,y2], if no such region found, reply [0,0,0,0].".format(query_text)
    pixel_bbox = [0, 0, 0, 0]
    if image.size[0] > 10 and image.size[1] > 10:
        try:
            bbox = _query(
                selected_model("bounding_box"),
                messages=[query_text],
                images=[image],
                timeout=QWEN_BBOX_TIMEOUT_SECONDS,
            )
        except APITimeoutError:
            print(
                f"Qwen bounding-box request exceeded "
                f"{QWEN_BBOX_TIMEOUT_SECONDS:g} seconds; returning [0, 0, 0, 0]."
            )
            return pixel_bbox

        elapsed = time.monotonic() - start_time
        if elapsed > QWEN_BBOX_TIMEOUT_SECONDS:
            print(
                f"Qwen bounding-box response arrived after {elapsed:.2f} seconds; "
                "returning [0, 0, 0, 0]."
            )
            return pixel_bbox

        x1, y1, x2, y2 = parse_bbox_list(bbox)

        # The model answers on a 0-1000 grid, so scale back to pixels.
        x1 = int(np.clip(x1 / 1000 * image_width, 0, image_width-1))
        y1 = int(np.clip(y1 / 1000 * image_height, 0, image_height-1))
        x2 = int(np.clip(x2 / 1000 * image_width, 0, image_width-1))
        y2 = int(np.clip(y2 / 1000 * image_height, 0, image_height-1))

        pixel_bbox = [x1, y1, x2, y2]
        end_time = time.monotonic()
        print("Bounding box received in {:.2f} seconds: {}".format(end_time - start_time, pixel_bbox))

    return pixel_bbox

def parse_bbox_list(text):
    pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
    match = re.search(pattern, text)
    if match:
        bbox = [int(match.group(i)) for i in range(1, 5)]
        return bbox
    else:
        print("No valid bounding box found in the text.")
        return [0, 0, 0, 0]
