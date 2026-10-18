import time
import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os

class CameraClient:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # depth native 640x480
        # self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        # color doesn't support 640x480 on L515; pick 960x540 then align-to-depth => 640x480
        # self.config.enable_stream(rs.stream.color, 960, 540, rs.format.rgb8, 30)
        self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        # Warmup
        for _ in range(10):
            self.pipeline.wait_for_frames()
        time.sleep(0.2)
        print("Camera initialized.")

    def get_observation(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_scale = depth_frame.get_units()
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_image = depth_raw.astype(np.float32) * depth_scale  # meters

        intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        K = np.array([[intr.fx, 0.0, intr.ppx],
                      [0.0, intr.fy, intr.ppy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        
        print("Got observation: color image {}, depth image {}, depth scale {:.6f} m/unit".format(
            color_image.shape, depth_image.shape, depth_scale
        ))
        return K, color_image, depth_image
    
    def stop(self):
        self.pipeline.stop()


def main(save_dir="obs", fps=5, num_frames=None):
    """
    - depth: meters, saved as .npy
    - depth_vis: uint8 visualization, saved as PNG
    - intrinsics: saved once as intrinsics.json (for aligned color frame)
    """
    os.makedirs(save_dir, exist_ok=True)
    camera_client = CameraClient()
    period = 1.0 / float(fps)
    next_t = time.perf_counter()
    i = 0
    K_saved = False
    try:
        while True:
            if num_frames is not None and i >= num_frames:
                break
            K, color_image, depth_image = camera_client.get_observation()
            if not K_saved:
                with open(os.path.join(save_dir, "intrinsics.json"), "w") as f:
                    json.dump(K.tolist(), f)
                print("K:\n", K)
                K_saved = True
            color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f"color_image_{i}.png"), color_bgr)
            np.save(os.path.join(save_dir, f"depth_image_{i}.npy"), depth_image)
            dmin, dmax = float(depth_image.min()), float(depth_image.max())
            depth_vis = (depth_image - dmin) / (dmax - dmin + 1e-6) * 255.0
            cv2.imwrite(os.path.join(save_dir, f"depth_image_vis_{i}.png"), depth_vis.astype(np.uint8))
            i += 1

            # Throttle to target fps
            next_t += period
            sleep_t = next_t - time.perf_counter()
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        camera_client.stop()
        print(f"Saved {i} frames to {save_dir}")

if __name__ == "__main__":
    # num_frames=None means run forever until Ctrl+C
    main(save_dir="obs", fps=5, num_frames=None)
