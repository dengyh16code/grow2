import time
import cv2
import sapien.core as sapien
import numpy as np
import json
from improvised_tooluse.geom_utils import trimesh_to_open3d_mesh, numpy_to_o3d
from improvised_tooluse.sapien_utils import get_actor_mesh, compute_xy_rect, rect_intersect
import os
from PIL import Image
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import axangle2quat, qmult
import copy

from improvised_tooluse import config

# ROBOT_NAME='panda'
ROBOT_NAME='fr3'
VIEW_CAMERA_POSITION = (1.7, 0.0, 2.1)
VIEW_CAMERA_RPY = (0.0, -np.arctan2(1.7, 2.1), np.pi)
VIEW_CAMERA_FOVY = 0.80
VIEW_CAMERA_NEAR = 0.05
VIEW_CAMERA_FAR = 100
VIEW_CAMERA_SIZE = (1080, 1080)

# seed = int(time.time() * 1000) % 2**32
# np.random.seed(seed)

class VideoRecorder:
    def __init__(self, record_path: str = 'video.mp4'):
        self.record_path = record_path
        if os.path.dirname(record_path):
            os.makedirs(os.path.dirname(record_path), exist_ok=True)
        self.images = []

    def capture(self, camera):
        camera.take_picture()
        rgba = camera.get_picture("Color")
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        self.images.append(Image.fromarray(rgba_img))
    
    def reset(self):
        self.images = []
    
    def save(self, fps: float = 30.0):
        if not self.images:
            return

        w, h = self.images[0].size  # PIL: (width, height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.record_path, fourcc, fps, (w, h))

        for img in self.images:
            frame = np.array(img)[..., :3]  # drop alpha, RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB -> BGR for OpenCV
            writer.write(frame_bgr)

        writer.release()
        print(f"Saved video to {self.record_path}")


class simu_env:
    def __init__(self, record_mp4_path: str = None, visualize: bool = True):
        # visualize=False is a hard headless switch: no window is ever opened,
        # not even briefly, so the pipeline runs on a display-less server.
        self.visualize = visualize
        self.viewer = None
        self.video_recorder = VideoRecorder(record_mp4_path) if record_mp4_path is not None else None
        self.recording_camera = None
        self.recording_active = False
        self.initialize_scene()
        self.create_table()
        self.create_robot_base()
        self.load_robot()
        sapien.render.set_camera_shader_dir('rt')
        sapien.render.set_ray_tracing_denoiser("optix") # change to "optix" or "oidn"
        self.camera_initializition()

    def initialize_scene(self):
        self.scene = sapien.Scene()  # Create an instance of simulation world (aka scene)
        self.scene.set_timestep(1 / 100.0)  # Set the simulation frequency

        self.scene.set_ambient_light([0.5, 0.5, 0.5])
        self.scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
        self.scene.add_point_light([1, 2, 2], [1, 1, 1])
        self.scene.add_point_light([1, -2, 2], [1, 1, 1])
        self.scene.add_point_light([-1, 0, 1], [1, 1, 1])
        
        
        self.create_viewer()
        self.scene.add_ground(altitude=0)  # Add a ground

        # Where the meshes, URDFs and info.json live -- kept in config.yaml.
        self.asset_dir = config.get("paths", "asset_dir")
        obj_info_json = os.path.join(self.asset_dir, "info.json")
        self.obj_info_dic = json.load(open(obj_info_json,'r'))
        self.obj_root_folder = os.path.join(self.asset_dir, "objects")
        self.OCCUPANCY_MAP = {}
        self.table_range = [0.15, 0.5, -0.35, 0.35]  # xmin, xmax, ymin, ymax
        self.timestep = self.scene.get_timestep()

    def create_viewer(self):
        """Open the on-screen viewer, or do nothing when running headless.

        A no-op if visualize=False, so the tasks can ask for a viewer to replay
        a solution without breaking a run on a machine with no display. An
        already-open viewer is reused rather than duplicated.
        """
        if not self.visualize:
            return None
        if self.viewer_open:
            return self.viewer

        viewer = self.scene.create_viewer()  # Create a viewer (window)
        viewer.set_camera_xyz(
            x=VIEW_CAMERA_POSITION[0],
            y=VIEW_CAMERA_POSITION[1],
            z=VIEW_CAMERA_POSITION[2],
        )
        viewer.set_camera_rpy(
            r=VIEW_CAMERA_RPY[0],
            p=VIEW_CAMERA_RPY[1],
            y=VIEW_CAMERA_RPY[2],
        )
        viewer.window.set_camera_parameters(
            near=VIEW_CAMERA_NEAR,
            far=VIEW_CAMERA_FAR,
            fovy=VIEW_CAMERA_FOVY,
        )
        viewer.control_window.toggle_camera_lines(False)
        viewer.window.resize(*VIEW_CAMERA_SIZE)
        self.viewer = viewer
        return self.viewer

    def create_recording_camera(self):
        """Create a rasterized offscreen camera matching the viewer's initial pose."""
        roll, pitch, yaw = VIEW_CAMERA_RPY
        forward = np.array([1.0, 0.0, 0.0])
        left = np.array([0.0, 1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        quat_wxyz = qmult(
            qmult(axangle2quat(up, -yaw), axangle2quat(left, -pitch)),
            axangle2quat(forward, roll),
        )

        sapien.render.set_camera_shader_dir("default")
        try:
            self.recording_camera = self.scene.add_camera(
                name="recording_camera",
                width=VIEW_CAMERA_SIZE[0],
                height=VIEW_CAMERA_SIZE[1],
                fovy=VIEW_CAMERA_FOVY,
                near=VIEW_CAMERA_NEAR,
                far=VIEW_CAMERA_FAR,
            )
        finally:
            # Keep subsequently created perception cameras on the ray-tracing pipeline.
            sapien.render.set_camera_shader_dir("rt")

        self.recording_camera.entity.set_pose(
            sapien.Pose(p=VIEW_CAMERA_POSITION, q=quat_wxyz)
        )
        return self.recording_camera

    @property
    def viewer_open(self) -> bool:
        """Whether there is a viewer to render into."""
        return self.viewer is not None and not self.viewer.closed

    def close_viewer(self):
        if self.viewer_open:
            self.viewer.close()

    def render_viewer(self, record_img: bool = True):
        """Capture an offscreen frame and render the viewer when enabled."""
        if record_img and self.recording_active:
            self.video_recorder.capture(self.recording_camera)
        if self.viewer_open:
            self.viewer.render()

    def start_recording(self):
        """Start a fresh recording, creating the offscreen camera lazily."""
        if self.video_recorder is None:
            return
        if self.recording_camera is None:
            self.create_recording_camera()
        self.video_recorder.reset()
        self.recording_active = True

    def reset_recording(self):
        self.recording_active = False
        if self.video_recorder is not None:
            self.video_recorder.reset()

    def save_recording(self, fps: float = 30.0):
        self.recording_active = False
        if self.video_recorder is not None:
            self.video_recorder.save(fps=fps)

    def idle(self, record_img: bool = False):
        """Step and render until the user closes the viewer; returns immediately
        when running headless."""
        while self.viewer_open:
            self.scene.step()
            self.scene.update_render()
            self.render_viewer(record_img=record_img)


    def build_experiment_env(self, test_case_dic, light_object_names: list = [], heavy_object_names: list = []):   
        for obj_name in test_case_dic:
            if "wooden_box" in obj_name:
                box_pos = test_case_dic["wooden_box"]["position"]
                box_rot = test_case_dic["wooden_box"]["quaternion"]
                self.create_wooden_box(pose=box_pos, rotation=box_rot)
            elif "support" in obj_name:
                support_pos = test_case_dic["support"]["position"]
                self.add_cylinder(support_pos)
            else:
                obj_pos = test_case_dic[obj_name]["position"]
                obj_rot = test_case_dic[obj_name]["quaternion"]
                static = test_case_dic[obj_name]["static"]
                if obj_name in light_object_names:
                    density = 10
                elif obj_name in heavy_object_names:
                    density = 10000
                else:
                    density = 1000 # default
                self.load_actor_from_mesh(
                    name=obj_name,
                    pose=obj_pos,
                    rotation=obj_rot,
                    adjust_height=False,
                    density=density,
                    static=static
                )

        self.simulate(time=2.0, record_img=False) # let the objects settle down

    def export_scene_as_json(self):
        scene_dict = {}
        for obj_name in self.OCCUPANCY_MAP.keys():
            static = self.OCCUPANCY_MAP[obj_name]["static"]
            actor = self.OCCUPANCY_MAP[obj_name]["actor"]
            pose = actor.get_pose()
            pos = pose.p.tolist()
            quat = pose.q.tolist()
            scene_dict[obj_name] = {
                "position": pos,
                "quaternion": quat,
                "static": static
            }
        return scene_dict


    def create_table(self):
        self.table_model_path = os.path.join(self.asset_dir, "table/table.glb")
        self.table_length = 1.38
        self.table_width = 0.691
        self.table_height = 0.525

        builder = self.scene.create_actor_builder()

        visual_offset = sapien.Pose(p=[0, 0, -self.table_height/2])
        builder.add_visual_from_file(filename=self.table_model_path, scale=[1.0,1.0,1.0], pose=visual_offset)

        collision_pose = sapien.Pose(p=[0, 0, 0])
        builder.add_box_collision(collision_pose, [self.table_width/2, self.table_length/2, self.table_height/2])

        self.table = builder.build_static(name="table")
        self.table.set_pose(sapien.Pose(p=[self.table_width/2, 0, self.table_height/2]))
    
    def create_frame(self, pose=None,rotation=0):
        actor = self.load_actor_from_mesh("frame_0", pose=pose, rotation=rotation, static=True)
        return actor
    
    def create_wooden_box(self, pose=None, rotation=0):
        collision_model_path = os.path.join(self.asset_dir, "table/wooden_box_vhacd.obj")
        visual_model_path = os.path.join(self.asset_dir, "table/wooden_box.glb")
        scale = [0.25, 0.4, 0.25]
        self.wooden_box_height = 0.16*scale[0]
        density = 1000

        if isinstance(rotation, list):  #provide quaternion directly
            quat_wxyz = np.array(rotation)
        else:  #provide rotation angle around z axis
            rotation_angle = rotation
            quat_wxyz = np.array([0.707, 0.707, 0, 0]) #default quat
            qz_xyzw = R.from_euler("z", rotation_angle, degrees=True).as_quat()
            base_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            new_xyzw = (R.from_quat(qz_xyzw) * R.from_quat(base_xyzw)).as_quat()
            quat_wxyz = np.array([new_xyzw[3], new_xyzw[0], new_xyzw[1], new_xyzw[2]])

        if pose is not None:
            pos_np = np.array(pose)
            rect = compute_xy_rect(collision_model_path, scale, pos_np, quat_wxyz)
            default_pose = sapien.Pose(p=pos_np, q=quat_wxyz)
            actor = self._build_actor("wooden_box", collision_model_path, visual_model_path, scale, default_pose, density, static=True)
            self.OCCUPANCY_MAP["wooden_box"] = {"bbox": rect, "actor": actor, "static": True}
            return actor
        
        z = self.table_height + 0.1
        max_trials = 200
        x_min, x_max, y_min, y_max = self.table_range

        for _ in range(max_trials):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            pos_np = np.array([x, y, z], dtype=float)
            new_rect = compute_xy_rect(collision_model_path, scale, pos_np, quat_wxyz)

            collision = False #Collision checking
            for obj in self.OCCUPANCY_MAP.keys():
                if rect_intersect(new_rect, self.OCCUPANCY_MAP[obj]["bbox"]):
                    collision = True
                    break
            
            if not collision:
                default_pose = sapien.Pose(p=pos_np, q=quat_wxyz)
                actor = self._build_actor("wooden_box", collision_model_path, visual_model_path, scale, default_pose, density, static=True)
                self.OCCUPANCY_MAP["wooden_box"] = {"bbox": new_rect,  "actor": actor, "static": True}
                return actor

        return actor
    

    def add_cylinder(
        self,
        pose,
        radius=0.02,
        half_length=0.075,
        color=None,
        name="support",
    ) -> sapien.Entity:

        builder = self.scene.create_actor_builder()

        pose[2] = self.table_height+half_length # on the table
        sapien_pose = sapien.Pose(p=pose, q=[0, np.sin(np.pi/4), 0, np.cos(np.pi/4)])

        material = sapien.physx.get_default_material()
        material.restitution = 0.05

        builder.add_cylinder_collision(
            radius=radius,
            half_length=half_length,
            material=material
        )

        builder.add_cylinder_visual(
            radius=radius,
            half_length=half_length,
            material=color,
        )

        cyl = builder.build_static(name=name)
        cyl.set_pose(sapien_pose)
        rect = [pose[0]-radius, pose[1]-radius, pose[0]+radius, pose[1]+radius]

        self.OCCUPANCY_MAP["support"] = {"bbox": rect,  "actor": cyl, "static": True}
        return cyl

    def create_robot_base(self):
        self.base_size = 0.3
        self.base_height = self.table_height
        box_pose = sapien.Pose([-0.3, 0, self.base_height/2])
        box_half_size = [self.base_size/2, self.base_size/2, self.base_height/2]
        self.create_static_box(box_half_size, box_pose, "robot_base")

    def create_static_box(self, half_size, pose, name, color=(0.95, 0.95, 0.95)):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=half_size)
        builder.add_box_visual(half_size=half_size, material=color)
        box = builder.build_static(name=name)
        box.set_pose(pose)
        return box
    
    def load_robot(self):
        if ROBOT_NAME == 'panda':
            self.home_joints = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        elif ROBOT_NAME == 'fr3':
            self.home_joints = [-0.0030637685314503847, -0.7778934457717781, 0.01585533499243391, -2.3693353790401894, -0.004060005459262234, 1.554097367289531, 0.7704814906387026]
        else:
            raise ValueError(f"Unsupported ROBOT_NAME: {ROBOT_NAME}")
        self.gripper_open_joint = [0.04, 0.04]
        self.gripper_closed_joint = [0.01, 0.01]
        robot_base_pose = [-0.3, 0, self.base_height]
        self.robot = self.load_franka_arm(robot_base_pose)
        self.end_effector = self.robot.find_link_by_name(f"{ROBOT_NAME}_hand")
        self.drive = None
        self._attached_solver_state = None

    def load_franka_arm(self, robot_base_pose):
        urdf_path = os.path.join(self.asset_dir, f"robots/{ROBOT_NAME}/{ROBOT_NAME}.urdf")
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        robot = loader.load(urdf_path)
        robot.set_root_pose(sapien.Pose(robot_base_pose, [1, 0, 0, 0])) 

        initial_joints = self.home_joints + self.gripper_open_joint
        robot.set_qpos(initial_joints)
        
        active_joints = robot.get_active_joints()
        settings = config.get_config().get("simulation", {})
        arm_stiffness = float(settings.get("arm_stiffness", 10000))
        arm_damping = float(settings.get("arm_damping", 300))
        arm_force_limit = float(settings.get("arm_force_limit", 100))
        position_iterations = int(settings.get("solver_position_iterations", 32))
        velocity_iterations = int(settings.get("solver_velocity_iterations", 2))
        if not (1 <= position_iterations <= 255 and 1 <= velocity_iterations <= 255):
            raise ValueError("Simulation solver iteration counts must be between 1 and 255")
        # Resolve the coupled arm, grasp constraint, and contacts accurately.
        robot.solver_position_iterations = position_iterations
        robot.solver_velocity_iterations = velocity_iterations

        gripper_stiffness = 1e3
        gripper_damping   = 1e2
        gripper_force     = 100  

        for j, joint in enumerate(active_joints[:7]):
            joint.set_drive_property(stiffness=arm_stiffness,
                                    damping=arm_damping,
                                    force_limit=arm_force_limit)
            joint.set_drive_target(initial_joints[j])
        
        for j, joint in enumerate(active_joints[7:]):
            joint.set_drive_property(stiffness=gripper_stiffness,
                                    damping=gripper_damping,
                                    force_limit=gripper_force)
            joint.set_drive_target(initial_joints[7 + j])
        
        return robot
    
    def load_actor_from_mesh(self, name, pose=None, rotation=0, density=1000, static=False, adjust_height=True, z_offset=0.0):
        """
        - If pose AND rotation angle are both given → use them directly.
        - Otherwise → sample a collision-free pose + using default quat using occupancy.
        """
        obj_category = name.split("_")[0] # Extract scale settings
        default_quat = self.obj_info_dic[name][0]
        scale_factor = self.obj_info_dic[name][1]

        scale = [scale_factor] * 3
        collision_model_path = f"{self.obj_root_folder}/{obj_category}/{name}.obj"
        visual_model_path = f"{self.obj_root_folder}/{obj_category}/{name}.glb"

        if isinstance(rotation, list):  #provide quaternion directly
            quat_wxyz = np.array(rotation)
        else:  #provide rotation angle around z axis
            quat_wxyz = np.array(default_quat)
            rotation_angle = rotation
            qz_xyzw = R.from_euler("z", rotation_angle, degrees=True).as_quat()
            base_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            new_xyzw = (R.from_quat(qz_xyzw) * R.from_quat(base_xyzw)).as_quat()
            quat_wxyz = np.array([new_xyzw[3], new_xyzw[0], new_xyzw[1], new_xyzw[2]])
        
        if pose is not None:
            pos_np = np.array(pose)
            rect = compute_xy_rect(collision_model_path, scale, pos_np, quat_wxyz)
            default_pose = sapien.Pose(p=pos_np, q=quat_wxyz)
            actor = self._build_actor(name, collision_model_path, visual_model_path, scale, default_pose, density, static, adjust_height, z_offset=z_offset)

            self.OCCUPANCY_MAP[name] = {"bbox": rect, "actor": actor, "static": static}
            return actor
        
        z = self.table_height + scale_factor + 0.01
        max_trials = 200
        x_min, x_max, y_min, y_max = self.table_range

        for _ in range(max_trials):
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(y_min, y_max)
            pos_np = np.array([x, y, z], dtype=float)
            new_rect = compute_xy_rect(collision_model_path, scale, pos_np, quat_wxyz)

            collision = False #Collision checking
            for obj in self.OCCUPANCY_MAP.keys():
                if rect_intersect(new_rect, self.OCCUPANCY_MAP[obj]["bbox"]):
                    collision = True
                    break
            
            if not collision:
                default_pose = sapien.Pose(p=pos_np, q=quat_wxyz)
                actor = self._build_actor(name, collision_model_path, visual_model_path, scale, default_pose, density, static, adjust_height, z_offset=z_offset)
                self.OCCUPANCY_MAP[name] = {"bbox": new_rect, "actor": actor, "static": static}
                return actor

        print(f"Failed to place {name} without collision after {max_trials} samples.")
        return None
    
    def load_actor_from_mesh_with_clamps(
        self,
        name,
        pose=None,
        rotation=None,
        density=1000,
        static=False,
        clamp_half_size=(0.0025, 0.02, 0.01),
        clamp_margin=0.01,
        clamp_z_lift=0.002,
    ):
        actor = self.load_actor_from_mesh(
            name=name,
            pose=pose,
            rotation=rotation,
            density=density,
            static=static,
        )

        if actor is None:
            return None
        actor_mesh = get_actor_mesh(actor, to_world_frame=True, visual=False)
        o3d_mesh = trimesh_to_open3d_mesh(actor_mesh)

        obb = o3d_mesh.get_oriented_bounding_box()
        center = np.asarray(obb.center)
        R_obb = np.asarray(obb.R)          # columns = local axes
        extents = np.asarray(obb.extent) * 0.5  # half extents

        axes_xy = []
        for i in range(3):
            axis = R_obb[:, i].copy()
            axis[2] = 0.0
            n = np.linalg.norm(axis)
            if n > 1e-6:
                axes_xy.append((i, axis / n, extents[i]))

        # Long axis = blade direction
        long_axis_idx, long_dir_xy, _ = max(axes_xy, key=lambda x: x[2])

        # Short axis = perpendicular (clamp squeeze)
        short_dir_xy = np.array([-long_dir_xy[1], long_dir_xy[0]])
        short_dir_xy /= np.linalg.norm(short_dir_xy)

        half_width = min(extents)
        clamp_distance = half_width + clamp_margin + clamp_half_size[0]
        print("clamp distance:", clamp_distance)

        clamp_xy_1 = center[:2] + short_dir_xy * clamp_distance
        clamp_xy_2 = center[:2] - short_dir_xy * clamp_distance

        yaw = np.arctan2(long_dir_xy[1], long_dir_xy[0]) + np.pi / 2
        q_xyzw = R.from_euler("z", yaw).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        z = self.table_height + clamp_half_size[2] + clamp_z_lift

        clamps = []
        for i, (x, y) in enumerate([clamp_xy_1, clamp_xy_2]):
            clamp_pose = sapien.Pose(
                p=[x, y, z],
                q=q_wxyz,
            )
            clamp = self.create_static_box(
                half_size=clamp_half_size,
                pose=clamp_pose,
                name=f"{name}_clamp_{i}",
                color=(0.8, 0.8, 0.8),
            )
            clamps.append(clamp)

        self.OCCUPANCY_MAP[name]["clamps"] = clamps
        return actor

    def remove_actor(self, name):
        if name in self.OCCUPANCY_MAP.keys():
            actor = self.OCCUPANCY_MAP[name]["actor"]
            self.scene.remove_actor(actor)

            if "clamps" in self.OCCUPANCY_MAP[name]:
                clamps = self.OCCUPANCY_MAP[name]["clamps"]
                for clamp in clamps:
                    self.scene.remove_actor(clamp)

            del self.OCCUPANCY_MAP[name]
            print(f"Removed actor and clamps: {name}")
        else:
            print(f"{name} not found.")
    
    def move_to_home(self):
        self.move_to_joint(self.home_joints)
        self.simulate(time=1.0) # wait for 1 seconds to let the robot move to home position

    def move_to_joint(self, joint_pos, joint_vel=None):
        assert len(joint_pos) == 7
        qf = self.robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        self.robot.set_qf(qf)
        active_joints = self.robot.get_active_joints()
        for joint_idx, joint in enumerate(active_joints[:7]):
            joint.set_drive_target(joint_pos[joint_idx])
            if joint_vel is not None:
                joint.set_drive_velocity_target(joint_vel[joint_idx])

    def follow_path(self, traj, render_every=10, final_wait=0):
        '''
        Trajetory format: {
            "position": np.ndarray of shape (n_step, 7),
            "velocity": np.ndarray of shape (n_step, 7)
        }
        '''
        print("Following arm trajectory with {} steps.".format(traj["position"].shape[0]))
        for i in range(traj["position"].shape[0]):
            self.move_to_joint(
                joint_pos=traj["position"][i],
                joint_vel=traj["velocity"][i]
            )
            self.scene.step()
            if i % render_every == 0:
                self.scene.update_render()
                self.render_viewer()

        if final_wait > 0:
            self.simulate(time=final_wait)  # wait for final_wait seconds after following the path

    def open_gripper(self):
        active_joints = self.robot.get_active_joints()
        for joint_idx, joint in enumerate(active_joints[7:]):
            joint.set_drive_target(self.gripper_open_joint[joint_idx])

        self.simulate(time=1) # wait for 1 seconds to let the gripper open

    def close_gripper(self):
        active_joints = self.robot.get_active_joints()
        for joint_idx, joint in enumerate(active_joints[7:]):
            joint.set_drive_target(self.gripper_closed_joint[joint_idx])

        self.simulate(time=1) # wait for 1 seconds to let the gripper close

    @staticmethod
    def lock_drive(drive: sapien.physx.PhysxDriveComponent):
        drive.set_drive_property_x(stiffness=1e9, damping=1e8, force_limit=1e10)
        drive.set_drive_property_y(stiffness=1e9, damping=1e8, force_limit=1e10)
        drive.set_drive_property_z(stiffness=1e9, damping=1e8, force_limit=1e10)
        # Angular limits alone do not lock every rotational axis in SAPIEN.
        # Keep orientation drives stiff, with damping tuned for the 10 ms step.
        settings = config.get_config().get("simulation", {})
        angular_damping = float(settings.get("attachment_angular_damping", 10000))
        drive.set_drive_property_twist(stiffness=1e9, damping=angular_damping, force_limit=1e10)
        drive.set_drive_property_slerp(stiffness=1e9, damping=angular_damping, force_limit=1e10)
        drive.set_drive_property_swing(stiffness=1e9, damping=angular_damping, force_limit=1e10)
        drive.set_limit_x(0, 0)
        drive.set_limit_y(0, 0)
        drive.set_limit_z(0, 0)
        drive.set_limit_twist(0, 0)
        drive.set_limit_cone(0, 0)
        drive.set_limit_pyramid(0, 0, 0, 0)
        drive.set_drive_velocity_target([0,0,0], [0,0,0])
        drive.set_drive_target(sapien.Pose())
        
    def attach_object(self, attached_object: sapien.Entity):
        assert self.drive is None, "Drive already exists for the arm."
        drive = self.scene.create_drive(
            self.end_effector, sapien.Pose(),
            attached_object, attached_object.pose.inv() * self.end_effector.entity_pose
        )
        self.lock_drive(drive)
        body = attached_object.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
        if body is not None:
            self._attached_solver_state = (
                body, body.solver_position_iterations, body.solver_velocity_iterations
            )
            body.solver_position_iterations = max(
                body.solver_position_iterations, self.robot.solver_position_iterations
            )
            body.solver_velocity_iterations = max(
                body.solver_velocity_iterations, self.robot.solver_velocity_iterations
            )
        self.drive = drive

    def detach_object(self):
        assert self.drive is not None, "No drive exists for the arm."
        self.drive.disable()
        self.drive = None
        if self._attached_solver_state is not None:
            body, position_iterations, velocity_iterations = self._attached_solver_state
            body.solver_position_iterations = position_iterations
            body.solver_velocity_iterations = velocity_iterations
            self._attached_solver_state = None

    def get_grasping_actor(self, grasp_image):
        camera = self.front_camera
        self.scene.update_render()
        camera.take_picture()
        seg_labels = camera.get_picture("Segmentation")
        actor_label = seg_labels[..., 1].astype(np.uint8)
        table_id = self.table.get_global_id()
        actor_label[actor_label == table_id] = 0  # Remove table from segmentation
        
        if 'wooden_box' in self.OCCUPANCY_MAP:
            wooden_box_id = self.OCCUPANCY_MAP['wooden_box']['actor'].get_global_id()
            actor_label[actor_label == wooden_box_id] = 0  # Remove wooden box from segmentation

        grasp_aff_map = np.array(grasp_image).astype(np.uint8)
        if np.sum(grasp_aff_map) == 0:
            print("No grasp detected.")
            return None
        grasp_actor_map = actor_label * grasp_aff_map
        values, counts = np.unique(grasp_actor_map, return_counts=True)
        mask = values != 0
        values = values[mask]
        counts = counts[mask]
        grasp_actor_id = values[np.argmax(counts)]

        for obj_name in self.OCCUPANCY_MAP.keys():
            actor = self.OCCUPANCY_MAP[obj_name]["actor"]
            if actor.get_global_id() == grasp_actor_id:
                print(f"Grasping actor is {obj_name}")
                return actor
        
        return None
    
    def get_obj_pointcloud(self, obj_name):
        obj_mask = self.get_obj_mask(obj_name)
        self.scene.update_render()
        _, _, front_pointmap, _ = self.render_camera_image(self.front_camera)
        points_xyz = front_pointmap[obj_mask > 0]
        
        N = points_xyz.shape[0]
        colors  = np.tile(np.array([[0., 0., 1.]]), (N, 1))
        normals = np.zeros((N, 3), dtype=np.float64)
        points_full = np.concatenate([points_xyz, colors, normals], axis=1)
        pc_o3d = numpy_to_o3d(points_full)
        
        # import open3d as o3d
        # o3d.visualization.draw_geometries([pc_o3d])
        return pc_o3d


    
    def get_obj_mask(self, obj_name):
        camera = self.front_camera
        self.scene.update_render()
        camera.take_picture()
        seg_labels = camera.get_picture("Segmentation")
        actor_label = seg_labels[..., 1].astype(np.uint8)

        actor = self.OCCUPANCY_MAP[obj_name]["actor"]
        actor_id = actor.get_global_id()
        obj_mask = (actor_label == actor_id).astype(np.uint8)
        # height, width = actor_label.shape
        # affordance_feature_map = np.zeros((height, width, 3), dtype=np.uint8)
        # affordance_feature_map[:, :, 2] = obj_mask * 255  # R channel
        return obj_mask

    def _build_actor(self, name, collision_model_path, visual_model_path, scale=np.array([1.0]*3), default_pose=sapien.Pose(), density=1000, static=False, adjust_height=True, z_offset=0.0):
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(filename=collision_model_path, scale=scale, density=density)
        builder.add_visual_from_file(filename=visual_model_path, scale=scale)
        if static:
            actor = builder.build_static(name=name)
        else:
            actor = builder.build(name=name)
        actor.set_pose(default_pose)
        actor_mesh = get_actor_mesh(actor, to_world_frame=True, visual=False)
        z_min = actor_mesh.bounds[0][2]
        if adjust_height:
            height = self.table_height - z_min
            adjusted_pose = sapien.Pose(p=[0, 0, height+z_offset]) * default_pose
            actor.set_pose(adjusted_pose)
        return actor
    
    def camera_initializition(self, random_view=False):
        if random_view:
            random_seed = time.time_ns() % (2**32)
            print(f"Using random seed: {random_seed}")
            rng = np.random.default_rng(random_seed)
            random_index = rng.integers(0, 4)  
        else:
            random_index = 0
        front_camera_pose_0 = sapien.Pose(p=[1.05, 0, 1.47], q=[-0.00218263, 0.487585, -0.00121894, -0.873072])
        front_camera_pose_1 = sapien.Pose([1.79934, 0.0314925, 0.971168], [-0.00740078, -0.162038, -0.00121532, 0.986756])
        front_camera_pose_2 = sapien.Pose([0.359889, -0.598236, 1.09493], [0.656825, -0.260277, 0.25968, 0.658334])
        front_camera_pose_3 = sapien.Pose([0.359889, 0.598236, 1.09493], [-0.656825, -0.260277, -0.25968, 0.658334])

        front_camera_pose_list = [front_camera_pose_0, front_camera_pose_1, front_camera_pose_2, front_camera_pose_3]

        self.front_camera = self.add_L515_camera(name="front_camera", pose=front_camera_pose_list[random_index])


    def add_L515_camera(self, name, pose):
        camera = self.scene.add_camera(name=name, width=1280, height=720, fovy=0.75049, near=0.25, far=100)
        camera.entity.set_pose(pose)
        return camera
    
    def get_observations(self):
        observations = {}
        observations["front_intrinsic"] = self.front_camera.get_intrinsic_matrix()
        self.scene.update_render()
        front_image, front_depth, front_pointmap, front_pointmap_cam = self.render_camera_image(self.front_camera)
        observations["front_color"] = front_image
        observations["front_depth"] = front_depth
        observations["front_pointmap"] = front_pointmap
        observations["front_pointmap_cam"] = front_pointmap_cam
        observations["front_extrinsic"] = self.front_camera.get_model_matrix()
        return observations
      
    def render_camera_image(self, camera):
        camera.take_picture()
        rgba = camera.get_picture("Color")  
        rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
        rgba_pil = Image.fromarray(rgba_img)
        rgba_pil = rgba_pil.convert("RGB") # ensure 3 channels
        # ----- Camera-frame point map -----
        position = camera.get_picture("Position")
        xyz_cam = copy.deepcopy(position[..., :3])                 # (H, W, 3)

        # ----- Transform to world frame -----
        cam2world = camera.get_model_matrix()       # 4×4 matrix
        R = cam2world[:3, :3]
        t = cam2world[:3, 3]

        H, W, _ = xyz_cam.shape
        xyz_world = xyz_cam.reshape(-1, 3) @ R.T + t
        xyz_world = xyz_world.reshape(H, W, 3)

        depth = copy.deepcopy(-position[..., 2])
        # depth_image = (depth * 1000.0).astype(np.uint16)
        # depth_pil = Image.fromarray(depth_image)
        
        return rgba_pil, depth, xyz_world, xyz_cam
    
    def simulate(self, time=1, record_img: bool = True, render_every=4): #seconds
        steps = int(time/self.timestep)
        for step in range(steps):
            self.scene.step()
            self.scene.update_render()
            if step % render_every == 0:
                self.render_viewer(record_img=record_img)
    
    def get_actor_by_name(self, name: str) -> sapien.Entity:
        for actor in self.scene.get_all_actors():
            if actor.name == name:
                return actor
        raise ValueError(f"Actor with name {name} not found.")
    
    def get_articulation_by_name(self, name: str) -> sapien.physx.PhysxArticulation:
        for articulation in self.scene.get_all_articulations():
            if articulation.name == name:
                return articulation
        raise ValueError(f"Articulation with name {name} not found.")
    
    def pack(self) -> dict:
        state_dict = dict()
        state_dict["actors"] = dict()
        state_dict["articulations"] = dict()
        state_dict["articulation_drives"] = dict()
        state_dict["actor_velocities"] = dict()
        for actor in self.scene.get_all_actors():
            state_dict["actors"][actor.name] = actor.pose
            body = actor.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
            if body is not None:
                state_dict["actor_velocities"][actor.name] = (
                    body.linear_velocity.copy(), body.angular_velocity.copy()
                )
        for articulation in self.scene.get_all_articulations():
            joint_positions = articulation.get_qpos()
            state_dict["articulations"][articulation.name] = joint_positions
            joints = articulation.get_active_joints()
            state_dict["articulation_drives"][articulation.name] = {
                "qvel": articulation.get_qvel().copy(),
                "targets": [joint.get_drive_target().copy() for joint in joints],
                "velocities": [joint.get_drive_velocity_target().copy() for joint in joints],
            }
        if len(state_dict["actors"]) == 0:
            del state_dict["actors"]
        if len(state_dict["articulations"]) == 0:
            del state_dict["articulations"]
        return state_dict
           
    def unpack(self, state_dict: dict):
        if "actors" in state_dict:
            for actor_name, pose in state_dict["actors"].items():
                actor = self.get_actor_by_name(actor_name)
                actor.set_pose(pose)
                comp: sapien.physx.PhysxRigidBodyComponent = actor.find_component_by_type(sapien.physx.PhysxRigidBodyComponent)
                if comp is not None:
                    linear, angular = state_dict.get("actor_velocities", {}).get(
                        actor_name, (np.zeros(3), np.zeros(3))
                    )
                    comp.linear_velocity = linear
                    comp.angular_velocity = angular
        if "articulations" in state_dict:
            for articulation_name, joint_positions in state_dict["articulations"].items():
                articulation = self.get_articulation_by_name(articulation_name)
                articulation.set_qpos(joint_positions)
                drives = state_dict.get("articulation_drives", {}).get(articulation_name)
                articulation.set_qvel(drives["qvel"] if drives else np.zeros_like(joint_positions))
                for i, joint in enumerate(articulation.get_active_joints()):
                    joint.set_drive_target(drives["targets"][i] if drives else joint_positions[i])
                    joint.set_drive_velocity_target(drives["velocities"][i] if drives else 0.0)

if __name__ == "__main__":
    simu_environment = simu_env()
    teapot = simu_environment.load_actor_from_mesh(
        name="teapot_1",
        pose=[0.45, -0.2, 0.6],
        rotation= 90,
        density=100
    )
    screwdriver = simu_environment.load_actor_from_mesh(
        name="screwdriver_2",
        pose=[0.25, -0.4, 0.6],
        rotation= -90,
    )
    simu_environment.simulate(time=2.0) # let the objects settle down
    obs = simu_environment.get_observations()

    # Simulation loop
    simu_environment.idle()

    # obs = simu_environment.get_observations()
    # obs["front_color"].save("front_color.png")
    # simu_environment.export_scene_as_json()
    
