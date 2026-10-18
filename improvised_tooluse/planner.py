from contextlib import contextmanager
import numpy as np
import sapien
from improvised_tooluse.sapien_utils import get_actor_mesh
from improvised_tooluse.simu_env import ROBOT_NAME, simu_env
from mplib.sapien_utils.conversion import SapienPlanningWorld, convert_object_name
from sapien.pysapien.physx import PhysxArticulation
from sapien import Pose
import mplib
from improvised_tooluse.screw_planner import CheckedSapienPlanner

class PandaPlanner:
    def __init__(self, env: simu_env):
        self.env = env
        self.robot: PhysxArticulation = env.robot
        self.planning_world = SapienPlanningWorld(self.env.scene, [self.robot])
        self.timestep = env.timestep
        robot_base_name = None
        for name in self.planning_world.get_object_names():
            if 'robot_base' in name:
                robot_base_name = name
                break
        self.planning_world.get_allowed_collision_matrix().set_entry(f"{ROBOT_NAME}_link0", robot_base_name, True)
        self.planner = CheckedSapienPlanner(
            self.planning_world, 
            f'{ROBOT_NAME}_hand',
            joint_vel_limits=np.ones(7) * 0.3,
            joint_acc_limits=np.ones(7) * 0.3,
        )
        self.attached_object = None
    
    def update_from_simulation(self):
        self.planning_world.update_from_simulation()
        
    def add_pcd_collision(self, pcd: np.ndarray, resolution: float = 1e-3, name: str = 'scene_pcd'):
        self.planner.update_point_cloud(pcd, resolution, name)
    
    def remove_pcd_collision(self, name: str = 'scene_pcd'):
        self.planner.remove_point_cloud(name)

    @contextmanager
    def ignore_collisions(self):
        acm = self.planning_world.get_allowed_collision_matrix()
        original_entries = {}
        for name1 in acm.get_all_entry_names():
            for name2 in acm.get_all_entry_names():
                original_entries[(name1, name2)] = acm.get_entry(name1, name2)

        eef_link_names = [f'{ROBOT_NAME}_hand', f'{ROBOT_NAME}_leftfinger', f'{ROBOT_NAME}_rightfinger']
        if self.attached_object is not None:
            eef_link_names.append(self.attached_object)
        env_obj_names = self.planning_world.get_object_names()

        for eef_link_name in eef_link_names:
            for env_obj_name in env_obj_names:
                acm.set_entry(eef_link_name, env_obj_name, True)
        try:
            yield
        finally:
            acm.clear()
            for (name1, name2), allowed in original_entries.items():
                if allowed is not None:
                    acm.set_entry(name1, name2, allowed.value)
        
    def _move_to_pose(self, pose: Pose, use_screw: bool = False, tolerance: float = 1e-3, qpos_step: float = 0.1, verbose: bool = False, speed_up_factor: float = None, return_traj: bool = False):
        if speed_up_factor is not None:
            current_vel_limits = self.planner.joint_vel_limits
            current_acc_limits = self.planner.joint_acc_limits
            self.planner.joint_vel_limits = speed_up_factor * current_vel_limits
            self.planner.joint_acc_limits = speed_up_factor * current_acc_limits
        
        if use_screw:
            result = self.planner.plan_screw(pose, self.robot.get_qpos(), time_step=self.timestep, qpos_step=qpos_step, verbose=verbose)
        else: # rrt-connect
            result = self.planner.plan_pose(pose, self.robot.get_qpos(), time_step=self.timestep, constraint_tolerance=tolerance, verbose=verbose, simplify=True)
        
        if speed_up_factor is not None:
            self.planner.joint_vel_limits = current_vel_limits
            self.planner.joint_acc_limits = current_acc_limits

        if result["status"] != "Success":
            print(result["status"])
            return -1 if not return_traj else ( -1, None)

        self.env.follow_path(result)
        return 0 if not return_traj else (0, result)

    def move_to_pose(
        self, 
        pose: Pose, 
        use_screw: bool = False, 
        tolerance: float = 1e-3, 
        qpos_step: float = 0.1, 
        verbose: bool = False, 
        ignore_collisions: bool = False, 
        speed_up_factor: float = None,
        return_traj: bool = False
    ):
        if ignore_collisions:
            with self.ignore_collisions():
                return self._move_to_pose(pose, use_screw, tolerance, qpos_step, verbose, speed_up_factor=speed_up_factor, return_traj=return_traj)
        else:
            return self._move_to_pose(pose, use_screw, tolerance, qpos_step, verbose, speed_up_factor=speed_up_factor, return_traj=return_traj)
    
    def attach_object(self, attached_object: sapien.Entity):
        # These links intentionally contact a grasped tool. Inferring touch links
        # from overlap at attachment can omit a finger that touches it later.
        touch_links = [f'{ROBOT_NAME}_hand', f'{ROBOT_NAME}_leftfinger',
                       f'{ROBOT_NAME}_rightfinger']
        self.planning_world.attach_object(
            attached_object, self.robot,
            self.robot.find_link_by_name(f'{ROBOT_NAME}_hand'),
            touch_links=touch_links,
        )
        self.attached_object = convert_object_name(attached_object)
    
    def detach_object(self):
        assert self.attached_object is not None, "No object is attached."
        self.planning_world.detach_object(self.attached_object)
        self.attached_object = None
    

def main():
    simu_environment = simu_env()
    can = simu_environment.load_actor_from_mesh(
        name="mug_1",
        pose=[0.4, 0.1, 0.6],
        quat=[0,0,0.707,0.707],
    )
    
    # add obstacle for collision avoidance test
    builder = simu_environment.scene.create_actor_builder()
    builder.add_box_collision(Pose(), [0.05, 0.1, 0.1])
    builder.add_box_visual(Pose(), [0.02, 0.1, 0.1], material=[1,0,0])
    obstacle = builder.build_static(name="obstacle")
    obstacle.set_pose(Pose([0.25, 0, 0.525 + 0.05]))

    simu_environment.load_actor_from_mesh(name="fork_1")   
    simu_environment.load_actor_from_mesh(name="screwdriver_2")
    for _ in range(100): # wait for a while to let the objects settle
        simu_environment.scene.step()
        simu_environment.scene.update_render()
    
    # start planning here! please check the usage of the planner below
    planner = PandaPlanner(simu_environment)

    can_bb_min, can_bb_max = get_actor_mesh(can, to_world_frame=True).bounds
    center = (can_bb_min + can_bb_max) / 2
    extents = (can_bb_max - can_bb_min) / 2
    grasp_pose = Pose([center[0], center[1], center[2] + extents[2]-0.035], [0, 1, 0, 0])
    pre_grasp_pose = Pose([0, 0, 0.1]) * grasp_pose
    post_grasp_pose = Pose([0, 0, 0.1]) * grasp_pose
    pre_place_pose = Pose([-0.3, -0.3, 0.]) * post_grasp_pose
    place_pose = Pose([0, 0, -0.05]) * pre_place_pose
    
    planner.move_to_pose(pre_grasp_pose)
    planner.update_from_simulation()

    while True:
        result = planner.move_to_pose(grasp_pose, tolerance=10) # large tolerance because the gap is small
        if result == 0:
            break
    planner.update_from_simulation()

    simu_environment.close_gripper()

    simu_environment.attach_object(can)
    planner.update_from_simulation()
    planner.attach_object(can)  # try commenting this line to see the difference

    planner.move_to_pose(post_grasp_pose)
    planner.update_from_simulation()

    while True:
        result = planner.move_to_pose(pre_place_pose)
        if result == 0:
            break
    planner.update_from_simulation()

    planner.move_to_pose(place_pose)
    planner.update_from_simulation()

    simu_environment.open_gripper()
    planner.update_from_simulation()
    planner.detach_object()
    simu_environment.detach_object()


    simu_environment.idle()

if __name__ == "__main__":
    main()