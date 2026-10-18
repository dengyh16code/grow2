from improvised_tooluse import config
from improvised_tooluse.simu_env import simu_env
from improvised_tooluse.affordance import Perception_module
from improvised_tooluse.geom_utils import visualize_perception_results
from improvised_tooluse.planner import PandaPlanner
import improvised_tooluse.tasks as tasks
import os
import json
import argparse
from pathlib import Path
import traceback


def save_result(filepath, result):
    if filepath is None:
        return
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", default=None,
                           help="config.yaml to run with (default: {})".format(config.default_config_path()))
    argparser.add_argument("--case_id", type=int, required=True, help="test_case_id")
    argparser.add_argument("--load_cache", action='store_true',
                           help="whether to load existing perception results")
    argparser.add_argument("--preview_scene", action='store_true',
                           help="open the scene viewer without running perception or task execution")
    argparser.add_argument("--perception_only", action='store_true',
                           help="whether to run the planning")
    argparser.add_argument("--verbose", action='store_true',
                           help="whether to visualize perception details")
    argparser.add_argument("--render", action='store_true',
                           help="whether to open the simulation viewer (off = fully headless)")
    argparser.add_argument("--record", action='store_true',
                           help="save task execution to trial_<case_id>.mp4 (works headlessly)")
    argparser.add_argument("--result_file", default=None,
                           help="write machine-readable task status for the batch runner")
    return argparser.parse_args()


def run_case(args, result):
    config.load_config(args.config)
    print("loaded config from {}".format(config.config_path()))

    test_case_id = args.case_id
    test_case = os.path.join(config.get("paths", "test_case_dir"),
                             "case_{}.json".format(test_case_id))

    task_info = json.load(open(test_case, 'r'))
    instruction = task_info["instruction"]
    task_type = task_info["task_type"]
    scene_dict = task_info["scene"]
    target_obj = instruction.split("the ", 1)[1]
    target_obj_name = None
    for obj_name in scene_dict.keys():
        if target_obj in obj_name:
            target_obj_name = obj_name
            break
    print("task instruction:{}, target object:{}".format(instruction, target_obj_name))
    if task_type == "lift":
        light_object_names = [target_obj_name]
        # other objects set to heavy
        heavy_object_names = []
        for obj_name in scene_dict.keys():
            if obj_name != target_obj_name:
                heavy_object_names.append(obj_name)
    else:
        light_object_names = []
        heavy_object_names = []
    enable_viewer = args.render or args.preview_scene
    record_path = 'trial_{}.mp4'.format(test_case_id) if args.record else None
    simu_environment = simu_env(record_mp4_path=record_path, visualize=enable_viewer)
    simu_environment.build_experiment_env(scene_dict, light_object_names=light_object_names, heavy_object_names=heavy_object_names)
    result_dir = task_type + "_result"
    os.makedirs(result_dir, exist_ok=True)
    cache_dir = os.path.join(result_dir, "perception_results_{}.pkl".format(test_case_id))
    reconstruction_cache_dir = os.path.join(result_dir, "reconstruction_results_{}.pkl".format(test_case_id))

    if not args.preview_scene:
        detector = Perception_module()
        if args.load_cache and os.path.exists(cache_dir):
            print("Loading existing perception results...")
            detector.load_perception_results(cache_dir, reconstruction_cache_dir)
        else:
            detector.update_observations(simu_environment.get_observations())
            selected_tool_dic = detector.object_and_affordance_selection(instruction)
            result["tool_selection_dict"] = selected_tool_dic
            # Preserve the selection even if reconstruction or native code crashes.
            save_result(args.result_file, result)
            specified_target_object_list = ["screw", "donut", "moth", "cake", "pizza", "bowl", "paperball"]
            target_affordance_point = None
            if target_obj in specified_target_object_list:
                target_affordance_point = simu_environment.get_obj_pointcloud(target_obj_name)
            
            detector.affordance_map_extraction(selected_tool_dic, target_affordance_point,reconstruction_cache_dir)
            detector.affordance_postprocessing()
            detector.grasping_pose_extraction(selected_tool_dic["tool_object"])
            detector.save_perception_results(cache_dir) 
  
        if args.verbose:
            pcd_visualize = visualize_perception_results(detector.obs["front_pointmap"], 
                                        None,
                                        detector.tool_aff_pcd,
                                        detector.target_aff_pcd,
                                        detector.pred_grasps_world,
                                        apply_density = False,
                                        )
            
        if not args.perception_only:
            tool = simu_environment.get_grasping_actor(detector.grasp_image)
            if tool is None:
                print("No valid grasping tool found. Exiting.")
                result["failure_reason"] = "No valid grasping tool found"
            else:
                target = simu_environment.get_actor_by_name(target_obj_name)
                motion_planner = PandaPlanner(simu_environment)
                task = tasks.names[task_type](simu_environment, motion_planner, detector, tool, target, verbose=args.verbose)
                task_status = task.execute_task()
                result["success"] = task_status == 0
                if not result["success"]:
                    result["failure_reason"] = (
                        f"Task execution failed (returned {task_status}); see the case log for planning/evaluation details"
                    )

    if args.preview_scene or args.perception_only:
        result["success"] = True

    if args.preview_scene:
        simu_environment.idle()

    if args.record:
        simu_environment.save_recording()


def main():
    args = parse_args()
    result = {
        "test_case": f"case_{args.case_id}.json",
        "success": False,
        "tool_selection_dict": None,
        "exception_error": None,
        "failure_reason": None,
        "status": "running",
    }
    save_result(args.result_file, result)
    try:
        run_case(args, result)
        result["status"] = "success" if result["success"] else "failed"
    except Exception as error:
        result.update(success=False, status="error",
                      exception_error=f"{type(error).__name__}: {error}",
                      failure_reason=f"{type(error).__name__}: {error}")
        traceback.print_exc()
    finally:
        save_result(args.result_file, result)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
