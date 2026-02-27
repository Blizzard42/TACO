"""
Test script for visualizing joint-space primitive actions.

Executes a fixed sequence of primitive actions (no policy warmup):
  Forward → Down → Right → Left → Up → Retreat
Each primitive runs for primitive_horizon steps (default 50).
"""
import sys
import os
import subprocess

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
import cv2

import yaml
from datetime import datetime
import importlib
import argparse

from generate_episode_instructions import *

from steering.utils import generate_primitives_qpos

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")
    assert os.path.isfile(camera_config_path), "task config file is missing"
    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def annotate_frame(obs, text, color=(0, 0, 255)):
    """Burn text onto the head_camera rgb in the observation dict (in-place).

    The environment writes self.now_obs head_camera rgb to ffmpeg on the next
    take_action call, so modifying it here makes the annotation appear in video.
    """
    img = obs["observation"]["head_camera"]["rgb"]  # (H, W, 3) uint8 RGB
    # OpenCV needs BGR
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Draw a dark background bar for readability
    h, w = img_bgr.shape[:2]
    cv2.rectangle(img_bgr, (0, h - 50), (w, h), (0, 0, 0), -1)
    cv2.putText(img_bgr, text, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Convert back to RGB and write in-place
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    obs["observation"]["head_camera"]["rgb"] = img_rgb


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    tag = usr_args["tag"]

    nudge_distance = usr_args.get("nudge_distance", 0.1)
    primitive_horizon = usr_args.get("primitive_horizon", 50)
    camera_name = usr_args.get("camera_name", "head_camera")

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files")
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise RuntimeError("embodiment items should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    save_dir = Path(f"eval_result/{tag}/primitive_test/{task_name}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    video_save_dir = None
    video_size = None
    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Primitive Test Config =============")
    print(f"Task:            {task_name}")
    print(f"Nudge distance:  {nudge_distance}")
    print(f"Prim horizon:    {primitive_horizon}")
    print(f"Camera:          {camera_name}")
    print(f"Save dir:        {save_dir}")
    print("=================================================\n")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]
    st_seed = 100000 * (1 + seed)

    # ---- Find a valid seed ----
    now_seed = st_seed
    args["eval_mode"] = True

    for attempt in range(200):
        args["render_freq"] = 0
        try:
            TASK_ENV.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **args)
            episode_info = TASK_ENV.play_once()
            TASK_ENV.close_env()
        except UnStableError:
            TASK_ENV.close_env()
            now_seed += 1
            continue
        except Exception as e:
            TASK_ENV.close_env()
            now_seed += 1
            print(e)
            continue

        if TASK_ENV.plan_success and TASK_ENV.check_success():
            break
        now_seed += 1
    else:
        print("ERROR: Could not find a valid seed after 200 attempts.")
        return

    print(f"\nUsing seed {now_seed}\n")

    # ---- Setup episode for evaluation ----
    args["render_freq"] = args.get("render_freq", 0)
    TASK_ENV.setup_demo(now_ep_num=0, seed=now_seed, is_test=True, **args)
    episode_info_list = [episode_info["info"]]
    results = generate_episode_descriptions(args["task_name"], episode_info_list, 1)
    instruction = np.random.choice(results[0][instruction_type])
    TASK_ENV.set_instruction(instruction=instruction)

    # Setup video recording
    ffmpeg = None
    if TASK_ENV.eval_video_path is not None and video_size:
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", video_size, "-framerate", "10",
                "-i", "-", "-pix_fmt", "yuv420p",
                "-vcodec", "libx264", "-crf", "23",
                f"{TASK_ENV.eval_video_path}/primitive_test.mp4",
            ],
            stdin=subprocess.PIPE,
        )
        TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

    # ---- Execute primitives sequentially (no policy warmup) ----
    # Fixed execution order: Forward → Down → Right → Left → Up → Retreat
    primitive_order = [
        "Nudge Up",
        "Nudge Forward",
        "Nudge Down",
        "Nudge Right",
        "Nudge Left",
        "Retreat",
    ]

    # Color map for primitives (BGR for OpenCV display)
    prim_colors_bgr = {
        "Nudge Left":    (255, 0, 0),     # Blue in BGR
        "Nudge Right":   (0, 165, 255),   # Orange
        "Nudge Up":      (0, 255, 0),     # Green
        "Nudge Down":    (0, 0, 255),     # Red
        "Nudge Forward": (255, 255, 0),   # Cyan
        "Retreat":       (255, 0, 255),   # Magenta
        "Stay":          (200, 200, 200), # Gray
    }

    # Save initial state
    init_obs = TASK_ENV.get_obs()
    init_state = init_obs["joint_action"]["vector"].copy()

    executed_names = []

    for prim_idx, prim_name in enumerate(primitive_order):
        print(f"\n--- Primitive {prim_idx + 1}/{len(primitive_order)}: {prim_name} ---")
        print(f"  Steps remaining: {TASK_ENV.step_lim - TASK_ENV.take_action_cnt}")

        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            print("  Step limit reached, stopping.")
            break

        # Generate primitives from current state
        observation = TASK_ENV.get_obs()
        primitive_trajs, primitive_names = generate_primitives_qpos(
            TASK_ENV, observation, camera_name, nudge_distance, primitive_horizon, arm_tag="both"
        )

        # Find the requested primitive
        if prim_name not in primitive_names:
            print(f"  WARNING: '{prim_name}' not available (IK failed). Executing hold trajectory.")
            # Generate a hold-position (no-op) trajectory so frames are still recorded
            current_state = observation["joint_action"]["vector"]
            traj = np.tile(current_state, (primitive_horizon, 1))
            executed_names.append(f"{prim_name} (IK FAILED)")
        else:
            traj_idx = primitive_names.index(prim_name)
            traj = primitive_trajs[traj_idx]
            executed_names.append(prim_name)

        print(f"  Trajectory shape: {traj.shape}")

        ik_failed = prim_name not in primitive_names
        color = prim_colors_bgr.get(prim_name, (255, 255, 255))
        # Convert BGR to RGB for annotation function
        color_rgb = (color[2], color[1], color[0])
        if ik_failed:
            color_rgb = (128, 128, 128)  # Gray for failed primitives

        for step_i in range(len(traj)):
            if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                break

            # Annotate the current frame
            ik_tag = " [IK FAILED]" if ik_failed else ""
            label = f"PRIM [{prim_idx+1}/{len(primitive_order)}] {prim_name}{ik_tag} ({step_i+1}/{len(traj)})"
            annotate_frame(TASK_ENV.now_obs, label, color=color_rgb)

            TASK_ENV.take_action(traj[step_i])

            # Get fresh obs (updates now_obs for next frame write)
            if step_i < len(traj) - 1:
                TASK_ENV.get_obs()

        # Log state change
        post_obs = TASK_ENV.get_obs()
        post_state = post_obs["joint_action"]["vector"]
        delta = post_state - init_state
        print(f"  Joint delta from init (L2): {np.linalg.norm(delta):.4f}")
        print(f"  Left arm delta:   {delta[:6]}")
        print(f"  Right arm delta:  {delta[7:13]}")

        # Log EE positions
        left_ee = np.array(TASK_ENV.robot.get_left_ee_pose()[:3])
        right_ee = np.array(TASK_ENV.robot.get_right_ee_pose()[:3])
        print(f"  Left EE pos:  {left_ee}")
        print(f"  Right EE pos: {right_ee}")

    # ---- Cleanup ----
    if ffmpeg is not None:
        TASK_ENV._del_eval_video_ffmpeg()

    TASK_ENV.close_env()

    # Save summary
    summary_path = save_dir / "primitive_test_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Seed: {now_seed}\n")
        f.write(f"Nudge distance: {nudge_distance}\n")
        f.write(f"Primitive horizon: {primitive_horizon}\n")
        f.write(f"Camera: {camera_name}\n")
        f.write(f"Instruction: {instruction}\n\n")
        f.write(f"Primitive execution order:\n")
        for name in executed_names:
            f.write(f"  - {name}\n")

    print(f"\nDone! Results saved to {save_dir}")
    if video_save_dir:
        print(f"Video: {TASK_ENV.eval_video_path}/primitive_test.mp4")


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()
    main(usr_args)
