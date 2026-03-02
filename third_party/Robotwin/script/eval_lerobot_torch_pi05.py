import sys
import os
import subprocess
import time as time_lib

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback
import matplotlib.pyplot as plt # Added for MMD plotting

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

from policy.pi05 import pi05_model_torch
# from policy.pi05 import deploy_policy # Removed to use custom steered evaluation loop

import torch

from steering.steerer import PivotSteerer, PrimitiveSteerer
from steering.utils import visualize_and_save_trajectory, compute_temporal_error
from steering.vlm_client import VLMClient

import debugpy

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


def get_task_ckpt_dir(base_dir: str, task: str) -> str:
    task_dir = os.path.join(base_dir, task)

    if not os.path.exists(task_dir):
        raise FileNotFoundError(f"Task 目录不存在: {task_dir}")

    return os.path.join(task_dir, "checkpoints/030000/pretrained_model")


# Helper from secondary file to ensure local eval loop works
def encode_obs(observation):
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]

    return input_rgb_arr, input_state


def report_episode_statistics(log_dir, episode_id, mmd_scores, vlm_intervention_count, 
                              vlm_intervention_steps, success, act_steps, num_mmd_samples, mmd_gamma, mmd_threshold):
    """Report MMD and VLM statistics for a single episode."""
    
    # Save MMD scores to file
    mmd_log_path = os.path.join(log_dir, f"mmd_scores_episode_{episode_id}.txt")
    with open(mmd_log_path, 'w') as f:
        f.write(f"Episode ID: {episode_id}\n")
        f.write(f"Success: {success}\n")
        f.write(f"Number of samples: {num_mmd_samples}\n")
        f.write(f"Gamma: {mmd_gamma}\n")
        f.write(f"Mean MMD: {np.mean(mmd_scores) if mmd_scores else 0:.6f}\n")
        f.write(f"Max MMD: {np.max(mmd_scores) if mmd_scores else 0:.6f}\n\n")
        
        f.write(f"MMD Threshold: {mmd_threshold}\n")
        f.write(f"Total VLM Interventions: {vlm_intervention_count}\n")
        if vlm_intervention_steps:
            f.write(f"VLM Intervention Steps: {vlm_intervention_steps}\n")
        f.write("\nPer-step MMD scores:\n")
        for i, score in enumerate(mmd_scores):
            vlm_marker = " [VLM]" if (i + 1) in vlm_intervention_steps else ""
            f.write(f"Step {i+1}: {score:.6f}{vlm_marker}\n")
    
    # Plot MMD scores
    if not mmd_scores: return
    timesteps = np.arange(1, len(mmd_scores) + 1) * act_steps
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(timesteps, mmd_scores, 'b-', marker='o', label='MMD Score')
    ax.axhline(y=mmd_threshold, color='orange', linestyle=':', label=f'Threshold: {mmd_threshold}')
    
    if vlm_intervention_steps:
        # Convert step numbers to indices (1-based to 0-based)
        intervention_indices = [s - 1 for s in vlm_intervention_steps if 0 < s <= len(mmd_scores)]
        if intervention_indices:
            intervention_times = [timesteps[i] for i in intervention_indices]
            intervention_vals = [mmd_scores[i] for i in intervention_indices]
            ax.scatter(intervention_times, intervention_vals, color='red', s=100, marker='*', zorder=5, label='Intervention')
            
    ax.set_title(f'Episode {episode_id} MMD (Success: {success})')
    ax.legend()
    plt.savefig(os.path.join(log_dir, f"mmd_plot_episode_{episode_id}.png"))
    plt.close()


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    save_dir = None
    video_save_dir = None
    video_size = None

    tag = usr_args["tag"]
    policy_path = usr_args["policy_path"]

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    if usr_args.get("debug", False):
        print("WAITING FOR DEBUGGER TO CONNECT")
        debugpy.listen(("0.0.0.0", 8567))
        debugpy.wait_for_client()
        print("DEBUGGER CONNECTED")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
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
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{tag}/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])
    
    # Store save_dir in args for use in eval_policy
    usr_args["log_dir"] = str(save_dir)

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = 100
    topk = 1

    ckpt_dir = policy_path

    model = pi05_model_torch.Lerobot_torch_PI05(
        task_name, 
        ckpt_dir,
    )

    torch.manual_seed(42)

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   usr_args=usr_args) # Pass usr_args for steering config
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                usr_args=None):
    
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    # ================= Steering Configuration =================
    compute_mmd = usr_args.get("compute_mmd", False)
    num_mmd_samples = usr_args.get("num_mmd_samples", 10)
    mmd_gamma = usr_args.get("mmd_gamma", "median")
    
    use_pivot_steering = usr_args.get("use_pivot_steering", False)
    use_primitive_steering = usr_args.get("use_primitive_steering", False)
    guidance_scale = usr_args.get("guidance_scale", 3.0)
    ensemble_weights = usr_args.get("ensemble_weights", [0.5, 0.5])
    mmd_threshold = usr_args.get("mmd_threshold", 0.8)
    
    vlm_server_url = usr_args.get("vlm_server_url", "")
    vlm_model_name = usr_args.get("vlm_model_name", "Qwen/Qwen2.5-VL-72B-Instruct")
    # TODO: Add specific prompt paths or use defaults
    vlm_prompt_path = usr_args.get("vlm_prompt_path", None) 
    
    log_dir = usr_args.get("log_dir", "./eval_result")
    
    # Steering constants
    act_steps = 50
    horizon_steps = 50 # Assuming this aligns with model cfg

    vlm_client = None
    if use_pivot_steering or use_primitive_steering:
        vlm_client = VLMClient(vlm_server_url, vlm_model_name)

    pivot_steerer = None
    if use_pivot_steering:
        pivot_steerer = PivotSteerer(
            vlm_client=vlm_client,
            save_dir=os.path.join(log_dir, "vlm_steering"),
            camera_name="head_camera",
            prompt_template_path=vlm_prompt_path,
            traj_std_perturb=0.0
        )
        print(f"Initialized PivotSteerer with VLM server: {vlm_server_url}")

    primitive_steerer = None
    if use_primitive_steering:
        primitive_steerer = PrimitiveSteerer(
            vlm_client=vlm_client,
            save_dir=os.path.join(log_dir, "primitive_steering"),
            camera_name="head_camera",
            prompt_template_path=vlm_prompt_path,
            horizon_steps=horizon_steps,
            nudge_distance=0.05,
            use_gripper_control=False,
            arm_tag="both"
        )
        print(f"Initialized PrimitiveSteerer (qpos mode) with VLM server: {vlm_server_url}")

    # ==========================================================

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    # Note: We are replacing deploy_policy.eval and reset_model with local logic
    # eval_func = deploy_policy.eval 
    # reset_func = deploy_policy.reset_model

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    # Global Tracking
    all_episode_mmd_scores = []
    all_episode_vlm_interventions = []

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(e)
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        
        # ---- Evaluation Loop Start ----
        # Replaces reset_func(model)
        model.reset_obsrvationwindows()
        if model.observation_window is None:
            model.set_language(instruction)

        # Episode-specific MMD tracking
        prev_action_samples = None
        mmd_scores = []
        vlm_intervention_count = 0
        vlm_intervention_steps = []
        cnt_step = 0 # Step counter for MMD logging (corresponds to chunks)

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            observation = TASK_ENV.get_obs()
            
            # Replaces: input_rgb_arr, input_state = encode_obs(observation)
            #           model.update_observation_window(...)
            input_rgb_arr, input_state = encode_obs(observation)
            model.update_observation_window(input_rgb_arr, input_state)

            # ======== Steered Action Generation Logic ========
            actions = None
            if compute_mmd:
                # 1. Sample actions for MMD
                # model.get_action supports num_samples (returns [num_samples, horizon, dim])
                with torch.inference_mode():
                    action_samples = model.get_action(num_samples=num_mmd_samples)
                    
                    # Ensure tensor for calculation (get_action likely returns tensor or numpy)
                    if isinstance(action_samples, np.ndarray):
                        action_samples = torch.from_numpy(action_samples)
                    
                    if num_mmd_samples == 1 and action_samples.ndim == 2:
                        action_samples = action_samples.unsqueeze(0)

                # 2. Compute MMD
                if prev_action_samples is not None:
                    # Reshape for compute_temporal_error: [num_envs=1, num_samples, horizon, dim]
                    curr_actions_t = action_samples.unsqueeze(0)
                    prev_actions_t = prev_action_samples.unsqueeze(0)
                    
                    # Handle Gamma string/float conversion
                    gamma_val = mmd_gamma
                    try:
                        gamma_val = float(mmd_gamma)
                    except: pass

                    mmd_score = compute_temporal_error(
                        curr_actions_t,
                        prev_actions_t,
                        exec_horizon=act_steps if act_steps != horizon_steps else act_steps // 2, # Use act_steps as exec horizon
                        gamma=gamma_val
                    )[0]
                    
                    mmd_scores.append(mmd_score.item())
                    # print(f"Step {cnt_step}: MMD = {mmd_score.item():.4f}")

                    # 3. VLM Steering Intervention
                    guidance_traj_list = []
                    
                    if mmd_score > mmd_threshold:
                        print("Attempting Steering")
                        print(f"\033[93mMMD Trigger ({mmd_score:.4f} > {mmd_threshold}). Steering...\033[0m")
                        
                        # Prepare data for steerer
                        # Note: Steerers expect CPU numpy usually
                        action_samples_np = action_samples.cpu().numpy()
                        
                        if pivot_steerer:
                            (selected_idx, selected_idx_right), _ = pivot_steerer.select_trajectory(
                                env=TASK_ENV,
                                obs=observation,
                                action_samples=action_samples, # Pass tensor or numpy depending on steerer impl
                                step_num=cnt_step,
                                episode_id=TASK_ENV.test_num,
                                mmd_score=mmd_score,
                                task_description=instruction,
                                num_trajectories=5 # Default pivot num
                            )
                            # Get guidance trajectory [horizon, dim]
                            guidance_traj = action_samples[selected_idx]
                            guidance_traj_list.append(guidance_traj)
                            
                        if primitive_steerer:
                            prim_traj, _ = primitive_steerer.select_trajectory(
                                env=TASK_ENV,
                                obs=observation,
                                action_samples=action_samples,
                                step_num=cnt_step,
                                episode_id=TASK_ENV.test_num,
                                mmd_score=mmd_score,
                                task_description=instruction
                            )
                            # Convert primitive numpy to tensor (already 14-dim qpos in qpos mode)
                            prim_traj_t = torch.from_numpy(prim_traj).to(action_samples.device)
                            guidance_traj_list.append(prim_traj_t)

                        # 4. Apply Guidance
                        if len(guidance_traj_list) > 0:
                            vlm_intervention_count += 1
                            vlm_intervention_steps.append(cnt_step + 1)
                            
                            # Ensemble weights
                            if len(guidance_traj_list) > 1:
                                stacked_trajs = torch.stack(guidance_traj_list)
                                weights = torch.tensor(ensemble_weights[:len(guidance_traj_list)], 
                                                     device=stacked_trajs.device)
                                weights = weights / weights.sum()
                                weights = weights.view(-1, 1, 1)
                                final_guidance = (stacked_trajs * weights).sum(dim=0)
                            else:
                                final_guidance = guidance_traj_list[0]

                            # Guided Inference
                            with torch.inference_mode():
                                actions = model.get_action(
                                    num_samples=1,
                                    guidance_actions=final_guidance,
                                    guidance_scale=guidance_scale,
                                    gripper_guidance=True
                                )
                                # Ensure actions is [horizon, dim]
                                if actions.ndim == 3: actions = actions.squeeze(0)

                # Update Previous Samples
                prev_action_samples = action_samples.clone()
                
                # If no steering happened, pick first sample or mean
                if actions is None:
                    actions = action_samples[0]
                
                # Visualization (Optional if MMD computed)
                if compute_mmd:
                    # Create directory
                    episode_rollout_dir = os.path.join(log_dir, "rollout_img", f"episode_{TASK_ENV.test_num}")
                    os.makedirs(episode_rollout_dir, exist_ok=True)
                    traj_save_path = os.path.join(episode_rollout_dir, f"step_{cnt_step}.png")
                    
                    # Needs adaptation: visualize expects specific env/obs format. 
                    # Passing TASK_ENV directly.
                    try:
                        # Assuming action_samples is [N, T, D]
                        visualize_and_save_trajectory(
                            TASK_ENV, 
                            observation, 
                            action_samples.cpu().numpy(), 
                            cnt_step,
                            save_path=traj_save_path,
                            mmd_mode=True,
                            mmd_score=mmd_scores[-1] if mmd_scores else None,
                            camera_name="head_camera"
                        )
                    except Exception as e:
                        # print(f"Viz failed: {e}") 
                        pass # visualization shouldn't crash eval

            else:
                # Standard Inference (No Steering/MMD)
                actions = model.get_action() # Returns [horizon, dim]

            # Execute Action Chunk
            # model.pi0_step usually defines execution horizon (e.g. 10 or 50)
            exec_steps = 25 # model.pi0_step
            # Handle if actions is tensor
            if isinstance(actions, torch.Tensor):
                actions = actions.cpu().numpy()

            for i, action in enumerate(actions[:exec_steps]):
                TASK_ENV.take_action(action)

                # We need to update observation window for every step in the chunk
                # (except the very last one where we loop back to top)
                if i < exec_steps - 1:
                    observation = TASK_ENV.get_obs()
                    input_rgb_arr, input_state = encode_obs(observation)
                    model.update_observation_window(input_rgb_arr, input_state)

                # Mid-chunk prediction: when act_steps == horizon_steps, sample at the midpoint
                # so the next MMD comparison uses only the second half of the horizon
                if compute_mmd and act_steps == horizon_steps and i == act_steps // 2:
                    with torch.inference_mode():
                        mid_action_samples = model.get_action(num_samples=num_mmd_samples)
                        if isinstance(mid_action_samples, np.ndarray):
                            mid_action_samples = torch.from_numpy(mid_action_samples)
                        if num_mmd_samples == 1 and mid_action_samples.ndim == 2:
                            mid_action_samples = mid_action_samples.unsqueeze(0)
                    prev_action_samples = mid_action_samples

            cnt_step += 1 # Increment chunk step counter

            # Check success (replacing TASK_ENV.eval_success check inside loop)
            if TASK_ENV.eval_success:
                succ = True
                break
        
        # ---- Evaluation Loop End ----

        # Report stats
        if compute_mmd and len(mmd_scores) > 0:
            report_episode_statistics(log_dir, TASK_ENV.test_num, mmd_scores, 
                                      vlm_intervention_count, vlm_intervention_steps, succ,
                                      act_steps, num_mmd_samples, mmd_gamma, mmd_threshold)
            all_episode_mmd_scores.append(mmd_scores)
            all_episode_vlm_interventions.append(vlm_intervention_count)

        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1
    
    # Final Statistics
    if compute_mmd and len(all_episode_mmd_scores) > 0:
        # Simple aggregated report
        with open(os.path.join(log_dir, "final_steering_stats.txt"), 'w') as f:
            f.write(f"Total Episodes: {len(all_episode_mmd_scores)}\n")
            f.write(f"Total Interventions: {sum(all_episode_vlm_interventions)}\n")

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
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