import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import numpy as np
import cv2
import torch
from PIL import Image
import sapien.core as sapien
from sklearn.metrics.pairwise import cosine_similarity
import json, re, random

from .vlm_client import VLMClient
from .utils import project_3d_to_2d, compute_future_ee_poses_using_controller, generate_primitives_qpos

import logging
log = logging.getLogger(__name__)


def select_representative_trajectories(action_trajectories: np.ndarray, num_trajectories: int = 3) -> np.ndarray:
    """
    Selects representative trajectories from a batch with maximum spread.
    
    This function uses an efficient O(k*n) Farthest Point Sampling (FPS) implementation.

    Strategy:
    1. Start with the "average" (medoid) trajectory — the one most similar overall.
    2. Iteratively select the trajectory most dissimilar to the selected set,
       using a greedy "farthest-point" criterion based on cosine distance.

    Args:
        action_trajectories (np.ndarray): Array of shape (num_samples, horizon, 3) containing 3D positions.
        num_trajectories (int): Number of representative trajectories to select.

    Returns:
        np.ndarray: Indices of selected trajectories.
    """
    num_samples = action_trajectories.shape[0]

    # Handle edge cases
    if num_trajectories <= 0:
        return np.array([], dtype=int)
    if num_samples <= num_trajectories:
        return np.arange(num_samples)

    # Flatten each trajectory: (num_samples, horizon * 3)
    flattened = action_trajectories.reshape(num_samples, -1)

    # Normalize to avoid zero-vector issues in cosine similarity
    norms = np.linalg.norm(flattened, axis=1, keepdims=True)
    flattened = np.where(norms == 0, 0, flattened / np.maximum(norms, 1e-8))

    # Compute cosine similarity and convert to distance
    # sim_matrix[i, j] = similarity between trajectory i and j
    sim_matrix = cosine_similarity(flattened)
    dist_matrix = 1 - sim_matrix  # cosine distance (0 = identical, 2 = opposite)

    # Step 1: Select the medoid (most central trajectory)
    # This is the point with the minimum *total* distance to all other points.
    total_dissimilarity = dist_matrix.sum(axis=1)
    medoid_idx = np.argmin(total_dissimilarity)

    selected_indices = [medoid_idx]

    # --- Optimized Step 2 ---
    # We maintain an array of the minimum distance from each point
    # to the *currently selected set*.
    # Initialize it with distances to the first selected point (the medoid).
    min_dists_to_selected = dist_matrix[medoid_idx, :].copy()
    
    # Mark the medoid as "selected" by setting its min_dist to a value
    # that np.argmax will never pick (since distances are >= 0).
    min_dists_to_selected[medoid_idx] = -1.0 

    for _ in range(num_trajectories - 1):
        # 1. Find the point with the maximum "minimum distance"
        # This is the point farthest from its closest selected neighbor.
        farthest_idx = np.argmax(min_dists_to_selected)
        
        # This check handles the case where we've selected all valid points
        if min_dists_to_selected[farthest_idx] == -1.0:
            break

        # 2. Add this point to our set
        selected_indices.append(farthest_idx)
        
        # 3. Mark this new point as selected
        min_dists_to_selected[farthest_idx] = -1.0
        
        # 4. Update the min_dists array.
        # For all remaining points, their new minimum distance is
        # min(their_current_min_dist, their_dist_to_the_new_point).
        dists_to_new_point = dist_matrix[farthest_idx, :]
        min_dists_to_selected = np.minimum(min_dists_to_selected, dists_to_new_point)
    
    return np.array(selected_indices, dtype=int)


class PivotSteerer:
    """
    Computes trajectory selection using VLM guidance when MMD threshold is exceeded.
    Draws multiple sampled trajectories on the current camera view and queries a VLM
    to select the best trajectory.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        save_dir: Optional[str] = None,
        camera_name: str = "head_camera",
        prompt_template_path: Optional[str] = None,
        traj_std_perturb: float = 0.0,
        line_thickness: int = 2
    ):
        """
        Initialize the PivotSteerer.
        
        Args:
            vlm_server_url: URL of the VLM server.
            save_dir: Directory to save annotated images and responses.
            camera_name: Name of the camera to use for visualization.
            prompt_template_path: Path to the prompt template file. If None, uses default template.
            traj_std_perturb: Standard deviation for Gaussian perturbation applied to trajectory clones.
                             If 0.0, no perturbation is applied.
        """
        self.vlm_client = vlm_client
        self.save_dir = Path(save_dir) if save_dir else None
        self.camera_name = camera_name
        self.step_count = 0
        self.traj_std_perturb = traj_std_perturb
        self.line_thickness = line_thickness
        
        # Load prompt template from file
        if prompt_template_path is None:
            # Use default template path relative to this file
            prompt_template_path = Path(__file__).parent / "vlm_prompt_template.txt"
        
        with open(prompt_template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "pivot_images").mkdir(exist_ok=True)

    def visualize_trajectories_on_camera(
        self, 
        env, 
        obs: Dict, 
        env_actions_list: list,
        num_trajectories: int = 3
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Visualize representative trajectories on camera image.
        
        Args:
            env: The environment
            obs: Current observation dict
            env_actions_list: List of (N, 14) action arrays
            num_trajectories: Number of representative trajectories to visualize
            
        Returns:
            annotated_image: BGR image with trajectories drawn
            selected_indices: List containing indices of selected trajectories for both arms
        """
        
        # Get current EE pose for both arms
        current_ee_pose = np.stack([env.robot.get_left_ee_pose()[:3], env.robot.get_right_ee_pose()[:3]], axis=0)
        
        # Compute future end-effector positions for all trajectories
        all_trajectories = []
        for env_actions in env_actions_list:
            ee_positions_world = compute_future_ee_poses_using_controller(
                env, env_actions, obs
            )
            # Add current position as the starting point
            all_positions = np.vstack([current_ee_pose[np.newaxis], ee_positions_world])
            
            if self.traj_std_perturb > 0.0:
                # Apply perturbation only to trajectory points, not the origin (first point)
                perturbation = np.random.normal(0, self.traj_std_perturb, all_positions[1:].shape)
                all_positions[1:] = all_positions[1:] + perturbation
                
            all_trajectories.append(all_positions)
        
        # Convert list to numpy array for selection algorithm
        # Shape transition: (num_samples, horizon, 2, 3) -> (2, num_samples, horizon, 3)
        all_trajectories = np.array(all_trajectories).transpose(2, 0, 1, 3)
        
        # Select representative trajectories for both arms
        selected_indices_left = select_representative_trajectories(
            all_trajectories[0], 
            num_trajectories=num_trajectories
        )

        selected_indices_right = select_representative_trajectories(
            all_trajectories[1], 
            num_trajectories=num_trajectories
        )
        
        # Get camera parameters
        cam_params = obs['observation'][self.camera_name]
        extrinsic = cam_params['extrinsic_cv']
        intrinsic = cam_params['intrinsic_cv']
        
        # Get image
        img = obs['observation'][self.camera_name]['rgb'].copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Define distinct colors for trajectories
        predefined_colors = [
            (0, 0, 255),    # Red
            (0, 165, 255),  # Orange
            (255, 0, 0),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
        ]
        
        # Draw selected representative trajectories for both arms
        # arm_idx 0: Left, arm_idx 1: Right
        arm_selected_indices = [selected_indices_left, selected_indices_right]
        
        for arm_idx, selected_indices in enumerate(arm_selected_indices):
            for i, traj_idx in enumerate(selected_indices):
                # Take the first 16 steps or horizon limit as per original logic
                all_positions = all_trajectories[arm_idx, traj_idx]
                
                # Project 3D positions to 2D
                points_2d, depths = project_3d_to_2d(all_positions, extrinsic, intrinsic)
                
                # Filter points behind camera
                valid_mask = depths > 0
                traj_color = predefined_colors[i % len(predefined_colors)]
                
                # Track valid points for arrow head
                last_valid_idx = None
                second_last_valid_idx = None
                
                # Draw trajectory
                for j in range(len(points_2d)):
                    if not valid_mask[j]:
                        continue
                    
                    x, y = int(points_2d[j, 0]), int(points_2d[j, 1])
                    
                    # Check if point is within image bounds
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        if j == 0 and i == 0:  # Only draw current position once per arm
                            # Current position - larger circle with white outline
                            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                            cv2.circle(img, (x, y), 3, (255, 255, 255), 2)
                        elif j > 0:
                            # Future positions
                            cv2.circle(img, (x, y), 1, traj_color, -1)
                        
                        # Track valid points for arrow head
                        second_last_valid_idx = last_valid_idx
                        last_valid_idx = j
                    
                    # Draw line connecting points
                    if j > 0 and valid_mask[j-1]:
                        x_prev, y_prev = int(points_2d[j-1, 0]), int(points_2d[j-1, 1])
                        if (0 <= x_prev < img.shape[1] and 0 <= y_prev < img.shape[0] and
                            0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
                            cv2.line(img, (x_prev, y_prev), (x, y), traj_color, self.line_thickness)
                
                # Draw arrow head at the end of trajectory
                # if last_valid_idx is not None and second_last_valid_idx is not None:
                #     x_last, y_last = int(points_2d[last_valid_idx, 0]), int(points_2d[last_valid_idx, 1])
                #     x_prev, y_prev = int(points_2d[second_last_valid_idx, 0]), int(points_2d[second_last_valid_idx, 1])
                    
                #     if (0 <= x_last < img.shape[1] and 0 <= y_last < img.shape[0]):
                #         dx, dy = x_last - x_prev, y_last - y_prev
                #         length = np.sqrt(dx**2 + dy**2)
                        
                #         if length > 0:
                #             dx, dy = dx / length, dy / length
                #             arrow_length, arrow_angle = 15, np.pi / 6
                            
                #             arrow_tip = (x_last, y_last)
                #             arrow_left = (
                #                 int(x_last - arrow_length * (dx * np.cos(arrow_angle) + dy * np.sin(arrow_angle))),
                #                 int(y_last - arrow_length * (dy * np.cos(arrow_angle) - dx * np.sin(arrow_angle)))
                #             )
                #             arrow_right = (
                #                 int(x_last - arrow_length * (dx * np.cos(arrow_angle) - dy * np.sin(arrow_angle))),
                #                 int(y_last - arrow_length * (dy * np.cos(arrow_angle) + dx * np.sin(arrow_angle)))
                #             )
                            
                #             pts = np.array([arrow_tip, arrow_left, arrow_right], np.int32)
                #             cv2.fillPoly(img, [pts], traj_color)
                #             cv2.polylines(img, [pts], True, traj_color, self.line_thickness)
            
        # Add legend with color names
        # color_names = ["Red", "Orange", "Blue", "Cyan", "Magenta"]
        # legend_y = 30
        # for i, color in enumerate(predefined_colors[:num_trajectories]):
        #     cv2.circle(img, (30, legend_y), 2, color, -1)
        #     cv2.putText(img, color_names[i], (45, legend_y + 5),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.25, (41, 19, 6), 2)
        #     legend_y += 25
        
        return img, (selected_indices_left, selected_indices_right)

    def select_trajectory(
        self,
        env,
        obs: Dict,
        action_samples: torch.Tensor,
        step_num: int,
        episode_id: int,
        mmd_score: float,
        task_description: Optional[str] = None,
        num_trajectories: int = 5
    ) -> tuple[tuple[int, int], str]:
        """
        Visualize representative trajectories and query VLM to select the best one for each arm.
        
        Args:
            env: The environment
            obs: Current observation
            action_samples: [num_samples, batch_size, horizon_steps, action_dim]
            step_num: Current step number
            mmd_score: The MMD score that triggered this call
            task_description: Task-specific description to insert into the prompt template
            num_trajectories: Number of representative trajectories to visualize (default: 5)
            
        Returns:
            tuple: ((sel_idx_left, sel_idx_right), vlm_response_text)
                   Note: indices are the original indices in action_samples
        """
        self.step_count = step_num
        
        # Convert all sampled actions to env_actions format
        all_env_actions = [action_samples[i].float().cpu().numpy() for i in range(action_samples.shape[0])]
        
        # Visualize representative trajectories - returns img and tuple of (left_indices, right_indices)
        annotated_img_bgr, (sel_indices_l, sel_indices_r) = self.visualize_trajectories_on_camera(
            env, obs, all_env_actions, num_trajectories=num_trajectories
        )
        
        # Add MMD score to image
        text = f"MMD Score: {mmd_score:.6f}"
        cv2.putText(annotated_img_bgr, text, (10, annotated_img_bgr.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (41, 19, 6), 2)
        
        # Convert to PIL Image for VLM
        annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_img_rgb)
        
        # Prepare prompt by substituting task description placeholder
        prompt = self.prompt_template
        if task_description:
            prompt = prompt.replace("<TASK_DESCRIPTION/>", task_description)
        else:
            # Remove the placeholder if no task description provided
            prompt = prompt.replace("<TASK_DESCRIPTION/>", "")
        
        # Query VLM to select 1 index for the left arm and 1 for the right arm
        # We assume the VLM client returns a tuple/list of two visual indices
        (vis_idx_l, vis_idx_r), vlm_response = self.vlm_client.select_trajectories(
            pil_image, 
            prompt,
            num_trajectories=num_trajectories
        )
        
        # Map back to original trajectory indices
        orig_idx_l = sel_indices_l[vis_idx_l]
        orig_idx_r = sel_indices_r[vis_idx_r]
        
        # Save annotated image and response if save_dir is set
        if self.save_dir:
            img_save_dir_tmp = self.save_dir / "pivot_images" / f"episode_{episode_id}"
            img_save_dir_tmp.mkdir(parents=True, exist_ok=True)
            img_save_path = img_save_dir_tmp / f"step_{step_num}_mmd_{mmd_score:.4f}.png"
            pil_image.save(img_save_path)
            
            response_path = img_save_dir_tmp / f"step_{step_num}_response.txt"
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(f"Step: {step_num}\n")
                f.write(f"MMD Score: {mmd_score:.6f}\n")
                f.write(f"Selected (visual) - Left: {vis_idx_l}, Right: {vis_idx_r}\n")
                f.write(f"Selected (original) - Left: {orig_idx_l}, Right: {orig_idx_r}\n")
                f.write(f"Rep Indices Left: {sel_indices_l.tolist()}\n")
                f.write(f"Rep Indices Right: {sel_indices_r.tolist()}\n\n")
                f.write(f"Prompt used:\n{prompt}\n\n")
                f.write("VLM Response:\n")
                f.write(vlm_response)
            
            print(f"Saved pivot visualization to {img_save_path}")
            print(f"Saved VLM response to {response_path}")
        
        print(f"VLM selected: L={vis_idx_l} (orig: {orig_idx_l}), R={vis_idx_r} (orig: {orig_idx_r})")
        
        return (orig_idx_l, orig_idx_r), vlm_response
    
    
class PrimitiveSteerer:
    """
    Computes recovery trajectory selection using VLM guidance when MMD threshold is exceeded.
    Generates hardcoded spatial primitives (nudges) based on camera view, draws them,
    and queries a VLM to select the best recovery action.
    """

    def __init__(
        self,
        vlm_client: VLMClient,
        save_dir: Optional[str] = None,
        camera_name: str = "overhead_camera",
        prompt_template_path: Optional[str] = None,
        horizon_steps: int = 6,  # How many steps the nudge action takes
        nudge_distance: float = 0.1,  # 5cm nudge
        line_thickness: int = 1,
        draw_trajectories: bool = False,
        primitive_traj_std_dev: float = 0.0,
        use_gripper_control: bool = True,
        arm_tag: str = "both"  # "left", "right", or "both" (only for qpos mode)
    ):
        """
        Initialize the PrimitiveSteerer.

        Args:
            vlm_server_url: URL of the VLM server.
            save_dir: Directory to save annotated images and responses.
            camera_name: Name of the camera to use for visualization.
            prompt_template_path: Path to the prompt template file.
            horizon_steps: Number of timesteps to execute the primitive (smoothness).
            nudge_distance: Physical distance in meters for spatial primitives.
            arm_tag: Which arm(s) to generate primitives for in qpos mode.
        """
        self.vlm_client = vlm_client
        self.save_dir = Path(save_dir) if save_dir else None
        self.camera_name = camera_name
        self.step_count = 0
        self.horizon_steps = horizon_steps
        self.nudge_dist = nudge_distance
        self.line_thickness = line_thickness
        self.draw_trajectories = draw_trajectories
        self.primitive_traj_std_dev = primitive_traj_std_dev
        self.use_gripper_control = use_gripper_control
        self.arm_tag = arm_tag
        
        # Load prompt template from file or default
        if prompt_template_path is None:
            # Default prompt for recovery
            prompt_template_path = Path(__file__).parent / "prompts/vlm_recovery_prompt.txt"
            
        # If file doesn't exist, use a hardcoded fallback string to prevent crash
        prompt_template_path = Path(prompt_template_path)
        if prompt_template_path.exists():
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
        else:
            raise ValueError(f"Prompt template file {prompt_template_path} does not exist")
        
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            (self.save_dir / "primitive_images").mkdir(exist_ok=True)

    def visualize_trajectories_on_camera(
        self, 
        env, 
        obs: Dict, 
        env_actions_list: list, 
        primitive_names: list
    ) -> np.ndarray:
        """
        Visualize primitives with outlines and offsets for clarity.
        UPDATED: Compact legend size.
        """
        # 1. Compute Future Poses
        controller = env.env.env.env.agent.controller.controllers['arm']
        current_ee_pose_at_base = controller.ee_pose_at_base
        robot_base_pose = env.env.env.env.agent.robot.pose
        current_tcp_world = (robot_base_pose * current_ee_pose_at_base).p
        
        all_trajectories = []
        for env_actions in env_actions_list:
            ee_positions_world = compute_future_ee_poses_using_controller(
                env, env_actions[:, :6], obs
            )
            all_positions = np.vstack([current_tcp_world, ee_positions_world])
            all_trajectories.append(all_positions)
        
        # 2. Get Camera Params
        cam_params = obs['camera_param'][self.camera_name]
        extrinsic = cam_params.get('extrinsic_cv', cam_params.get('extrinsic'))
        intrinsic = cam_params.get('intrinsic_cv', cam_params.get('intrinsic'))
        
        # 3. Prepare Image
        img = obs['image'][self.camera_name]['rgb'].copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.dtype == np.float32:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
        # Colors
        colors = {
            "Red": (0, 0, 255), "Orange": (0, 165, 255), 
            "Blue": (255, 0, 0), "Cyan": (255, 255, 0), 
            "Magenta": (255, 0, 255), "Green": (0, 255, 0)
        }
        ordered_colors = list(colors.values())

        # 4. Draw Trajectories (with offset start and outlines)
        # Origin Dot
        origin_2d, d = project_3d_to_2d(np.array([current_tcp_world]), extrinsic, intrinsic)
        ox, oy = int(origin_2d[0,0]), int(origin_2d[0,1])
        cv2.circle(img, (ox, oy), 9, (0, 0, 0), -1)
        cv2.circle(img, (ox, oy), 7, (255, 255, 255), -1)

        for i, all_positions in enumerate(all_trajectories):
            points_2d, depths = project_3d_to_2d(all_positions, extrinsic, intrinsic)
            valid = depths > 0
            color = ordered_colors[i % len(ordered_colors)]
            
            # Start drawing slightly away from origin to reduce clutter
            start_idx = max(1, int(len(points_2d) * 0.2))
            
            # Draw Lines
            for j in range(start_idx, len(points_2d)):
                if not valid[j] or not valid[j-1]: continue
                p1 = (int(points_2d[j-1, 0]), int(points_2d[j-1, 1]))
                p2 = (int(points_2d[j, 0]), int(points_2d[j, 1]))
                
                if 0 <= p1[0] < img.shape[1] and 0 <= p1[1] < img.shape[0]:
                    cv2.line(img, p1, p2, (0,0,0), self.line_thickness + 2)
                    cv2.line(img, p1, p2, color, self.line_thickness)

            # Draw Arrow Head
            if len(points_2d) > 1 and valid[-1] and valid[-2]:
                last, prev = points_2d[-1], points_2d[-2]
                lx, ly, px, py = int(last[0]), int(last[1]), int(prev[0]), int(prev[1])
                dx, dy = lx - px, ly - py
                mag = np.sqrt(dx*dx + dy*dy)
                
                if mag > 0:
                    dx, dy = dx/mag, dy/mag
                    alen, angle = 20, np.pi / 5
                    p_tip = (lx, ly)
                    p_left = (int(lx - alen*(dx*np.cos(angle) + dy*np.sin(angle))),
                              int(ly - alen*(dy*np.cos(angle) - dx*np.sin(angle))))
                    p_right = (int(lx - alen*(dx*np.cos(angle) - dy*np.sin(angle))),
                               int(ly - alen*(dy*np.cos(angle) + dx*np.sin(angle))))
                    
                    triangle = np.array([p_tip, p_left, p_right], np.int32)
                    cv2.polylines(img, [triangle], True, (0,0,0), 2)
                    cv2.fillPoly(img, [triangle], color)

        # 5. Draw Compact Legend
        legend_names = ["Red", "Orange", "Blue", "Cyan", "Magenta", "Green"]
        
        # Compact visual settings
        font_scale = 0.4
        line_spacing = 16
        dot_radius = 4
        text_offset_x = 15
        box_padding = 5
        
        legend_x, legend_y = 10, 10
        box_w = 160 # Narrower width
        box_h = len(primitive_names) * line_spacing + (box_padding * 2)
        
        # Background
        overlay = img.copy()
        cv2.rectangle(overlay, (legend_x, legend_y), (legend_x + box_w, legend_y + box_h), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        
        current_y = legend_y + box_padding + 8 # Initial text baseline
        
        for i, name in enumerate(primitive_names):
            color = ordered_colors[i % len(ordered_colors)]
            c_name = legend_names[i % len(legend_names)]
            
            # Icon
            cv2.circle(img, (legend_x + 8, current_y - 4), dot_radius, color, -1)
            
            # Text
            text = f"{c_name}: {name}"
            cv2.putText(img, text, (legend_x + text_offset_x + 5, current_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (41, 19, 6), 1)
            current_y += line_spacing
            
        return img

    def _extract_vlm_response(self, response_text: str) -> dict:
        """
        Extracts 'chosen_trajectory' and 'gripper_state' from VLM response.
        Handles standard JSON and Markdown formatting.
        
        Args:
            response_text (str): The raw string returned by the VLM.
            
        Returns:
            dict: {'chosen_trajectory': str, 'gripper_state': str}
                Values are None if extraction fails.
        """
        # Default return structure
        extracted_data = {
            "chosen_trajectory": None,
            "gripper_state": None
        }

        # 1. Clean up Markdown code blocks (e.g., ```json ... ```)
        clean_text = response_text.strip()
        if clean_text.startswith("```json"):
            clean_text = clean_text[7:]
        elif clean_text.startswith("```"):
            clean_text = clean_text[3:]
        
        if clean_text.endswith("```"):
            clean_text = clean_text[:-3]
        
        clean_text = clean_text.strip()

        # 2. Attempt strict JSON parsing
        try:
            data = json.loads(clean_text)
            extracted_data["chosen_trajectory"] = data.get("chosen_trajectory")
            extracted_data["gripper_state"] = data.get("gripper_state")
            return extracted_data
        except json.JSONDecodeError:
            log.warning(f"JSON parse failed for VLM response. Attempting Regex fallback. Response: {response_text[:50]}...")

        # 3. Fallback: Regex extraction
        # This handles cases where the VLM adds extra text outside the JSON block
        
        # Extract chosen_trajectory
        traj_pattern = r'"chosen_trajectory"\s*:\s*"([^"]+)"'
        traj_match = re.search(traj_pattern, response_text)
        if traj_match:
            extracted_data["chosen_trajectory"] = traj_match.group(1)

        # Extract gripper_state
        state_pattern = r'"gripper_state"\s*:\s*"([^"]+)"'
        state_match = re.search(state_pattern, response_text)
        if state_match:
            extracted_data["gripper_state"] = state_match.group(1)

        return extracted_data

    def select_trajectory(
        self,
        env,
        obs: Dict,
        action_samples: torch.Tensor,
        step_num: int = 0,
        episode_id: int = 0,
        mmd_score: float = 0.0,
        task_description: Optional[str] = None,
    ) -> tuple[np.ndarray, str]:
        """
        Generate primitives, visualize them, and query VLM to select one.
        
        Returns:
            tuple: (selected_trajectory_array, vlm_response_text)
            
            IMPORTANT: The returned trajectory is (Horizon, 7) PHYSICAL DELTAS.
            The EvalAgent must skip post-processing for this result.
        """
        self.step_count = step_num
        
        # 1. Generate Primitives
        # Returns list of (Horizon, 14) arrays and list of strings
        primitive_trajs, primitive_names = generate_primitives_qpos(
            env, obs, self.camera_name, self.nudge_dist, self.horizon_steps, self.arm_tag
        )
        
        if self.primitive_traj_std_dev > 0.0:
            for i in range(len(primitive_trajs)):
                perturbation = np.random.normal(0, self.primitive_traj_std_dev, primitive_trajs[i][1:].shape)
                primitive_trajs[i][1:] = primitive_trajs[i][1:] + perturbation
        
        if self.draw_trajectories:
            # 2. Visualize
            annotated_img_bgr = self.visualize_trajectories_on_camera(
                env, obs, primitive_trajs, primitive_names
            )
            
            # Add MMD Alert
            text = f"UNCERTAINTY {mmd_score:.2f} - RECOVERY MODE"
            cv2.putText(annotated_img_bgr, text, (10, annotated_img_bgr.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 3. Prepare VLM Inputs
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(annotated_img_rgb)
        else:
            pil_image = Image.fromarray(obs['observation'][self.camera_name]['rgb'].copy())
        
        # Format Prompt
        prompt = self.prompt_template
        if task_description:
            prompt = prompt.replace("<TASK_DESCRIPTION/>", task_description)
        else:
            prompt = prompt.replace("<TASK_DESCRIPTION/>", "Fix robot state")
        
        if self.draw_trajectories:
            # List primitives in prompt
            color_names = ["Red", "Orange", "Blue", "Cyan", "Magenta", "Green"]
            prim_list_str = ""
            for i, name in enumerate(primitive_names):
                c_name = color_names[i % len(color_names)]
                prim_list_str += f"- {c_name}: {name}\n"
            prompt = prompt.replace("<PRIMITIVE_LIST/>", prim_list_str)
        
        # 4. Query VLM
        # We use select_trajectory method from client, but since we need specific
        # primitive mapping, we might need 'generate' or assume select_trajectory 
        # returns an index 0-N matching the list.
        # Assuming vlm_client.select_trajectory returns (index, text)
        _, vlm_response = self.vlm_client.select_trajectory(
            pil_image, 
            prompt,
            num_trajectories=len(primitive_names),
            primitive=True
        )
        
        # 5. Retrieve Selected Action
        try:
            extracted_info = self._extract_vlm_response(vlm_response)
            selected_name = extracted_info["chosen_trajectory"]
            selected_gripper_state = extracted_info["gripper_state"]
            selected_vis_idx = primitive_names.index(selected_name)
            selected_traj = primitive_trajs[selected_vis_idx]
        except Exception as e:
            log.error(f"Error extracting Primitive VLM response: {e}")
            selected_name = primitive_names[random.randint(0, len(primitive_names) - 1)]
            selected_gripper_state = "open" if random.random() < 0.5 else "close"
            selected_vis_idx = primitive_names.index(selected_name)
            selected_traj = primitive_trajs[selected_vis_idx]
        
        # Set gripper state
        # In qpos mode, gripper values are at indices 6 (left) and 13 (right)
        # They are already set to current values by generate_primitives_qpos
        # Optionally override with VLM-selected gripper state
        if self.use_gripper_control and selected_gripper_state is not None:
            gripper_val = None
            if selected_gripper_state == "open":
                gripper_val = 1.0
            elif selected_gripper_state == "close":
                gripper_val = 0.0
            # "keep" leaves them as-is (already set to current values)
            if gripper_val is not None:
                left_arm_dim = 6  # TODO: make configurable if needed
                selected_traj[:, left_arm_dim] = gripper_val          # left gripper
                selected_traj[:, left_arm_dim + 1 + 6] = gripper_val  # right gripper
        
        # 6. Logging / Saving
        if self.save_dir:
            img_save_dir_tmp = self.save_dir / "primitive_images" / f"episode_{episode_id}"
            img_save_dir_tmp.mkdir(parents=True, exist_ok=True)
            img_save_path = img_save_dir_tmp / f"step_{step_num}_mmd_{mmd_score:.4f}.png"
            pil_image.save(img_save_path)
            
            response_path = img_save_dir_tmp / f"step_{step_num}_response.txt"
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(f"Step: {step_num}\n")
                f.write(f"MMD Score: {mmd_score:.6f}\n")
                f.write(f"Selected Primitive: {selected_name} (Idx: {selected_vis_idx})\n")
                f.write(f"Prompt used:\n{prompt}\n\n")
                f.write("VLM Response:\n")
                f.write(vlm_response)
            
            log.info(f"Saved primitive visualization to {img_save_path}")

        log.info(f"VLM Primitive Selection: {selected_name} (Idx: {selected_vis_idx})")
        
        return selected_traj, vlm_response