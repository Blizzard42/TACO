import numpy as np
from typing import Optional, Union, Literal, cast, Dict, Tuple, List
import transforms3d as t3d

import torch
from sklearn import metrics
import sapien.core as sapien

import cv2
import logging
log = logging.getLogger(__name__)

def compute_mmd_rbf(
    x: np.ndarray, y: np.ndarray, gamma: Optional[Union[float, Literal['median', 'max_eig']]] = None
) -> np.float64:
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Args:
        x: [N, D] matrix.
        y: [M, D] matrix.
        gamma: rbf kernel parameter.

    Returns:
        MMD value.
    """
    assert x.ndim == 2 and y.ndim == 2

    if isinstance(gamma, str):
        if gamma == "median":
            z = np.vstack([x, y])
            distances = np.sum((z[:, np.newaxis, :] - z[np.newaxis, :, :]) ** 2, axis=2)
            gamma = cast(float, 1.0 / (2 * np.median(distances[distances > 0])))
        elif gamma == "max_eig":
            z = np.vstack([x, y])
            cov = np.cov(z.T)
            max_eig = np.max(np.linalg.eigvalsh(cov))
            gamma = cast(float, 1.0 / max_eig)
        else:
            raise ValueError(f"Gamma {gamma} is not supported.")

    xx = metrics.pairwise.rbf_kernel(x, x, gamma)
    yy = metrics.pairwise.rbf_kernel(y, y, gamma)
    xy = metrics.pairwise.rbf_kernel(x, y, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_temporal_error(
    curr_action: torch.Tensor,
    prev_action: torch.Tensor,
    exec_horizon: int,
    gamma: Optional[Union[float, Literal['median', 'max_eig']]],
    envs_to_ignore: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute temporal consistency error.

    Args:
        curr_action: Current action (num_envs, batch_size, pred_horizon, action_dim).
        prev_action: Previous action (num_envs, batch_size, pred_horizon, action_dim).
        exec_horizon: Robot execution horizon.
        gamma: value or method to adjust RBF by
        envs_to_ignore: Optional array of booleans indicating which envs to ignore

    Returns:
        Array of MMD scores for each environment.
    """
    assert curr_action.ndim == 4 and prev_action.ndim == 4  # Dimensions are consistent with expectations
    assert curr_action.shape[0] == prev_action.shape[0]  # Number of envs is consistent
    assert curr_action.shape[1] == prev_action.shape[1]  # Number of MMD trajectories is consistent
    assert curr_action.shape[2:] == prev_action.shape[2:]  # Trajectory shape is consistent
    pred_horizon = curr_action.shape[2]
    assert pred_horizon - exec_horizon > 0, "No overlap in action prediction horizon."

    if envs_to_ignore is not None:
        assert len(envs_to_ignore) == curr_action.shape[0]

    prev = prev_action[:, :, exec_horizon:]
    curr = curr_action[:, :, : pred_horizon - exec_horizon]
    res = []

    for i in range(curr_action.shape[0]):
        if envs_to_ignore is None or not envs_to_ignore[i]:
            curr_trajectories = curr[i].reshape(curr[i].shape[0], -1).to(torch.float32).cpu().numpy()
            prev_trajectories = prev[i].reshape(prev[i].shape[0], -1).to(torch.float32).cpu().numpy()
            res.append(compute_mmd_rbf(curr_trajectories, prev_trajectories, gamma))
        else:
            res.append(0)

    return np.array(res)


def visualize_and_save_trajectory(env, obs, env_actions_list, cnt_step, save_path=None, mmd_mode=False, mmd_score=None, camera_name="overhead_camera", save=True):
    """
    Convenience function to visualize and optionally save trajectory.
    
    Args:
        env: Environment
        obs: Current observation
        env_actions_list: List of action trajectories [(N, 7), ...] if mmd_mode=True, 
                         or single (N, 7) array if mmd_mode=False
        cnt_step: Current step count
        save_path: Path to save the visualization
        mmd_mode: If True, visualize all trajectories with different colors
        mmd_score: Optional MMD score to display in the plot
        camera_name: Name of the camera to visualize
        save: If True, save the visualization to a file
    """
    import matplotlib.pyplot as plt
    
    annotated_images = visualize_trajectory_on_cameras(env, obs, env_actions_list, mmd_mode=mmd_mode, camera_name=camera_name)
    
    # Create figure with only overhead camera
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if camera_name in annotated_images:
        ax.imshow(cv2.cvtColor(annotated_images[camera_name], cv2.COLOR_BGR2RGB))
        ax.set_title(f'{camera_name} - Step {cnt_step}')
        ax.axis('off')
        
        # Add MMD score text in top-left corner if available
        if mmd_score is not None:
            ax.text(0.02, 0.98, f'MMD: {mmd_score:.6f}', 
                   transform=ax.transAxes,
                   fontsize=14,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=2))
    else:
        raise ValueError(f"Camera {camera_name} not found in annotated images")
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'trajectory_step_{cnt_step}.png'
    if save:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        log.debug(f"Saved trajectory visualization to {save_path}")
    
    return annotated_images
        
    
def project_3d_to_2d(points_3d, extrinsic, intrinsic):
    """
    Project 3D points in world coordinates to 2D pixel coordinates.
    
    Args:
        points_3d: (N, 3) array of 3D points in world coordinates
        extrinsic: (4, 4) camera extrinsic matrix [R|t]
        intrinsic: (3, 3) camera intrinsic matrix
    
    Returns:
        points_2d: (N, 2) array of 2D pixel coordinates
        depths: (N,) array of depths (for filtering points behind camera)
    """
    # Convert to homogeneous coordinates
    points_3d_homo = np.concatenate([points_3d, np.ones((len(points_3d), 1))], axis=1)
    
    # Transform to camera coordinates
    points_cam = (extrinsic @ points_3d_homo.T).T
    
    # Extract x, y, z in camera frame
    x_cam = points_cam[:, 0]
    y_cam = points_cam[:, 1]
    z_cam = points_cam[:, 2]
    
    # Project to image plane using intrinsics
    points_2d_homo = (intrinsic @ points_cam[:, :3].T).T
    
    # Normalize by depth
    points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
    
    return points_2d, z_cam


def compute_future_ee_poses_using_controller(env, env_actions, obs):
    left_planner = env.robot.left_mplib_planner.planner
    right_planner = env.robot.right_mplib_planner.planner
    
    robot_base_pose_left = left_planner.robot.get_base_pose()
    robot_base_pose_right = right_planner.robot.get_base_pose()
    # Get current state
    
    ee_positions = []
    current_qpos_left = env.robot.left_entity.get_qpos()
    current_qpos_right = env.robot.right_entity.get_qpos()

    arm_indices_left = left_planner.move_group_joint_indices
    arm_indices_right = right_planner.move_group_joint_indices

    eef_link_index_left = left_planner.pinocchio_model.get_link_names().index('fl_link6') # Should be 42 from experience
    eef_link_index_right = right_planner.pinocchio_model.get_link_names().index('fr_link6') # Should be 43 from experience

    def get_tcp_pos(world_pose, arm_tag):
        mat = t3d.quaternions.quat2mat(world_pose.q)
        g_trans = env.robot.left_global_trans_matrix if arm_tag == "left" else env.robot.right_global_trans_matrix
        delta = env.robot.left_delta_matrix if arm_tag == "left" else env.robot.right_delta_matrix
        bias = env.robot.left_gripper_bias if arm_tag == "left" else env.robot.right_gripper_bias
        
        # Combine rotations
        combined_rot = mat @ g_trans @ delta
        # Offset by (bias - 0.12) along the local X axis of the combined frame
        offset = combined_rot @ np.array([bias, 0, 0])
        return world_pose.p + offset
    
    # Iterate through actions and compute cumulative poses
    for action in env_actions:
        left_arm_actions = action[:6]
        right_arm_actions = action[7:13]
        target_qpos_left = current_qpos_left.copy()
        target_qpos_right = current_qpos_right.copy()
        target_qpos_left[arm_indices_left] = left_arm_actions
        target_qpos_right[arm_indices_right] = right_arm_actions

        # Use the controller's actual compute_target_pose method
        left_planner.pinocchio_model.compute_forward_kinematics(target_qpos_left)
        left_target_pose = left_planner.pinocchio_model.get_link_pose(eef_link_index_left) 

        right_planner.pinocchio_model.compute_forward_kinematics(target_qpos_right)
        right_target_pose = right_planner.pinocchio_model.get_link_pose(eef_link_index_right)
        
        # Extract position from the pose
        ee_positions.append(
            (get_tcp_pos(robot_base_pose_left * left_target_pose, "left"), 
             get_tcp_pos(robot_base_pose_right * right_target_pose, "right")))
    
    return np.array(ee_positions)


def visualize_trajectory_on_cameras(env, obs, env_actions_list, mmd_mode=False, camera_name="head_camera"):
    """
    Visualize the predicted trajectory on camera images.
    Uses the actual controller's compute_target_pose to ensure accuracy.
    
    Args:
        env: The environment
        obs: Current observation dict with keys ['observation']
        env_actions_list: List of (N, 14) action arrays if mmd_mode=True, 
                         or single (N, 14) array if mmd_mode=False
        mmd_mode: If True, visualize all trajectories with different colors
    
    Returns:
        annotated_images: Dict with camera name keys and annotated images
    """
    # Get current EE pose from robot and stack for dual-arm handling
    current_ee_pose = np.stack([env.robot.get_left_ee_pose()[:3], env.robot.get_right_ee_pose()[:3]], axis=0)
    current_tcp_world = current_ee_pose # Shape: (2, 3)
    
    # Convert to list if not in MMD mode
    if not mmd_mode:
        env_actions_list = [env_actions_list]
    
    # Compute future end-effector positions for all trajectories
    all_trajectories = []
    for env_actions in env_actions_list:
        ee_positions_world = compute_future_ee_poses_using_controller(env, env_actions, obs)
        # Add current position as the starting point (expanding dims for vstack compatibility)
        all_positions = np.vstack([current_tcp_world[np.newaxis], ee_positions_world])
        all_trajectories.append(all_positions)
    
    # Reshape to (num_arms, num_samples, horizon, 3)
    all_trajectories_array = np.array(all_trajectories).transpose(2, 0, 1, 3)
    num_arms = all_trajectories_array.shape[0]
    num_trajectories = all_trajectories_array.shape[1]

    annotated_images = {}
    
    # Generate distinct colors for each trajectory
    trajectory_colors = []
    if mmd_mode:
        # Use HSV color space for better color distribution
        for i in range(num_trajectories):
            hue = int(180 * i / max(1, num_trajectories))
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
            trajectory_colors.append(tuple(int(c) for c in color_bgr))
    else:
        # Single trajectory: placeholder for gradient
        trajectory_colors = [(0, 255, 0)]
    
    for cam_name in [camera_name]:
        # Get camera parameters from the updated observation structure
        cam_params = obs['observation'][cam_name]
        
        # Try to get extrinsic and intrinsic (prioritizing _cv suffix per New Function 1)
        extrinsic = cam_params.get('extrinsic_cv', cam_params.get('extrinsic'))
        intrinsic = cam_params.get('intrinsic_cv', cam_params.get('intrinsic'))
        
        if extrinsic is None or intrinsic is None:
            continue
        
        # Get image
        img = obs['observation'][cam_name]['rgb'].copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Ensure image is in correct format (uint8 BGR)
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        # Draw all trajectories for both arms
        for arm_idx in range(num_arms):
            for traj_idx in range(num_trajectories):
                all_positions = all_trajectories_array[arm_idx, traj_idx]
                
                # Project 3D positions to 2D
                points_2d, depths = project_3d_to_2d(all_positions, extrinsic, intrinsic)
                
                # Filter points behind camera
                valid_mask = depths > 0
                
                # Get color for this trajectory
                if mmd_mode:
                    traj_color = trajectory_colors[traj_idx]
                else:
                    traj_color = None  # Will use per-point colors
                
                # Draw trajectory
                for i in range(len(points_2d)):
                    if not valid_mask[i]:
                        continue
                    
                    x, y = int(points_2d[i, 0]), int(points_2d[i, 1])
                    
                    # Check if point is within image bounds
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        # Determine color
                        if mmd_mode:
                            point_color = traj_color
                        else:
                            # Gradient from green (current) to red (future)
                            ratio = i / max(1, len(points_2d) - 1)
                            b = 0
                            g = int(255 * (1 - ratio))
                            r = int(255 * ratio)
                            point_color = (b, g, r)
                        
                        # Draw point
                        if i == 0 and traj_idx == 0:  # Current position
                            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
                            cv2.circle(img, (x, y), 3, (255, 255, 255), 1)
                            if not mmd_mode:
                                label = "Left" if arm_idx == 0 else "Right"
                                cv2.putText(img, label, (x + 12, y + 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        elif i > 0:
                            # Future positions
                            circle_radius = 1 if mmd_mode else 2
                            cv2.circle(img, (x, y), circle_radius, point_color, -1)
                            if not mmd_mode:
                                cv2.circle(img, (x, y), 2, (255, 255, 255), 1)
                                cv2.putText(img, f"t+{i}", (x + 10, y - 10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, point_color, 2)
                    
                    # Draw line connecting points
                    if i > 0 and valid_mask[i-1]:
                        x_prev, y_prev = int(points_2d[i-1, 0]), int(points_2d[i-1, 1])
                        if (0 <= x_prev < img.shape[1] and 0 <= y_prev < img.shape[0] and
                            0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
                            line_width = 2 if mmd_mode else 3
                            line_color = traj_color if mmd_mode else point_color
                            cv2.line(img, (x_prev, y_prev), (x, y), line_color, line_width)
        
        annotated_images[cam_name] = img
    
    return annotated_images


def generate_primitives_qpos(
    env,
    obs: Dict,
    camera_name: str,
    nudge_distance: float,
    horizon_steps: int,
    arm_tag: str = "both"
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Generate 14-dim qpos action primitives for the bimanual ALOHA robot.

    Computes Cartesian displacement primitives (left, right, up, down, forward, retreat),
    converts them to target EE poses, then uses IK to get joint-space trajectories.

    Args:
        env: TASK_ENV (Base_Task instance) with robot and planner access.
        obs: Current observation dict from env.get_obs().
        camera_name: Camera name for determining frame-relative directions.
        nudge_distance: Displacement magnitude in meters (e.g., 0.05).
        horizon_steps: Number of timesteps for the trajectory.
        arm_tag: "left", "right", or "both" - which arm(s) to generate primitives for.

    Returns:
        Tuple of (trajectories, names):
            trajectories: List of (horizon_steps, 14) numpy arrays in qpos format.
            names: List of primitive names (e.g., "Nudge Left").
    """
    # Get camera extrinsic for frame-relative directions
    cam_params = obs['observation'][camera_name]
    extrinsic = cam_params.get('extrinsic_cv')
    if extrinsic is None:
        raise ValueError(f"Could not find extrinsic_cv for {camera_name}")

    # extrinsic is W2C. We need C2W rotation.
    R_c2w = extrinsic[:3, :3].T

    # Direction vectors in camera frame (standard OpenCV: X=right, Y=down, Z=forward).
    # Down and Forward have small off-axis components to avoid pushing the EE into
    # extreme poses (table collision / workspace boundary) that cause IK failure.
    direction_vecs = {
        "Nudge Left":     np.array([-1, 0, 0]),
        "Nudge Right":    np.array([1, 0, 0]),
        "Nudge Up":       np.array([0, -1, 0]),
        "Nudge Down":     np.array([0, 0.8, -0.2]),
        "Nudge Forward":  np.array([0, 0, 1]),
        "Retreat":        np.array([0, 0, -1]),
        # "None":           np.array([0, 0, 0])
    }

    # Get current state
    current_state = obs["joint_action"]["vector"]  # 14-dim
    left_arm_dim = len(obs["joint_action"]["left_arm"])   # 6
    right_arm_dim = len(obs["joint_action"]["right_arm"]) # 6

    current_left_joints = current_state[:left_arm_dim]
    current_left_gripper = current_state[left_arm_dim]
    current_right_joints = current_state[left_arm_dim + 1:left_arm_dim + 1 + right_arm_dim]
    current_right_gripper = current_state[left_arm_dim + 1 + right_arm_dim]

    # Get current EE poses
    current_left_ee = np.array(env.robot.get_left_ee_pose(), dtype=np.float64)
    current_right_ee = np.array(env.robot.get_right_ee_pose(), dtype=np.float64)

    trajectories = []
    names = []

    for name, cam_vec in direction_vecs.items():
        # Transform direction to world frame
        world_displacement = R_c2w @ (cam_vec * nudge_distance)

        # Compute target EE poses
        target_left_ee = current_left_ee.copy()
        target_right_ee = current_right_ee.copy()

        target_left_joints = current_left_joints.copy()
        target_right_joints = current_right_joints.copy()

        ik_success = False

        if arm_tag in ("left", "both"):
            target_left_ee[:3] += world_displacement
            try:
                left_result = env.robot.left_plan_path(target_left_ee.tolist())
                if left_result["status"] == "Success":
                    target_left_joints = left_result["position"][-1][:left_arm_dim]
                    ik_success = True
                else:
                    print(f"IK failed for left arm primitive '{name}': {left_result['status']}")
                    continue
            except Exception as e:
                print(f"IK exception for left arm primitive '{name}': {e}")
                continue

        if arm_tag in ("right", "both"):
            target_right_ee[:3] += world_displacement
            try:
                right_result = env.robot.right_plan_path(target_right_ee.tolist())
                if right_result["status"] == "Success":
                    target_right_joints = right_result["position"][-1][:right_arm_dim]
                    ik_success = True
                else:
                    print(f"IK failed for right arm primitive '{name}': {right_result['status']}")
                    continue
            except Exception as e:
                print(f"IK exception for right arm primitive '{name}': {e}")
                continue

        if not ik_success:
            continue

        # Interpolate from current to target over horizon_steps
        traj = np.zeros((horizon_steps, len(current_state)))  # (horizon_steps, 14)

        for t in range(horizon_steps):
            alpha = (t + 1) / horizon_steps  # 0 -> 1 linear interpolation

            interp_left = current_left_joints * (1 - alpha) + target_left_joints * alpha
            interp_right = current_right_joints * (1 - alpha) + target_right_joints * alpha

            traj[t, :left_arm_dim] = interp_left
            traj[t, left_arm_dim] = current_left_gripper  # keep gripper unchanged
            traj[t, left_arm_dim + 1:left_arm_dim + 1 + right_arm_dim] = interp_right
            traj[t, left_arm_dim + 1 + right_arm_dim] = current_right_gripper  # keep gripper unchanged

        trajectories.append(traj)
        names.append(name)

    if len(trajectories) == 0:
        print("All IK solutions failed. Returning no-op primitive.")
        noop_traj = np.tile(current_state, (horizon_steps, 1))
        trajectories.append(noop_traj)
        names.append("Stay")

    return trajectories, names