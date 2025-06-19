# train_eval/evaluate_libero.py

from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from config.config import PolicyConfig
from model.predictor.policy import HierarchicalAutoregressivePolicy
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark_dict
from model.modeling_bidirectional_rtdiffusion import HierarchicalPolicy
import torch
import numpy as np
import imageio
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
import draccus
import os
import math
from collections import deque

from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

from model.predictor.policy import HierarchicalAutoregressivePolicy
from config.config import PolicyConfig
from model.modeling_bidirectional_rtdiffusion import HierarchicalPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features

# 토크나이저 병렬 처리 경고 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# [MODIFIED] Add the quaternion to axis-angle conversion helper function


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# --- 1. 평가 설정을 위한 dataclass 정의 (가이드의 GenerateConfig 역할) ---


@dataclass
class EvalConfig:
    # --- 필수 설정 ---
    planner_checkpoint_path: str = "outputs/train/bidirectional_transformerl/model_final.pth"

    # --- LIBERO 벤치마크 설정 ---
    benchmark_name: str = "libero_object"
    task_order_index: int = 2  # 사용할 태스크 순서 인덱스

    # --- 평가 관련 설정 ---
    num_trials_per_task: int = 10
    seed: int = 42

    # --- 결과 저장 및 로깅 설정 ---
    output_dir: str = "outputs/eval/libero_benchmark"

    # [MODIFIED] Added parameters for stabilization and replanning
    num_steps_wait: int = 10
    replan_steps: int = 4


@draccus.wrap()
def main(cfg: EvalConfig):
    """ 메인 평가 함수 """
    # 결과 저장 폴더 생성
    output_directory = Path(cfg.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models and metadata...")

    # --- [FIXED] Load the config saved during training ---
    planner_checkpoint_path = Path(cfg.planner_checkpoint_path)
    planner_checkpoint_dir = planner_checkpoint_path.parent
    policy_cfg = PolicyConfig.from_pretrained(planner_checkpoint_dir)
    print(
        f"Loaded training config from {planner_checkpoint_dir} with state_dim={policy_cfg.hierarchical_transformer.state_dim}")
    # --- END FIX ---

    # 데이터셋 메타데이터 로드 (정규화 통계 등을 위해 필요)
    metadata = LeRobotDatasetMetadata(policy_cfg.data.dataset_repo_id)
    features = dataset_to_policy_features(metadata.features)

    # a. 계획 모델 (Planner) 로딩
    planner_model = HierarchicalAutoregressivePolicy(config=policy_cfg)
    planner_model.load_state_dict(torch.load(
        cfg.planner_checkpoint_path, map_location=device))
    planner_model.to(device).eval()
    print(f"Planner model loaded from: {cfg.planner_checkpoint_path}")

    # c. 통합 정책 파이프라인 생성 (Wrapper class usage)
    policy = HierarchicalPolicy(
        policy_model=planner_model,  # The end-to-end model is passed here
        dataset_stats=metadata.stats,
        output_features={"action": features["action"]},
        config=policy_cfg
    )
    print("Policy wrapper with per-step replanning is ready.")

    policy.to(device)

    # --- LIBERO 벤치마크 및 환경 설정 ---
    benchmark_dict = get_benchmark_dict()
    benchmark_cls = benchmark_dict[cfg.benchmark_name]

    # [CORRECT] Correct instantiation
    task_suite = benchmark_cls(task_order_index=cfg.task_order_index)

    n_tasks = task_suite.n_tasks
    # [FIXED] Reverted to the original, correct iteration logic
    task_names = task_suite.get_task_names()

    print(
        f"Starting evaluation on benchmark '{cfg.benchmark_name}' with {n_tasks} tasks.")

    total_successes = 0
    total_episodes = 0

    # [MODIFIED] Set max_steps based on task suite name, like in Seer
    if cfg.benchmark_name == "libero_spatial":
        max_steps = 220
    elif cfg.benchmark_name == "libero_object":
        max_steps = 580
    elif cfg.benchmark_name == "libero_goal":
        max_steps = 300
    elif cfg.benchmark_name == "libero_10":
        max_steps = 520
    else:
        max_steps = 400  # Default for other suites like libero_90

    for task_idx, task_name in enumerate(task_names):
        print(
            f"\n===== Evaluating Task {task_idx+1}/{n_tasks}: {task_name} =====")

        task_successes = 0
        task = task_suite.get_task(task_idx)
        task_description = task.language

        # --- 바로 이 부분을 수정했습니다 ---
        task_init_states = task_suite.get_task_init_states(
            task_idx)
        # --- 여기까지 수정 ---

        for trial_idx in tqdm(range(cfg.num_trials_per_task), desc=f"Task '{task_name}'"):
            total_episodes += 1
            env = OffScreenRenderEnv(
                bddl_file_name=task_suite.get_task_bddl_file_path(task_idx),
                control_freq=20,
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=256,
                camera_widths=256
            )
            env.seed(cfg.seed + trial_idx)

            init_state = task_init_states[trial_idx % len(task_init_states)]
            obs = env.set_init_state(init_state)

            # [MODIFIED] Wait for simulation to stabilize here in the main loop
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(np.zeros(7))

            # Pass the stabilized obs and max_steps to run_episode
            success, frames = run_episode(
                policy=policy,
                env=env,
                task_description=task_description,
                initial_obs=obs,
                max_steps=max_steps,
                device=device
            )

            if success:
                task_successes += 1
                total_successes += 1

            video_path = output_directory / \
                f"{task_name.replace(' ', '_')}_trial_{trial_idx}_{'success' if success else 'fail'}.mp4"
            imageio.mimsave(str(video_path), frames, fps=30)

            env.close()
        task_success_rate = (task_successes / cfg.num_trials_per_task) * 100
        print(
            f"Result for '{task_name}': {task_successes}/{cfg.num_trials_per_task} ({task_success_rate:.1f}%) successful.")

    overall_success_rate = (total_successes / total_episodes) * 100
    print(f"\n===== Overall Benchmark Result =====")
    print(f"Benchmark: {cfg.benchmark_name}")
    print(
        f"Total Success Rate: {total_successes}/{total_episodes} ({overall_success_rate:.1f}%)")


def run_episode(policy: HierarchicalPolicy, env, task_description, initial_obs, max_steps, device):
    """ [MODIFIED] The evaluation loop is now simpler and more robust. """
    policy.reset()

    obs = initial_obs
    frames = []

    # Record first frame after stabilization
    concatenated_frame = np.concatenate(
        [obs["robot0_eye_in_hand_image"], obs["agentview_image"]], axis=1)
    frames.append(np.flip(concatenated_frame, axis=0))

    t = 0
    terminated = False
    info = {}

    while t < max_steps:
        state_np = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(
            obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
        state = torch.from_numpy(state_np).to(device, dtype=torch.float32)

        front_img = torch.from_numpy(obs["agentview_image"]).to(
            device, non_blocking=True).permute(2, 0, 1).to(torch.float32)
        wrist_img = torch.from_numpy(obs["robot0_eye_in_hand_image"]).to(
            device, non_blocking=True).permute(2, 0, 1).to(torch.float32)
        image = torch.stack([front_img, wrist_img], dim=0)

        observation_dict = {
            "observation.image": image, "observation.state": state, "language_instruction": [task_description]
        }

        # Replanning happens inside select_action on every step
        action_tensor = policy.select_action(observation_dict)
        action = action_tensor.cpu().numpy()
        obs, _, terminated, info = env.step(action)

        frames.append(np.flip(np.concatenate(
            [obs["robot0_eye_in_hand_image"], obs["agentview_image"]], axis=1), 0))

        if terminated:
            break

        t += 1

    return info.get("success", False), frames


if __name__ == "__main__":
    main()
