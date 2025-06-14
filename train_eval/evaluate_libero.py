# train_eval/evaluate_libero.py

from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from model.modeling_bidirectional_rtdiffusion import HierarchicalPolicy
from model.invdyn.invdyn import MlpInvDynamic
from config.config import PolicyConfig
from model.predictor.policy import HierarchicalAutoregressivePolicy
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.benchmark import get_benchmark_dict
import torch
import numpy as np
import imageio
import tqdm
from pathlib import Path
from dataclasses import dataclass, field
import draccus
import os
import math
from collections import deque

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
    planner_checkpoint_path: str = "outputs/train/bidirectional_transformerf/model_final.pth"
    invdyn_checkpoint_path: str = "outputs/train/invdyn_only/invdyn_final.pth"

    # --- LIBERO 벤치마크 설정 ---
    benchmark_name: str = "libero_goal"
    task_order_index: int = 0  # 사용할 태스크 순서 인덱스

    # --- 평가 관련 설정 ---
    num_trials_per_task: int = 10
    seed: int = 42

    # --- 결과 저장 및 로깅 설정 ---
    output_dir: str = "outputs/eval/libero_benchmark"

    # [MODIFIED] Added parameters for stabilization and replanning
    num_steps_wait: int = 10
    replan_steps: int = 5


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
        planner_checkpoint_path, map_location=device))
    planner_model.to(device).eval()
    print(f"Planner model loaded from: {cfg.planner_checkpoint_path}")

    # b. 행동 모델 (Controller / Inverse Dynamics) 로딩
    invdyn_model = MlpInvDynamic(
        o_dim=policy_cfg.hierarchical_transformer.state_dim,
        a_dim=features["action"].shape[0],
        hidden_dim=policy_cfg.inverse_dynamics.hidden_dim,
    )
    invdyn_model.load_state_dict(torch.load(
        cfg.invdyn_checkpoint_path, map_location=device))
    invdyn_model.to(device).eval()
    print(f"Inverse dynamics model loaded from: {cfg.invdyn_checkpoint_path}")

    # c. 통합 정책 파이프라인 생성
    combined_policy = HierarchicalPolicy(
        bidirectional_transformer=planner_model,
        inverse_dynamics_model=invdyn_model,
        dataset_stats=metadata.stats,
        all_dataset_features=metadata.features,
        n_obs_steps=policy_cfg.data.n_obs_steps,
        output_features={"action": features["action"]}
    )

    # --- [FIXED] Move the entire policy object to the selected device ---
    combined_policy.to(device)
    combined_policy.eval()
    print("Combined policy pipeline is ready.")

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

        for trial_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task '{task_name}'"):
            total_episodes += 1
            # --- 바로 이 부분을 수정하시면 됩니다 ---
            env = OffScreenRenderEnv(
                bddl_file_name=task_suite.get_task_bddl_file_path(task_idx),
                control_freq=20,
                # 전면, 손목 카메라 활성화
                camera_names=["frontview", "robot0_eye_in_hand"],
                camera_heights=224,  # 경고 메시지 방지를 위해 16의 배수로 수정
                camera_widths=224
            )
            # --- 여기까지 수정 ---
            env.seed(cfg.seed + trial_idx)
            env.reset()

            init_state = task_init_states[trial_idx % len(task_init_states)]
            success, frames = run_episode(
                policy=combined_policy,
                env=env,
                task_description=task_description,
                initial_state=init_state,
                device=device,
                cfg=cfg
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


def run_episode(policy, env, task_description, initial_state, device, cfg: EvalConfig):
    """ 한 에피소드를 실행하고 성공 여부와 렌더링된 프레임을 반환합니다. """
    policy.reset()
    env.set_init_state(initial_state)

    # action_plan is now managed here
    action_plan = deque()

    obs, _, _, _ = env.step(np.zeros(7))
    frames = []

    max_steps = env.env.horizon
    t = 0

    while t < max_steps-1:
        # 1. Wait for simulation to stabilize
        if t < cfg.num_steps_wait:
            obs, _, terminated, info = env.step(np.zeros(7))
            if t == cfg.num_steps_wait - 1:
                # Start recording frames after waiting
                concatenated_frame = np.concatenate(
                    [obs["frontview_image"], obs["robot0_eye_in_hand_image"]], axis=1)
                frames.append(np.flip(concatenated_frame, axis=0))
            t += 1
            continue

        # 2. Check if we need to replan (every replan_steps)
        if (t - cfg.num_steps_wait) % cfg.replan_steps == 0:
            # Construct observation dict for the policy
            state_np = np.concatenate([obs["robot0_eef_pos"], _quat2axisangle(
                obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]])
            state = torch.from_numpy(state_np).to(
                device, dtype=torch.float32).unsqueeze(0)

            front_img = torch.from_numpy(obs["frontview_image"]).permute(
                2, 0, 1).to(device, dtype=torch.float32) / 255.0
            wrist_img = torch.from_numpy(obs["robot0_eye_in_hand_image"]).permute(
                2, 0, 1).to(device, dtype=torch.float32) / 255.0

            image = torch.stack([front_img, wrist_img], dim=0).unsqueeze(0)

            observation_dict = {
                "observation.image": image,
                "observation.state": state,
                "language_instruction": [task_description]
            }

            # Generate a new chunk of actions
            action_chunk = policy.plan_actions(observation_dict)
            action_plan.extend(action_chunk)

        # 3. Get the next action from the plan
        if not action_plan:
            # If plan is empty (e.g., not enough history yet), take a dummy action
            action = np.zeros(7)
            action[-1] = -1  # Gripper open
        else:
            action = action_plan.popleft().cpu().numpy()

        obs, _, terminated, info = env.step(action)

        # Save frame for video
        concatenated_frame = np.concatenate(
            [obs["robot0_eye_in_hand_image"], obs["frontview_image"]], axis=1)
        frames.append(np.flip(concatenated_frame, axis=0))

        if terminated:
            break

        t += 1

    if "success" not in info:
        info["success"] = False

    return info["success"], frames


if __name__ == "__main__":
    main()
