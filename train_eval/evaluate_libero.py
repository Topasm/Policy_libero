# train_eval/evaluate_libero.py

import torch
import numpy as np
import imageio
import tqdm
from pathlib import Path
from dataclasses import dataclass
import draccus

from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv

from model.predictor.policy import HierarchicalAutoregressivePolicy
from model.predictor.config import PolicyConfig
from model.invdyn.invdyn import MlpInvDynamic
from model.modeling_bidirectional_rtdiffusion import HierarchicalPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features

# --- 1. 평가 설정을 위한 dataclass 정의 (가이드의 GenerateConfig 역할) ---


@dataclass
class EvalConfig:
    # --- 필수 설정 ---
    planner_checkpoint_path: str = "outputs/train/bidirectional_transformer/model_final.pth"
    invdyn_checkpoint_path: str = "outputs/train/invdyn_only/invdyn_final.pth"

    # --- LIBERO 벤치마크 설정 ---
    benchmark_name: str = "libero_spatial"
    task_order_index: int = 0  # 사용할 태스크 순서 인덱스

    # --- 평가 관련 설정 ---
    num_trials_per_task: int = 10
    seed: int = 42

    # --- 결과 저장 및 로깅 설정 ---
    output_dir: str = "outputs/eval/libero_benchmark"


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
    # --- 여기까지 수정 ---

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
                camera_heights=84,                         # 이미지 높이
                camera_widths=84                           # 이미지 너비
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


def run_episode(policy, env, task_description, initial_state, device):
    """ 한 에피소드를 실행하고 성공 여부와 렌더링된 프레임을 반환합니다. """

    policy.reset()
    env.set_init_state(initial_state)
    obs, _, _, _ = env.step(np.zeros(7))
    frames = [np.concatenate(
        [obs["frontview_image"], obs["robot0_eye_in_hand_image"]], axis=1)]

    terminated = False

    while not terminated:
        # --- [FIXED] Manually construct the 8D state vector to match training data ---
        # The model was trained on an 8D state. We construct it from the env's obs dict.
        # The most likely 8D state is [eef_pos (3), eef_quat (4), gripper (1)].
        eef_pos = obs["robot0_eef_pos"]
        eef_quat = obs["robot0_eef_quat"]
        gripper_qpos = obs["robot0_gripper_qpos"][0:1]

        # Concatenate to form the 8D state vector
        state_np = np.concatenate([eef_pos, eef_quat, gripper_qpos])
        state = torch.from_numpy(state_np).to(
            device, dtype=torch.float32).unsqueeze(0)
        # --- END FIX ---

        front_img = torch.from_numpy(obs["frontview_image"]).permute(
            2, 0, 1).to(device, dtype=torch.float32) / 255.0
        wrist_img = torch.from_numpy(obs["robot0_eye_in_hand_image"]).permute(
            2, 0, 1).to(device, dtype=torch.float32) / 255.0

        image = torch.cat([front_img, wrist_img], dim=0).unsqueeze(0)

        observation_dict = {
            "observation.image": image,
            "observation.state": state,  # Pass the correctly shaped 8D state
            "language_instruction": [task_description]
        }

        action = policy.select_action(observation_dict).cpu().numpy()
        obs, _, terminated, info = env.step(action)

        frames.append(np.concatenate(
            [obs["robot0_eye_in_hand_image"], obs["frontview_image"]], axis=1))

        if terminated:
            break

    return info["success"], frames


if __name__ == "__main__":
    main()
