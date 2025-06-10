# train_eval/evaluate_libero.py

import torch
import numpy as np
import imageio
import tqdm
from pathlib import Path
from dataclasses import dataclass
import draccus

# LIBERO 벤치마크 및 환경 관련 import
from libero.libero.benchmark import get_benchmark_dict, BENCHMARK_MAPPING
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

# 모델 및 설정 관련 import
from model.predictor.policy import HierarchicalAutoregressivePolicy
from model.predictor.config import PolicyConfig
from model.invdyn.invdyn import MlpInvDynamic
from model.modeling_bidirectional_rtdiffusion import HierarchicalPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features

# --- 1. 평가 설정을 위한 dataclass 정의 (가이드의 GenerateConfig 역할) ---


@dataclass
class EvalConfig:
    # --- 필수 설정: 학습된 모델의 체크포인트 경로 ---
    planner_checkpoint_path: str
    invdyn_checkpoint_path: str

    # --- LIBERO 벤치마크 설정 ---
    # 예: "libero_spatial", "libero_object", "libero_goal"
    benchmark_name: str = "libero_spatial"

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

    # --- 2. 모델 로딩 (가이드의 MyModel.__init__ 역할) ---
    print("Loading models and metadata...")

    # 데이터셋 메타데이터 로드 (정규화 통계 등을 위해 필요)
    # PolicyConfig의 기본 dataset_repo_id를 사용
    policy_cfg = PolicyConfig()
    metadata = LeRobotDatasetMetadata(policy_cfg.data.dataset_repo_id)
    features = dataset_to_policy_features(metadata.features)

    # a. 계획 모델 (Planner) 로딩
    planner_model = HierarchicalAutoregressivePolicy(config=policy_cfg)
    planner_model.load_state_dict(torch.load(
        cfg.planner_checkpoint_path, map_location=device))
    planner_model.to(device).eval()
    print(f"Planner model loaded from: {cfg.planner_checkpoint_path}")

    # b. 행동 모델 (Controller / Inverse Dynamics) 로딩
    invdyn_model = MlpInvDynamic(
        o_dim=features["observation.state"].shape[0],
        a_dim=features["action"].shape[0],
        hidden_dim=policy_cfg.inverse_dynamics.hidden_dim,
    )
    invdyn_model.load_state_dict(torch.load(
        cfg.invdyn_checkpoint_path, map_location=device))
    invdyn_model.to(device).eval()
    print(f"Inverse dynamics model loaded from: {cfg.invdyn_checkpoint_path}")

    # c. 통합 정책 파이프라인 생성 (모델 래퍼)
    combined_policy = HierarchicalPolicy(
        bidirectional_transformer=planner_model,
        inverse_dynamics_model=invdyn_model,
        dataset_stats=metadata.stats,
        all_dataset_features=metadata.features,
        n_obs_steps=policy_cfg.data.n_obs_steps,
        output_features={"action": features["action"]}
    )
    print("Combined policy pipeline is ready.")

    # --- 3. LIBERO 벤치마크 및 환경 설정 ---
    benchmark_dict = get_benchmark_dict()
    task_suite = benchmark_dict[cfg.benchmark_name]
    n_tasks = task_suite.n_tasks
    task_names = task_suite.get_task_names()

    print(
        f"Starting evaluation on benchmark '{cfg.benchmark_name}' with {n_tasks} tasks.")

    total_successes = 0
    total_episodes = 0

    # --- 4. 태스크 순회 및 평가 실행 ---
    for task_idx, task_name in enumerate(task_names):
        print(
            f"\n===== Evaluating Task {task_idx+1}/{n_tasks}: {task_name} =====")
        task_successes = 0

        # LIBERO 태스크에서 환경 인스턴스와 초기 상태 가져오기
        task = task_suite.get_task(task_idx)
        task_description = task.language
        task_init_states = task.get_init_states()

        for trial_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task '{task_name}'"):
            total_episodes += 1

            # 환경 생성 및 초기화
            env = OffScreenRenderEnv(task.env_name, control_freq=20)
            env.seed(cfg.seed + trial_idx)
            env.reset()

            # 에피소드 실행
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

            # 결과 비디오 저장
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

    policy.reset()  # 에피소드 시작 전 정책 내부 큐 초기화

    # 환경을 지정된 초기 상태로 설정
    env.set_init_state(initial_state)

    # LIBERO 환경의 관측값은 state, robot_state, front_image, wrist_image 등을 포함
    obs, _, _, _ = env.step(np.zeros(7))

    frames = [obs["wrist_image"], obs["front_image"]]  # 초기 프레임 저장
    terminated = False

    while not terminated:
        # --- 5. 관측값 전처리 (가이드의 predict_action 역할) ---
        # LIBERO 환경의 관측값을 모델이 기대하는 형식으로 변환
        state = torch.from_numpy(obs["robot_state"]).to(
            device, dtype=torch.float32).unsqueeze(0)

        front_img = torch.from_numpy(obs["front_image"]).permute(
            2, 0, 1).to(device, dtype=torch.float32) / 255.0
        wrist_img = torch.from_numpy(obs["wrist_image"]).permute(
            2, 0, 1).to(device, dtype=torch.float32) / 255.0

        # 두 이미지를 채널 차원에서 합쳐 6채널 이미지 생성
        image = torch.cat([front_img, wrist_img], dim=0).unsqueeze(0)

        # 모델 입력 딕셔너리 생성
        observation_dict = {
            "observation.image": image,
            "observation.state": state,
            "language_instruction": [task_description]  # 배치 처리를 위해 리스트로 전달
        }

        # --- 6. 행동 예측 및 실행 ---
        action = policy.select_action(observation_dict).cpu().numpy()
        obs, _, terminated, info = env.step(action)

        # 렌더링된 프레임 저장 (결과 비디오를 위해)
        # LIBERO 환경은 RGB 순서로 이미지를 반환
        frames.append(np.concatenate(
            [obs["wrist_image"], obs["front_image"]], axis=1))

    return info["success"], frames


if __name__ == "__main__":
    main()
