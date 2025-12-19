"""
강화학습 에이전트 성능 검증 스크립트

다음과 같은 검증 방법을 제공합니다:
1. 고정된 환경 조건에서 성능 평가 (baseline)
2. 랜덤 환경 조건에서 일반화 성능 평가
3. 특정 물리 파라미터 범위에서 robustness 테스트
4. 통계적 분석 (평균, 표준편차, 성공률 등)
5. 시각화 (학습 곡선, 분포 등)
"""

import os
import argparse
import json
from collections import deque
import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy import stats
import imageio

# 기존 모듈 import
from test_rnn_v2 import (
    RecurrentActorCritic, 
    set_env_physics, 
    load_config,
    sample_physics_from_ranges,
    DEFAULT_CONFIG
)


class AgentValidator:
    """에이전트 검증 클래스"""
    
    def __init__(self, model_path, config_path=None, device="cuda"):
        """
        Args:
            model_path: 학습된 모델 경로
            config_path: 설정 파일 경로
            device: 연산 디바이스
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.cfg = load_config(config_path)
        
        # 모델 로드
        env = gym.make(self.cfg["env_id"])
        obs_dim = env.observation_space.shape[0] * self.cfg.get("frame_stack", 1)
        action_dim = env.action_space.n
        
        self.model = RecurrentActorCritic(
            obs_dim, action_dim, 
            hidden_size=self.cfg.get("hidden_size", 256),
            device=self.device
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        env.close()
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Device: {self.device}")
    
    def run_episode(self, env, deterministic=True, render=False):
        """
        단일 에피소드 실행
        
        Args:
            env: Gymnasium 환경
            deterministic: True면 argmax 액션, False면 샘플링
            render: 렌더링 프레임 수집 여부
            
        Returns:
            dict: 에피소드 결과 (reward, length, actions, frames 등)
        """
        state, _ = env.reset()
        hidden = self.model.init_hidden(batch_size=1)
        state_buffer = deque(maxlen=self.cfg.get("frame_stack", 1))
        
        total_reward = 0.0
        done = False
        actions = []
        rewards = []
        frames = []
        
        step_count = 0
        while not done:
            if render:
                frames.append(env.render())
            
            # State stacking
            state_buffer.append(state)
            if len(state_buffer) < self.cfg.get("frame_stack", 1):
                pad = [np.zeros_like(state) for _ in range(
                    self.cfg.get("frame_stack", 1) - len(state_buffer))]
                stacked = np.concatenate(pad + list(state_buffer), axis=0)
            else:
                stacked = np.concatenate(list(state_buffer), axis=0)
            
            # Action selection
            with torch.no_grad():
                state_t = torch.FloatTensor(stacked).unsqueeze(0).to(self.device)
                logits, _, hidden = self.model(state_t, hidden)
                
                if logits.dim() == 3:
                    logits = logits.squeeze(0)[-1]
                elif logits.dim() == 2:
                    logits = logits[-1]
                
                if deterministic:
                    action = int(torch.argmax(logits).cpu().item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = int(dist.sample().cpu().item())
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            actions.append(action)
            rewards.append(float(reward))
            total_reward += reward
            state = next_state
            step_count += 1
        
        return {
            "total_reward": total_reward,
            "length": step_count,
            "actions": actions,
            "rewards": rewards,
            "frames": frames,
            "success": total_reward >= 200  # LunarLander 기준
        }
    
    def validate_fixed_env(self, n_episodes=100, gravity=-10.0, 
                          wind_power=10.0, turbulence_power=1.0,
                          deterministic=True):
        """
        고정된 환경 조건에서 성능 평가
        
        Args:
            n_episodes: 평가 에피소드 수
            gravity, wind_power, turbulence_power: 환경 물리 파라미터
            deterministic: 결정론적 정책 사용 여부
            
        Returns:
            dict: 평가 결과 통계
        """
        print(f"\n{'='*60}")
        print(f"고정 환경 평가 (n={n_episodes})")
        print(f"Gravity: {gravity}, Wind: {wind_power}, Turbulence: {turbulence_power}")
        print(f"{'='*60}")
        
        env = gym.make(self.cfg["env_id"])
        set_env_physics(env, gravity=gravity, enable_wind=True,
                       wind_power=wind_power, turbulence_power=turbulence_power)
        
        results = []
        for _ in tqdm(range(n_episodes), desc="Evaluating"):
            result = self.run_episode(env, deterministic=deterministic)
            results.append(result)
        
        env.close()
        
        # 통계 계산
        rewards = [r["total_reward"] for r in results]
        lengths = [r["length"] for r in results]
        successes = [r["success"] for r in results]
        
        stats = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "median_reward": np.median(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
            "success_rate": np.mean(successes) * 100,
            "rewards": rewards,
            "lengths": lengths,
            "successes": successes
        }
        
        self._print_stats(stats)
        return stats
    
    def validate_random_env(self, n_episodes=100, deterministic=True):
        """
        랜덤 환경 조건에서 일반화 성능 평가
        
        Args:
            n_episodes: 평가 에피소드 수
            deterministic: 결정론적 정책 사용 여부
            
        Returns:
            dict: 평가 결과 통계 및 물리 파라미터 정보
        """
        print(f"\n{'='*60}")
        print(f"랜덤 환경 평가 (n={n_episodes})")
        print(f"{'='*60}")
        
        env = gym.make(self.cfg["env_id"])
        
        results = []
        physics_params = []
        
        for _ in tqdm(range(n_episodes), desc="Evaluating"):
            # 랜덤 물리 파라미터 샘플링
            sampled = sample_physics_from_ranges(self.cfg)
            set_env_physics(env,
                          gravity=sampled.get("box2d_gravity"),
                          wind_power=sampled.get("wind_power"),
                          turbulence_power=sampled.get("turbulence_power"))
            
            result = self.run_episode(env, deterministic=deterministic)
            result["physics"] = sampled.copy()
            results.append(result)
            physics_params.append(sampled)
        
        env.close()
        
        # 통계 계산
        rewards = [r["total_reward"] for r in results]
        lengths = [r["length"] for r in results]
        successes = [r["success"] for r in results]
        
        stats = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "median_reward": np.median(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "mean_length": np.mean(lengths),
            "success_rate": np.mean(successes) * 100,
            "rewards": rewards,
            "lengths": lengths,
            "successes": successes,
            "physics_params": physics_params
        }
        
        self._print_stats(stats)
        return stats
    
    def validate_physics_sweep(self, param_name, param_range, n_episodes_per_value=20,
                              fixed_params=None, deterministic=True):
        """
        특정 물리 파라미터를 변화시키며 robustness 테스트
        
        Args:
            param_name: 테스트할 파라미터 이름 ('box2d_gravity', 'wind_power', 'turbulence_power')
            param_range: 테스트할 값들의 리스트
            n_episodes_per_value: 각 값당 실행할 에피소드 수
            fixed_params: 고정할 다른 파라미터들
            deterministic: 결정론적 정책 사용 여부
            
        Returns:
            dict: 파라미터별 평가 결과
        """
        print(f"\n{'='*60}")
        print(f"물리 파라미터 스윕 테스트: {param_name}")
        print(f"Range: {param_range}")
        print(f"{'='*60}")
        
        if fixed_params is None:
            fixed_params = {
                "box2d_gravity": -10.0,
                "wind_power": 10.0,
                "turbulence_power": 1.0
            }
        
        env = gym.make(self.cfg["env_id"])
        
        sweep_results = []
        
        for param_value in tqdm(param_range, desc=f"Sweeping {param_name}"):
            # 파라미터 설정
            params = fixed_params.copy()
            params[param_name] = param_value
            
            set_env_physics(env,
                          gravity=params.get("box2d_gravity"),
                          enable_wind=True,
                          wind_power=params.get("wind_power"),
                          turbulence_power=params.get("turbulence_power"))
            
            # 여러 에피소드 실행
            episode_rewards = []
            episode_lengths = []
            episode_successes = []
            
            for _ in range(n_episodes_per_value):
                result = self.run_episode(env, deterministic=deterministic)
                episode_rewards.append(result["total_reward"])
                episode_lengths.append(result["length"])
                episode_successes.append(result["success"])
            
            sweep_results.append({
                "param_value": param_value,
                "mean_reward": np.mean(episode_rewards),
                "std_reward": np.std(episode_rewards),
                "mean_length": np.mean(episode_lengths),
                "success_rate": np.mean(episode_successes) * 100,
                "rewards": episode_rewards
            })
        
        env.close()
        
        return {
            "param_name": param_name,
            "param_range": param_range,
            "results": sweep_results
        }
    
    def compare_with_baseline(self, baseline_stats, test_stats):
        """
        베이스라인과 테스트 결과 비교
        
        Args:
            baseline_stats: 베이스라인 통계
            test_stats: 테스트 통계
            
        Returns:
            dict: 비교 결과 및 통계적 유의성
        """
        print(f"\n{'='*60}")
        print(f"베이스라인 vs 테스트 비교")
        print(f"{'='*60}")
        
        # T-test for reward
        t_stat, p_value = stats.ttest_ind(
            baseline_stats["rewards"],
            test_stats["rewards"]
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (baseline_stats["std_reward"]**2 + test_stats["std_reward"]**2) / 2
        )
        cohens_d = (test_stats["mean_reward"] - baseline_stats["mean_reward"]) / pooled_std
        
        comparison = {
            "baseline_mean": baseline_stats["mean_reward"],
            "test_mean": test_stats["mean_reward"],
            "difference": test_stats["mean_reward"] - baseline_stats["mean_reward"],
            "percent_change": ((test_stats["mean_reward"] - baseline_stats["mean_reward"]) 
                              / baseline_stats["mean_reward"] * 100),
            "t_statistic": t_stat,
            "p_value": p_value,
            "cohens_d": cohens_d,
            "significant": p_value < 0.05,
            "baseline_success_rate": baseline_stats["success_rate"],
            "test_success_rate": test_stats["success_rate"]
        }
        
        print(f"Baseline Mean Reward: {comparison['baseline_mean']:.2f}")
        print(f"Test Mean Reward:     {comparison['test_mean']:.2f}")
        print(f"Difference:           {comparison['difference']:.2f} ({comparison['percent_change']:.1f}%)")
        print(f"P-value:              {comparison['p_value']:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        print(f"Cohen's d:            {comparison['cohens_d']:.3f}")
        print(f"Success Rate Change:  {comparison['baseline_success_rate']:.1f}% → {comparison['test_success_rate']:.1f}%")
        
        return comparison
    
    def visualize_results(self, stats_dict, save_dir="validation_plots"):
        """
        검증 결과 시각화
        
        Args:
            stats_dict: 검증 통계 딕셔너리
            save_dir: 저장 경로
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Reward 분포
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        for name, stats in stats_dict.items():
            plt.hist(stats["rewards"], bins=30, alpha=0.5, label=name)
        plt.xlabel("Total Reward")
        plt.ylabel("Frequency")
        plt.title("Reward Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Box plot
        plt.subplot(1, 3, 2)
        data = [stats["rewards"] for stats in stats_dict.values()]
        labels = list(stats_dict.keys())
        plt.boxplot(data, labels=labels)
        plt.ylabel("Total Reward")
        plt.title("Reward Comparison")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 3. Success rate
        plt.subplot(1, 3, 3)
        success_rates = [stats["success_rate"] for stats in stats_dict.values()]
        plt.bar(labels, success_rates)
        plt.ylabel("Success Rate (%)")
        plt.title("Success Rate Comparison")
        plt.ylim(0, 100)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "validation_comparison.png"), dpi=150)
        print(f"✓ Saved: {os.path.join(save_dir, 'validation_comparison.png')}")
        plt.close()
    
    def visualize_physics_sweep(self, sweep_result, save_dir="validation_plots"):
        """
        물리 파라미터 스윕 결과 시각화
        
        Args:
            sweep_result: sweep 결과 딕셔너리
            save_dir: 저장 경로
        """
        os.makedirs(save_dir, exist_ok=True)
        
        param_name = sweep_result["param_name"]
        param_values = [r["param_value"] for r in sweep_result["results"]]
        mean_rewards = [r["mean_reward"] for r in sweep_result["results"]]
        std_rewards = [r["std_reward"] for r in sweep_result["results"]]
        success_rates = [r["success_rate"] for r in sweep_result["results"]]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Reward vs parameter
        ax1.errorbar(param_values, mean_rewards, yerr=std_rewards, 
                    marker='o', capsize=5, capthick=2, linewidth=2)
        ax1.set_xlabel(param_name)
        ax1.set_ylabel("Mean Reward")
        ax1.set_title(f"Performance vs {param_name}")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=200, color='r', linestyle='--', label='Success threshold')
        ax1.legend()
        
        # Success rate vs parameter
        ax2.plot(param_values, success_rates, marker='o', linewidth=2)
        ax2.set_xlabel(param_name)
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title(f"Success Rate vs {param_name}")
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sweep_{param_name}.png")
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def visualize_physics_correlation(self, random_stats, save_dir="validation_plots"):
        """
        물리 파라미터와 성능 간의 상관관계 분석
        
        Args:
            random_stats: 랜덤 환경 평가 결과
            save_dir: 저장 경로
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 데이터프레임 생성
        df_data = []
        for i, params in enumerate(random_stats["physics_params"]):
            df_data.append({
                "gravity": params.get("box2d_gravity", 0),
                "wind_power": params.get("wind_power", 0),
                "turbulence_power": params.get("turbulence_power", 0),
                "reward": random_stats["rewards"][i],
                "success": random_stats["successes"][i]
            })
        df = pd.DataFrame(df_data)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Scatter plots
        for idx, param in enumerate(["gravity", "wind_power", "turbulence_power"]):
            ax = axes[0, idx]
            scatter = ax.scatter(df[param], df["reward"], 
                               c=df["success"], cmap="RdYlGn", alpha=0.6)
            ax.set_xlabel(param)
            ax.set_ylabel("Reward")
            ax.set_title(f"Reward vs {param}")
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label="Success")
            
            # 상관계수 계산
            corr = df[param].corr(df["reward"])
            ax.text(0.05, 0.95, f"r = {corr:.3f}", transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        
        # Distribution plots
        for idx, param in enumerate(["gravity", "wind_power", "turbulence_power"]):
            ax = axes[1, idx]
            success_vals = df[df["success"] == True][param]
            fail_vals = df[df["success"] == False][param]
            
            ax.hist([success_vals, fail_vals], bins=20, label=["Success", "Fail"], alpha=0.6)
            ax.set_xlabel(param)
            ax.set_ylabel("Frequency")
            ax.set_title(f"{param} Distribution by Outcome")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, "physics_correlation.png")
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved: {save_path}")
        plt.close()
        
        # Correlation matrix
        plt.figure(figsize=(8, 6))
        corr_matrix = df[["gravity", "wind_power", "turbulence_power", "reward"]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, 
                   square=True, linewidths=1)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        save_path = os.path.join(save_dir, "correlation_matrix.png")
        plt.savefig(save_path, dpi=150)
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def save_results_to_json(self, results, filename="validation_results.json"):
        """
        검증 결과를 JSON으로 저장
        
        Args:
            results: 저장할 결과 딕셔너리
            filename: 저장 파일명
        """
        # NumPy 타입을 Python 기본 타입으로 변환
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(filename, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"✓ Results saved to: {filename}")
    
    def _print_stats(self, stats):
        """통계 출력"""
        print(f"\n결과 통계:")
        print(f"  Mean Reward:   {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Median Reward: {stats['median_reward']:.2f}")
        print(f"  Min Reward:    {stats['min_reward']:.2f}")
        print(f"  Max Reward:    {stats['max_reward']:.2f}")
        print(f"  Mean Length:   {stats['mean_length']:.1f}")
        print(f"  Success Rate:  {stats['success_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="강화학습 에이전트 검증")
    parser.add_argument("--model", type=str, required=True, help="모델 파일 경로")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--mode", type=str, default="all", 
                       choices=["fixed", "random", "sweep", "all"],
                       help="검증 모드")
    parser.add_argument("--n-episodes", type=int, default=1000, 
                       help="에피소드 수 (기본: 1000)")
    parser.add_argument("--output-dir", type=str, default="validation_results",
                       help="결과 저장 디렉토리")
    parser.add_argument("--device", type=str, default="cuda",
                       help="연산 디바이스")
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Validator 초기화
    validator = AgentValidator(args.model, args.config, args.device)
    
    all_results = {}
    
    # 1. 고정 환경 평가 (baseline)
    if args.mode in ["fixed", "all"]:
        print("\n" + "="*60)
        print("1. 고정 환경 평가 (Baseline)")
        print("="*60)
        fixed_stats = validator.validate_fixed_env(
            n_episodes=args.n_episodes,
            gravity=-10.0,
            wind_power=10.0,
            turbulence_power=1.0
        )
        all_results["fixed_baseline"] = fixed_stats
    
    # 2. 랜덤 환경 평가 (일반화 성능)
    if args.mode in ["random", "all"]:
        print("\n" + "="*60)
        print("2. 랜덤 환경 평가 (일반화 성능)")
        print("="*60)
        random_stats = validator.validate_random_env(n_episodes=args.n_episodes)
        all_results["random_env"] = random_stats
        
        # 물리 파라미터 상관관계 분석
        validator.visualize_physics_correlation(
            random_stats, 
            save_dir=args.output_dir
        )
    
    # 3. 물리 파라미터 스윕 테스트
    if args.mode in ["sweep", "all"]:
        print("\n" + "="*60)
        print("3. 물리 파라미터 스윕 테스트")
        print("="*60)
        
        # Gravity sweep
        gravity_sweep = validator.validate_physics_sweep(
            param_name="box2d_gravity",
            param_range=np.linspace(-15, -5, 11),
            n_episodes_per_value=20
        )
        all_results["gravity_sweep"] = gravity_sweep
        validator.visualize_physics_sweep(gravity_sweep, save_dir=args.output_dir)
        
        # Wind power sweep
        wind_sweep = validator.validate_physics_sweep(
            param_name="wind_power",
            param_range=np.linspace(0, 20, 11),
            n_episodes_per_value=20
        )
        all_results["wind_sweep"] = wind_sweep
        validator.visualize_physics_sweep(wind_sweep, save_dir=args.output_dir)
        
        # Turbulence sweep
        turb_sweep = validator.validate_physics_sweep(
            param_name="turbulence_power",
            param_range=np.linspace(0, 5, 11),
            n_episodes_per_value=20
        )
        all_results["turbulence_sweep"] = turb_sweep
        validator.visualize_physics_sweep(turb_sweep, save_dir=args.output_dir)
    
    # 4. 결과 비교 및 시각화
    if args.mode == "all" and "fixed_baseline" in all_results and "random_env" in all_results:
        print("\n" + "="*60)
        print("4. 베이스라인 vs 랜덤 환경 비교")
        print("="*60)
        comparison = validator.compare_with_baseline(
            all_results["fixed_baseline"],
            all_results["random_env"]
        )
        all_results["comparison"] = comparison
        
        # 시각화
        stats_dict = {
            "Fixed (Baseline)": all_results["fixed_baseline"],
            "Random Env": all_results["random_env"]
        }
        validator.visualize_results(stats_dict, save_dir=args.output_dir)
    
    # 5. 결과 저장
    output_file = os.path.join(args.output_dir, "validation_results.json")
    validator.save_results_to_json(all_results, output_file)
    
    print("\n" + "="*60)
    print("검증 완료!")
    print(f"결과가 {args.output_dir}/ 에 저장되었습니다.")
    print("="*60)


if __name__ == "__main__":
    main()
