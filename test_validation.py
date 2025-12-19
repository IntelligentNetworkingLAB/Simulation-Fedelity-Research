"""
빠른 validation 테스트 스크립트
실제 validation 전에 모델 로딩과 기본 동작을 확인합니다.
"""

import os
import torch
import gymnasium as gym
from validate_agent import AgentValidator

def test_model_loading(model_path, config_path=None):
    """모델이 제대로 로드되는지 테스트"""
    print(f"\n{'='*60}")
    print(f"모델 로딩 테스트")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 에러: 모델 파일을 찾을 수 없습니다: {model_path}")
        return False
    
    try:
        validator = AgentValidator(model_path, config_path, device="cuda")
        print(f"✓ 모델 로딩 성공!")
        print(f"✓ 모델 구조:")
        print(f"  - Observation dim: {validator.model.obs_dim}")
        print(f"  - Action dim: {validator.model.action_dim}")
        print(f"  - Hidden size: {validator.model.hidden_size}")
        print(f"  - Device: {validator.model.device}")
        return validator
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_test(validator, n_episodes=5):
    """빠른 성능 테스트 (5 에피소드)"""
    print(f"\n{'='*60}")
    print(f"빠른 성능 테스트 (n={n_episodes})")
    print(f"{'='*60}")
    
    if validator is None:
        print(f"❌ Validator가 초기화되지 않았습니다.")
        return False
    
    try:
        stats = validator.validate_fixed_env(
            n_episodes=n_episodes,
            gravity=-10.0,
            wind_power=10.0,
            turbulence_power=1.0,
            deterministic=True
        )
        
        print(f"\n✓ 테스트 완료!")
        print(f"  평균 보상: {stats['mean_reward']:.2f}")
        print(f"  성공률: {stats['success_rate']:.1f}%")
        
        if stats['mean_reward'] > 0:
            print(f"\n✅ 모델이 정상적으로 작동합니다!")
            return True
        else:
            print(f"\n⚠️  경고: 평균 보상이 낮습니다. 모델이 제대로 학습되지 않았을 수 있습니다.")
            return False
            
    except Exception as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validation 테스트")
    parser.add_argument("--model", type=str, required=True, help="모델 파일 경로")
    parser.add_argument("--config", type=str, help="설정 파일 경로")
    parser.add_argument("--quick", action="store_true", help="빠른 테스트만 수행 (5 에피소드)")
    parser.add_argument("--n-episodes", type=int, default=None, help="테스트 에피소드 수 (기본: 100, --quick시 5)")
    args = parser.parse_args()

    print("="*60)
    print("Validation 시스템 테스트")
    print("="*60)

    # 1. 모델 로딩 테스트
    validator = test_model_loading(args.model, args.config)
    if validator is None:
        print("\n❌ 모델 로딩에 실패했습니다.")
        return

    # 에피소드 수 결정
    if args.n_episodes is not None:
        n_episodes = args.n_episodes
    elif args.quick:
        n_episodes = 5
    else:
        n_episodes = 100

    # 2. 성능 테스트
    success = quick_test(validator, n_episodes=n_episodes)

    if success and not args.quick:
        print(f"\n{'='*60}")
        print("전체 validation을 실행하시겠습니까?")
        print("다음 명령어를 사용하세요:")
        print(f"{'='*60}")
        print()
        print(f"python validate_agent.py \\")
        print(f"    --model {args.model} \\")
        if args.config:
            print(f"    --config {args.config} \\")
        print(f"    --mode all \\")
        print(f"    --n-episodes 1000")
        print()


if __name__ == "__main__":
    main()
