# 강화학습 에이전트 검증 가이드

이 가이드는 학습이 완료된 강화학습 에이전트의 성능을 체계적으로 검증하는 방법을 설명합니다.

## 📋 목차
1. [검증 방법 개요](#검증-방법-개요)
2. [사용 방법](#사용-방법)
3. [검증 메트릭](#검증-메트릭)
4. [결과 해석](#결과-해석)

---

## 검증 방법 개요

### 1️⃣ 고정 환경 평가 (Fixed Environment Validation)
**목적**: 학습 시 사용한 기본 환경에서의 성능 확인

- 물리 파라미터를 고정 (예: gravity=-10.0, wind=10.0, turbulence=1.0)
- 여러 에피소드 실행하여 성능의 일관성 확인
- **사용 시기**: 학습이 제대로 되었는지 확인할 때

```bash
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode fixed \
    --n-episodes 100
```

**출력 지표**:
- Mean Reward (평균 보상)
- Standard Deviation (표준편차) - 낮을수록 안정적
- Success Rate (성공률) - LunarLander의 경우 reward > 200
- Episode Length (에피소드 길이)

---

### 2️⃣ 랜덤 환경 평가 (Random Environment Validation)
**목적**: 다양한 환경 조건에서의 일반화 성능 확인

- 매 에피소드마다 물리 파라미터를 랜덤하게 샘플링
- 학습 시 보지 못한 조건에서도 잘 작동하는지 확인
- **사용 시기**: 실제 배포 전 robustness 확인

```bash
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode random \
    --n-episodes 100
```

**분석 내용**:
- 고정 환경 대비 성능 하락 정도
- 물리 파라미터와 성능 간의 상관관계
- 특정 조건에서의 실패 패턴

---

### 3️⃣ 물리 파라미터 스윕 (Physics Parameter Sweep)
**목적**: 특정 파라미터가 성능에 미치는 영향 분석

- 한 파라미터를 체계적으로 변화시키며 성능 측정
- 에이전트가 작동하는 파라미터 범위 파악
- **사용 시기**: 에이전트의 한계와 강점 분석

```bash
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode sweep \
    --n-episodes 20  # 각 파라미터 값당
```

**테스트 파라미터**:
- **Gravity** (중력): -15 ~ -5 범위
- **Wind Power** (바람 세기): 0 ~ 20 범위
- **Turbulence Power** (난류): 0 ~ 5 범위

---

### 4️⃣ 전체 검증 (All Validations)
**목적**: 포괄적인 성능 평가

모든 검증을 한번에 수행하고 결과를 비교 분석합니다.

```bash
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode all \
    --n-episodes 100 \
    --output-dir validation_results
```

---

## 사용 방법

### 기본 사용법

```bash
# 1. 고정 환경에서 100 에피소드 평가
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode fixed \
    --n-episodes 100

# 2. 랜덤 환경에서 평가 (config 파일 사용)
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --config config.json \
    --mode random \
    --n-episodes 100

# 3. 전체 검증 수행
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode all \
    --output-dir my_validation_results
```

### 명령행 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--model` | 학습된 모델 파일 경로 (필수) | - |
| `--config` | 설정 파일 경로 (선택) | None |
| `--mode` | 검증 모드 (fixed/random/sweep/all) | all |
| `--n-episodes` | 평가할 에피소드 수 | 100 |
| `--output-dir` | 결과 저장 디렉토리 | validation_results |
| `--device` | 연산 디바이스 (cuda/cpu) | cuda |

---

## 검증 메트릭

### 주요 지표

1. **Mean Reward** (평균 보상)
   - 에피소드별 총 보상의 평균
   - LunarLander 기준: 200 이상이면 "성공"

2. **Standard Deviation** (표준편차)
   - 성능의 일관성을 나타냄
   - 낮을수록 안정적인 에이전트

3. **Success Rate** (성공률)
   - 목표를 달성한 에피소드 비율
   - LunarLander: reward > 200인 비율

4. **Episode Length** (에피소드 길이)
   - 에피소드 종료까지 걸린 스텝 수
   - 너무 짧으면 조기 종료, 너무 길면 비효율적

### 통계적 비교

- **T-test**: 두 조건 간 성능 차이의 통계적 유의성
- **Cohen's d**: 효과 크기 (0.2=small, 0.5=medium, 0.8=large)
- **P-value < 0.05**: 통계적으로 유의미한 차이

---

## 결과 해석

### 출력 파일

검증 후 `output_dir/` 폴더에 다음 파일들이 생성됩니다:

```
validation_results/
├── validation_results.json          # 모든 수치 결과
├── validation_comparison.png        # 조건별 성능 비교
├── sweep_box2d_gravity.png         # 중력 스윕 결과
├── sweep_wind_power.png            # 바람 스윕 결과
├── sweep_turbulence_power.png      # 난류 스윕 결과
├── physics_correlation.png         # 파라미터-성능 상관관계
└── correlation_matrix.png          # 상관관계 행렬
```

### 좋은 에이전트의 기준

#### ✅ 우수한 에이전트
- 고정 환경: Mean Reward > 200, Success Rate > 90%
- 랜덤 환경: 고정 환경 대비 **10% 이내** 성능 하락
- Std Reward < 50 (일관된 성능)
- 파라미터 스윕: 넓은 범위에서 안정적 성능

#### ⚠️ 개선 필요
- 고정 환경: Mean Reward < 150
- 랜덤 환경: 고정 환경 대비 **30% 이상** 성능 하락
- Std Reward > 100 (성능 편차 큼)
- 특정 파라미터 범위에서 급격한 성능 저하

### 실제 예시

```
=== 고정 환경 평가 ===
Mean Reward:   245.32 ± 35.21
Success Rate:  94.0%
→ 해석: 학습이 잘 되었음

=== 랜덤 환경 평가 ===
Mean Reward:   218.45 ± 52.18
Success Rate:  85.0%
Difference:    -26.87 (-10.9%)
→ 해석: 일반화 성능 우수 (10% 내 하락)

=== 물리 파라미터 스윕 ===
Gravity -15.0: Mean Reward 180.3 (Success 75%)
Gravity -10.0: Mean Reward 245.3 (Success 94%)
Gravity -5.0:  Mean Reward 210.5 (Success 88%)
→ 해석: 강한 중력(-15)에서 다소 어려움, 
        전반적으로 넓은 범위에서 작동
```

---

## 고급 사용법

### Python 스크립트에서 사용

```python
from validate_agent import AgentValidator

# Validator 초기화
validator = AgentValidator(
    model_path="models/ppo_rnn_lunar.pt",
    config_path="config.json",
    device="cuda"
)

# 1. 고정 환경 평가
fixed_stats = validator.validate_fixed_env(
    n_episodes=100,
    gravity=-10.0,
    wind_power=10.0,
    turbulence_power=1.0
)

# 2. 랜덤 환경 평가
random_stats = validator.validate_random_env(n_episodes=100)

# 3. 비교 분석
comparison = validator.compare_with_baseline(fixed_stats, random_stats)

# 4. 시각화
validator.visualize_results({
    "Fixed": fixed_stats,
    "Random": random_stats
}, save_dir="my_plots")
```

### 커스텀 검증

```python
import numpy as np

# 특정 중력 범위만 테스트
gravity_sweep = validator.validate_physics_sweep(
    param_name="box2d_gravity",
    param_range=np.linspace(-12, -8, 9),  # 더 세밀한 범위
    n_episodes_per_value=50,               # 더 많은 에피소드
    fixed_params={
        "wind_power": 15.0,                # 바람 고정
        "turbulence_power": 2.0            # 난류 고정
    }
)

validator.visualize_physics_sweep(gravity_sweep)
```

---

## 체크리스트

학습 완료 후 다음 순서로 검증하세요:

- [ ] **Step 1**: 고정 환경에서 기본 성능 확인
  - Mean Reward가 목표치 이상인가?
  - Success Rate가 충분히 높은가?
  
- [ ] **Step 2**: 랜덤 환경에서 일반화 성능 확인
  - 성능 하락이 허용 범위 내인가?
  - 특정 조건에서만 실패하는가?
  
- [ ] **Step 3**: 파라미터 스윕으로 작동 범위 파악
  - 어느 범위에서 안정적으로 작동하는가?
  - 실패 조건은 무엇인가?
  
- [ ] **Step 4**: 결과 분석 및 문서화
  - 시각화 자료 확인
  - 개선이 필요한 부분 파악

---

## 문제 해결

### Q: "CUDA out of memory" 에러
```bash
# CPU로 실행
python validate_agent.py --model models/ppo_rnn_lunar.pt --device cpu
```

### Q: 에피소드가 너무 오래 걸림
```bash
# 에피소드 수 줄이기
python validate_agent.py --model models/ppo_rnn_lunar.pt --n-episodes 50
```

### Q: 결과가 일관되지 않음
- 더 많은 에피소드로 평가 (--n-episodes 200+)
- Seed 고정이 필요한 경우 코드 수정 필요

---

## 참고 자료

- **통계적 유의성**: p-value < 0.05를 기준으로 사용
- **효과 크기**: Cohen's d를 통해 실용적 의미 파악
- **성공 기준**: 도메인에 따라 다르므로 조정 필요

---

**작성일**: 2025-11-06
**버전**: 1.0
