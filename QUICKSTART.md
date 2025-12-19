# 🚀 빠른 시작 가이드

학습된 `.pt` 모델 파일로 validation을 수행하는 방법입니다.

## 📝 현재 상황 확인

당신은 이미 다음과 같은 모델들을 가지고 있습니다:

```bash
# 확인된 모델 파일들
models/ppo_rnn_lunar.pt                                    # 메인 모델
RecurrentPolicy/ND_Default_v2/models/ppo_rnn_lunar.pt    # Default 버전
RecurrentPolicy/ND_hignmean_v2/models/ppo_rnn_lunar.pt   # High mean 버전
RecurrentPolicy/Oracle_ver/models/ppo_rnn_lunar.pt       # Oracle 버전
# ... 그 외 다수
```

## ✅ 1단계: 모델 확인 (빠른 테스트)

먼저 모델이 제대로 로드되는지 확인하세요:

```bash
# 메인 모델 테스트
python test_validation.py --model models/ppo_rnn_lunar.pt --quick

# 특정 버전 테스트
python test_validation.py \
    --model RecurrentPolicy/ND_hignmean_v2/models/ppo_rnn_lunar.pt \
    --quick
```

**출력 예시:**
```
==============================================================
모델 로딩 테스트
==============================================================
✓ 모델 로딩 성공!
✓ 모델 구조:
  - Observation dim: 8
  - Action dim: 4
  - Hidden size: 256

==============================================================
빠른 성능 테스트 (n=5)
==============================================================
Evaluating: 100%|██████████| 5/5 [00:02<00:00,  2.03it/s]

결과 통계:
  Mean Reward:   245.32 ± 35.21
  Success Rate:  100.0%

✅ 모델이 정상적으로 작동합니다!
```

## ✅ 2단계: 전체 Validation 수행

모델이 정상 작동하면 전체 검증을 실행하세요:

```bash
# 기본 실행 (모든 검증)
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode all \
    --n-episodes 100 \
    --output-dir validation_results
```

### 선택적 검증

특정 검증만 수행할 수도 있습니다:

```bash
# 1. 고정 환경만 (가장 빠름)
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode fixed \
    --n-episodes 100

# 2. 랜덤 환경 (일반화 성능)
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode random \
    --n-episodes 100

# 3. 파라미터 스윕 (robustness)
python validate_agent.py \
    --model models/ppo_rnn_lunar.pt \
    --mode sweep \
    --n-episodes 20
```

## 📊 3단계: 결과 확인

검증이 완료되면 다음 파일들이 생성됩니다:

```
validation_results/
├── validation_results.json              # 📄 수치 결과
├── validation_comparison.png            # 📊 성능 비교 그래프
├── sweep_box2d_gravity.png             # 📈 중력 영향 분석
├── sweep_wind_power.png                # 📈 바람 영향 분석
├── sweep_turbulence_power.png          # 📈 난류 영향 분석
├── physics_correlation.png             # 📉 상관관계 분석
└── correlation_matrix.png              # 🔢 상관계수 행렬
```

## 🔍 결과 해석 예시

### 좋은 모델 ✅
```
=== 고정 환경 평가 ===
Mean Reward:   245.32 ± 35.21
Success Rate:  94.0%

=== 랜덤 환경 평가 ===
Mean Reward:   218.45 ± 52.18
Success Rate:  85.0%
→ 성능 하락: -10.9% (허용 범위 내)

✅ 일반화 성능 우수!
```

### 개선 필요 ⚠️
```
=== 고정 환경 평가 ===
Mean Reward:   180.50 ± 68.42
Success Rate:  65.0%

=== 랜덤 환경 평가 ===
Mean Reward:   95.23 ± 85.11
Success Rate:  25.0%
→ 성능 하락: -47.2% (과도한 하락)

⚠️  일반화 성능 부족. 더 다양한 환경에서 학습 필요.
```

## 💡 여러 모델 비교하기

여러 버전의 모델을 비교하려면:

```bash
# 1. 각 모델 평가
python validate_agent.py \
    --model RecurrentPolicy/ND_Default_v2/models/ppo_rnn_lunar.pt \
    --mode all --output-dir results_default

python validate_agent.py \
    --model RecurrentPolicy/ND_hignmean_v2/models/ppo_rnn_lunar.pt \
    --mode all --output-dir results_highmean

python validate_agent.py \
    --model RecurrentPolicy/Oracle_ver/models/ppo_rnn_lunar.pt \
    --mode all --output-dir results_oracle

# 2. 결과 비교
# validation_results.json 파일들을 비교하면 됩니다
```

## 🐛 문제 해결

### 문제: "CUDA out of memory"
```bash
# 해결: CPU 사용
python validate_agent.py --model models/ppo_rnn_lunar.pt --device cpu
```

### 문제: "Model file not found"
```bash
# 해결: 절대 경로 사용
python validate_agent.py \
    --model /home/yjs/SimulationFidelity/LunarRender/models/ppo_rnn_lunar.pt
```

### 문제: 검증이 너무 오래 걸림
```bash
# 해결: 에피소드 수 줄이기
python validate_agent.py --model models/ppo_rnn_lunar.pt --n-episodes 50
```

### 문제: Config 파일이 필요한 경우
```bash
# 학습 시 사용한 config 파일 지정
python validate_agent.py \
    --model RecurrentPolicy/ND_hignmean_v2/models/ppo_rnn_lunar.pt \
    --config RecurrentPolicy/ND_hignmean_v2/config.json
```

## ⚡ 권장 워크플로우

```bash
# 1. 빠른 테스트 (30초)
python test_validation.py --model models/ppo_rnn_lunar.pt --quick

# 2. 고정 환경 평가 (2분)
python validate_agent.py --model models/ppo_rnn_lunar.pt --mode fixed --n-episodes 50

# 3. 결과가 좋으면 전체 검증 (10-15분)
python validate_agent.py --model models/ppo_rnn_lunar.pt --mode all --n-episodes 100
```

## 📌 체크리스트

- [ ] `test_validation.py`로 모델 로딩 확인
- [ ] `--mode fixed`로 기본 성능 확인
- [ ] 성공률 > 80% 확인
- [ ] `--mode random`으로 일반화 성능 확인
- [ ] 성능 하락 < 20% 확인
- [ ] `--mode sweep`로 robustness 확인
- [ ] 시각화 결과 분석
- [ ] 결과 문서화

## 🎯 요약

**당신의 .pt 파일로 바로 검증 가능합니다!**

```bash
# 이 한 줄이면 충분합니다:
python validate_agent.py --model models/ppo_rnn_lunar.pt --mode all
```

결과는 `validation_results/` 폴더에 저장됩니다.
