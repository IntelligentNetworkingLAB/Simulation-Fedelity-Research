"""
ND_history_labcom 폴더 구조에서 실험 파라미터 추출 및 MSE/Log-Likelihood 계산

폴더 구조:
- ND_history_labcom/
  - ND_MEAN_{g}_{w}_{t}_o/
    - ND_VAR_{vg}_{vw}_{vt}/

출력:
- experiments.csv: 모든 실험의 mean/var 파라미터
- results.csv: 각 실험의 sum_mse, log_likelihood
"""

import os
import re
import csv
import json
import math
from typing import List, Dict, Tuple
import numpy as np


def parse_mean_folder(folder_name: str) -> Tuple[float, float, float]:
    """ND_MEAN_-10_10_1_o 형태에서 (gravity, wind, turbulence) 추출"""
    # ND_MEAN_{g}_{w}_{t}_o 또는 ND_MEAN_{g}_{w}_{t}
    match = re.match(r'ND_MEAN_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)(?:_o)?', folder_name)
    if match:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
    raise ValueError(f"Cannot parse mean folder: {folder_name}")


def parse_var_folder(folder_name: str) -> Tuple[float, float, float]:
    """ND_VAR_2_3_0.5 형태에서 (var_gravity, var_wind, var_turbulence) 추출"""
    match = re.match(r'ND_VAR_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)', folder_name)
    if match:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
    raise ValueError(f"Cannot parse var folder: {folder_name}")


def scan_experiments(base_dir: str) -> List[Dict]:
    """ND_history_labcom 폴더를 스캔하여 모든 실험 파라미터 추출"""
    experiments = []
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    for mean_folder in os.listdir(base_dir):
        if not mean_folder.startswith("ND_MEAN_"):
            continue
        
        mean_path = os.path.join(base_dir, mean_folder)
        if not os.path.isdir(mean_path):
            continue
        
        try:
            mean_g, mean_w, mean_t = parse_mean_folder(mean_folder)
        except ValueError as e:
            print(f"[Warn] {e}")
            continue
        
        # 하위 ND_VAR_ 폴더 스캔
        for var_folder in os.listdir(mean_path):
            if not var_folder.startswith("ND_VAR_"):
                continue
            
            var_path = os.path.join(mean_path, var_folder)
            if not os.path.isdir(var_path):
                continue
            
            try:
                var_g, var_w, var_t = parse_var_folder(var_folder)
            except ValueError as e:
                print(f"[Warn] {e}")
                continue
            
            experiments.append({
                "experiment_name": f"{mean_folder}/{var_folder}",
                "mean_folder": mean_folder,
                "var_folder": var_folder,
                "mean_gravity": mean_g,
                "mean_wind": mean_w,
                "mean_turbulence": mean_t,
                "var_gravity": var_g,
                "var_wind": var_w,
                "var_turbulence": var_t,
                "path": var_path
            })
    
    return experiments


def compute_mse_vector(
    mean_true: np.ndarray,
    var_true: np.ndarray,
    mean_pred: np.ndarray,
    var_pred: np.ndarray
) -> Dict:
    """
    여러 방식의 MSE 계산:
    1. sum_mse (기존): Σ[(Δμ)² + v_true + v_pred] - 분포 샘플 간 거리
    2. mse_mean_vector: ||μ_true - μ_pred||² - Mean 벡터 간 MSE
    3. mse_var_vector: ||var_true - var_pred||² - Variance 벡터 간 MSE
    4. mse_combined: mse_mean_vector + mse_var_vector - 파라미터 벡터 전체 MSE
    """
    # 기존 방식 (분포 샘플 거리)
    mse_per_dim = (mean_true - mean_pred)**2 + var_true + var_pred
    
    # Mean 벡터 MSE
    mse_mean_per_dim = (mean_true - mean_pred)**2
    mse_mean_vector = float(np.sum(mse_mean_per_dim))
    
    # Variance 벡터 MSE
    mse_var_per_dim = (var_true - var_pred)**2
    mse_var_vector = float(np.sum(mse_var_per_dim))
    
    # 합산 (6차원 벡터로 볼 때)
    mse_combined = mse_mean_vector + mse_var_vector
    
    return {
        "mse_per_dim": mse_per_dim.tolist(),
        "sum_mse": float(np.sum(mse_per_dim)),
        "mean_mse": float(np.mean(mse_per_dim)),
        "mse_mean_vector": mse_mean_vector,
        "mse_var_vector": mse_var_vector,
        "mse_combined": mse_combined,
        "mse_mean_per_dim": mse_mean_per_dim.tolist(),
        "mse_var_per_dim": mse_var_per_dim.tolist()
    }


def compute_log_likelihood(
    x_true: np.ndarray,
    mu_pred: np.ndarray,
    var_pred: np.ndarray
) -> float:
    """
    정규분포 가정 하에 log p(x_true | μ_pred, σ²_pred)
    
    각 차원 독립이면:
    log L = Σ_i log N(x_true[i] | μ_pred[i], σ²_pred[i])
          = Σ_i [ -0.5 log(2π σ²) - (x - μ)² / (2σ²) ]
    
    Note: x_true를 μ_true로 해석 (평균이 실제 관측값 대표)
    """
    # var_pred가 0이면 무한대 log-likelihood (수치 안정성 위해 클리핑)
    var_pred_safe = np.maximum(var_pred, 1e-9)
    
    log_prob = -0.5 * np.log(2 * np.pi * var_pred_safe) - (x_true - mu_pred)**2 / (2 * var_pred_safe)
    return float(np.sum(log_prob))


def compute_kl_divergence(
    mu_true: np.ndarray,
    var_true: np.ndarray,
    mu_pred: np.ndarray,
    var_pred: np.ndarray
) -> float:
    """
    KL Divergence between two multivariate Gaussians (diagonal covariance)
    
    KL(P||Q) = KL(N(μ_true, Σ_true) || N(μ_pred, Σ_pred))
    
    For diagonal covariance (independent dimensions):
    KL = Σ_i KL(N(μ_true[i], σ²_true[i]) || N(μ_pred[i], σ²_pred[i]))
    
    KL for 1D Gaussians:
    KL(N(μ1,σ1²) || N(μ2,σ2²)) = 0.5 * [log(σ2²/σ1²) + (σ1² + (μ1-μ2)²)/σ2² - 1]
    
    Properties:
    - KL >= 0 (항상 양수 또는 0)
    - KL = 0 if and only if distributions are identical
    - 비대칭: KL(P||Q) ≠ KL(Q||P)
    """
    # 수치 안정성
    var_true_safe = np.maximum(var_true, 1e-9)
    var_pred_safe = np.maximum(var_pred, 1e-9)
    
    # 각 차원별 KL divergence
    kl_per_dim = 0.5 * (
        np.log(var_pred_safe / var_true_safe) +
        (var_true_safe + (mu_true - mu_pred)**2) / var_pred_safe -
        1.0
    )
    
    return float(np.sum(kl_per_dim))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract experiments and compute metrics")
    parser.add_argument("--base-dir", type=str, default="ND_history_labcom",
                        help="ND_history_labcom 폴더 경로")
    parser.add_argument("--true-mean", type=str, default="-10,10,1",
                        help="Ground truth mean (gravity,wind,turbulence)")
    parser.add_argument("--true-var", type=str, default="2.0,3.0,0.5",
                        help="Ground truth variance")
    parser.add_argument("--output-experiments", type=str, default="experiments.csv",
                        help="실험 파라미터 CSV 출력 경로")
    parser.add_argument("--output-results", type=str, default="results.csv",
                        help="MSE/Log-likelihood 결과 CSV 출력 경로")
    parser.add_argument("--json", type=str, default=None,
                        help="JSON 출력 경로 (선택)")
    
    args = parser.parse_args()
    
    # Ground truth 파싱
    true_mean = np.array([float(x.strip()) for x in args.true_mean.split(",")])
    true_var = np.array([float(x.strip()) for x in args.true_var.split(",")])
    
    print(f"Ground Truth: mean={true_mean}, var={true_var}")
    print(f"Scanning: {args.base_dir}")
    
    # 실험 스캔
    experiments = scan_experiments(args.base_dir)
    print(f"Found {len(experiments)} experiments")
    
    # experiments.csv 저장
    exp_fields = ["experiment_name", "mean_folder", "var_folder",
                  "mean_gravity", "mean_wind", "mean_turbulence",
                  "var_gravity", "var_wind", "var_turbulence", "path"]
    
    with open(args.output_experiments, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=exp_fields)
        writer.writeheader()
        writer.writerows(experiments)
    
    print(f"✓ Saved experiments to: {args.output_experiments}")
    
    # MSE / KL Divergence 계산
    results = []
    for exp in experiments:
        pred_mean = np.array([exp["mean_gravity"], exp["mean_wind"], exp["mean_turbulence"]])
        pred_var = np.array([exp["var_gravity"], exp["var_wind"], exp["var_turbulence"]])
        
        mse_result = compute_mse_vector(true_mean, true_var, pred_mean, pred_var)
        log_lik = compute_log_likelihood(true_mean, pred_mean, pred_var)
        kl_div = compute_kl_divergence(true_mean, true_var, pred_mean, pred_var)
        
        results.append({
            "experiment_name": exp["experiment_name"],
            "sum_mse": mse_result["sum_mse"],
            "mean_mse": mse_result["mean_mse"],
            "mse_mean_vector": mse_result["mse_mean_vector"],
            "mse_var_vector": mse_result["mse_var_vector"],
            "mse_combined": mse_result["mse_combined"],
            "kl_divergence": kl_div,
            "log_likelihood": log_lik,
            "mse_gravity": mse_result["mse_per_dim"][0],
            "mse_wind": mse_result["mse_per_dim"][1],
            "mse_turbulence": mse_result["mse_per_dim"][2],
            "mse_mean_gravity": mse_result["mse_mean_per_dim"][0],
            "mse_mean_wind": mse_result["mse_mean_per_dim"][1],
            "mse_mean_turbulence": mse_result["mse_mean_per_dim"][2],
            "mse_var_gravity": mse_result["mse_var_per_dim"][0],
            "mse_var_wind": mse_result["mse_var_per_dim"][1],
            "mse_var_turbulence": mse_result["mse_var_per_dim"][2]
        })
    
    # results.csv 저장
    result_fields = ["experiment_name", "sum_mse", "mean_mse", 
                     "mse_mean_vector", "mse_var_vector", "mse_combined",
                     "kl_divergence", "log_likelihood",
                     "mse_gravity", "mse_wind", "mse_turbulence",
                     "mse_mean_gravity", "mse_mean_wind", "mse_mean_turbulence",
                     "mse_var_gravity", "mse_var_wind", "mse_var_turbulence"]
    
    with open(args.output_results, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ Saved results to: {args.output_results}")
    
    # 핵심 지표만 담은 간단한 CSV 생성
    summary_fields = ["experiment_name", "mse_mean_vector", "mse_var_vector", "mse_combined", "kl_divergence"]
    summary_output = args.output_results.replace(".csv", "_summary.csv")
    
    with open(summary_output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in summary_fields})
    
    print(f"✓ Saved summary to: {summary_output}")
    
    # JSON 저장 (선택)
    if args.json:
        output_data = {
            "ground_truth": {
                "mean": true_mean.tolist(),
                "var": true_var.tolist()
            },
            "experiments": experiments,
            "results": results
        }
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✓ Saved JSON to: {args.json}")
    
    # 요약 통계
    results_sorted = sorted(results, key=lambda x: x["mse_combined"])
    print(f"\n{'='*60}")
    print("Top 5 Best (lowest mse_combined = mean+var vector MSE):")
    print(f"{'='*60}")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i}. {r['experiment_name']}")
        print(f"   Combined MSE: {r['mse_combined']:.4f} (Mean: {r['mse_mean_vector']:.4f} + Var: {r['mse_var_vector']:.4f})")
        print(f"   KL Divergence: {r['kl_divergence']:.6f} | Log-Likelihood: {r['log_likelihood']:.4f}")
    
    results_sorted_kl = sorted(results, key=lambda x: x["kl_divergence"])
    print(f"\n{'='*60}")
    print("Top 5 Best (lowest KL divergence):")
    print(f"{'='*60}")
    for i, r in enumerate(results_sorted_kl[:5], 1):
        print(f"{i}. {r['experiment_name']}")
        print(f"   KL Divergence: {r['kl_divergence']:.6f} | Combined MSE: {r['mse_combined']:.4f}")


if __name__ == "__main__":
    main()
