"""
MSE 계산 유틸리티

기능:
- 두 정규분포 X~N(m1, v1), Y~N(m2, v2)에 대해, 독립 가정 시 E[(X-Y)^2] = (m1-m2)^2 + v1 + v2 로 MSE 계산
- 스칼라 또는 벡터 입력(콤마로 구분) 지원
- CSV 파일 입력 지원(컬럼: mean_true,var_true,mean_pred,var_pred)
- 벡터 입력 시 집계값 제공:
	- mean_mse: 각 차원별 MSE의 평균
	- sum_mse: 각 차원별 MSE의 합 = E[||X - Y||^2] (독립, 대각 공분산 가정)

사용 예시 (PowerShell):
- 단일 값:
  python graphtool.py --true-mean -10 --true-var 2.0 --pred-mean -8 --pred-var 3.0

- 벡터(콤마 구분):
  python graphtool.py --true-mean -10,-12 --true-var 2.0,1.5 --pred-mean -8,-11 --pred-var 3.0,2.5

- CSV 파일:
  python graphtool.py --csv input.csv
  # input.csv 예시 헤더: mean_true,var_true,mean_pred,var_pred
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Iterable, List, Tuple, Union

import numpy as np


def _parse_number_list(s: str) -> np.ndarray:
	""""1,2,3" 형태를 float ndarray로 파싱. 공백 허용."""
	parts = [p.strip() for p in s.split(",") if p.strip() != ""]
	return np.array([float(p) for p in parts], dtype=float)


def mse_from_means_vars(
	mean_true: Union[float, np.ndarray],
	var_true: Union[float, np.ndarray],
	mean_pred: Union[float, np.ndarray],
	var_pred: Union[float, np.ndarray],
	cov_xy: Union[float, np.ndarray] = 0.0,
) -> np.ndarray:
	"""
	독립 가정하에서 두 정규분포 X~N(mean_true,var_true), Y~N(mean_pred,var_pred)에 대해
	E[(X-Y)^2] = (mean_true-mean_pred)^2 + var_true + var_pred - 2*cov_xy

	일반적으로 독립이면 cov_xy=0 이므로: (Δμ)^2 + v_true + v_pred

	입력은 스칼라 또는 동일 길이의 배열(브로드캐스팅 가능) 모두 지원.
	var_* < 0 인 값이 있으면 경고를 띄우고 |var|로 보정합니다.
	"""
	mt = np.asarray(mean_true, dtype=float)
	vt = np.asarray(var_true, dtype=float)
	mp = np.asarray(mean_pred, dtype=float)
	vp = np.asarray(var_pred, dtype=float)
	cov = np.asarray(cov_xy, dtype=float)

	# 음수 variance 방지(입력 실수 방지용). 실제론 에러 처리하는 게 엄격하지만, 편의상 경고+보정.
	if np.any(vt < 0) or np.any(vp < 0):
		print("[Warn] Variance should be >= 0. Using absolute value for negative entries.")
		vt = np.abs(vt)
		vp = np.abs(vp)

	return (mt - mp) ** 2 + vt + vp - 2.0 * cov


def _load_csv_rows(path: str) -> List[Tuple[float, float, float, float]]:
	rows: List[Tuple[float, float, float, float]] = []
	with open(path, "r", newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		required = {"mean_true", "var_true", "mean_pred", "var_pred"}
		if not required.issubset(set(reader.fieldnames or [])):
			raise ValueError(
				f"CSV must contain columns: {sorted(required)}; got {reader.fieldnames}"
			)
		for r in reader:
			rows.append(
				(
					float(r["mean_true"]),
					float(r["var_true"]),
					float(r["mean_pred"]),
					float(r["var_pred"]),
				)
			)
	return rows


def main():
	parser = argparse.ArgumentParser(description="Compute MSE from means and variances of two Normal distributions")
	parser.add_argument("--true-mean", type=str, help="실제 분포의 mean (콤마 구분 가능)")
	parser.add_argument("--true-var", type=str, help="실제 분포의 variance (콤마 구분 가능)")
	parser.add_argument("--pred-mean", type=str, help="추정 분포의 mean (콤마 구분 가능)")
	parser.add_argument("--pred-var", type=str, help="추정 분포의 variance (콤마 구분 가능)")
	parser.add_argument("--cov", type=str, default=None, help="공분산 값(또는 벡터). 생략 시 0 가정")
	parser.add_argument("--csv", type=str, help="CSV 파일 경로 (컬럼: mean_true,var_true,mean_pred,var_pred)")
	parser.add_argument("--json", type=str, default=None, help="결과를 JSON으로 저장할 경로")

	args = parser.parse_args()

	results = {}

	if args.csv:
		rows = _load_csv_rows(args.csv)
		m_true = np.array([r[0] for r in rows], dtype=float)
		v_true = np.array([r[1] for r in rows], dtype=float)
		m_pred = np.array([r[2] for r in rows], dtype=float)
		v_pred = np.array([r[3] for r in rows], dtype=float)
		mse_vals = mse_from_means_vars(m_true, v_true, m_pred, v_pred)
		results = {
			"count": int(mse_vals.size),
			"mse_per_row": mse_vals.tolist(),
			"mean_mse": float(np.mean(mse_vals)),
			"sum_mse": float(np.sum(mse_vals)),
			"std_mse": float(np.std(mse_vals)),
		}
		print(
			f"Rows: {len(rows)} | Mean MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f} | "
			f"Sum MSE (vector): {results['sum_mse']:.6f}"
		)
	else:
		# 개별 인자 모드
		required = [args.true_mean, args.true_var, args.pred_mean, args.pred_var]
		if any(x is None for x in required):
			parser.error("Either provide --csv or all of --true-mean --true-var --pred-mean --pred-var")

		mt = _parse_number_list(args.true_mean)
		vt = _parse_number_list(args.true_var)
		mp = _parse_number_list(args.pred_mean)
		vp = _parse_number_list(args.pred_var)
		if args.cov is not None:
			cov = _parse_number_list(args.cov)
		else:
			cov = 0.0

		mse_vals = mse_from_means_vars(mt, vt, mp, vp, cov)
		results = {
			"count": int(np.size(mse_vals)),
			"mse": mse_vals.tolist() if np.size(mse_vals) > 1 else float(np.squeeze(mse_vals)),
			"mean_mse": float(np.mean(mse_vals)),
			"sum_mse": float(np.sum(mse_vals)),
			"std_mse": float(np.std(mse_vals)),
		}
		print(json.dumps(results, ensure_ascii=False, indent=2))

	if args.json:
		with open(args.json, "w", encoding="utf-8") as f:
			json.dump(results, f, ensure_ascii=False, indent=2)
		print(f"Saved results to {args.json}")


if __name__ == "__main__":
	main()

