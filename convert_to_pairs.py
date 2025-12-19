"""
experiments.csv를 graphtool.py 입력 형식(pairs.csv)으로 변환

입력: experiments.csv (mean_gravity, mean_wind, mean_turbulence, var_gravity, var_wind, var_turbulence)
출력: pairs_*.csv (mean_true, var_true, mean_pred, var_pred) - 각 차원별로 행 분리
"""

import csv
import argparse


def convert_experiments_to_pairs(input_path, output_path, true_mean, true_var):
    """
    experiments.csv를 pairs.csv 형식으로 변환
    
    각 실험의 3개 파라미터를 3개 행으로 펼침:
    - row1: gravity  (mean_true[0], var_true[0], mean_pred[gravity], var_pred[gravity])
    - row2: wind     (mean_true[1], var_true[1], mean_pred[wind], var_pred[wind])
    - row3: turbulence (mean_true[2], var_true[2], mean_pred[turbulence], var_pred[turbulence])
    """
    
    with open(input_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        experiments = list(reader)
    
    # 각 실험마다 pairs_<experiment_name>.csv 생성
    for exp in experiments:
        exp_name = exp['experiment_name'].replace('/', '_').replace('\\', '_')
        exp_output = output_path.replace('.csv', f'_{exp_name}.csv')
        
        pairs = [
            {
                'mean_true': true_mean[0],
                'var_true': true_var[0],
                'mean_pred': float(exp['mean_gravity']),
                'var_pred': float(exp['var_gravity'])
            },
            {
                'mean_true': true_mean[1],
                'var_true': true_var[1],
                'mean_pred': float(exp['mean_wind']),
                'var_pred': float(exp['var_wind'])
            },
            {
                'mean_true': true_mean[2],
                'var_true': true_var[2],
                'mean_pred': float(exp['mean_turbulence']),
                'var_pred': float(exp['var_turbulence'])
            }
        ]
        
        with open(exp_output, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=['mean_true', 'var_true', 'mean_pred', 'var_pred'])
            writer.writeheader()
            writer.writerows(pairs)
        
        print(f"✓ Created: {exp_output}")
    
    # 통합 버전 (모든 실험 x 3 파라미터)
    all_pairs = []
    for exp in experiments:
        all_pairs.extend([
            {
                'experiment': exp['experiment_name'],
                'param': 'gravity',
                'mean_true': true_mean[0],
                'var_true': true_var[0],
                'mean_pred': float(exp['mean_gravity']),
                'var_pred': float(exp['var_gravity'])
            },
            {
                'experiment': exp['experiment_name'],
                'param': 'wind',
                'mean_true': true_mean[1],
                'var_true': true_var[1],
                'mean_pred': float(exp['mean_wind']),
                'var_pred': float(exp['var_wind'])
            },
            {
                'experiment': exp['experiment_name'],
                'param': 'turbulence',
                'mean_true': true_mean[2],
                'var_true': true_var[2],
                'mean_pred': float(exp['mean_turbulence']),
                'var_pred': float(exp['var_turbulence'])
            }
        ])
    
    all_output = output_path.replace('.csv', '_all.csv')
    with open(all_output, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=['experiment', 'param', 'mean_true', 'var_true', 'mean_pred', 'var_pred'])
        writer.writeheader()
        writer.writerows(all_pairs)
    
    print(f"✓ Created unified: {all_output}")


def main():
    parser = argparse.ArgumentParser(description="Convert experiments.csv to graphtool pairs format")
    parser.add_argument("--input", type=str, default="experiments.csv", help="입력 experiments.csv")
    parser.add_argument("--output", type=str, default="pairs.csv", help="출력 pairs.csv 베이스명")
    parser.add_argument("--true-mean", type=str, default="-10,10,1", help="Ground truth mean")
    parser.add_argument("--true-var", type=str, default="2.0,3.0,0.5", help="Ground truth variance")
    
    args = parser.parse_args()
    
    true_mean = [float(x.strip()) for x in args.true_mean.split(',')]
    true_var = [float(x.strip()) for x in args.true_var.split(',')]
    
    convert_experiments_to_pairs(args.input, args.output, true_mean, true_var)
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
