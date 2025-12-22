"""
Team-wise LSTM 추론 스크립트

Usage:
    # temporal_test 평가 (정답이 있는 데이터)
    python team_infer.py --eval

    # 실제 테스트 데이터 추론 (제출용)
    python team_infer.py --submit
"""
import os
import argparse
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn

from config import (
    DEVICE,
    HIDDEN_DIM,
    TEAM_MODEL_DIR,
    TEST_META_PATH,
    SAMPLE_SUB_PATH,
    SUBMIT_PATH,
    FIELD_X,
    FIELD_Y,
    log_device,
)
from models import LSTMBaseline


# =========================
# 경로 설정 (temporal_test 평가용)
# =========================
TEMPORAL_TEST_META_PATH = "../../data/temporal_test.csv"
TEMPORAL_TEST_DATA_ROOT = "../../data"

# 실제 테스트용 경로
REAL_TEST_META_PATH = "../../data/basic/base_test.csv"
REAL_TEST_DATA_ROOT = "../../data"


# ===========================
# 모델 로딩
# ===========================

def load_team_models(
    team_model_dir: str = TEAM_MODEL_DIR,
    device: str = DEVICE,
) -> Dict[int, nn.Module]:
    """
    team_model_dir 내의 team_{id}.pth 파일들을 모두 읽어서
    team_id -> model dict 로 반환
    """
    models: Dict[int, nn.Module] = {}

    if not os.path.isdir(team_model_dir):
        print(f"Team model dir not found: {team_model_dir}")
        return models

    for fname in os.listdir(team_model_dir):
        if not fname.startswith("team_") or not fname.endswith(".pth"):
            continue

        try:
            team_id = int(fname[len("team_"):-len(".pth")])
        except ValueError:
            continue

        path = os.path.join(team_model_dir, fname)
        state_dict = torch.load(path, map_location=device)

        model = LSTMBaseline(input_dim=2, hidden_dim=HIDDEN_DIM).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        models[team_id] = model
        print(f"Loaded model for team {team_id}")

    return models


# ===========================
# 추론 시퀀스 생성
# ===========================

def build_inference_sequence(use_df: pd.DataFrame, exclude_last_end: bool = True) -> np.ndarray:
    """
    추론용 시퀀스 생성
    exclude_last_end: True면 마지막 행의 end_x, end_y 제외 (예측 대상)
    """
    sx = use_df["start_x"].values / FIELD_X
    sy = use_df["start_y"].values / FIELD_Y
    ex = use_df["end_x"].values / FIELD_X
    ey = use_df["end_y"].values / FIELD_Y

    coords = []
    for i in range(len(use_df)):
        coords.append([sx[i], sy[i]])
        # 마지막 행의 end는 예측 대상이므로 제외
        if i < len(use_df) - 1:
            coords.append([ex[i], ey[i]])

    return np.array(coords, dtype="float32")


def predict_single_episode(
    model: nn.Module,
    use_df: pd.DataFrame,
) -> Tuple[float, float]:
    """단일 episode에 대해 예측 수행"""
    seq = build_inference_sequence(use_df)
    x = torch.tensor(seq).unsqueeze(0).to(DEVICE)
    length = torch.tensor([seq.shape[0]]).to(DEVICE)

    with torch.no_grad():
        pred = model(x, length).cpu().numpy()[0]

    pred_x = pred[0] * FIELD_X
    pred_y = pred[1] * FIELD_Y

    return pred_x, pred_y


# ===========================
# Temporal Test 평가 (정답 비교)
# ===========================

def evaluate_temporal_test(
    models: Dict[int, nn.Module],
    meta_path: str = TEMPORAL_TEST_META_PATH,
    data_root: str = TEMPORAL_TEST_DATA_ROOT,
) -> None:
    """
    temporal_test 데이터에 대해 추론 후 정답과 비교하여 평균 유클리드 거리 계산
    """
    print("\n========== Temporal Test Evaluation ==========")

    meta_df = pd.read_csv(meta_path)
    print(f"Total episodes to evaluate: {len(meta_df)}")

    preds_x, preds_y = [], []
    true_x, true_y = [], []
    skipped = 0

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Evaluating"):
        # episode 데이터 로드
        csv_path = os.path.join(data_root, row["path"][2:])  # "./temporal_test/..." -> "temporal_test/..."
        g = pd.read_csv(csv_path)
        g = g.sort_values("time_seconds").reset_index(drop=True)

        # 정답: 마지막 행의 end_x, end_y
        last_row = g.iloc[-1]
        gt_x = last_row["end_x"]
        gt_y = last_row["end_y"]

        # NaN 체크 (정답이 없으면 스킵)
        if pd.isna(gt_x) or pd.isna(gt_y):
            skipped += 1
            continue

        # episode의 마지막 행 기준 team_id 사용
        last_team = int(last_row["team_id"])
        model = models.get(last_team)

        if model is None:
            print(f"[WARN] No model for team_id={last_team}, skipping")
            skipped += 1
            continue

        # 동일 episode 내에서 마지막 행과 team_id가 같은 데이터만 활용
        g_team = g[g["team_id"] == last_team].reset_index(drop=True)
        use_df = g_team if len(g_team) >= 2 else g  # fallback

        # 예측
        pred_x_val, pred_y_val = predict_single_episode(model, use_df)

        preds_x.append(pred_x_val)
        preds_y.append(pred_y_val)
        true_x.append(gt_x)
        true_y.append(gt_y)

    # 평가 지표 계산
    preds_x = np.array(preds_x)
    preds_y = np.array(preds_y)
    true_x = np.array(true_x)
    true_y = np.array(true_y)

    distances = np.sqrt((preds_x - true_x) ** 2 + (preds_y - true_y) ** 2)
    mean_dist = distances.mean()
    std_dist = distances.std()
    median_dist = np.median(distances)
    min_dist = distances.min()
    max_dist = distances.max()

    print("\n========== Evaluation Results ==========")
    print(f"Evaluated episodes: {len(distances)}")
    print(f"Skipped episodes: {skipped}")
    print(f"Mean Euclidean Distance: {mean_dist:.4f}")
    print(f"Std Euclidean Distance: {std_dist:.4f}")
    print(f"Median Euclidean Distance: {median_dist:.4f}")
    print(f"Min Distance: {min_dist:.4f}")
    print(f"Max Distance: {max_dist:.4f}")
    print("=" * 45)


# ===========================
# 실제 테스트 추론 (제출용)
# ===========================

def run_inference_for_submission(
    models: Dict[int, nn.Module],
    test_meta_path: str = REAL_TEST_META_PATH,
    data_root: str = REAL_TEST_DATA_ROOT,
    submit_path: str = SUBMIT_PATH,
) -> None:
    """실제 테스트 데이터에 대해 추론 수행 및 제출 파일 생성"""
    print("\n========== Inference for Submission ==========")

    test_meta = pd.read_csv(test_meta_path)
    print(f"Total episodes to predict: {len(test_meta)}")

    results = []

    for _, row in tqdm(test_meta.iterrows(), total=len(test_meta), desc="Predicting"):
        game_episode = row["game_episode"]

        # episode 데이터 로드
        csv_path = os.path.join(data_root, row["path"][2:])
        g = pd.read_csv(csv_path)
        g = g.sort_values("time_seconds").reset_index(drop=True)

        # episode의 마지막 행 기준 team_id 사용
        last_team = int(g.iloc[-1]["team_id"])
        model = models.get(last_team)

        if model is None:
            print(f"[WARN] No model for team_id={last_team}, output (0, 0)")
            results.append({
                "game_episode": game_episode,
                "end_x": 0.0,
                "end_y": 0.0,
            })
            continue

        # 동일 episode 내에서 마지막 행과 team_id가 같은 데이터만 활용
        g_team = g[g["team_id"] == last_team].reset_index(drop=True)
        use_df = g_team if len(g_team) >= 2 else g

        # 예측
        pred_x, pred_y = predict_single_episode(model, use_df)

        results.append({
            "game_episode": game_episode,
            "end_x": pred_x,
            "end_y": pred_y,
        })

    # 결과 저장
    submission = pd.DataFrame(results)
    os.makedirs(os.path.dirname(submit_path), exist_ok=True)
    submission.to_csv(submit_path, index=False)
    print(f"\nSubmission saved to: {submit_path}")
    print(f"Total predictions: {len(submission)}")


# ===========================
# 메인 함수
# ===========================

def parse_args():
    parser = argparse.ArgumentParser(description="Team-wise LSTM Inference")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="temporal_test 데이터로 평가 수행 (정답 비교)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="실제 테스트 데이터 추론 (제출용 파일 생성)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log_device()

    models = load_team_models()
    if not models:
        print("No team models found. Please run team_train.py first.")
        return

    # 둘 다 지정 안 했으면 기본으로 eval 수행
    if not args.eval and not args.submit:
        args.eval = True

    if args.eval:
        evaluate_temporal_test(models)

    if args.submit:
        run_inference_for_submission(models)


if __name__ == "__main__":
    main()
