"""
Transformer 좌표 예측 모델 추론 스크립트

Usage:
    # temporal_test 평가 (정답이 있는 데이터)
    python infer.py --eval
    
    # 실제 테스트 데이터 추론 (제출용)
    python infer.py --submit
    
    # 둘 다 수행
    python infer.py --eval --submit
"""
import os
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

from config import (
    DEVICE,
    MODEL_PATH,
    TEST_META_PATH,
    SAMPLE_SUB_PATH,
    SUBMIT_PATH,
    FIELD_X,
    FIELD_Y,
    D_MODEL,
    NHEAD,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
    log_device,
)
from dataset import build_inference_sequence
from models import TransformerEncoder


# =========================
# 경로 설정
# =========================
TEMPORAL_TEST_META_PATH = "../data/temporal_test.csv"
TEMPORAL_TEST_DATA_ROOT = "../data"

REAL_TEST_META_PATH = "../data/basic/base_test.csv"
REAL_TEST_DATA_ROOT = "../data"


def load_model(
    model_path: str = MODEL_PATH,
    model_type: str = "encoder",
) -> torch.nn.Module:
    """
    학습된 모델 로드
    """
    if model_type == "encoder":
        model = TransformerEncoder(
            input_dim=2,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
        ).to(DEVICE)
    else:
        from models import TransformerWithPooling
        model = TransformerWithPooling(
            input_dim=2,
            d_model=D_MODEL,
            nhead=NHEAD,
            num_layers=NUM_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
        ).to(DEVICE)
    
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    return model


def predict_single_episode(
    model: torch.nn.Module,
    use_df: pd.DataFrame,
) -> Tuple[float, float]:
    """
    단일 episode에 대해 예측 수행
    
    Args:
        model: 학습된 모델
        use_df: episode 데이터프레임
        
    Returns:
        pred_x, pred_y: 예측 좌표
    """
    # 시퀀스 생성
    seq = build_inference_sequence(use_df)
    
    # 텐서 변환
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, T, 2]
    length = torch.tensor([seq.shape[0]]).to(DEVICE)  # [1]
    
    # 패딩 마스크 (단일 샘플이므로 모두 False)
    padding_mask = torch.zeros(1, seq.shape[0], dtype=torch.bool).to(DEVICE)
    
    # 예측
    with torch.no_grad():
        pred = model(x, length, src_key_padding_mask=padding_mask)
        pred = pred.cpu().numpy()[0]
    
    pred_x = pred[0] * FIELD_X
    pred_y = pred[1] * FIELD_Y
    
    return pred_x, pred_y


def evaluate_temporal_test(
    model: torch.nn.Module,
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
        csv_path = os.path.join(data_root, row["path"][2:])
        
        if not os.path.exists(csv_path):
            skipped += 1
            continue
            
        g = pd.read_csv(csv_path)
        g = g.sort_values("time_seconds").reset_index(drop=True)
        
        if len(g) < 1:
            skipped += 1
            continue
        
        # 정답: 마지막 행의 end_x, end_y
        last_row = g.iloc[-1]
        gt_x = last_row["end_x"]
        gt_y = last_row["end_y"]
        
        # NaN 체크 (정답이 없으면 스킵)
        if pd.isna(gt_x) or pd.isna(gt_y):
            skipped += 1
            continue
        
        # 예측
        try:
            pred_x_val, pred_y_val = predict_single_episode(model, g)
        except Exception as e:
            print(f"Error predicting {row['game_episode']}: {e}")
            skipped += 1
            continue
        
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
    
    return mean_dist


def run_inference_for_submission(
    model: torch.nn.Module,
    test_meta_path: str = REAL_TEST_META_PATH,
    data_root: str = REAL_TEST_DATA_ROOT,
    submit_path: str = SUBMIT_PATH,
) -> None:
    """
    실제 테스트 데이터에 대해 추론 수행 및 제출 파일 생성
    """
    print("\n========== Inference for Submission ==========")
    
    if not os.path.exists(test_meta_path):
        print(f"Test meta file not found: {test_meta_path}")
        return
    
    test_meta = pd.read_csv(test_meta_path)
    print(f"Total episodes to predict: {len(test_meta)}")
    
    results = []
    
    for _, row in tqdm(test_meta.iterrows(), total=len(test_meta), desc="Predicting"):
        game_episode = row["game_episode"]
        
        # episode 데이터 로드
        csv_path = os.path.join(data_root, row["path"][2:])
        
        if not os.path.exists(csv_path):
            print(f"[WARN] File not found: {csv_path}")
            results.append({
                "game_episode": game_episode,
                "end_x": 52.5,  # 필드 중앙
                "end_y": 34.0,
            })
            continue
        
        g = pd.read_csv(csv_path)
        g = g.sort_values("time_seconds").reset_index(drop=True)
        
        if len(g) < 1:
            results.append({
                "game_episode": game_episode,
                "end_x": 52.5,
                "end_y": 34.0,
            })
            continue
        
        # 예측
        try:
            pred_x, pred_y = predict_single_episode(model, g)
        except Exception as e:
            print(f"Error predicting {game_episode}: {e}")
            pred_x, pred_y = 52.5, 34.0
        
        results.append({
            "game_episode": game_episode,
            "end_x": pred_x,
            "end_y": pred_y,
        })
    
    # 결과 저장
    submission = pd.DataFrame(results)
    os.makedirs(os.path.dirname(submit_path) if os.path.dirname(submit_path) else ".", exist_ok=True)
    submission.to_csv(submit_path, index=False)
    
    print(f"\nSubmission saved to: {submit_path}")
    print(f"Total predictions: {len(submission)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer 좌표 예측 모델 추론")
    
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
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help=f"모델 경로 (default: {MODEL_PATH})"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="encoder",
        choices=["encoder", "pooling"],
        help="모델 타입"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    log_device()
    
    # 모델 로드
    if not os.path.exists(args.model_path):
        print(f"Model not found: {args.model_path}")
        print("Please run train.py first.")
        return
    
    model = load_model(args.model_path, args.model_type)
    
    # 둘 다 지정 안 했으면 기본으로 eval 수행
    if not args.eval and not args.submit:
        args.eval = True
    
    if args.eval:
        evaluate_temporal_test(model)
    
    if args.submit:
        run_inference_for_submission(model)


if __name__ == "__main__":
    main()

