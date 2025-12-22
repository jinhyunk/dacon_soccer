"""
Team-wise LSTM 추론 스크립트

Usage:
    python team_infer.py
"""
import os
from typing import Dict

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

def build_inference_sequence(use_df: pd.DataFrame) -> np.ndarray:
    """추론용 시퀀스 생성"""
    sx = use_df["start_x"].values / FIELD_X
    sy = use_df["start_y"].values / FIELD_Y
    ex = use_df["end_x"].values / FIELD_X
    ey = use_df["end_y"].values / FIELD_Y

    coords = []
    for i in range(len(use_df)):
        coords.append([sx[i], sy[i]])
        # 마지막 행은 end_x가 NaN (또는 타깃)이어서 제외
        if i < len(use_df) - 1:
            coords.append([ex[i], ey[i]])

    return np.array(coords, dtype="float32")


# ===========================
# 추론 실행
# ===========================

def run_inference(
    models: Dict[int, nn.Module],
    test_meta_path: str = TEST_META_PATH,
    sample_sub_path: str = SAMPLE_SUB_PATH,
    submit_path: str = SUBMIT_PATH,
) -> None:
    """테스트 데이터에 대해 추론 수행 및 제출 파일 생성"""
    test_meta = pd.read_csv(test_meta_path)
    submission = pd.read_csv(sample_sub_path)
    submission = submission.merge(test_meta, on="game_episode", how="left")

    preds_x, preds_y = [], []

    for _, row in tqdm(submission.iterrows(), total=len(submission)):
        # episode 데이터 로드
        g = pd.read_csv("../data" + row["path"][1:]).reset_index(drop=True)
        g = g.sort_values("time_seconds").reset_index(drop=True)

        # episode의 마지막 행 기준 team_id 사용
        last_team = int(g.iloc[-1]["team_id"])
        model = models.get(last_team)

        if model is None:
            print(f"[WARN] No model for team_id={last_team}, output (0, 0)")
            preds_x.append(0.0)
            preds_y.append(0.0)
            continue

        # 동일 episode 내에서 마지막 행과 team_id가 같은 데이터만 활용
        g_team = g[g["team_id"] == last_team].reset_index(drop=True)
        use_df = g_team if len(g_team) >= 2 else g  # fallback

        # 시퀀스 생성 및 예측
        seq = build_inference_sequence(use_df)
        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)
        length = torch.tensor([seq.shape[0]]).to(DEVICE)

        with torch.no_grad():
            pred = model(x, length).cpu().numpy()[0]

        preds_x.append(pred[0] * FIELD_X)
        preds_y.append(pred[1] * FIELD_Y)

    print("Inference Done.")

    # 결과 저장
    submission["end_x"] = preds_x
    submission["end_y"] = preds_y

    os.makedirs(os.path.dirname(submit_path), exist_ok=True)
    submission[["game_episode", "end_x", "end_y"]].to_csv(submit_path, index=False)
    print(f"Saved: {submit_path}")


def main():
    log_device()

    models = load_team_models()
    if not models:
        print("No team models found. Please run team_train.py first.")
        return

    run_inference(models)


if __name__ == "__main__":
    main()
