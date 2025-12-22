import os
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn


# =========================
# 1. 하이퍼파라미터 / 경로
# =========================

TEAM_MODEL_DIR = "../data/team_models"
TEST_META_PATH = "../data/test.csv"
SAMPLE_SUB_PATH = "../data/sample_submission.csv"
SUBMIT_PATH = "../data/team_phase_submit.csv"

HIDDEN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_device() -> None:
    print("Using device:", DEVICE)


# ===========================
# 2. LSTM 베이스라인 모델
# ===========================

class LSTMBaseline(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 2)  # (x_norm, y_norm)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 2], lengths: [B]
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]  # [B, H]
        out = self.fc(h_last)  # [B, 2]
        return out


def load_team_models_from_dir(
    team_model_dir: str, device: str = DEVICE
) -> Dict[int, nn.Module]:
    """
    TEAM_MODEL_DIR 내의 team_{id}.pth 파일들을 모두 읽어서
    team_id -> model dict 로 반환한다.
    """
    models: Dict[int, nn.Module] = {}
    if not os.path.isdir(team_model_dir):
        print(f"Team model dir not found: {team_model_dir}")
        return models

    for fname in os.listdir(team_model_dir):
        if not fname.startswith("team_") or not fname.endswith(".pth"):
            continue
        try:
            team_id = int(fname[len("team_") : -len(".pth")])
        except ValueError:
            continue

        path = os.path.join(team_model_dir, fname)
        state_dict = torch.load(path, map_location=device)

        model = LSTMBaseline(input_dim=2, hidden_dim=HIDDEN_DIM).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        models[team_id] = model
        print(f"Loaded model for team {team_id} from {path}")

    return models


# ===========================
# 3. 평가 데이터셋 추론
# ===========================

def run_inference(
    models: Dict[int, nn.Module],
    test_meta_path: str = TEST_META_PATH,
    sample_sub_path: str = SAMPLE_SUB_PATH,
    submit_path: str = SUBMIT_PATH,
) -> None:
    test_meta = pd.read_csv(test_meta_path)
    submission = pd.read_csv(sample_sub_path)

    # game_episode 기준 메타 정보 합치기
    submission = submission.merge(test_meta, on="game_episode", how="left")

    preds_x, preds_y = [], []

    for _, row in tqdm(submission.iterrows(), total=len(submission)):
        # episode 전체 데이터를 로드
        g = pd.read_csv("../data" + row["path"][1:]).reset_index(drop=True)
        g = g.sort_values("time_seconds").reset_index(drop=True)

        # episode 의 마지막 행 기준 team_id 사용
        last_row = g.iloc[-1]
        last_team = int(last_row["team_id"])

        model = models.get(last_team)
        if model is None:
            # 해당 team 의 모델이 없으면 전체 episode 를 사용해도 되지만,
            # 여기서는 일단 스킵하고 (0,0) 을 넣는 방식으로 안전하게 처리
            print(f"[WARN] No model for team_id={last_team}, output (0, 0)")
            preds_x.append(0.0)
            preds_y.append(0.0)
            continue

        # 동일 episode 내에서 마지막 행과 team_id 가 같은 데이터만 활용
        g_team = g[g["team_id"] == last_team].reset_index(drop=True)
        # 데이터가 너무 적으면 fallback 으로 episode 전체 사용
        use_df = g_team if len(g_team) >= 2 else g

        # 정규화된 좌표 준비
        sx = use_df["start_x"].values / 105.0
        sy = use_df["start_y"].values / 68.0
        ex = use_df["end_x"].values / 105.0
        ey = use_df["end_y"].values / 68.0

        coords = []
        for i in range(len(use_df)):
            # start 는 항상 존재
            coords.append([sx[i], sy[i]])
            # 마지막 행은 end_x 가 NaN (또는 타깃) 이어서 제외
            if i < len(use_df) - 1:
                coords.append([ex[i], ey[i]])

        seq = np.array(coords, dtype="float32")  # [T, 2]

        x = torch.tensor(seq).unsqueeze(0).to(DEVICE)  # [1, T, 2]
        length = torch.tensor([seq.shape[0]]).to(DEVICE)  # [1]

        with torch.no_grad():
            pred = model(x, length).cpu().numpy()[0]  # [2], 정규화 좌표

        preds_x.append(pred[0] * 105.0)
        preds_y.append(pred[1] * 68.0)

    print("Inference Done.")

    submission["end_x"] = preds_x
    submission["end_y"] = preds_y

    os.makedirs(os.path.dirname(submit_path), exist_ok=True)
    submission[["game_episode", "end_x", "end_y"]].to_csv(submit_path, index=False)
    print(f"Saved: {submit_path}")


def main():
    log_device()
    models = load_team_models_from_dir(TEAM_MODEL_DIR, device=DEVICE)
    if not models:
        print("No team models found. Please run team_train.py first.")
        return

    run_inference(models)


if __name__ == "__main__":
    main()


