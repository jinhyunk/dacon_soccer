import os
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn


# =========================
# 1. 하이퍼파라미터 / 경로
# =========================

MODEL_PATH = "../data/poc_lstm_best.pth"
TEST_META_PATH = "../data/test.csv"
SAMPLE_SUB_PATH = "../data/sample_submission.csv"
SUBMIT_PATH = "../data/phase_submit2.csv"

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


def load_model(model_path: str = MODEL_PATH) -> nn.Module:
    model = LSTMBaseline(input_dim=2, hidden_dim=HIDDEN_DIM).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model weights from: {model_path}")
    return model


# ===========================
# 3. 평가 데이터셋 추론
# ===========================

def run_inference(
    model: nn.Module,
    test_meta_path: str = TEST_META_PATH,
    sample_sub_path: str = SAMPLE_SUB_PATH,
    submit_path: str = SUBMIT_PATH,
) -> None:
    test_meta = pd.read_csv(test_meta_path)
    submission = pd.read_csv(sample_sub_path)

    submission = submission.merge(test_meta, on="game_episode", how="left")

    preds_x, preds_y = [], []

    for _, row in tqdm(submission.iterrows(), total=len(submission)):
        g = pd.read_csv("../data" + row["path"][1:]).reset_index(drop=True)

        # 마지막 행 정보
        last_row = g.iloc[-1]
        last_phase = last_row["phase"]
        last_team = last_row["team_id"]

        # 마지막 phase 데이터
        phase_df = g[g["phase"] == last_phase]
        input_df = phase_df

        # 정규화된 좌표 준비
        sx = input_df["start_x"].values / 105.0
        sy = input_df["start_y"].values / 68.0
        ex = input_df["end_x"].values / 105.0
        ey = input_df["end_y"].values / 68.0

        coords = []
        for i in range(len(input_df)):
            # start는 항상 존재하므로 그대로 사용
            coords.append([sx[i], sy[i]])
            # 마지막 행은 end_x가 NaN이므로 자동으로 제외됨
            if i < len(input_df) - 1:
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
    model = load_model(MODEL_PATH)
    run_inference(model)


if __name__ == "__main__":
    main()


