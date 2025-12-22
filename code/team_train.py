import os
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. 하이퍼파라미터 / 경로
# =========================

TRAIN_PATH = "../data/phase_train.csv"
TEAM_MODEL_DIR = "../data/team_models"

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
HIDDEN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_device() -> None:
    print("Using device:", DEVICE)


# ==========================================
# 2. Episode 기반 시퀀스 생성 (team 전용)
# ==========================================

def build_episode_team_sequences(
    df: pd.DataFrame,
    team_id: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    episode 단위로 나눈 뒤,
    각 episode 내에서 time_seconds 기준 마지막 행의 team_id 가 주어진 team_id 인 경우만 사용.

    해당 episode에서 마지막 행과 동일한 team_id 의 데이터만 모아 시퀀스를 만들고,
    그 팀의 마지막 행 end_x, end_y 를 타깃으로 사용한다.
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    # episode 단위 그룹핑
    for _, g in tqdm(df.groupby("game_episode"), desc=f"Build epi seq (team={team_id})"):
        # 시간 순서 정렬
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        last_row = g.iloc[-1]
        last_team = last_row["team_id"]

        # 이 episode의 마지막 행의 team_id 가 우리가 학습하려는 team 이 아니면 스킵
        if last_team != team_id:
            continue

        # 같은 episode 내에서 마지막 행과 team_id 가 같은 데이터만 사용
        g_team = g[g["team_id"] == team_id].reset_index(drop=True)
        if len(g_team) < 2:
            # 시퀀스 최소 길이 보장을 위해 2 미만이면 사용하지 않음
            continue

        # 정규화된 좌표 준비
        sx = g_team["start_x"].values / 105.0
        sy = g_team["start_y"].values / 68.0
        ex = g_team["end_x"].values / 105.0
        ey = g_team["end_y"].values / 68.0

        coords = []
        for i in range(len(g_team)):
            # 항상 start는 들어감
            coords.append([sx[i], sy[i]])
            # 마지막 행 이전까지만 end를 넣음 (마지막 행의 end_x, end_y 는 타깃)
            if i < len(g_team) - 1:
                coords.append([ex[i], ey[i]])

        seq = np.array(coords, dtype="float32")  # [T, 2]
        target = np.array([ex[-1], ey[-1]], dtype="float32")  # 마지막 행 end_x, end_y

        episodes.append(seq)
        targets.append(target)

    print(f"[team {team_id}] episode 시퀀스 수:", len(episodes))
    return episodes, targets


# ==========================================
# 3. Custom Dataset / DataLoader 정의
# ==========================================

class EpisodeDataset(Dataset):
    def __init__(self, episodes: List[np.ndarray], targets: List[np.ndarray]):
        self.episodes = episodes
        self.targets = targets

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.episodes[idx])  # [T, 2]
        tgt = torch.tensor(self.targets[idx])  # [2]
        length = seq.size(0)
        return seq, length, tgt


def collate_fn(batch):
    seqs, lengths, tgts = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)  # [B, T, 2]
    tgts = torch.stack(tgts, dim=0)  # [B, 2]
    return padded, lengths, tgts


def build_team_dataloaders(
    df: pd.DataFrame,
    team_id: int,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    주어진 team_id 에 대해 episode 기반 시퀀스를 생성하고
    이를 train/valid 로 나누어 DataLoader 를 만든다.
    """
    episodes, targets = build_episode_team_sequences(df, team_id)

    if len(episodes) < 2:
        raise ValueError(f"[team {team_id}] episode 시퀀스가 너무 적어서 학습이 불가능합니다.")

    idx_train, idx_valid = train_test_split(
        np.arange(len(episodes)), test_size=0.2, random_state=42
    )

    episodes_train = [episodes[i] for i in idx_train]
    targets_train = [targets[i] for i in idx_train]
    episodes_valid = [episodes[i] for i in idx_valid]
    targets_valid = [targets[i] for i in idx_valid]

    train_loader = DataLoader(
        EpisodeDataset(episodes_train, targets_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_loader = DataLoader(
        EpisodeDataset(episodes_valid, targets_valid),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print(
        f"[team {team_id}] train episodes: {len(episodes_train)}, "
        f"valid episodes: {len(episodes_valid)}"
    )
    return train_loader, valid_loader


# ===========================
# 4. LSTM 베이스라인 모델
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
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]  # [B, H] 마지막 layer의 hidden state
        out = self.fc(h_last)  # [B, 2]
        return out


def create_model() -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    model = LSTMBaseline(input_dim=2, hidden_dim=HIDDEN_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    return model, criterion, optimizer


# ===========================
# 5. 학습 루프 (team 단위)
# ===========================

def train_team(
    team_id: int,
    df: pd.DataFrame,
) -> Tuple[nn.Module, Dict, float]:
    """
    한 개의 team_id 에 대해 모델을 학습하고,
    best state_dict 와 best metric (mean_dist) 를 반환한다.
    """
    print(f"\n========== Team {team_id} Training ==========")

    train_loader, valid_loader = build_team_dataloaders(df, team_id, batch_size=BATCH_SIZE)
    model, criterion, optimizer = create_model()

    best_dist = float("inf")
    best_model_state: Dict | None = None

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0

        for X, lengths, y in tqdm(train_loader, desc=f"[team {team_id}] train epoch {epoch}"):
            X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X, lengths)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # --- Valid: 평균 유클리드 거리 ---
        model.eval()
        dists = []

        with torch.no_grad():
            for X, lengths, y in tqdm(
            valid_loader, desc=f"[team {team_id}] valid epoch {epoch}"
            ):
                X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
                pred = model(X, lengths)

                pred_np = pred.cpu().numpy()
                true_np = y.cpu().numpy()

                pred_x = pred_np[:, 0] * 105.0
                pred_y = pred_np[:, 1] * 68.0
                true_x = true_np[:, 0] * 105.0
                true_y = true_np[:, 1] * 68.0

                dist = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
                dists.append(dist)

        mean_dist = np.concatenate(dists).mean()  # 평균 유클리드 거리

        print(
            f"[team {team_id}][Epoch {epoch}] "
            f"train_loss={train_loss:.4f} | "
            f"valid_mean_dist={mean_dist:.4f}"
        )

        # ----- BEST MODEL 업데이트 -----
        if mean_dist < best_dist:
            best_dist = mean_dist
            best_model_state = model.state_dict().copy()
            print(f" --> [team {team_id}] Best model updated! (dist={best_dist:.4f})")

    if best_model_state is None:
        # 이론상 첫 epoch에서 항상 갱신되므로 도달하지 않지만 안전하게 처리
        best_model_state = model.state_dict()

    return model, best_model_state, best_dist


def main():
    log_device()
    os.makedirs(TEAM_MODEL_DIR, exist_ok=True)

    # 학습 데이터 로드
    df = pd.read_csv(TRAIN_PATH)

    # 존재하는 team_id 목록
    team_ids = sorted(df["team_id"].unique())
    print("Found team_ids:", team_ids)

    for team_id in team_ids:
        try:
            model, best_state, best_dist = train_team(team_id=int(team_id), df=df)
        except ValueError as e:
            # 데이터가 너무 적어 학습이 불가능한 team 은 스킵
            print(f"[team {team_id}] skip: {e}")
            continue

        print(
            f"[team {team_id}] Training finished. "
            f"Best valid mean dist: {best_dist:.4f}"
        )

        # 팀별 best 모델 가중치 저장
        model_path = os.path.join(TEAM_MODEL_DIR, f"team_{int(team_id)}.pth")
        torch.save(best_state, model_path)
        print(f"[team {team_id}] Best model state_dict saved to: {model_path}")


if __name__ == "__main__":
    main()


