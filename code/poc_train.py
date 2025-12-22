from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader


# =========================
# 1. 하이퍼파라미터 세팅
# =========================

TRAIN_PATH = "../data/phase_train.csv"
MODEL_PATH = "../data/poc_lstm_best.pth"

BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
HIDDEN_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def log_device() -> None:
    print("Using device:", DEVICE)


# =========================================
# 2. 데이터 로드 및 전처리 유틸 함수 정의
# =========================================

def phase_wise_data(df: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby("phase")):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue

        # 정규화된 좌표 준비
        sx = g["start_x"].values / 105.0
        sy = g["start_y"].values / 68.0
        ex = g["end_x"].values / 105.0
        ey = g["end_y"].values / 68.0

        coords = []
        for i in range(len(g)):
            # 항상 start는 들어감
            coords.append([sx[i], sy[i]])
            # 마지막 행 이전까지만 end를 넣음 (마지막 행의 end_x, end_y는 타깃이므로)
            if i < len(g) - 1:
                coords.append([ex[i], ey[i]])

        seq = np.array(coords, dtype="float32")  # [T, 2]
        target = np.array([ex[-1], ey[-1]], dtype="float32")  # 마지막 행 end_x, end_y

        episodes.append(seq)
        targets.append(target)

    print("phase 수 : ", len(episodes))
    return episodes, targets


# ==========================================
# 3. Custom Dataset / DataLoader 정의 함수
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


def build_dataloaders(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader]:
    """
    노트북에서 사용하던 방식과 동일하게 phase 기준으로 시퀀스를 구성하고
    train/valid DataLoader 를 반환한다.
    """
    df = df.sort_values(["game_episode", "time_seconds"]).reset_index(drop=True)

    # phase 기준 시퀀스 생성
    episodes, targets = phase_wise_data(df)

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

    print("train episodes:", len(episodes_train), "valid episodes:", len(episodes_valid))
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
# 5. 학습 루프
# ===========================

def train_model(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = EPOCHS,
) -> Tuple[nn.Module, dict, float]:
    best_dist = float("inf")
    best_model_state = None

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0

        for X, lengths, y in tqdm(train_loader):
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
            for X, lengths, y in tqdm(valid_loader):
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
            f"[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} | "
            f"valid_mean_dist={mean_dist:.4f}"
        )

        # ----- BEST MODEL 업데이트 -----
        if mean_dist < best_dist:
            best_dist = mean_dist
            best_model_state = model.state_dict().copy()
            print(f" --> Best model updated! (dist={best_dist:.4f})")

    if best_model_state is None:
        # 이론상 첫 epoch에서 항상 갱신되므로 도달하지 않지만 안전하게 처리
        best_model_state = model.state_dict()

    return model, best_model_state, best_dist


def main():
    log_device()

    # 학습 데이터 로드
    df = pd.read_csv(TRAIN_PATH)

    # DataLoader 구성 (phase 기준)
    train_loader, valid_loader = build_dataloaders(df, batch_size=BATCH_SIZE)

    # 모델/손실함수/옵티마이저 정의
    model, criterion, optimizer = create_model()

    # 학습
    model, best_state, best_dist = train_model(
        model, criterion, optimizer, train_loader, valid_loader, epochs=EPOCHS
    )
    print(f"Training finished. Best valid mean dist: {best_dist:.4f}")

    # best 모델 가중치 저장
    torch.save(best_state, MODEL_PATH)
    print(f"Best model state_dict saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()


