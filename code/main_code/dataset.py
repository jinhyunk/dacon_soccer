"""
Dataset 및 시퀀스 생성 함수
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from config import BATCH_SIZE, FIELD_X, FIELD_Y


# ==========================================
# Dataset 클래스
# ==========================================

class EpisodeDataset(Dataset):
    """시퀀스-타깃 쌍을 위한 Dataset"""

    def __init__(self, episodes: List[np.ndarray], targets: List[np.ndarray]):
        self.episodes = episodes
        self.targets = targets

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int):
        seq = torch.tensor(self.episodes[idx])  # [T, 2]
        tgt = torch.tensor(self.targets[idx])   # [2]
        length = seq.size(0)
        return seq, length, tgt


def collate_fn(batch):
    """가변 길이 시퀀스를 패딩하여 배치로 만듦"""
    seqs, lengths, tgts = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)  # [B, T, 2]
    tgts = torch.stack(tgts, dim=0)                # [B, 2]
    return padded, lengths, tgts


# ==========================================
# DataLoader 빌더
# ==========================================

def build_dataloaders(
    episodes: List[np.ndarray],
    targets: List[np.ndarray],
    batch_size: int = BATCH_SIZE,
    test_size: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    시퀀스 리스트를 train/valid로 나누어 DataLoader 반환
    """
    if len(episodes) < 2:
        raise ValueError("시퀀스가 너무 적어서 학습이 불가능합니다.")

    idx_train, idx_valid = train_test_split(
        np.arange(len(episodes)), test_size=test_size, random_state=42
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

    print(f"train: {len(episodes_train)}, valid: {len(episodes_valid)}")
    return train_loader, valid_loader


# ==========================================
# 시퀀스 생성 유틸 함수
# ==========================================

def _build_sequence_from_group(g: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    DataFrame 그룹에서 좌표 시퀀스와 타깃을 생성하는 공통 로직
    """
    sx = g["start_x"].values / FIELD_X
    sy = g["start_y"].values / FIELD_Y
    ex = g["end_x"].values / FIELD_X
    ey = g["end_y"].values / FIELD_Y

    coords = []
    for i in range(len(g)):
        coords.append([sx[i], sy[i]])
        if i < len(g) - 1:
            coords.append([ex[i], ey[i]])

    seq = np.array(coords, dtype="float32")           # [T, 2]
    target = np.array([ex[-1], ey[-1]], dtype="float32")  # [2]

    return seq, target


# ==========================================
# Pretrain용 시퀀스 생성
# ==========================================

def build_pretrain_sequences_by_phase(
    df: pd.DataFrame,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """phase 기준으로 시퀀스 생성"""
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby("phase"), desc="Build pretrain seq (phase)"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        seq, target = _build_sequence_from_group(g)
        episodes.append(seq)
        targets.append(target)

    print(f"[pretrain-phase] 시퀀스 수: {len(episodes)}")
    return episodes, targets


def build_pretrain_sequences_by_team(
    df: pd.DataFrame,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """team_id 기준으로 시퀀스 생성 (game_episode + team_id 그룹)"""
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby(["game_episode", "team_id"]), desc="Build pretrain seq (team_id)"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        seq, target = _build_sequence_from_group(g)
        episodes.append(seq)
        targets.append(target)

    print(f"[pretrain-team_id] 시퀀스 수: {len(episodes)}")
    return episodes, targets


# ==========================================
# Fine-tune용 시퀀스 생성 (team 전용)
# ==========================================

def build_episode_team_sequences(
    df: pd.DataFrame,
    team_id: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    episode 단위로 나눈 뒤,
    각 episode 내에서 마지막 행의 team_id가 주어진 team_id인 경우만 사용.
    해당 episode에서 마지막 행과 동일한 team_id의 데이터만 모아 시퀀스를 만듦.
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby("game_episode"), desc=f"Build epi seq (team={team_id})"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        last_row = g.iloc[-1]
        last_team = last_row["team_id"]

        # 이 episode의 마지막 행의 team_id가 우리가 학습하려는 team이 아니면 스킵
        if last_team != team_id:
            continue

        # 같은 episode 내에서 마지막 행과 team_id가 같은 데이터만 사용
        g_team = g[g["team_id"] == team_id].reset_index(drop=True)
        if len(g_team) < 2:
            continue

        seq, target = _build_sequence_from_group(g_team)
        episodes.append(seq)
        targets.append(target)

    print(f"[team {team_id}] episode 시퀀스 수:", len(episodes))
    return episodes, targets

