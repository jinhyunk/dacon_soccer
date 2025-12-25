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

from config import BATCH_SIZE, FIELD_X, FIELD_Y, MAX_SEQ_LEN


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
        seq = torch.tensor(self.episodes[idx], dtype=torch.float32)  # [T, 2]
        tgt = torch.tensor(self.targets[idx], dtype=torch.float32)   # [2]
        length = seq.size(0)
        return seq, length, tgt


def collate_fn(batch):
    """가변 길이 시퀀스를 패딩하여 배치로 만듦"""
    seqs, lengths, tgts = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    # 패딩 (batch_first=True -> [B, T, 2])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    tgts = torch.stack(tgts, dim=0)  # [B, 2]
    
    # 패딩 마스크 생성 (True = 패딩 위치)
    max_len = padded.size(1)
    padding_mask = torch.arange(max_len).unsqueeze(0) >= lengths.unsqueeze(1)
    
    return padded, lengths, tgts, padding_mask


# ==========================================
# DataLoader 빌더
# ==========================================

def build_dataloaders(
    episodes: List[np.ndarray],
    targets: List[np.ndarray],
    game_ids: List = None,
    batch_size: int = BATCH_SIZE,
    test_size: float = 0.2,
    split_by_game: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    시퀀스 리스트를 train/valid로 나누어 DataLoader 반환
    
    Args:
        episodes: 시퀀스 리스트
        targets: 타깃 리스트
        game_ids: 각 에피소드의 game_id 리스트 (split_by_game=True일 때 필요)
        batch_size: 배치 크기
        test_size: 검증 데이터 비율
        split_by_game: True면 game_id 기준 분리, False면 에피소드 기준 분리
    """
    if len(episodes) < 2:
        raise ValueError("시퀀스가 너무 적어서 학습이 불가능합니다.")

    if split_by_game and game_ids is not None:
        # game_id 기준 분리: 같은 game_id는 train 또는 valid에만 존재
        unique_game_ids = np.array(list(set(game_ids)))
        train_games, valid_games = train_test_split(
            unique_game_ids, test_size=test_size, random_state=42
        )
        train_games_set = set(train_games)
        valid_games_set = set(valid_games)
        
        idx_train = [i for i, gid in enumerate(game_ids) if gid in train_games_set]
        idx_valid = [i for i, gid in enumerate(game_ids) if gid in valid_games_set]
        
        print(f"Split by game_id: {len(train_games)} train games, {len(valid_games)} valid games")
    else:
        # 에피소드 기준 분리 (기존 방식)
        idx_train, idx_valid = train_test_split(
            np.arange(len(episodes)), test_size=test_size, random_state=42
        )
        if split_by_game:
            print("Warning: game_ids not provided, falling back to episode-based split")

    episodes_train = [episodes[i] for i in idx_train]
    targets_train = [targets[i] for i in idx_train]
    episodes_valid = [episodes[i] for i in idx_valid]
    targets_valid = [targets[i] for i in idx_valid]

    train_loader = DataLoader(
        EpisodeDataset(episodes_train, targets_train),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        EpisodeDataset(episodes_valid, targets_valid),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    print(f"train: {len(episodes_train)}, valid: {len(episodes_valid)}")
    return train_loader, valid_loader


# ==========================================
# 시퀀스 생성 유틸 함수
# ==========================================

def _build_sequence_from_group(g: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    DataFrame 그룹에서 좌표 시퀀스와 타깃을 생성하는 공통 로직
    
    시퀀스 구성:
    - 모든 행의 start_x, start_y
    - 마지막 행 이전까지의 end_x, end_y
    
    타깃: 마지막 행의 end_x, end_y
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

    seq = np.array(coords, dtype="float32")  # [T, 2]
    target = np.array([ex[-1], ey[-1]], dtype="float32")  # [2]

    # 시퀀스 길이 제한
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[-MAX_SEQ_LEN:]

    return seq, target


# ==========================================
# 학습용 시퀀스 생성 (game_episode 기준)
# ==========================================

def build_train_sequences(
    df: pd.DataFrame,
    return_game_ids: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
    """
    game_episode 기준으로 시퀀스 생성
    
    Args:
        df: 학습 데이터프레임
        return_game_ids: True면 game_id 리스트도 반환
    
    Returns:
        episodes: 시퀀스 리스트
        targets: 타깃 리스트
        game_ids: 각 에피소드의 game_id 리스트 (return_game_ids=True일 때만)
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    game_ids: List = []

    for game_episode, g in tqdm(df.groupby("game_episode"), desc="Building train sequences"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        seq, target = _build_sequence_from_group(g)
        episodes.append(seq)
        targets.append(target)
        
        # game_episode에서 game_id 추출 (game_episode 형식: "game_id_episode_num")
        # 또는 데이터에 game_id 컬럼이 있다면 그것을 사용
        if "game_id" in g.columns:
            game_id = g["game_id"].iloc[0]
        else:
            # game_episode에서 game_id 추출 시도
            # 형식이 "gameid_episodenum" 이라고 가정
            game_id = "_".join(str(game_episode).split("_")[:-1]) if "_" in str(game_episode) else game_episode
        game_ids.append(game_id)

    print(f"Total sequences: {len(episodes)}")
    print(f"Unique game_ids: {len(set(game_ids))}")
    
    if return_game_ids:
        return episodes, targets, game_ids
    return episodes, targets


# ==========================================
# 추론용 시퀀스 생성
# ==========================================

def build_inference_sequence(use_df: pd.DataFrame) -> np.ndarray:
    """
    추론용 시퀀스 생성
    마지막 행의 end_x, end_y는 예측 대상이므로 제외
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

    seq = np.array(coords, dtype="float32")
    
    # 시퀀스 길이 제한
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[-MAX_SEQ_LEN:]
    
    return seq

