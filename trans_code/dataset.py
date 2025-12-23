"""
Dataset 및 시퀀스 생성 함수

데이터 모드:
- episode: game_episode 기준으로 시퀀스 생성 (기본)
- episode_phase: game_episode 내에서 phase별로 시퀀스 생성
- phase: phase 기준으로 시퀀스 생성 (pretrain용)
- team_id: (game_episode, team_id) 기준으로 시퀀스 생성 (pretrain용)
- episode_team: team별 episode 시퀀스 (team-wise finetune용)
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
# 학습용 시퀀스 생성 - episode 기준 (기본)
# ==========================================

def build_episode_sequences(
    df: pd.DataFrame,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    game_episode 기준으로 시퀀스 생성
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby("game_episode"), desc="Building episode sequences"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        seq, target = _build_sequence_from_group(g)
        episodes.append(seq)
        targets.append(target)

    print(f"[episode mode] Total sequences: {len(episodes)}")
    return episodes, targets


# ==========================================
# 학습용 시퀀스 생성 - episode + phase 기준
# ==========================================

def build_episode_phase_sequences(
    df: pd.DataFrame,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    game_episode 내에서 phase별로 시퀀스 생성
    각 (game_episode, phase) 조합마다 하나의 시퀀스
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby(["game_episode", "phase"]), desc="Building episode+phase sequences"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        seq, target = _build_sequence_from_group(g)
        episodes.append(seq)
        targets.append(target)

    print(f"[episode_phase mode] Total sequences: {len(episodes)}")
    return episodes, targets


# ==========================================
# 학습용 시퀀스 생성 - phase 기준 (pretrain용)
# ==========================================

def build_phase_sequences(
    df: pd.DataFrame,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    phase 기준으로 시퀀스 생성 (pretrain용)
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby("phase"), desc="Building phase sequences"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        seq, target = _build_sequence_from_group(g)
        episodes.append(seq)
        targets.append(target)

    print(f"[phase mode] Total sequences: {len(episodes)}")
    return episodes, targets


# ==========================================
# 학습용 시퀀스 생성 - team_id 기준 (pretrain용)
# ==========================================

def build_team_sequences(
    df: pd.DataFrame,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    (game_episode, team_id) 기준으로 시퀀스 생성 (pretrain용)
    main_code의 build_pretrain_sequences_by_team과 동일
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby(["game_episode", "team_id"]), desc="Building team sequences"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        seq, target = _build_sequence_from_group(g)
        episodes.append(seq)
        targets.append(target)

    print(f"[team_id mode] Total sequences: {len(episodes)}")
    return episodes, targets


# ==========================================
# 학습용 시퀀스 생성 - team별 episode (finetune용)
# ==========================================

def build_episode_team_sequences(
    df: pd.DataFrame,
    team_id: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    episode 단위로 나눈 뒤,
    각 episode 내에서 마지막 행의 team_id가 주어진 team_id인 경우만 사용.
    해당 episode에서 마지막 행과 동일한 team_id의 데이터만 모아 시퀀스를 만듦.
    
    main_code의 build_episode_team_sequences와 동일
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    for _, g in tqdm(df.groupby("game_episode"), desc=f"Building episode seq (team={team_id})"):
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

    print(f"[team {team_id}] episode sequences: {len(episodes)}")
    return episodes, targets


# ==========================================
# 통합 시퀀스 빌더
# ==========================================

def build_train_sequences(
    df: pd.DataFrame,
    data_mode: str = "episode",
    team_id: int = None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    data_mode에 따라 다른 방식으로 시퀀스 생성
    
    Args:
        df: 학습 데이터프레임
        data_mode: 
            - "episode": game_episode 기준 (기본)
            - "episode_phase": game_episode + phase 기준
            - "phase": phase 기준 (pretrain용)
            - "team_id": (game_episode, team_id) 기준 (pretrain용)
            - "episode_team": team별 episode 시퀀스 (team_id 필수)
    
    Returns:
        episodes, targets
    """
    if data_mode == "episode":
        return build_episode_sequences(df)
    elif data_mode == "episode_phase":
        return build_episode_phase_sequences(df)
    elif data_mode == "phase":
        return build_phase_sequences(df)
    elif data_mode == "team_id":
        return build_team_sequences(df)
    elif data_mode == "episode_team":
        if team_id is None:
            raise ValueError("data_mode='episode_team' requires team_id parameter.")
        return build_episode_team_sequences(df, team_id)
    else:
        raise ValueError(
            f"Unknown data_mode: {data_mode}. "
            "Use 'episode', 'episode_phase', 'phase', 'team_id', or 'episode_team'."
        )


# ==========================================
# 추론용 시퀀스 생성
# ==========================================

def build_inference_sequence(
    use_df: pd.DataFrame,
    infer_mode: str = "episode",
) -> np.ndarray:
    """
    추론용 시퀀스 생성
    마지막 행의 end_x, end_y는 예측 대상이므로 제외
    
    Args:
        use_df: 추론에 사용할 데이터프레임
        infer_mode: 
            - "episode": 전체 episode 데이터 사용
            - "episode_phase": 마지막 행과 같은 phase 데이터만 사용
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


def get_inference_dataframe(
    g: pd.DataFrame,
    infer_mode: str = "episode",
) -> pd.DataFrame:
    """
    추론 모드에 따라 사용할 데이터프레임 반환
    
    Args:
        g: episode 전체 데이터프레임 (time_seconds 정렬됨)
        infer_mode:
            - "episode": 전체 episode 데이터 사용
            - "episode_phase": 마지막 행과 같은 phase 데이터만 사용
            - "episode_team": 마지막 행과 같은 team_id 데이터만 사용
    
    Returns:
        추론에 사용할 데이터프레임
    """
    if infer_mode == "episode":
        return g
    elif infer_mode == "episode_phase":
        # 마지막 행의 phase와 동일한 phase 데이터만 사용
        last_phase = g.iloc[-1]["phase"]
        g_phase = g[g["phase"] == last_phase].reset_index(drop=True)
        # 데이터가 너무 적으면 전체 episode fallback
        if len(g_phase) < 2:
            return g
        return g_phase
    elif infer_mode == "episode_team":
        # 마지막 행의 team_id와 동일한 team_id 데이터만 사용
        last_team = g.iloc[-1]["team_id"]
        g_team = g[g["team_id"] == last_team].reset_index(drop=True)
        # 데이터가 너무 적으면 전체 episode fallback
        if len(g_team) < 2:
            return g
        return g_team
    else:
        raise ValueError(
            f"Unknown infer_mode: {infer_mode}. "
            "Use 'episode', 'episode_phase', or 'episode_team'."
        )
