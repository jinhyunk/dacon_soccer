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
# 학습용 시퀀스 생성 (game_episode 기준) - Episode 모드
# ==========================================

def build_train_sequences_by_episode(
    df: pd.DataFrame,
    return_game_ids: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
    """
    game_episode 기준으로 시퀀스 생성 (Episode 모드)
    
    한 에피소드 전체를 하나의 시퀀스로 사용하고, 
    마지막 액션의 도착 좌표를 예측
    
    Args:
        df: 학습 데이터프레임
        return_game_ids: True면 game_id 리스트도 반환
    
    Returns:
        episodes: 시퀀스 리스트
        targets: 타깃 리스트
        game_ids: 각 에피소드의 game_id 리스트
    """
    episodes: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    game_ids: List = []

    for game_episode, g in tqdm(df.groupby("game_episode"), desc="Building sequences (episode mode)"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue

        seq, target = _build_sequence_from_group(g)
        episodes.append(seq)
        targets.append(target)
        
        # game_id 추출
        if "game_id" in g.columns:
            game_id = g["game_id"].iloc[0]
        else:
            game_id = "_".join(str(game_episode).split("_")[:-1]) if "_" in str(game_episode) else game_episode
        game_ids.append(game_id)

    print(f"Total sequences: {len(episodes)}")
    print(f"Unique game_ids: {len(set(game_ids))}")
    
    if return_game_ids:
        return episodes, targets, game_ids
    return episodes, targets


# ==========================================
# 학습용 시퀀스 생성 (phase 기준) - Phase 모드
# ==========================================

def build_train_sequences_by_phase(
    df: pd.DataFrame,
    return_game_ids: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
    """
    Episode 내 phase 기준으로 시퀀스 생성 (Phase 모드)
    
    1. 먼저 game_episode로 나눔
    2. 각 episode 내에서 phase로 다시 나눔
    3. 각 phase를 하나의 시퀀스로 사용하고, 마지막 좌표를 예측
    
    Args:
        df: 학습 데이터프레임 (반드시 'phase' 컬럼 필요)
        return_game_ids: True면 game_id 리스트도 반환
    
    Returns:
        sequences: 시퀀스 리스트
        targets: 타깃 리스트
        game_ids: 각 시퀀스의 game_id 리스트
    """
    if "phase" not in df.columns:
        raise ValueError("Phase 모드를 사용하려면 'phase' 컬럼이 필요합니다.")
    
    sequences: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    game_ids: List = []
    
    total_episodes = 0
    total_phases = 0

    # 1단계: episode 기준으로 그룹화
    for game_episode, episode_df in tqdm(df.groupby("game_episode"), desc="Building sequences (phase mode)"):
        total_episodes += 1
        
        # game_id 추출 (episode 수준에서 한 번만)
        if "game_id" in episode_df.columns:
            game_id = episode_df["game_id"].iloc[0]
        else:
            game_id = "_".join(str(game_episode).split("_")[:-1]) if "_" in str(game_episode) else game_episode
        
        # 2단계: episode 내에서 phase 기준으로 그룹화
        for phase_id, phase_df in episode_df.groupby("phase"):
            phase_df = phase_df.sort_values("time_seconds").reset_index(drop=True)
            
            if len(phase_df) < 2:
                continue
            
            total_phases += 1
            
            seq, target = _build_sequence_from_group(phase_df)
            sequences.append(seq)
            targets.append(target)
            game_ids.append(game_id)

    print(f"Total episodes: {total_episodes}")
    print(f"Total phases (sequences): {total_phases}")
    print(f"Unique game_ids: {len(set(game_ids))}")
    
    if return_game_ids:
        return sequences, targets, game_ids
    return sequences, targets


# ==========================================
# 학습용 시퀀스 생성 (team_id 기준) - Team 모드
# ==========================================

def build_train_sequences_by_team(
    df: pd.DataFrame,
    return_game_ids: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
    """
    Episode 내 team_id 기준으로 시퀀스 생성 (Team 모드)
    
    1. 먼저 game_episode로 나눔
    2. 각 episode의 마지막 행의 team_id를 확인
    3. 해당 team_id의 데이터만 사용하여 시퀀스 생성
    4. 마지막 행의 좌표를 예측 타깃으로 사용
    
    Args:
        df: 학습 데이터프레임 (반드시 'team_id' 컬럼 필요)
        return_game_ids: True면 game_id 리스트도 반환
    
    Returns:
        sequences: 시퀀스 리스트
        targets: 타깃 리스트
        game_ids: 각 시퀀스의 game_id 리스트
    """
    if "team_id" not in df.columns:
        raise ValueError("Team 모드를 사용하려면 'team_id' 컬럼이 필요합니다.")
    
    sequences: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    game_ids: List = []
    
    total_episodes = 0
    skipped = 0

    for game_episode, episode_df in tqdm(df.groupby("game_episode"), desc="Building sequences (team mode)"):
        episode_df = episode_df.sort_values("time_seconds").reset_index(drop=True)
        total_episodes += 1
        
        if len(episode_df) < 2:
            skipped += 1
            continue
        
        # game_id 추출
        if "game_id" in episode_df.columns:
            game_id = episode_df["game_id"].iloc[0]
        else:
            game_id = "_".join(str(game_episode).split("_")[:-1]) if "_" in str(game_episode) else game_episode
        
        # 마지막 행의 team_id 확인
        target_team_id = episode_df.iloc[-1]["team_id"]
        
        # 해당 team_id의 데이터만 필터링
        team_df = episode_df[episode_df["team_id"] == target_team_id].sort_values("time_seconds").reset_index(drop=True)
        
        if len(team_df) < 2:
            skipped += 1
            continue
        
        seq, target = _build_sequence_from_group(team_df)
        sequences.append(seq)
        targets.append(target)
        game_ids.append(game_id)

    print(f"Total episodes: {total_episodes}")
    print(f"Valid sequences: {len(sequences)}")
    print(f"Skipped (insufficient data): {skipped}")
    print(f"Unique game_ids: {len(set(game_ids))}")
    
    if return_game_ids:
        return sequences, targets, game_ids
    return sequences, targets


# ==========================================
# 통합 시퀀스 생성 함수
# ==========================================

def build_train_sequences(
    df: pd.DataFrame,
    mode: str = "episode",
    return_game_ids: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray], List]:
    """
    시퀀스 모드에 따라 적절한 시퀀스 생성 함수 호출
    
    Args:
        df: 학습 데이터프레임
        mode: "episode", "phase", 또는 "team"
            - episode: game_episode 단위로 시퀀스 생성 (기본값)
            - phase: phase 단위로 시퀀스 생성
            - team: 마지막 행과 동일한 team_id의 데이터만 사용
        return_game_ids: True면 game_id 리스트도 반환
    
    Returns:
        sequences: 시퀀스 리스트
        targets: 타깃 리스트
        game_ids: 각 시퀀스의 game_id 리스트
    """
    print(f"Sequence mode: {mode}")
    
    if mode == "episode":
        return build_train_sequences_by_episode(df, return_game_ids)
    elif mode == "phase":
        return build_train_sequences_by_phase(df, return_game_ids)
    elif mode == "team":
        return build_train_sequences_by_team(df, return_game_ids)
    else:
        raise ValueError(f"Unknown sequence mode: {mode}. Use 'episode', 'phase', or 'team'")


# ==========================================
# 추론용 시퀀스 생성
# ==========================================

def build_inference_sequence(use_df: pd.DataFrame) -> np.ndarray:
    """
    추론용 시퀀스 생성 (Episode 모드)
    마지막 행의 end_x, end_y는 예측 대상이므로 제외
    
    Args:
        use_df: 추론할 에피소드의 데이터프레임
        
    Returns:
        seq: (T, 2) 정규화된 좌표 시퀀스
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


def build_inference_sequence_by_phase(
    use_df: pd.DataFrame, 
    target_phase: str
) -> np.ndarray:
    """
    추론용 시퀀스 생성 (Phase 모드)
    지정된 phase에 해당하는 데이터만 사용하여 시퀀스 생성
    
    Args:
        use_df: 추론할 에피소드의 전체 데이터프레임
        target_phase: 예측할 phase ID
        
    Returns:
        seq: (T, 2) 정규화된 좌표 시퀀스
    """
    if "phase" not in use_df.columns:
        raise ValueError("Phase 모드를 사용하려면 'phase' 컬럼이 필요합니다.")
    
    # 해당 phase 데이터만 추출
    phase_df = use_df[use_df["phase"] == target_phase].copy()
    phase_df = phase_df.sort_values("time_seconds").reset_index(drop=True)
    
    if len(phase_df) == 0:
        raise ValueError(f"Phase '{target_phase}'에 해당하는 데이터가 없습니다.")
    
    return build_inference_sequence(phase_df)


def get_phases_from_episode(use_df: pd.DataFrame) -> List[str]:
    """
    에피소드에서 모든 phase ID 목록 추출
    
    Args:
        use_df: 에피소드 데이터프레임
        
    Returns:
        phase_ids: phase ID 리스트 (시간 순서)
    """
    if "phase" not in use_df.columns:
        raise ValueError("'phase' 컬럼이 필요합니다.")
    
    # 각 phase의 첫 번째 time_seconds 기준으로 정렬
    phase_order = use_df.groupby("phase")["time_seconds"].min().sort_values()
    return phase_order.index.tolist()


def build_inference_sequence_by_team(use_df: pd.DataFrame) -> np.ndarray:
    """
    추론용 시퀀스 생성 (Team 모드)
    마지막 행과 동일한 team_id의 데이터만 사용하여 시퀀스 생성
    
    Args:
        use_df: 추론할 에피소드의 전체 데이터프레임
        
    Returns:
        seq: (T, 2) 정규화된 좌표 시퀀스
    """
    if "team_id" not in use_df.columns:
        raise ValueError("Team 모드를 사용하려면 'team_id' 컬럼이 필요합니다.")
    
    use_df = use_df.sort_values("time_seconds").reset_index(drop=True)
    
    # 마지막 행의 team_id 확인
    target_team_id = use_df.iloc[-1]["team_id"]
    
    # 해당 team_id의 데이터만 필터링
    team_df = use_df[use_df["team_id"] == target_team_id].sort_values("time_seconds").reset_index(drop=True)
    
    if len(team_df) == 0:
        raise ValueError(f"Team ID '{target_team_id}'에 해당하는 데이터가 없습니다.")
    
    return build_inference_sequence(team_df)

