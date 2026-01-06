"""
XGBoost용 피처 추출 및 데이터셋 생성
시퀀스 데이터를 tabular 피처로 변환
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import FIELD_X, FIELD_Y, LAST_N_COORDS, EXCLUDE_FEATURES, EXCLUDE_FEATURE_GROUPS


# ==========================================
# 피처 그룹 정의 (ablation study용)
# ==========================================
FEATURE_GROUPS = {
    "stats": ["n_actions", "start_x_mean", "start_y_mean", "start_x_std", "start_y_std",
              "end_x_mean", "end_y_mean", "end_x_std", "end_y_std"],
    "last_pos": ["last_start_x", "last_start_y"],
    "last_coords": [f"coord_{i}_{c}" for i in range(LAST_N_COORDS) for c in ["x", "y"]],
    "movement": ["total_distance", "mean_distance", "max_distance", "last_move_angle", "last_move_dist"],
    "direction": ["overall_direction_x", "overall_direction_y", "overall_angle", "overall_distance"],
    "range": ["x_range", "y_range", "x_min", "x_max", "y_min", "y_max"],
    "time": ["duration", "time_last", "avg_speed"],
    "context": ["is_home", "team_id", "last_action_is_pass", "last_action_is_carry",
                "last_action_is_duel", "pass_ratio", "carry_ratio", "last_result_success"],
}


def get_excluded_features() -> set:
    """제외할 피처 목록 반환"""
    excluded = set(EXCLUDE_FEATURES)
    
    # 그룹 단위 제외
    for group in EXCLUDE_FEATURE_GROUPS:
        if group in FEATURE_GROUPS:
            excluded.update(FEATURE_GROUPS[group])
    
    return excluded


def filter_features(features: Dict[str, float]) -> Dict[str, float]:
    """제외 목록에 있는 피처 제거"""
    excluded = get_excluded_features()
    if not excluded:
        return features
    
    return {k: v for k, v in features.items() if k not in excluded}


# ==========================================
# 피처 추출 함수
# ==========================================

def extract_features_from_sequence(
    df: pd.DataFrame,
    full_episode_df: pd.DataFrame = None,
    last_n: int = LAST_N_COORDS,
) -> Dict[str, float]:
    """
    에피소드/시퀀스 데이터프레임에서 피처 추출 (버전1 피처 + NaN 처리)
    
    Args:
        df: 시퀀스 데이터프레임 (time_seconds로 정렬됨)
        full_episode_df: 전체 에피소드 데이터프레임 (미사용, 호환성 유지)
        last_n: 마지막 N개 좌표를 피처로 사용
    
    Returns:
        피처 딕셔너리
    """
    features = {}
    
    n_rows = len(df)
    
    # 좌표 데이터 정규화 (start는 항상 존재)
    sx = df["start_x"].values / FIELD_X
    sy = df["start_y"].values / FIELD_Y
    
    # end_x, end_y는 마지막 행이 NaN일 수 있음 (제출용 데이터)
    # 마지막 행의 end는 예측 대상이므로 제외하고 처리
    ex_raw = df["end_x"].values / FIELD_X
    ey_raw = df["end_y"].values / FIELD_Y
    
    # 마지막 행을 제외한 end 좌표 (피처 계산용)
    if n_rows > 1:
        ex = ex_raw[:-1]  # 마지막 행 제외
        ey = ey_raw[:-1]
        sx_for_move = sx[:-1]  # 이동 계산용 start도 마지막 제외
        sy_for_move = sy[:-1]
    else:
        ex = np.array([])
        ey = np.array([])
        sx_for_move = np.array([])
        sy_for_move = np.array([])
    
    # ===== 1. 기본 통계 피처 =====
    features["n_actions"] = n_rows
    
    # 시작 좌표 통계
    features["start_x_mean"] = float(sx.mean())
    features["start_y_mean"] = float(sy.mean())
    features["start_x_std"] = float(sx.std()) if n_rows > 1 else 0.0
    features["start_y_std"] = float(sy.std()) if n_rows > 1 else 0.0
    
    # 종료 좌표 통계 (마지막 행 제외)
    if len(ex) > 0:
        features["end_x_mean"] = float(ex.mean())
        features["end_y_mean"] = float(ey.mean())
        features["end_x_std"] = float(ex.std()) if len(ex) > 1 else 0.0
        features["end_y_std"] = float(ey.std()) if len(ey) > 1 else 0.0
    else:
        features["end_x_mean"] = float(sx.mean())  # fallback to start
        features["end_y_mean"] = float(sy.mean())
        features["end_x_std"] = 0.0
        features["end_y_std"] = 0.0
    
    # ===== 2. 마지막 액션 피처 =====
    features["last_start_x"] = float(sx[-1])
    features["last_start_y"] = float(sy[-1])
    
    # ===== 3. 마지막 N개 좌표 (flatten) =====
    # 좌표 시퀀스: 모든 start + 마지막 제외한 end
    coords = []
    for i in range(n_rows):
        coords.append([sx[i], sy[i]])
        if i < n_rows - 1 and i < len(ex):  # 마지막 행의 end는 예측 대상이므로 제외
            coords.append([ex[i], ey[i]])
    
    coords = np.array(coords) if len(coords) > 0 else np.array([[0.5, 0.5]])
    
    if len(coords) >= last_n:
        last_coords = coords[-last_n:]
    else:
        pad_len = last_n - len(coords)
        padding = np.tile(coords[0:1], (pad_len, 1))
        last_coords = np.vstack([padding, coords])
    
    for i in range(last_n):
        features[f"coord_{i}_x"] = float(last_coords[i, 0])
        features[f"coord_{i}_y"] = float(last_coords[i, 1])
    
    # ===== 4. 이동 방향 및 거리 피처 =====
    if len(ex) > 0 and len(sx_for_move) > 0:
        move_x = ex - sx_for_move
        move_y = ey - sy_for_move
        move_dist = np.sqrt(move_x**2 + move_y**2)
        
        features["total_distance"] = float(move_dist.sum())
        features["mean_distance"] = float(move_dist.mean())
        features["max_distance"] = float(move_dist.max())
        
        # 마지막 이동 방향 (마지막-1 행의 이동)
        features["last_move_angle"] = float(np.arctan2(move_y[-1], move_x[-1]))
        features["last_move_dist"] = float(move_dist[-1])
    else:
        features["total_distance"] = 0.0
        features["mean_distance"] = 0.0
        features["max_distance"] = 0.0
        features["last_move_angle"] = 0.0
        features["last_move_dist"] = 0.0
    
    # 전체 진행 방향 (시작점 -> 마지막 시작점)
    if n_rows > 1:
        overall_dx = sx[-1] - sx[0]
        overall_dy = sy[-1] - sy[0]
        features["overall_direction_x"] = float(overall_dx)
        features["overall_direction_y"] = float(overall_dy)
        features["overall_angle"] = float(np.arctan2(overall_dy, overall_dx))
        features["overall_distance"] = float(np.sqrt(overall_dx**2 + overall_dy**2))
    else:
        features["overall_direction_x"] = 0.0
        features["overall_direction_y"] = 0.0
        features["overall_angle"] = 0.0
        features["overall_distance"] = 0.0
    
    # ===== 5. 위치 범위 피처 =====
    features["x_range"] = float(sx.max() - sx.min())
    features["y_range"] = float(sy.max() - sy.min())
    features["x_min"] = float(sx.min())
    features["x_max"] = float(sx.max())
    features["y_min"] = float(sy.min())
    features["y_max"] = float(sy.max())
    
    # ===== 6. 시간 관련 피처 =====
    if "time_seconds" in df.columns:
        time = df["time_seconds"].values
        features["duration"] = float(time[-1] - time[0]) if n_rows > 1 else 0.0
        features["time_last"] = float(time[-1])
        
        if features["duration"] > 0:
            features["avg_speed"] = features["total_distance"] / features["duration"]
        else:
            features["avg_speed"] = 0.0
    
    # ===== 7. 컨텍스트 피처 =====
    if "is_home" in df.columns:
        features["is_home"] = int(df["is_home"].iloc[-1])
    
    if "team_id" in df.columns:
        features["team_id"] = df["team_id"].iloc[-1]
    
    # 액션 타입 원핫 (마지막 액션)
    if "type_name" in df.columns:
        last_type = df["type_name"].iloc[-1]
        features["last_action_is_pass"] = int(last_type == "Pass")
        features["last_action_is_carry"] = int(last_type == "Carry")
        features["last_action_is_duel"] = int(last_type == "Duel")
        
        # 액션 타입 비율
        type_counts = df["type_name"].value_counts(normalize=True)
        features["pass_ratio"] = float(type_counts.get("Pass", 0))
        features["carry_ratio"] = float(type_counts.get("Carry", 0))
    
    # 결과 (마지막 액션)
    if "result_name" in df.columns:
        last_result = df["result_name"].iloc[-1]
        features["last_result_success"] = int(last_result == "Successful")
    
    # 제외 목록 적용
    return filter_features(features)


def get_target_from_sequence(df: pd.DataFrame) -> Tuple[float, float]:
    """
    시퀀스의 타깃 좌표 추출 (마지막 행의 end_x, end_y)
    
    Returns:
        (end_x, end_y) - 정규화된 좌표
    """
    last_row = df.iloc[-1]
    return last_row["end_x"] / FIELD_X, last_row["end_y"] / FIELD_Y


# ==========================================
# 시퀀스 모드별 데이터셋 생성
# ==========================================

def build_dataset_by_episode(
    df: pd.DataFrame,
    return_game_ids: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List]:
    """
    game_episode 기준으로 데이터셋 생성 (Episode 모드)
    
    Returns:
        features_df: 피처 데이터프레임
        targets: (N, 2) 타깃 배열 [x, y]
        game_ids: game_id 리스트
    """
    features_list = []
    targets = []
    game_ids = []
    episode_ids = []
    
    for game_episode, g in tqdm(df.groupby("game_episode"), desc="Building dataset (episode mode)"):
        g = g.sort_values("time_seconds").reset_index(drop=True)
        if len(g) < 2:
            continue
        
        # 피처 추출 (full_episode_df로 전체 에피소드 정보 전달)
        feats = extract_features_from_sequence(g, full_episode_df=g)
        features_list.append(feats)
        
        # 타깃 추출
        target_x, target_y = get_target_from_sequence(g)
        targets.append([target_x, target_y])
        
        # game_id 추출
        if "game_id" in g.columns:
            game_id = g["game_id"].iloc[0]
        else:
            game_id = "_".join(str(game_episode).split("_")[:-1]) if "_" in str(game_episode) else game_episode
        game_ids.append(game_id)
        episode_ids.append(game_episode)
    
    features_df = pd.DataFrame(features_list)
    features_df["game_episode"] = episode_ids
    targets = np.array(targets)
    
    print(f"Total samples: {len(features_df)}")
    print(f"Features: {features_df.shape[1] - 1}")
    print(f"Unique game_ids: {len(set(game_ids))}")
    
    if return_game_ids:
        return features_df, targets, game_ids
    return features_df, targets


def build_dataset_by_phase(
    df: pd.DataFrame,
    return_game_ids: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List]:
    """
    Phase 기준으로 데이터셋 생성 (Phase 모드)
    """
    if "phase" not in df.columns:
        raise ValueError("Phase 모드를 사용하려면 'phase' 컬럼이 필요합니다.")
    
    features_list = []
    targets = []
    game_ids = []
    episode_ids = []
    
    for game_episode, episode_df in tqdm(df.groupby("game_episode"), desc="Building dataset (phase mode)"):
        episode_df = episode_df.sort_values("time_seconds").reset_index(drop=True)
        
        # game_id 추출
        if "game_id" in episode_df.columns:
            game_id = episode_df["game_id"].iloc[0]
        else:
            game_id = "_".join(str(game_episode).split("_")[:-1]) if "_" in str(game_episode) else game_episode
        
        # phase별 처리
        for phase_id, phase_df in episode_df.groupby("phase"):
            phase_df = phase_df.sort_values("time_seconds").reset_index(drop=True)
            if len(phase_df) < 2:
                continue
            
            # full_episode_df로 전체 에피소드 정보 전달 (phase 정보 활용)
            feats = extract_features_from_sequence(phase_df, full_episode_df=episode_df)
            features_list.append(feats)
            
            target_x, target_y = get_target_from_sequence(phase_df)
            targets.append([target_x, target_y])
            
            game_ids.append(game_id)
            episode_ids.append(game_episode)
    
    features_df = pd.DataFrame(features_list)
    features_df["game_episode"] = episode_ids
    targets = np.array(targets)
    
    print(f"Total samples: {len(features_df)}")
    print(f"Features: {features_df.shape[1] - 1}")
    print(f"Unique game_ids: {len(set(game_ids))}")
    
    if return_game_ids:
        return features_df, targets, game_ids
    return features_df, targets


def build_dataset_by_team(
    df: pd.DataFrame,
    return_game_ids: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List]:
    """
    Team 기준으로 데이터셋 생성 (Team 모드)
    마지막 행과 동일한 team_id의 데이터만 사용
    """
    if "team_id" not in df.columns:
        raise ValueError("Team 모드를 사용하려면 'team_id' 컬럼이 필요합니다.")
    
    features_list = []
    targets = []
    game_ids = []
    episode_ids = []
    
    for game_episode, episode_df in tqdm(df.groupby("game_episode"), desc="Building dataset (team mode)"):
        episode_df = episode_df.sort_values("time_seconds").reset_index(drop=True)
        if len(episode_df) < 2:
            continue
        
        # game_id 추출
        if "game_id" in episode_df.columns:
            game_id = episode_df["game_id"].iloc[0]
        else:
            game_id = "_".join(str(game_episode).split("_")[:-1]) if "_" in str(game_episode) else game_episode
        
        # 마지막 행의 team_id
        target_team_id = episode_df.iloc[-1]["team_id"]
        
        # 해당 team_id 데이터만 필터링
        team_df = episode_df[episode_df["team_id"] == target_team_id].sort_values("time_seconds").reset_index(drop=True)
        
        if len(team_df) < 2:
            continue
        
        # full_episode_df로 전체 에피소드 정보 전달
        feats = extract_features_from_sequence(team_df, full_episode_df=episode_df)
        features_list.append(feats)
        
        target_x, target_y = get_target_from_sequence(team_df)
        targets.append([target_x, target_y])
        
        game_ids.append(game_id)
        episode_ids.append(game_episode)
    
    features_df = pd.DataFrame(features_list)
    features_df["game_episode"] = episode_ids
    targets = np.array(targets)
    
    print(f"Total samples: {len(features_df)}")
    print(f"Features: {features_df.shape[1] - 1}")
    print(f"Unique game_ids: {len(set(game_ids))}")
    
    if return_game_ids:
        return features_df, targets, game_ids
    return features_df, targets


def build_dataset(
    df: pd.DataFrame,
    mode: str = "episode",
    return_game_ids: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, List]:
    """
    통합 데이터셋 생성 함수
    
    Args:
        df: 학습 데이터프레임
        mode: "episode", "phase", "team" 중 하나
        return_game_ids: game_id 리스트 반환 여부
    
    Returns:
        features_df: 피처 데이터프레임
        targets: (N, 2) 타깃 배열
        game_ids: game_id 리스트
    """
    print(f"Dataset mode: {mode}")
    
    if mode == "episode":
        return build_dataset_by_episode(df, return_game_ids)
    elif mode == "phase":
        return build_dataset_by_phase(df, return_game_ids)
    elif mode == "team":
        return build_dataset_by_team(df, return_game_ids)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'episode', 'phase', or 'team'")


# ==========================================
# Train/Valid 분리
# ==========================================

def split_dataset(
    features_df: pd.DataFrame,
    targets: np.ndarray,
    game_ids: List,
    test_size: float = 0.2,
    split_by_game: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    데이터셋을 train/valid로 분리
    
    Args:
        features_df: 피처 데이터프레임
        targets: 타깃 배열
        game_ids: game_id 리스트
        test_size: 검증 데이터 비율
        split_by_game: True면 game_id 기준 분리
    
    Returns:
        X_train, X_valid, y_train, y_valid
    """
    # game_episode 컬럼 제외
    feature_cols = [c for c in features_df.columns if c != "game_episode"]
    X = features_df[feature_cols].values
    
    if split_by_game and game_ids is not None:
        # game_id 기준 분리
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
        # 샘플 기준 분리
        idx_train, idx_valid = train_test_split(
            np.arange(len(X)), test_size=test_size, random_state=42
        )
    
    X_train = X[idx_train]
    X_valid = X[idx_valid]
    y_train = targets[idx_train]
    y_valid = targets[idx_valid]
    
    print(f"Train: {len(X_train)}, Valid: {len(X_valid)}")
    
    return X_train, X_valid, y_train, y_valid, feature_cols


# ==========================================
# 추론용 피처 추출
# ==========================================

def extract_inference_features(
    df: pd.DataFrame,
    mode: str = "episode",
    feature_cols: List[str] = None,
) -> np.ndarray:
    """
    추론용 피처 추출
    
    Args:
        df: 에피소드 데이터프레임
        mode: 시퀀스 모드
        feature_cols: 피처 컬럼 리스트 (학습 시 사용한 순서와 동일해야 함)
    
    Returns:
        features: (1, n_features) 피처 배열
    """
    full_episode_df = df.sort_values("time_seconds").reset_index(drop=True)
    use_df = full_episode_df.copy()
    
    if mode == "team" and "team_id" in df.columns:
        # Team 모드: 마지막 행의 team_id 데이터만 사용
        target_team_id = full_episode_df.iloc[-1]["team_id"]
        use_df = full_episode_df[full_episode_df["team_id"] == target_team_id].sort_values("time_seconds").reset_index(drop=True)
    
    # 피처 추출 (full_episode_df 전달)
    feats = extract_features_from_sequence(use_df, full_episode_df=full_episode_df)
    
    if feature_cols is not None:
        # 학습 시 사용한 피처 컬럼 순서로 정렬
        features = [feats.get(col, 0) for col in feature_cols]
    else:
        features = list(feats.values())
    
    return np.array(features).reshape(1, -1)

