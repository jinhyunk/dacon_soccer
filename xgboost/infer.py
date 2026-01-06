"""
XGBoost 모델 추론 스크립트
좌표 예측 및 평가 지표 계산
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from config import (
    TEST_META_PATH,
    SUBMIT_META_PATH,
    SAMPLE_SUB_PATH,
    MODEL_PATH_X,
    MODEL_PATH_Y,
    SUBMIT_PATH,
    SEQUENCE_MODE,
    FIELD_X,
    FIELD_Y,
    log_config,
)
from dataset import extract_inference_features
from model import XGBCoordinatePredictor


def build_episode_path_map(meta_path: str) -> tuple:
    """메타데이터에서 game_episode -> 파일경로 매핑 생성"""
    test_meta = pd.read_csv(meta_path)
    base_dir = os.path.dirname(meta_path)
    data_dir = os.path.dirname(base_dir)
    
    episode_to_path = {}
    for _, row in test_meta.iterrows():
        ep = row["game_episode"]
        path = row["path"]
        # base_test.csv의 ./test/... -> basic_test/... 로 변환
        if path.startswith("./test/"):
            path = os.path.join(data_dir, "basic_test", path[7:])
        elif path.startswith("./temporal_test/"):
            path = os.path.join(base_dir, path[2:])
        elif path.startswith("./"):
            path = os.path.join(base_dir, path[2:])
        episode_to_path[ep] = path
    
    return episode_to_path, test_meta


def main(submit_mode: bool = False, sequence_mode: str = None):
    """
    메인 추론 함수
    
    Args:
        submit_mode: True면 제출용(base_test.csv), False면 검증용(temporal_test.csv)
        sequence_mode: 시퀀스 모드 (None이면 config의 SEQUENCE_MODE 사용)
    """
    log_config()
    
    # 시퀀스 모드 결정
    seq_mode = sequence_mode if sequence_mode else SEQUENCE_MODE
    print(f"\nSequence mode: {seq_mode}")
    
    if submit_mode:
        # 제출 모드
        meta_path = SUBMIT_META_PATH
        print(f"[SUBMIT MODE] Loading test metadata from {meta_path}...")
        episode_to_path, test_meta = build_episode_path_map(meta_path)
        
        print(f"Loading sample submission from {SAMPLE_SUB_PATH}...")
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        episode_list = sample_sub["game_episode"].values
        print(f"Episodes to predict: {len(episode_list)}")
        compute_distance = False
    else:
        # 검증 모드
        meta_path = TEST_META_PATH
        print(f"[VALIDATION MODE] Loading test metadata from {meta_path}...")
        episode_to_path, test_meta = build_episode_path_map(meta_path)
        
        episode_list = test_meta["game_episode"].values
        print(f"Episodes to predict: {len(episode_list)}")
        compute_distance = True
    
    # 모델 로드
    print("\nLoading XGBoost models...")
    model = XGBCoordinatePredictor()
    model.load(MODEL_PATH_X, MODEL_PATH_Y)
    
    feature_cols = model.feature_cols
    if feature_cols:
        print(f"Feature columns loaded: {len(feature_cols)}")
    
    # 예측
    print("\nRunning inference...")
    results = []
    distances = []
    
    for idx in tqdm(episode_list, desc=f"Predicting ({seq_mode} mode)"):
        if idx not in episode_to_path:
            # 에피소드가 없으면 중앙값으로 예측
            results.append({
                "game_episode": idx,
                "end_x": FIELD_X / 2,
                "end_y": FIELD_Y / 2,
            })
            continue
        
        episode_path = episode_to_path[idx]
        
        try:
            use_df = pd.read_csv(episode_path)
            use_df = use_df.sort_values("time_seconds").reset_index(drop=True)
        except Exception as e:
            print(f"Error loading {episode_path}: {e}")
            results.append({
                "game_episode": idx,
                "end_x": FIELD_X / 2,
                "end_y": FIELD_Y / 2,
            })
            continue
        
        if len(use_df) == 0:
            results.append({
                "game_episode": idx,
                "end_x": FIELD_X / 2,
                "end_y": FIELD_Y / 2,
            })
            continue
        
        # 피처 추출
        features = extract_inference_features(
            use_df, mode=seq_mode, feature_cols=feature_cols
        )
        
        # 예측
        pred = model.predict(features)  # (1, 2)
        pred_x = float(np.clip(pred[0, 0] * FIELD_X, 0, FIELD_X))
        pred_y = float(np.clip(pred[0, 1] * FIELD_Y, 0, FIELD_Y))
        
        # Ground truth와의 거리 계산 (검증 모드)
        if compute_distance:
            gt_x = use_df.iloc[-1]["end_x"]
            gt_y = use_df.iloc[-1]["end_y"]
            dist = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
            distances.append(dist)
        
        results.append({
            "game_episode": idx,
            "end_x": pred_x,
            "end_y": pred_y,
        })
    
    # 결과 저장
    result_df = pd.DataFrame(results)
    
    if submit_mode:
        result_df.to_csv(SUBMIT_PATH, index=False)
        print(f"\nSubmission saved to {SUBMIT_PATH}")
    
    print(f"Total predictions: {len(result_df)}")
    
    # 통계 출력
    print("\nPrediction statistics:")
    print(f"  end_x: mean={result_df['end_x'].mean():.2f}, std={result_df['end_x'].std():.2f}")
    print(f"  end_y: mean={result_df['end_y'].mean():.2f}, std={result_df['end_y'].std():.2f}")
    
    # Mean distance 출력 (검증 모드에서만)
    if len(distances) > 0:
        distances = np.array(distances)
        print(f"\n{'='*50}")
        print("[Distance Metrics]")
        print(f"{'='*50}")
        print(f"  Mean Distance: {distances.mean():.4f} m")
        print(f"  Std Distance:  {distances.std():.4f} m")
        print(f"  Min Distance:  {distances.min():.4f} m")
        print(f"  Max Distance:  {distances.max():.4f} m")
        print(f"  Median:        {np.median(distances):.4f} m")
        print(f"  90th Percentile: {np.percentile(distances, 90):.4f} m")
        print(f"{'='*50}")
    
    return result_df, distances


def compare_with_baseline(
    model_distances: np.ndarray,
    baseline_name: str = "Center",
) -> None:
    """베이스라인과 비교"""
    # 필드 중앙 예측 시 평균 거리 (대략적인 추정)
    # 실제 데이터로 계산해야 정확
    print(f"\n[Comparison with {baseline_name}]")
    print(f"  Model Mean Distance: {model_distances.mean():.4f} m")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="XGBoost Inference")
    parser.add_argument("--submit", action="store_true",
                        help="제출 모드 (base_test.csv + sample_submission.csv 사용)")
    parser.add_argument("--mode", type=str, default=None,
                        choices=["episode", "phase", "team"],
                        help="시퀀스 모드 (episode/phase/team). 미지정시 config 사용")
    
    args = parser.parse_args()
    
    result_df, distances = main(submit_mode=args.submit, sequence_mode=args.mode)
    
    if len(distances) > 0:
        compare_with_baseline(np.array(distances))

