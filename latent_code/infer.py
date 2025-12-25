"""
CVAE 모델 추론 스크립트
다중 미래 예측을 통한 패스 도착 지점 예측
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from config import (
    TEST_META_PATH,
    SUBMIT_META_PATH,
    SAMPLE_SUB_PATH,
    MODEL_PATH,
    SUBMIT_PATH,
    DEVICE,
    INPUT_DIM,
    HIDDEN_DIM,
    Z_DIM,
    NUM_SAMPLES,
    FIELD_X,
    FIELD_Y,
    ENCODER_TYPE,
    TRANSFORMER_NUM_LAYERS,
    TRANSFORMER_NUM_HEADS,
    MAX_SEQ_LEN,
    log_device,
)
from dataset import build_inference_sequence
from model import PassCVAE
from trainer import load_model


def predict_single(
    model: PassCVAE,
    seq: np.ndarray,
    num_samples: int = NUM_SAMPLES,
    aggregation: str = "mean"
) -> np.ndarray:
    """
    단일 시퀀스에 대해 다중 샘플링 후 집계하여 최종 예측
    
    Args:
        model: 학습된 CVAE 모델
        seq: (T, 2) - 입력 시퀀스 (정규화된 좌표)
        num_samples: 샘플링 개수
        aggregation: "mean" or "median"
    
    Returns:
        prediction: (2,) - 예측된 좌표 (정규화된)
    """
    model.eval()
    
    with torch.no_grad():
        # 배치 차원 추가
        x_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, 2)
        lengths = torch.tensor([seq.shape[0]], dtype=torch.long).to(DEVICE)  # (1,)
        
        # 패딩 마스크 생성 (단일 시퀀스라 패딩 없음)
        T = x_seq.size(1)
        padding_mask = torch.zeros(1, T, dtype=torch.bool, device=DEVICE)  # (1, T)
        
        # 다중 샘플링 예측
        predictions = model.sample_multiple(x_seq, lengths, num_samples, padding_mask=padding_mask)  # (1, K, 2)
        
        # 집계
        if aggregation == "mean":
            pred = predictions.mean(dim=1)  # (1, 2)
        elif aggregation == "median":
            pred = predictions.median(dim=1).values  # (1, 2)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return pred.squeeze(0).cpu().numpy()


def predict_with_samples(
    model: PassCVAE,
    seq: np.ndarray,
    num_samples: int = NUM_SAMPLES,
) -> np.ndarray:
    """
    단일 시퀀스에 대해 모든 샘플 예측 반환
    
    Returns:
        predictions: (K, 2) - K개의 다중 미래 예측
    """
    model.eval()
    
    with torch.no_grad():
        x_seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        lengths = torch.tensor([seq.shape[0]], dtype=torch.long).to(DEVICE)
        
        # 패딩 마스크 생성 (단일 시퀀스라 패딩 없음)
        T = x_seq.size(1)
        padding_mask = torch.zeros(1, T, dtype=torch.bool, device=DEVICE)
        
        predictions = model.sample_multiple(x_seq, lengths, num_samples, padding_mask=padding_mask)  # (1, K, 2)
        
        return predictions.squeeze(0).cpu().numpy()  # (K, 2)


def build_episode_path_map(meta_path: str) -> dict:
    """메타데이터에서 game_episode -> 파일경로 매핑 생성"""
    import os
    
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


def main(submit_mode: bool = False):
    """
    메인 추론 함수
    
    Args:
        submit_mode: True면 제출용(base_test.csv), False면 검증용(temporal_test.csv)
    """
    import os
    log_device()
    
    if submit_mode:
        # 제출 모드: base_test.csv + sample_submission.csv 사용
        meta_path = SUBMIT_META_PATH
        print(f"[SUBMIT MODE] Loading test metadata from {meta_path}...")
        episode_to_path, test_meta = build_episode_path_map(meta_path)
        
        # sample_submission에서 episode 목록 가져오기
        print(f"Loading sample submission from {SAMPLE_SUB_PATH}...")
        sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
        episode_list = sample_sub["game_episode"].values
        print(f"Episodes to predict: {len(episode_list)}")
        compute_distance = False  # 제출용은 GT가 없음
    else:
        # 검증 모드: temporal_test.csv 사용 (train에서 분리한 데이터)
        meta_path = TEST_META_PATH
        print(f"[VALIDATION MODE] Loading test metadata from {meta_path}...")
        episode_to_path, test_meta = build_episode_path_map(meta_path)
        
        # temporal_test의 모든 episode 사용
        episode_list = test_meta["game_episode"].values
        print(f"Episodes to predict: {len(episode_list)}")
        compute_distance = True  # 검증용은 GT 있음
    
    # 인코더 추가 설정 (Transformer용)
    encoder_kwargs = {}
    if ENCODER_TYPE == "transformer":
        encoder_kwargs = {
            "num_layers": TRANSFORMER_NUM_LAYERS,
            "num_heads": TRANSFORMER_NUM_HEADS,
            "max_seq_len": MAX_SEQ_LEN,
        }
    
    # 모델 로드
    print(f"Loading model (encoder: {ENCODER_TYPE})...")
    model = PassCVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM,
        use_learned_prior=True,
        encoder_type=ENCODER_TYPE,
        encoder_kwargs=encoder_kwargs
    ).to(DEVICE)
    model = load_model(model, MODEL_PATH)
    model.eval()
    
    # 예측
    print("\nRunning inference...")
    results = []
    distances = []  # 거리 저장용
    
    for idx in tqdm(episode_list, desc="Predicting"):
        # 해당 episode의 CSV 파일 경로 가져오기
        if idx not in episode_to_path:
            # 데이터가 없는 경우 기본값
            results.append({
                "game_episode": idx,
                "end_x": FIELD_X / 2,
                "end_y": FIELD_Y / 2
            })
            continue
        
        # 개별 episode CSV 파일 로드
        episode_path = episode_to_path[idx]
        use_df = pd.read_csv(episode_path)
        use_df = use_df.sort_values("time_seconds").reset_index(drop=True)
        
        if len(use_df) == 0:
            results.append({
                "game_episode": idx,
                "end_x": FIELD_X / 2,
                "end_y": FIELD_Y / 2
            })
            continue
        
        # 시퀀스 생성
        seq = build_inference_sequence(use_df)
        
        # 예측 (다중 샘플링 후 평균)
        pred = predict_single(model, seq, num_samples=NUM_SAMPLES, aggregation="mean")
        
        # 실제 좌표로 변환
        pred_x = float(pred[0] * FIELD_X)
        pred_y = float(pred[1] * FIELD_Y)
        
        # 좌표 범위 클리핑
        pred_x = np.clip(pred_x, 0, FIELD_X)
        pred_y = np.clip(pred_y, 0, FIELD_Y)
        
        # Ground truth와의 거리 계산 (검증 모드에서만)
        if compute_distance:
            gt_x = use_df.iloc[-1]["end_x"]
            gt_y = use_df.iloc[-1]["end_y"]
            dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
            distances.append(dist)
        
        results.append({
            "game_episode": idx,
            "end_x": pred_x,
            "end_y": pred_y
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
        print(f"\n[Distance Metrics]")
        print(f"  Mean Dist: {distances.mean():.4f} m")
        print(f"  Std Dist:  {distances.std():.4f} m")
        print(f"  Min Dist:  {distances.min():.4f} m")
        print(f"  Max Dist:  {distances.max():.4f} m")
        print(f"  Median:    {np.median(distances):.4f} m")


def visualize_samples(
    model: PassCVAE,
    episode_path: str,
    game_episode: str,
    num_samples: int = 50,
    save_path: str = None
):
    """
    다중 미래 예측 시각화
    
    Args:
        model: 학습된 모델
        episode_path: episode CSV 파일 경로
        game_episode: episode ID (제목용)
        num_samples: 샘플링 개수
        save_path: 저장 경로 (None이면 화면에 표시)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return
    
    # 시퀀스 추출
    use_df = pd.read_csv(episode_path)
    use_df = use_df.sort_values("time_seconds").reset_index(drop=True)
    
    if len(use_df) == 0:
        print(f"No data for episode {game_episode}")
        return
    
    seq = build_inference_sequence(use_df)
    
    # Ground truth (마지막 행의 end_x, end_y)
    gt_x = use_df.iloc[-1]["end_x"]
    gt_y = use_df.iloc[-1]["end_y"]
    
    # 다중 샘플링
    predictions = predict_with_samples(model, seq, num_samples)  # (K, 2)
    
    # 좌표 변환
    pred_x = predictions[:, 0] * FIELD_X
    pred_y = predictions[:, 1] * FIELD_Y
    
    seq_x = seq[:, 0] * FIELD_X
    seq_y = seq[:, 1] * FIELD_Y
    
    # 시각화
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 필드 배경 (연한 녹색)
    ax.add_patch(plt.Rectangle((0, 0), FIELD_X, FIELD_Y, 
                                facecolor='#90EE90', alpha=0.3, edgecolor='black', linewidth=2))
    
    # 필드 범위
    ax.set_xlim(-5, FIELD_X + 5)
    ax.set_ylim(-5, FIELD_Y + 5)
    ax.set_aspect('equal')
    
    # 입력 시퀀스 (경로)
    ax.plot(seq_x, seq_y, 'b-', linewidth=2, label='Input trajectory', alpha=0.7)
    ax.scatter(seq_x[:-1], seq_y[:-1], c='blue', s=30, alpha=0.5)
    ax.scatter(seq_x[-1], seq_y[-1], c='blue', s=100, marker='o', label='Current position')
    
    # 다중 예측 (투명도로 불확실성 표현)
    ax.scatter(pred_x, pred_y, c='red', alpha=0.3, s=50, label=f'Predictions (n={num_samples})')
    
    # 평균 예측
    mean_x = pred_x.mean()
    mean_y = pred_y.mean()
    ax.scatter(mean_x, mean_y, c='green', s=200, marker='*', 
               edgecolors='black', linewidths=2, label='Mean prediction', zorder=5)
    
    # Ground Truth
    ax.scatter(gt_x, gt_y, c='yellow', s=200, marker='X',
               edgecolors='black', linewidths=2, label='Ground Truth', zorder=5)
    
    # 거리 계산
    dist = np.sqrt((mean_x - gt_x) ** 2 + (mean_y - gt_y) ** 2)
    
    ax.legend(loc='upper right')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Multiple Future Predictions for {game_episode}\nMean Pred: ({mean_x:.1f}, {mean_y:.1f}) | GT: ({gt_x:.1f}, {gt_y:.1f}) | Dist: {dist:.2f}m')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_mode(game_episode: str, num_samples: int = 50, save_path: str = None):
    """
    시각화 모드 실행
    
    사용법:
        python infer.py --visualize 126292_1
        python infer.py --visualize 126292_1 --save viz.png
    """
    log_device()
    
    # 메타데이터에서 경로 찾기 (temporal_test.csv 사용)
    print(f"Loading test metadata from {TEST_META_PATH}...")
    episode_to_path, test_meta = build_episode_path_map(TEST_META_PATH)
    
    if game_episode not in episode_to_path:
        print(f"Episode {game_episode} not found in test data.")
        return
    
    path = episode_to_path[game_episode]
    
    # 인코더 추가 설정 (Transformer용)
    encoder_kwargs = {}
    if ENCODER_TYPE == "transformer":
        encoder_kwargs = {
            "num_layers": TRANSFORMER_NUM_LAYERS,
            "num_heads": TRANSFORMER_NUM_HEADS,
            "max_seq_len": MAX_SEQ_LEN,
        }
    
    # 모델 로드
    print(f"Loading model (encoder: {ENCODER_TYPE})...")
    model = PassCVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM,
        use_learned_prior=True,
        encoder_type=ENCODER_TYPE,
        encoder_kwargs=encoder_kwargs
    ).to(DEVICE)
    model = load_model(model, MODEL_PATH)
    model.eval()
    
    # 시각화
    print(f"Visualizing episode {game_episode}...")
    visualize_samples(model, path, game_episode, num_samples, save_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CVAE Inference")
    parser.add_argument("--submit", action="store_true",
                        help="제출 모드 (base_test.csv + sample_submission.csv 사용)")
    parser.add_argument("--visualize", type=str, default=None,
                        help="Visualize a specific game_episode (e.g., 126292_1)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples for visualization")
    parser.add_argument("--save", type=str, default=None,
                        help="Save visualization to file path")
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_mode(args.visualize, args.num_samples, args.save)
    else:
        main(submit_mode=args.submit)
