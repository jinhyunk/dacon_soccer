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
        
        # 다중 샘플링 예측
        predictions = model.sample_multiple(x_seq, lengths, num_samples)  # (1, K, 2)
        
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
        
        predictions = model.sample_multiple(x_seq, lengths, num_samples)  # (1, K, 2)
        
        return predictions.squeeze(0).cpu().numpy()  # (K, 2)


def main():
    """메인 추론 함수"""
    log_device()
    
    # 테스트 데이터 로드
    print(f"Loading test data from {TEST_META_PATH}...")
    test_df = pd.read_csv(TEST_META_PATH)
    print(f"Test data shape: {test_df.shape}")
    
    # 샘플 제출 파일 로드
    print(f"Loading sample submission from {SAMPLE_SUB_PATH}...")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    print(f"Sample submission shape: {sample_sub.shape}")
    
    # 모델 로드
    print("Loading model...")
    model = PassCVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM,
        use_learned_prior=True
    ).to(DEVICE)
    model = load_model(model, MODEL_PATH)
    model.eval()
    
    # 예측
    print("\nRunning inference...")
    results = []
    
    for idx in tqdm(sample_sub["id"].values, desc="Predicting"):
        # 해당 ID의 시퀀스 추출
        use_df = test_df[test_df["id"] == idx].copy()
        use_df = use_df.sort_values("time_seconds").reset_index(drop=True)
        
        if len(use_df) == 0:
            # 데이터가 없는 경우 기본값
            results.append({
                "id": idx,
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
        
        results.append({
            "id": idx,
            "end_x": pred_x,
            "end_y": pred_y
        })
    
    # 결과 저장
    result_df = pd.DataFrame(results)
    result_df.to_csv(SUBMIT_PATH, index=False)
    print(f"\nSubmission saved to {SUBMIT_PATH}")
    print(f"Total predictions: {len(result_df)}")
    
    # 통계 출력
    print("\nPrediction statistics:")
    print(f"  end_x: mean={result_df['end_x'].mean():.2f}, std={result_df['end_x'].std():.2f}")
    print(f"  end_y: mean={result_df['end_y'].mean():.2f}, std={result_df['end_y'].std():.2f}")


def visualize_samples(
    model: PassCVAE,
    test_df: pd.DataFrame,
    sample_id: int,
    num_samples: int = 50,
    save_path: str = None
):
    """
    다중 미래 예측 시각화 (선택적)
    
    Args:
        model: 학습된 모델
        test_df: 테스트 데이터프레임
        sample_id: 시각화할 샘플 ID
        num_samples: 샘플링 개수
        save_path: 저장 경로 (None이면 화면에 표시)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping visualization.")
        return
    
    # 시퀀스 추출
    use_df = test_df[test_df["id"] == sample_id].copy()
    use_df = use_df.sort_values("time_seconds").reset_index(drop=True)
    
    if len(use_df) == 0:
        print(f"No data for ID {sample_id}")
        return
    
    seq = build_inference_sequence(use_df)
    
    # 다중 샘플링
    predictions = predict_with_samples(model, seq, num_samples)  # (K, 2)
    
    # 좌표 변환
    pred_x = predictions[:, 0] * FIELD_X
    pred_y = predictions[:, 1] * FIELD_Y
    
    seq_x = seq[:, 0] * FIELD_X
    seq_y = seq[:, 1] * FIELD_Y
    
    # 시각화
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 필드 그리기
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
    
    ax.legend()
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(f'Multiple Future Predictions for ID {sample_id}')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == "__main__":
    main()
