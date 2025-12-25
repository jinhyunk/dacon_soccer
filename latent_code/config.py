"""
CVAE 모델 하이퍼파라미터 및 경로 설정
"""
import torch

# =========================
# 경로 설정
# =========================
TRAIN_PATH = "../data/train.csv"
TEST_META_PATH = "../data/temporal_test.csv"
SAMPLE_SUB_PATH = "../data/sample_submission.csv"
MODEL_DIR = "../data/cvae_models"
MODEL_PATH = "../data/cvae_models/cvae.pth"
SUBMIT_PATH = "../data/cvae_submit.csv"

# =========================
# 하이퍼파라미터
# =========================
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

# CVAE 파라미터
INPUT_DIM = 2           # 입력 차원 (x, y 좌표)
HIDDEN_DIM = 128        # LSTM hidden 차원
Z_DIM = 16              # 잠재 변수 차원
NUM_SAMPLES = 20        # 추론 시 샘플링 개수

# KL divergence 가중치 (beta-VAE)
BETA_START = 0.0        # 초기 beta 값
BETA_END = 1.0          # 최종 beta 값
BETA_WARMUP_EPOCHS = 10 # beta warmup 에폭 수

# 시퀀스 설정
MAX_SEQ_LEN = 512       # 최대 시퀀스 길이

# =========================
# Device 설정
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 좌표 정규화 상수
# =========================
FIELD_X = 105.0
FIELD_Y = 68.0


def log_device() -> None:
    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")


def get_beta(epoch: int) -> float:
    """KL annealing: beta 값을 에폭에 따라 선형 증가"""
    if epoch >= BETA_WARMUP_EPOCHS:
        return BETA_END
    return BETA_START + (BETA_END - BETA_START) * (epoch / BETA_WARMUP_EPOCHS)

