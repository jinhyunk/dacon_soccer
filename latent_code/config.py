"""
CVAE 모델 하이퍼파라미터 및 경로 설정
"""
import torch

# =========================
# 경로 설정
# =========================
TRAIN_PATH = "../data/train.csv"
TEST_META_PATH = "../data/temporal_test.csv"          # 검증용 (train에서 분리)
SUBMIT_META_PATH = "../data/basic/base_test.csv"      # 제출용
SAMPLE_SUB_PATH = "../data/sample_submission.csv"
MODEL_DIR = "../data/cvae_models"
MODEL_PATH = "../data/cvae_models/cvae.pth"
SUBMIT_PATH = "../data/cvae_submit.csv"

# =========================
# 하이퍼파라미터
# =========================
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3

# CVAE 파라미터
INPUT_DIM = 2           # 입력 차원 (x, y 좌표)
HIDDEN_DIM = 128        # 인코더 hidden 차원
Z_DIM = 16              # 잠재 변수 차원
NUM_SAMPLES = 20        # 추론 시 샘플링 개수

# 인코더 설정
# 지원 옵션: "lstm", "gru", "transformer"
# - lstm: LSTM 기반 (기본값, 순차적 의존성 학습에 적합)
# - gru: GRU 기반 (LSTM보다 가볍고 빠름)
# - transformer: Transformer 기반 (전역 의존성 학습, 긴 시퀀스에 효과적)
ENCODER_TYPE = "gru"
# ENCODER_TYPE = "transformer"
# ENCODER_TYPE = "lstm"

# Transformer 인코더 전용 파라미터
TRANSFORMER_NUM_LAYERS = 4    # Transformer encoder layer 개수
TRANSFORMER_NUM_HEADS = 4     # Multi-head attention의 head 개수

# KL divergence 가중치 (beta-VAE)
BETA_START = 0.0        # 초기 beta 값
BETA_END = 1.0          # 최종 beta 값
BETA_WARMUP_EPOCHS = 10 # beta warmup 에폭 수

# 시퀀스 설정
MAX_SEQ_LEN = 512       # 최대 시퀀스 길이

# 시퀀스 모드 설정
# 지원 옵션: "episode", "phase"
# - episode: game_episode 단위로 시퀀스 생성 (기본값)
#            한 에피소드 전체를 하나의 시퀀스로 사용, 마지막 좌표 예측
# - phase: game_episode 내에서 phase 단위로 시퀀스 생성
#          한 에피소드 내 각 phase를 별도 시퀀스로 사용, 각 phase의 마지막 좌표 예측
# SEQUENCE_MODE = "episode"
SEQUENCE_MODE = "phase"

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

