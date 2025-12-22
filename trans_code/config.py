"""
Transformer 모델 하이퍼파라미터 및 경로 설정
"""
import torch

# =========================
# 경로 설정
# =========================
TRAIN_PATH = "../data/train.csv"
TEST_META_PATH = "../data/temporal_test.csv"
SAMPLE_SUB_PATH = "../data/sample_submission.csv"
MODEL_DIR = "../data/trans_models"
MODEL_PATH = "../data/trans_models/transformer.pth"
SUBMIT_PATH = "../data/trans_submit.csv"

# =========================
# 하이퍼파라미터
# =========================
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4

# Transformer 파라미터
D_MODEL = 64          # 임베딩 차원
NHEAD = 4             # 멀티헤드 어텐션 헤드 수
NUM_LAYERS = 3        # Transformer 인코더 레이어 수
DIM_FEEDFORWARD = 256 # 피드포워드 차원
DROPOUT = 0.1         # 드롭아웃 비율
MAX_SEQ_LEN = 512     # 최대 시퀀스 길이

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

