"""
하이퍼파라미터 및 경로 설정
"""
import torch

# =========================
# 경로 설정
# =========================
TRAIN_PATH = "../../data/phase_train.csv"
TEST_META_PATH = "../../data/test.csv"
SAMPLE_SUB_PATH = "../../data/sample_submission.csv"
TEAM_MODEL_DIR = "../../data/team_models"
PRETRAIN_MODEL_PATH = "../../data/team_models/pretrain.pth"
SUBMIT_PATH = "../../data/submit_new.csv"

# =========================
# 하이퍼파라미터
# =========================
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 30
LR = 1e-3
HIDDEN_DIM = 64

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

