"""
XGBoost 모델 하이퍼파라미터 및 경로 설정
"""
import os

# =========================
# 경로 설정
# =========================
TRAIN_PATH = "../data/train.csv"
TEST_META_PATH = "../data/temporal_test.csv"          # 검증용 (train에서 분리)
SUBMIT_META_PATH = "../data/basic/base_test.csv"      # 제출용
SAMPLE_SUB_PATH = "../data/sample_submission.csv"
MODEL_DIR = "../data/xgboost_models"
MODEL_PATH_X = "../data/xgboost_models/xgb_x.json"    # X 좌표 모델 (separate)
MODEL_PATH_Y = "../data/xgboost_models/xgb_y.json"    # Y 좌표 모델 (separate)
MODEL_PATH_MULTI = "../data/xgboost_models/xgb_multi.json"  # Multi-output 모델
SUBMIT_PATH = "../data/xgboost_submit.csv"

# =========================
# Multi-Output 설정
# =========================
# True: 하나의 트리에서 X, Y 동시 예측 (권장)
# False: X, Y 별도 모델로 예측 (기존 방식)
USE_MULTI_OUTPUT = False

# =========================
# XGBoost 하이퍼파라미터
# =========================
XGB_PARAMS = {
    "n_estimators": 500,
    # max depth 시간에 따라 활용한 값 이력 : 8 -> 3
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    # min_child_weight 이력 : 3 -> 10
    "min_child_weight": 10,
    "gamma": 0,              # 리프 노드 분할에 필요한 최소 loss 감소량 (과적합 방지)
    # alpha & lambda 이력 : 0.1 / 1.0 -> 2 / 6
    "reg_alpha": 2,          # L1 regularization
    "reg_lambda": 6,         # L2 regularization
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 1,
}

# Early stopping 설정
EARLY_STOPPING_ROUNDS = 50

# =========================
# 피처 엔지니어링 설정
# =========================
# 마지막 N개 좌표를 피처로 사용
LAST_N_COORDS = 10

# 제외할 피처 리스트 (ablation study용)
# 사용 가능한 피처:
#   - 기본 통계: n_actions, start_x_mean, start_y_mean, start_x_std, start_y_std,
#                end_x_mean, end_y_mean, end_x_std, end_y_std
#   - 마지막 위치: last_start_x, last_start_y
#   - 마지막 N개 좌표: coord_0_x, coord_0_y, ..., coord_9_x, coord_9_y
#   - 이동 거리: total_distance, mean_distance, max_distance, last_move_angle, last_move_dist
#   - 진행 방향: overall_direction_x, overall_direction_y, overall_angle, overall_distance
#   - 위치 범위: x_range, y_range, x_min, x_max, y_min, y_max
#   - 시간: duration, time_last, avg_speed
#   - 컨텍스트: is_home, team_id, last_action_is_pass, last_action_is_carry,
#               last_action_is_duel, pass_ratio, carry_ratio, last_result_success
#
# 예시: EXCLUDE_FEATURES = ["n_actions", "time_last", "team_id"]
# EXCLUDE_FEATURES = ["last_start_x", "last_start_y", "coord_9_x", "coord_9_y"]
EXCLUDE_FEATURES = []

# 피처 그룹 단위 제외 (개별 피처보다 우선 적용)
# 사용 가능한 그룹: "stats", "last_coords", "movement", "direction", "range", "time", "context"
# 예시: EXCLUDE_FEATURE_GROUPS = ["time", "context"]
EXCLUDE_FEATURE_GROUPS = []

# 시퀀스 모드 설정
# 지원 옵션: "episode", "phase", "team"
# - episode: game_episode 단위로 시퀀스 생성 (기본값)
# - phase: phase 단위로 시퀀스 생성
# - team: 예측할 좌표와 동일한 team_id의 데이터만 사용
# SEQUENCE_MODE = "team"
SEQUENCE_MODE = "episode"

# =========================
# 좌표 정규화 상수
# =========================
FIELD_X = 105.0
FIELD_Y = 68.0


def ensure_model_dir():
    """모델 저장 디렉토리 생성"""
    os.makedirs(MODEL_DIR, exist_ok=True)


def log_config():
    """설정 출력"""
    print("=" * 50)
    print("XGBoost Configuration")
    print("=" * 50)
    print(f"Multi-Output Mode: {USE_MULTI_OUTPUT}")
    print(f"Sequence Mode: {SEQUENCE_MODE}")
    print(f"Last N Coords: {LAST_N_COORDS}")
    print(f"XGB Params:")
    for k, v in XGB_PARAMS.items():
        print(f"  {k}: {v}")
    if EXCLUDE_FEATURES:
        print(f"Excluded Features: {EXCLUDE_FEATURES}")
    if EXCLUDE_FEATURE_GROUPS:
        print(f"Excluded Feature Groups: {EXCLUDE_FEATURE_GROUPS}")
    print("=" * 50)

