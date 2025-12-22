"""
Team-wise LSTM 학습 메인 스크립트

Usage:
    # 기본 실행 (pretrain_mode=phase)
    python team_train.py

    # pretrain 기준을 team_id로 변경
    python team_train.py --pretrain_mode=team_id

    # epoch 수 조정
    python team_train.py --pretrain_epochs=5 --finetune_epochs=10

    # pretrain 건너뛰기 (기존 pretrain.pth 사용)
    python team_train.py --skip_pretrain
"""
import os
import argparse

import pandas as pd
import torch

from config import (
    TRAIN_PATH,
    TEAM_MODEL_DIR,
    PRETRAIN_MODEL_PATH,
    PRETRAIN_EPOCHS,
    FINETUNE_EPOCHS,
    DEVICE,
    log_device,
)
from trainer import pretrain, finetune_team


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="Team-wise LSTM Training with Pretrain")
    parser.add_argument(
        "--pretrain_mode",
        type=str,
        default="phase",
        choices=["phase", "team_id"],
        help="Pretrain 기준: 'phase' 또는 'team_id' (default: phase)"
    )
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=PRETRAIN_EPOCHS,
        help=f"Pretrain epoch 수 (default: {PRETRAIN_EPOCHS})"
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=FINETUNE_EPOCHS,
        help=f"Fine-tune epoch 수 (default: {FINETUNE_EPOCHS})"
    )
    parser.add_argument(
        "--skip_pretrain",
        action="store_true",
        help="Pretrain을 건너뛰고 기존 pretrain.pth를 로드 (파일이 있어야 함)"
    )
    return parser.parse_args()


def load_pretrain_state(args, df: pd.DataFrame) -> dict:
    """Pretrain 가중치 로드 또는 학습"""
    if args.skip_pretrain:
        if not os.path.exists(PRETRAIN_MODEL_PATH):
            raise FileNotFoundError(
                f"skip_pretrain 옵션이 활성화되었지만 {PRETRAIN_MODEL_PATH} 파일이 없습니다."
            )
        print(f"\n[skip_pretrain] Loading pretrain weights from: {PRETRAIN_MODEL_PATH}")
        return torch.load(PRETRAIN_MODEL_PATH, map_location=DEVICE)

    return pretrain(
        df,
        pretrain_mode=args.pretrain_mode,
        epochs=args.pretrain_epochs,
    )


def train_all_teams(df: pd.DataFrame, pretrain_state: dict, finetune_epochs: int):
    """모든 팀에 대해 fine-tuning 수행"""
    team_ids = sorted(df["team_id"].unique())
    print("\nFound team_ids:", team_ids)

    for team_id in team_ids:
        try:
            model, best_state, best_dist = finetune_team(
                team_id=int(team_id),
                df=df,
                pretrain_state=pretrain_state,
                epochs=finetune_epochs,
            )
        except ValueError as e:
            print(f"[team {team_id}] skip: {e}")
            continue

        print(
            f"[team {team_id}] Fine-tuning finished. "
            f"Best valid mean dist: {best_dist:.4f}"
        )

        # 팀별 best 모델 가중치 저장
        model_path = os.path.join(TEAM_MODEL_DIR, f"team_{int(team_id)}.pth")
        torch.save(best_state, model_path)
        print(f"[team {team_id}] Best model saved to: {model_path}")


def main():
    args = parse_args()

    log_device()
    os.makedirs(TEAM_MODEL_DIR, exist_ok=True)

    # 학습 데이터 로드
    print(f"\nLoading data from: {TRAIN_PATH}")
    df = pd.read_csv(TRAIN_PATH)

    # Pretrain
    pretrain_state = load_pretrain_state(args, df)

    # Fine-tune per team
    train_all_teams(df, pretrain_state, args.finetune_epochs)

    print("\n========== All training completed! ==========")


if __name__ == "__main__":
    main()
