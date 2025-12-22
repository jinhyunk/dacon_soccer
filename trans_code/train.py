"""
Transformer 좌표 예측 모델 학습 스크립트

Usage:
    # 기본 실행
    python train.py
    
    # 에폭 수 변경
    python train.py --epochs=50
    
    # 모델 타입 변경 (encoder 또는 pooling)
    python train.py --model_type=pooling
    
    # 조기 종료 patience 설정
    python train.py --patience=15
"""
import os
import argparse

import pandas as pd
import torch

from config import (
    TRAIN_PATH,
    MODEL_DIR,
    MODEL_PATH,
    EPOCHS,
    BATCH_SIZE,
    D_MODEL,
    NHEAD,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
    LR,
    log_device,
)
from dataset import build_train_sequences, build_dataloaders
from models import create_model
from trainer import train_loop, save_model


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="Transformer 좌표 예측 모델 학습")
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"학습 에폭 수 (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help=f"배치 크기 (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="encoder",
        choices=["encoder", "pooling"],
        help="모델 타입: 'encoder' (마지막 hidden) 또는 'pooling' (평균 풀링)"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=D_MODEL,
        help=f"모델 차원 (default: {D_MODEL})"
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=NHEAD,
        help=f"어텐션 헤드 수 (default: {NHEAD})"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=NUM_LAYERS,
        help=f"Transformer 레이어 수 (default: {NUM_LAYERS})"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"학습률 (default: {LR})"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="조기 종료 patience (default: 10)"
    )
    parser.add_argument(
        "--no_scheduler",
        action="store_true",
        help="학습률 스케줄러 비활성화"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Device 확인
    log_device()
    
    # 모델 저장 디렉토리 생성
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # =========================
    # 데이터 로드
    # =========================
    print(f"\n========== Loading Data ==========")
    print(f"Train path: {TRAIN_PATH}")
    
    df = pd.read_csv(TRAIN_PATH)
    print(f"Total rows: {len(df):,}")
    print(f"Unique game_episodes: {df['game_episode'].nunique():,}")
    
    # =========================
    # 시퀀스 생성
    # =========================
    print(f"\n========== Building Sequences ==========")
    episodes, targets = build_train_sequences(df)
    
    # =========================
    # DataLoader 생성
    # =========================
    print(f"\n========== Creating DataLoaders ==========")
    train_loader, valid_loader = build_dataloaders(
        episodes, targets, batch_size=args.batch_size
    )
    
    # =========================
    # 모델 생성
    # =========================
    print(f"\n========== Creating Model ==========")
    model, criterion, optimizer = create_model(
        model_type=args.model_type,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        lr=args.lr,
    )
    
    # =========================
    # 학습
    # =========================
    print(f"\n========== Training ==========")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Use scheduler: {not args.no_scheduler}")
    
    model, best_state, best_dist = train_loop(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=args.epochs,
        use_scheduler=not args.no_scheduler,
        early_stop_patience=args.patience,
        desc_prefix="[Transformer]",
    )
    
    # =========================
    # 모델 저장
    # =========================
    print(f"\n========== Saving Model ==========")
    save_model(best_state, MODEL_PATH)
    
    print(f"\n========== Training Complete ==========")
    print(f"Best validation distance: {best_dist:.4f}")


if __name__ == "__main__":
    main()

