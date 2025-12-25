"""
CVAE 모델 학습 메인 스크립트
"""
import pandas as pd
import torch

from config import (
    TRAIN_PATH,
    MODEL_PATH,
    DEVICE,
    EPOCHS,
    LR,
    BATCH_SIZE,
    INPUT_DIM,
    HIDDEN_DIM,
    Z_DIM,
    log_device,
)
from dataset import build_train_sequences, build_dataloaders
from model import PassCVAE
from trainer import train_loop, save_model


def main():
    # Device 확인
    log_device()
    
    # 데이터 로드
    print(f"Loading data from {TRAIN_PATH}...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"Data shape: {df.shape}")
    
    # 시퀀스 생성
    print("Building sequences...")
    episodes, targets = build_train_sequences(df)
    
    # DataLoader 생성
    print("Building dataloaders...")
    train_loader, valid_loader = build_dataloaders(
        episodes, targets,
        batch_size=BATCH_SIZE,
        test_size=0.2
    )
    
    # 모델 생성
    print("Creating CVAE model...")
    model = PassCVAE(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM,
        use_learned_prior=True  # 학습된 prior p(z|X) 사용
    ).to(DEVICE)
    
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR,
        weight_decay=1e-4
    )
    
    # 학습
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    model, best_state, best_dist = train_loop(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        epochs=EPOCHS,
        use_scheduler=True,
        early_stop_patience=15,
    )
    
    # 모델 저장
    save_model(best_state, MODEL_PATH)
    print(f"\nBest validation distance: {best_dist:.4f}")
    print("Training completed!")


if __name__ == "__main__":
    main()
