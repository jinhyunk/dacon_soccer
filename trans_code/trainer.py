"""
학습 루프 함수 (wandb 연동 지원)
"""
from typing import Dict, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    DEVICE,
    EPOCHS,
    FIELD_X,
    FIELD_Y,
    MODEL_PATH,
)


def train_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    desc: str = "Train",
) -> float:
    """
    한 에폭 학습
    
    Returns:
        train_loss: 평균 학습 손실
    """
    model.train()
    total_loss = 0.0
    
    for X, lengths, y, padding_mask in tqdm(train_loader, desc=desc):
        X = X.to(DEVICE)
        lengths = lengths.to(DEVICE)
        y = y.to(DEVICE)
        padding_mask = padding_mask.to(DEVICE)
        
        optimizer.zero_grad()
        pred = model(X, lengths, src_key_padding_mask=padding_mask)
        loss = criterion(pred, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    train_loss = total_loss / len(train_loader.dataset)
    return train_loss


def validate(
    model: nn.Module,
    valid_loader: DataLoader,
    desc: str = "Valid",
) -> Tuple[float, float]:
    """
    검증 수행
    
    Returns:
        mean_dist: 평균 유클리드 거리
        std_dist: 표준편차
    """
    model.eval()
    dists = []
    
    with torch.no_grad():
        for X, lengths, y, padding_mask in tqdm(valid_loader, desc=desc):
            X = X.to(DEVICE)
            lengths = lengths.to(DEVICE)
            y = y.to(DEVICE)
            padding_mask = padding_mask.to(DEVICE)
            
            pred = model(X, lengths, src_key_padding_mask=padding_mask)
            
            pred_np = pred.cpu().numpy()
            true_np = y.cpu().numpy()
            
            # 정규화된 좌표를 실제 좌표로 변환
            pred_x = pred_np[:, 0] * FIELD_X
            pred_y = pred_np[:, 1] * FIELD_Y
            true_x = true_np[:, 0] * FIELD_X
            true_y = true_np[:, 1] * FIELD_Y
            
            dist = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            dists.append(dist)
    
    all_dists = np.concatenate(dists)
    mean_dist = all_dists.mean()
    std_dist = all_dists.std()
    
    return mean_dist, std_dist


def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = EPOCHS,
    use_scheduler: bool = True,
    early_stop_patience: int = 10,
    desc_prefix: str = "",
    use_wandb: bool = False,
) -> Tuple[nn.Module, Dict, float]:
    """
    전체 학습 루프
    
    Args:
        model: 학습할 모델
        criterion: 손실 함수
        optimizer: 옵티마이저
        train_loader: 학습 데이터 로더
        valid_loader: 검증 데이터 로더
        epochs: 학습 에폭 수
        use_scheduler: 학습률 스케줄러 사용 여부
        early_stop_patience: 조기 종료 patience
        desc_prefix: 로그 출력 접두어
        use_wandb: wandb 로깅 사용 여부
        
    Returns:
        model, best_model_state, best_dist
    """
    # wandb import (optional)
    wandb = None
    if use_wandb:
        try:
            import wandb as wb
            wandb = wb
        except ImportError:
            print("[WARN] wandb not installed. Skipping wandb logging.")
            use_wandb = False
    
    best_dist = float("inf")
    best_model_state: Optional[Dict] = None
    patience_counter = 0
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_epoch(
            model, criterion, optimizer, train_loader,
            desc=f"{desc_prefix} train epoch {epoch}"
        )
        
        # Validate
        mean_dist, std_dist = validate(
            model, valid_loader,
            desc=f"{desc_prefix} valid epoch {epoch}"
        )
        
        # Learning rate 업데이트
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"{desc_prefix}[Epoch {epoch:3d}/{epochs}] "
            f"train_loss={train_loss:.6f} | "
            f"valid_dist={mean_dist:.4f} ± {std_dist:.4f} | "
            f"lr={current_lr:.2e}"
        )
        
        # wandb 로깅
        if use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "valid/mean_dist": mean_dist,
                "valid/std_dist": std_dist,
                "train/learning_rate": current_lr,
                "valid/best_dist": best_dist,
            })
        
        # Best model 업데이트
        if mean_dist < best_dist:
            best_dist = mean_dist
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f" --> {desc_prefix} Best model updated! (dist={best_dist:.4f})")
            
            # wandb에 best metric 기록
            if use_wandb and wandb is not None:
                wandb.run.summary["best_valid_dist"] = best_dist
                wandb.run.summary["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f" --> Early stopping at epoch {epoch} (patience={early_stop_patience})")
                break
    
    if best_model_state is None:
        best_model_state = model.state_dict()
    
    return model, best_model_state, best_dist


def save_model(state_dict: Dict, path: str = MODEL_PATH) -> None:
    """모델 저장"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    print(f"Model saved to: {path}")


def load_model(model: nn.Module, path: str = MODEL_PATH) -> nn.Module:
    """모델 로드"""
    state_dict = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"Model loaded from: {path}")
    return model
