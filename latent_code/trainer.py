"""
CVAE 학습 루프 및 손실 함수
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
    get_beta,
    NUM_SAMPLES,
)


def kl_divergence(
    mu: torch.Tensor, 
    logvar: torch.Tensor,
    prior_mu: torch.Tensor = None,
    prior_logvar: torch.Tensor = None
) -> torch.Tensor:
    """
    KL divergence: KL(q(z|X,Y) || p(z|X))
    
    If prior_mu and prior_logvar are None, assumes standard normal prior N(0, I)
    """
    if prior_mu is None or prior_logvar is None:
        # KL(N(mu, var) || N(0, I))
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    else:
        # KL(N(mu, var) || N(prior_mu, prior_var))
        # = 0.5 * sum(log(prior_var/var) + (var + (mu - prior_mu)^2)/prior_var - 1)
        prior_var = prior_logvar.exp()
        var = logvar.exp()
        kl = 0.5 * torch.sum(
            prior_logvar - logvar + 
            (var + (mu - prior_mu).pow(2)) / prior_var - 1,
            dim=-1
        )
    
    return kl.mean()


def cvae_loss(
    y_pred: torch.Tensor,
    y_gt: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    prior_mu: torch.Tensor = None,
    prior_logvar: torch.Tensor = None,
    beta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CVAE ELBO Loss = Reconstruction Loss + beta * KL Divergence
    
    Args:
        y_pred: (B, 2) - 예측 좌표
        y_gt: (B, 2) - 타깃 좌표
        mu, logvar: posterior q(z|X,Y)의 파라미터
        prior_mu, prior_logvar: prior p(z|X)의 파라미터 (옵션)
        beta: KL 가중치 (beta-VAE)
    
    Returns:
        total_loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(y_pred, y_gt)
    
    # KL divergence
    kl_loss = kl_divergence(mu, logvar, prior_mu, prior_logvar)
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    beta: float = 1.0,
    desc: str = "Train",
) -> Tuple[float, float, float]:
    """
    한 에폭 학습
    
    Returns:
        avg_loss, avg_recon, avg_kl
    """
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    n_samples = 0
    
    for X, lengths, y, padding_mask in tqdm(train_loader, desc=desc):
        X = X.to(DEVICE)
        lengths = lengths.to(DEVICE)
        y = y.to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass (학습 모드)
        y_pred, mu, logvar, prior_mu, prior_logvar = model(X, lengths, y_gt=y)
        
        # Compute loss
        loss, recon, kl = cvae_loss(
            y_pred, y, mu, logvar, 
            prior_mu, prior_logvar, 
            beta=beta
        )
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_size = X.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon.item() * batch_size
        total_kl += kl.item() * batch_size
        n_samples += batch_size
    
    return total_loss / n_samples, total_recon / n_samples, total_kl / n_samples


def validate(
    model: nn.Module,
    valid_loader: DataLoader,
    num_samples: int = NUM_SAMPLES,
    desc: str = "Valid",
) -> Tuple[float, float, float, float]:
    """
    검증 수행 (다중 샘플링으로 평균 예측)
    
    Returns:
        mean_dist, std_dist, min_ade, fde
    """
    model.eval()
    
    # 단일 예측 (mean aggregation) 기준 거리
    single_dists = []
    # 다중 샘플 중 최소 거리 (minADE)
    min_dists = []
    
    with torch.no_grad():
        for X, lengths, y, padding_mask in tqdm(valid_loader, desc=desc):
            X = X.to(DEVICE)
            lengths = lengths.to(DEVICE)
            y = y.to(DEVICE)
            
            # 다중 샘플링 예측
            predictions = model.sample_multiple(X, lengths, num_samples)  # (B, K, 2)
            
            # 평균 예측
            mean_pred = predictions.mean(dim=1)  # (B, 2)
            
            # 실제 좌표로 변환
            mean_pred_np = mean_pred.cpu().numpy()
            true_np = y.cpu().numpy()
            
            pred_x = mean_pred_np[:, 0] * FIELD_X
            pred_y = mean_pred_np[:, 1] * FIELD_Y
            true_x = true_np[:, 0] * FIELD_X
            true_y = true_np[:, 1] * FIELD_Y
            
            dist = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            single_dists.append(dist)
            
            # minADE: 각 샘플 중 가장 가까운 예측의 거리
            predictions_np = predictions.cpu().numpy()  # (B, K, 2)
            for i in range(predictions_np.shape[0]):
                sample_preds = predictions_np[i]  # (K, 2)
                sample_x = sample_preds[:, 0] * FIELD_X
                sample_y = sample_preds[:, 1] * FIELD_Y
                
                sample_dists = np.sqrt(
                    (sample_x - true_x[i]) ** 2 + 
                    (sample_y - true_y[i]) ** 2
                )
                min_dists.append(sample_dists.min())
    
    all_single_dists = np.concatenate(single_dists)
    all_min_dists = np.array(min_dists)
    
    mean_dist = all_single_dists.mean()
    std_dist = all_single_dists.std()
    min_ade = all_min_dists.mean()
    
    return mean_dist, std_dist, min_ade, 0.0


def train_loop(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = EPOCHS,
    use_scheduler: bool = True,
    early_stop_patience: int = 15,
    desc_prefix: str = "",
) -> Tuple[nn.Module, Dict, float]:
    """
    전체 학습 루프
    
    Returns:
        model, best_model_state, best_dist
    """
    best_dist = float("inf")
    best_min_ade = float("inf")
    best_model_state: Optional[Dict] = None
    patience_counter = 0
    
    # Learning rate scheduler
    scheduler = None
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    for epoch in range(1, epochs + 1):
        # KL annealing
        beta = get_beta(epoch)
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, optimizer, train_loader,
            beta=beta,
            desc=f"{desc_prefix}Train {epoch}/{epochs}"
        )
        
        # Validate
        mean_dist, std_dist, min_ade, _ = validate(
            model, valid_loader,
            desc=f"{desc_prefix}Valid {epoch}/{epochs}"
        )
        
        # Learning rate 업데이트
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        print(
            f"{desc_prefix}[Epoch {epoch:3d}/{epochs}] "
            f"loss={train_loss:.6f} (recon={train_recon:.6f}, kl={train_kl:.6f}) | "
            f"dist={mean_dist:.4f}±{std_dist:.4f} | "
            f"minADE={min_ade:.4f} | "
            f"beta={beta:.3f} | lr={current_lr:.2e}"
        )
        
        # Best model 업데이트 (minADE 기준)
        if min_ade < best_min_ade:
            best_min_ade = min_ade
            best_dist = mean_dist
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f" --> Best model updated! (minADE={best_min_ade:.4f}, dist={best_dist:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f" --> Early stopping at epoch {epoch}")
                break
    
    if best_model_state is None:
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    return model, best_model_state, best_dist


def save_model(state_dict: Dict, path: str = MODEL_PATH) -> None:
    """모델 저장"""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)
    print(f"Model saved to: {path}")


def load_model(model: nn.Module, path: str = MODEL_PATH) -> nn.Module:
    """모델 로드"""
    state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model loaded from: {path}")
    return model

