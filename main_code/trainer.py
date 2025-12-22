"""
학습 루프 및 pretrain/finetune 함수
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from config import (
    DEVICE,
    BATCH_SIZE,
    PRETRAIN_EPOCHS,
    FINETUNE_EPOCHS,
    PRETRAIN_MODEL_PATH,
    FIELD_X,
    FIELD_Y,
)
from models import create_model
from dataset import (
    build_dataloaders,
    build_pretrain_sequences_by_phase,
    build_pretrain_sequences_by_team,
    build_episode_team_sequences,
)


# ===========================
# 공통 학습 루프
# ===========================

def train_loop(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int,
    desc_prefix: str = "",
) -> Tuple[nn.Module, Dict, float]:
    """
    공통 학습 루프. pretrain 및 finetune 모두 사용.
    """
    best_dist = float("inf")
    best_model_state: Dict | None = None

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        total_loss = 0.0

        for X, lengths, y in tqdm(train_loader, desc=f"{desc_prefix} train epoch {epoch}"):
            X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X, lengths)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # --- Valid: 평균 유클리드 거리 ---
        model.eval()
        dists = []

        with torch.no_grad():
            for X, lengths, y in tqdm(valid_loader, desc=f"{desc_prefix} valid epoch {epoch}"):
                X, lengths, y = X.to(DEVICE), lengths.to(DEVICE), y.to(DEVICE)
                pred = model(X, lengths)

                pred_np = pred.cpu().numpy()
                true_np = y.cpu().numpy()

                pred_x = pred_np[:, 0] * FIELD_X
                pred_y = pred_np[:, 1] * FIELD_Y
                true_x = true_np[:, 0] * FIELD_X
                true_y = true_np[:, 1] * FIELD_Y

                dist = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
                dists.append(dist)

        mean_dist = np.concatenate(dists).mean()

        print(
            f"{desc_prefix}[Epoch {epoch}] "
            f"train_loss={train_loss:.4f} | "
            f"valid_mean_dist={mean_dist:.4f}"
        )

        if mean_dist < best_dist:
            best_dist = mean_dist
            best_model_state = model.state_dict().copy()
            print(f" --> {desc_prefix} Best model updated! (dist={best_dist:.4f})")

    if best_model_state is None:
        best_model_state = model.state_dict()

    return model, best_model_state, best_dist


# ===========================
# Pretrain 함수
# ===========================

def pretrain(
    df: pd.DataFrame,
    pretrain_mode: str = "phase",
    epochs: int = PRETRAIN_EPOCHS,
) -> Dict:
    """
    phase 또는 team_id 기준으로 pretrain 수행.

    Args:
        df: 학습 데이터
        pretrain_mode: "phase" 또는 "team_id"
        epochs: pretrain epoch 수

    Returns:
        best_state: 최적의 모델 가중치
    """
    print(f"\n========== Pretrain (mode={pretrain_mode}) ==========")

    if pretrain_mode == "phase":
        episodes, targets = build_pretrain_sequences_by_phase(df)
    elif pretrain_mode == "team_id":
        episodes, targets = build_pretrain_sequences_by_team(df)
    else:
        raise ValueError(f"Unknown pretrain_mode: {pretrain_mode}. Use 'phase' or 'team_id'.")

    train_loader, valid_loader = build_dataloaders(episodes, targets, batch_size=BATCH_SIZE)

    model, criterion, optimizer = create_model(pretrain_state=None)

    _, best_state, best_dist = train_loop(
        model, criterion, optimizer,
        train_loader, valid_loader,
        epochs=epochs,
        desc_prefix=f"[pretrain-{pretrain_mode}]",
    )

    print(f"[pretrain-{pretrain_mode}] Finished. Best valid mean dist: {best_dist:.4f}")

    # pretrain 모델 저장
    torch.save(best_state, PRETRAIN_MODEL_PATH)
    print(f"[pretrain] Saved to: {PRETRAIN_MODEL_PATH}")

    return best_state


# ===========================
# Fine-tune 함수 (team 단위)
# ===========================

def finetune_team(
    team_id: int,
    df: pd.DataFrame,
    pretrain_state: Dict,
    epochs: int = FINETUNE_EPOCHS,
) -> Tuple[nn.Module, Dict, float]:
    """
    Pretrain된 가중치를 기반으로 특정 team_id에 대해 fine-tune 수행.

    Args:
        team_id: 학습할 팀 ID
        df: 학습 데이터
        pretrain_state: pretrain된 모델 가중치
        epochs: finetune epoch 수

    Returns:
        model, best_state, best_dist
    """
    print(f"\n========== Team {team_id} Fine-tuning ==========")

    episodes, targets = build_episode_team_sequences(df, team_id)

    if len(episodes) < 2:
        raise ValueError(f"[team {team_id}] episode 시퀀스가 너무 적어서 학습이 불가능합니다.")

    train_loader, valid_loader = build_dataloaders(episodes, targets, batch_size=BATCH_SIZE)

    # pretrain 가중치로 초기화
    model, criterion, optimizer = create_model(pretrain_state=pretrain_state)

    _, best_state, best_dist = train_loop(
        model, criterion, optimizer,
        train_loader, valid_loader,
        epochs=epochs,
        desc_prefix=f"[team {team_id}]",
    )

    return model, best_state, best_dist

