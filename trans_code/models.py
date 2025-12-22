"""
Transformer 모델 정의
"""
import math
from typing import Dict, Tuple, Optional

import torch
from torch import nn

from config import (
    DEVICE,
    D_MODEL,
    NHEAD,
    NUM_LAYERS,
    DIM_FEEDFORWARD,
    DROPOUT,
    MAX_SEQ_LEN,
    LR,
)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding
    """
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 계산
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # [max_len, d_model] -> [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, D]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    좌표 예측을 위한 Transformer Encoder 모델
    
    입력: (x, y) 좌표 시퀀스 [B, T, 2]
    출력: 마지막 위치의 예측 좌표 [B, 2]
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 입력 임베딩: 2D 좌표 -> d_model 차원
        self.input_embed = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # 출력 레이어: 마지막 hidden state -> 2D 좌표
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, 2] - 패딩된 좌표 시퀀스
            lengths: [B] - 각 시퀀스의 실제 길이
            src_key_padding_mask: [B, T] - 패딩 마스크 (True = 패딩)
        
        Returns:
            [B, 2] - 예측 좌표 (정규화됨)
        """
        batch_size = x.size(0)
        
        # 입력 임베딩
        x = self.input_embed(x)  # [B, T, d_model]
        
        # 스케일링 (Transformer 논문)
        x = x * math.sqrt(self.d_model)
        
        # Positional Encoding
        x = self.pos_encoder(x)  # [B, T, d_model]
        
        # Transformer Encoder
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )  # [B, T, d_model]
        
        # 각 시퀀스의 마지막 유효 위치의 hidden state 추출
        # lengths[i] - 1 이 마지막 유효 인덱스
        last_indices = (lengths - 1).view(batch_size, 1, 1).expand(-1, -1, self.d_model)
        last_hidden = encoded.gather(1, last_indices).squeeze(1)  # [B, d_model]
        
        # 출력 레이어
        out = self.output_layer(last_hidden)  # [B, 2]
        
        return out


class TransformerWithPooling(nn.Module):
    """
    평균 풀링을 사용하는 Transformer 모델 (대안)
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # 입력 임베딩
        self.input_embed = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """평균 풀링을 사용한 forward"""
        batch_size = x.size(0)
        
        # 입력 임베딩
        x = self.input_embed(x)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Transformer Encoder
        encoded = self.transformer_encoder(
            x,
            src_key_padding_mask=src_key_padding_mask,
        )
        
        # 마스크를 고려한 평균 풀링
        if src_key_padding_mask is not None:
            # 패딩 위치는 0으로 설정
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, T, 1]
            sum_hidden = (encoded * mask).sum(dim=1)  # [B, d_model]
            avg_hidden = sum_hidden / lengths.float().unsqueeze(-1)  # [B, d_model]
        else:
            avg_hidden = encoded.mean(dim=1)
        
        out = self.output_layer(avg_hidden)
        
        return out


def create_model(
    model_type: str = "encoder",
    pretrain_state: Optional[Dict] = None,
    d_model: int = D_MODEL,
    nhead: int = NHEAD,
    num_layers: int = NUM_LAYERS,
    dim_feedforward: int = DIM_FEEDFORWARD,
    dropout: float = DROPOUT,
    lr: float = LR,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """
    모델, 손실함수, 옵티마이저 생성
    
    Args:
        model_type: "encoder" (마지막 hidden) 또는 "pooling" (평균 풀링)
        pretrain_state: 사전학습된 가중치 (선택적)
        
    Returns:
        model, criterion, optimizer
    """
    if model_type == "encoder":
        model = TransformerEncoder(
            input_dim=2,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(DEVICE)
    elif model_type == "pooling":
        model = TransformerWithPooling(
            input_dim=2,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    if pretrain_state is not None:
        model.load_state_dict(pretrain_state)
        print("  (loaded pretrain weights)")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, criterion, optimizer

