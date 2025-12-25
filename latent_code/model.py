"""
Conditional VAE for Pass Destination Prediction
구조:
- EncoderX: 입력 시퀀스를 인코딩 → h_X (context)
  - 지원 인코더: LSTM, GRU, Transformer
- EncoderZ: q(z | X, Y) - posterior (학습 시 사용)
- PriorNet: p(z | X) - prior (추론 시 사용, 옵션)
- Decoder: p(Y | X, z) - 디코더
"""
import math
from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import INPUT_DIM, HIDDEN_DIM, Z_DIM

# 지원하는 인코더 타입
EncoderType = Literal["lstm", "gru", "transformer"]


# ==========================================
# Positional Encoding (Transformer용)
# ==========================================

class PositionalEncoding(nn.Module):
    """Transformer를 위한 Sinusoidal Positional Encoding"""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Sinusoidal position encoding 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==========================================
# LSTM Encoder
# ==========================================

class LSTMEncoderX(nn.Module):
    """
    LSTM 기반 인코더
    입력 시퀀스를 인코딩하여 context vector h_X를 생성
    """
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, input_dim) - 패딩된 시퀀스
            lengths: (B,) - 각 시퀀스의 실제 길이
            padding_mask: (B, T) - 패딩 마스크 (미사용, 인터페이스 통일용)
        
        Returns:
            h_X: (B, hidden_dim) - context vector
        """
        # CPU에서 lengths 필요 (pack_padded_sequence 요구사항)
        lengths_cpu = lengths.cpu()
        
        # Pack padded sequence
        packed = pack_padded_sequence(
            x_seq, lengths_cpu, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # LSTM forward
        _, (h_n, _) = self.lstm(packed)
        
        # 마지막 레이어의 hidden state 사용
        h_X = h_n[-1]  # (B, hidden_dim)
        h_X = self.layer_norm(h_X)
        
        return h_X


# ==========================================
# GRU Encoder
# ==========================================

class GRUEncoderX(nn.Module):
    """
    GRU 기반 인코더
    LSTM보다 가볍고 빠르면서 유사한 성능
    """
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=False
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, input_dim)
            lengths: (B,)
            padding_mask: (B, T) - 패딩 마스크 (미사용)
        
        Returns:
            h_X: (B, hidden_dim)
        """
        lengths_cpu = lengths.cpu()
        
        packed = pack_padded_sequence(
            x_seq, lengths_cpu, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        _, h_n = self.gru(packed)
        
        h_X = h_n[-1]  # (B, hidden_dim)
        h_X = self.layer_norm(h_X)
        
        return h_X


# ==========================================
# Transformer Encoder
# ==========================================

class TransformerEncoderX(nn.Module):
    """
    Transformer 기반 인코더
    Self-attention으로 시퀀스의 전역적인 의존성 학습
    """
    def __init__(
        self, 
        input_dim: int = INPUT_DIM, 
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (+1 for CLS token)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len + 1, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable [CLS] token for sequence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, input_dim)
            lengths: (B,) - 각 시퀀스의 실제 길이
            padding_mask: (B, T) - True = 패딩 위치
        
        Returns:
            h_X: (B, hidden_dim)
        """
        B, T, _ = x_seq.shape
        
        # Input projection
        x = self.input_proj(x_seq)  # (B, T, hidden_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, hidden_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Update padding mask for CLS token
        if padding_mask is not None:
            # CLS token은 패딩이 아님
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, T+1)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, T+1, hidden_dim)
        
        # CLS token의 representation 사용
        h_X = x[:, 0, :]  # (B, hidden_dim)
        h_X = self.layer_norm(h_X)
        
        return h_X


# ==========================================
# Encoder Factory
# ==========================================

def create_encoder_x(
    encoder_type: EncoderType = "lstm",
    input_dim: int = INPUT_DIM,
    hidden_dim: int = HIDDEN_DIM,
    **kwargs
) -> nn.Module:
    """
    인코더 타입에 따라 적절한 EncoderX 생성
    
    Args:
        encoder_type: "lstm", "gru", "transformer" 중 하나
        input_dim: 입력 차원
        hidden_dim: hidden 차원
        **kwargs: 추가 인자 (transformer의 경우 num_layers, num_heads 등)
    
    Returns:
        EncoderX 인스턴스
    """
    if encoder_type == "lstm":
        return LSTMEncoderX(input_dim, hidden_dim)
    elif encoder_type == "gru":
        return GRUEncoderX(input_dim, hidden_dim)
    elif encoder_type == "transformer":
        return TransformerEncoderX(
            input_dim, 
            hidden_dim,
            num_layers=kwargs.get('num_layers', 4),
            num_heads=kwargs.get('num_heads', 4),
            dropout=kwargs.get('dropout', 0.1),
            max_seq_len=kwargs.get('max_seq_len', 512)
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. "
                         f"Supported: 'lstm', 'gru', 'transformer'")


# 기존 호환성을 위한 alias
EncoderX = LSTMEncoderX


class EncoderZ(nn.Module):
    """
    Posterior encoder: q(z | X, Y)
    학습 시 사용: 입력 X와 타깃 Y를 조건으로 잠재 변수 z의 분포를 학습
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        
        # 입력: h_X (hidden_dim) + y (2) → z의 mu, logvar
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, z_dim)

    def forward(self, h_X: torch.Tensor, y_gt: torch.Tensor) -> tuple:
        """
        Args:
            h_X: (B, hidden_dim) - context vector from EncoderX
            y_gt: (B, 2) - ground truth target coordinates
        
        Returns:
            mu: (B, z_dim) - mean of q(z|X,Y)
            logvar: (B, z_dim) - log variance of q(z|X,Y)
        """
        h = torch.cat([h_X, y_gt], dim=-1)  # (B, hidden_dim + 2)
        h = self.fc(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class PriorNet(nn.Module):
    """
    Prior network: p(z | X)
    추론 시 사용: 입력 X만으로 z의 사전 분포를 추정
    (옵션: 사용하지 않으면 N(0, I)를 prior로 사용)
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, z_dim)

    def forward(self, h_X: torch.Tensor) -> tuple:
        """
        Args:
            h_X: (B, hidden_dim)
        
        Returns:
            mu: (B, z_dim)
            logvar: (B, z_dim)
        """
        h = self.fc(h_X)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Decoder: p(Y | X, z)
    context vector h_X와 잠재 변수 z를 결합하여 타깃 좌표 예측
    """
    def __init__(self, hidden_dim: int = HIDDEN_DIM, z_dim: int = Z_DIM):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # (x, y) 좌표
            nn.Sigmoid()  # 정규화된 좌표 [0, 1]
        )

    def forward(self, h_X: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_X: (B, hidden_dim)
            z: (B, z_dim)
        
        Returns:
            y_pred: (B, 2) - predicted coordinates
        """
        h = torch.cat([h_X, z], dim=-1)
        y_pred = self.fc(h)
        return y_pred


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick: z = mu + std * eps
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class PassCVAE(nn.Module):
    """
    Conditional VAE for Pass Destination Prediction
    
    학습:
        - EncoderX로 입력 시퀀스를 인코딩 → h_X
        - EncoderZ로 q(z|X,Y) 학습 → mu, logvar
        - Reparameterization trick으로 z 샘플링
        - Decoder로 (h_X, z) → Y 예측
        - ELBO loss = Reconstruction loss + KL divergence
    
    추론:
        - EncoderX로 입력 시퀀스를 인코딩 → h_X
        - PriorNet으로 p(z|X) 샘플링 또는 N(0,I)에서 z 샘플링
        - Decoder로 (h_X, z) → Y 예측
        - 여러 z를 샘플링하여 다중 미래 예측 가능
    
    지원 인코더:
        - "lstm": LSTM 기반 (기본값)
        - "gru": GRU 기반 (더 가벼움)
        - "transformer": Transformer 기반 (전역 의존성 학습)
    """
    def __init__(
        self, 
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM, 
        z_dim: int = Z_DIM,
        use_learned_prior: bool = True,
        encoder_type: EncoderType = "lstm",
        encoder_kwargs: dict = None
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_learned_prior = use_learned_prior
        self.encoder_type = encoder_type
        
        # 인코더 생성 (encoder_type에 따라 다른 인코더 사용)
        encoder_kwargs = encoder_kwargs or {}
        self.encoder_x = create_encoder_x(
            encoder_type=encoder_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **encoder_kwargs
        )
        self.encoder_z = EncoderZ(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        
        # 학습된 prior 사용 여부
        if use_learned_prior:
            self.prior_net = PriorNet(hidden_dim, z_dim)

    def forward(
        self, 
        x_seq: torch.Tensor, 
        lengths: torch.Tensor,
        y_gt: torch.Tensor = None,
        padding_mask: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            x_seq: (B, T, input_dim) - 입력 시퀀스
            lengths: (B,) - 각 시퀀스의 실제 길이
            y_gt: (B, 2) or None - 타깃 좌표 (학습 시에만 제공)
            padding_mask: (B, T) or None - 패딩 마스크 (True = 패딩 위치)
        
        Returns:
            학습 시: (y_pred, mu, logvar, prior_mu, prior_logvar)
            추론 시: y_pred
        """
        # Context vector 추출
        h_X = self.encoder_x(x_seq, lengths, padding_mask)  # (B, hidden_dim)
        
        if y_gt is not None:
            # ========== TRAINING ==========
            # Posterior: q(z|X,Y)
            mu, logvar = self.encoder_z(h_X, y_gt)
            z = reparameterize(mu, logvar)
            
            # Decode
            y_pred = self.decoder(h_X, z)
            
            # Prior: p(z|X) or N(0,I)
            if self.use_learned_prior:
                prior_mu, prior_logvar = self.prior_net(h_X)
            else:
                prior_mu = torch.zeros_like(mu)
                prior_logvar = torch.zeros_like(logvar)
            
            return y_pred, mu, logvar, prior_mu, prior_logvar
        
        else:
            # ========== INFERENCE ==========
            B = x_seq.size(0)
            
            if self.use_learned_prior:
                # 학습된 prior에서 샘플링
                prior_mu, prior_logvar = self.prior_net(h_X)
                z = reparameterize(prior_mu, prior_logvar)
            else:
                # N(0, I)에서 샘플링
                z = torch.randn(B, self.z_dim, device=x_seq.device)
            
            y_pred = self.decoder(h_X, z)
            return y_pred
    
    def sample_multiple(
        self, 
        x_seq: torch.Tensor, 
        lengths: torch.Tensor,
        num_samples: int = 20,
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        다중 미래 예측을 위해 여러 z를 샘플링
        
        Args:
            x_seq: (B, T, input_dim)
            lengths: (B,)
            num_samples: 샘플링 개수
            padding_mask: (B, T) or None - 패딩 마스크
        
        Returns:
            predictions: (B, num_samples, 2) - 각 샘플에 대한 예측 좌표
        """
        self.eval()
        with torch.no_grad():
            h_X = self.encoder_x(x_seq, lengths, padding_mask)  # (B, hidden_dim)
            B = x_seq.size(0)
            
            predictions = []
            
            for _ in range(num_samples):
                if self.use_learned_prior:
                    prior_mu, prior_logvar = self.prior_net(h_X)
                    z = reparameterize(prior_mu, prior_logvar)
                else:
                    z = torch.randn(B, self.z_dim, device=x_seq.device)
                
                y_pred = self.decoder(h_X, z)
                predictions.append(y_pred)
            
            # (B, num_samples, 2)
            predictions = torch.stack(predictions, dim=1)
        
        return predictions
    
    def get_best_prediction(
        self,
        x_seq: torch.Tensor,
        lengths: torch.Tensor,
        num_samples: int = 20,
        aggregation: str = "mean",
        padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        다중 샘플의 평균 또는 중앙값을 최종 예측으로 사용
        
        Args:
            aggregation: "mean" or "median"
            padding_mask: (B, T) or None - 패딩 마스크
        
        Returns:
            y_pred: (B, 2)
        """
        predictions = self.sample_multiple(x_seq, lengths, num_samples, padding_mask)
        
        if aggregation == "mean":
            return predictions.mean(dim=1)
        elif aggregation == "median":
            return predictions.median(dim=1).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
