"""
Conditional VAE for Pass Destination Prediction
구조:
- EncoderX: LSTM으로 입력 시퀀스를 인코딩 → h_X (context)
- EncoderZ: q(z | X, Y) - posterior (학습 시 사용)
- PriorNet: p(z | X) - prior (추론 시 사용, 옵션)
- Decoder: p(Y | X, z) - 디코더
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from config import INPUT_DIM, HIDDEN_DIM, Z_DIM


class EncoderX(nn.Module):
    """
    입력 시퀀스를 인코딩하여 context vector h_X를 생성
    LSTM 사용, 가변 길이 시퀀스 처리
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

    def forward(self, x_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_seq: (B, T, input_dim) - 패딩된 시퀀스
            lengths: (B,) - 각 시퀀스의 실제 길이
        
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
    """
    def __init__(
        self, 
        input_dim: int = INPUT_DIM,
        hidden_dim: int = HIDDEN_DIM, 
        z_dim: int = Z_DIM,
        use_learned_prior: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.use_learned_prior = use_learned_prior
        
        self.encoder_x = EncoderX(input_dim, hidden_dim)
        self.encoder_z = EncoderZ(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)
        
        # 학습된 prior 사용 여부
        if use_learned_prior:
            self.prior_net = PriorNet(hidden_dim, z_dim)

    def forward(
        self, 
        x_seq: torch.Tensor, 
        lengths: torch.Tensor,
        y_gt: torch.Tensor = None
    ) -> tuple:
        """
        Args:
            x_seq: (B, T, input_dim) - 입력 시퀀스
            lengths: (B,) - 각 시퀀스의 실제 길이
            y_gt: (B, 2) or None - 타깃 좌표 (학습 시에만 제공)
        
        Returns:
            학습 시: (y_pred, mu, logvar, prior_mu, prior_logvar)
            추론 시: y_pred
        """
        # Context vector 추출
        h_X = self.encoder_x(x_seq, lengths)  # (B, hidden_dim)
        
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
        num_samples: int = 20
    ) -> torch.Tensor:
        """
        다중 미래 예측을 위해 여러 z를 샘플링
        
        Args:
            x_seq: (B, T, input_dim)
            lengths: (B,)
            num_samples: 샘플링 개수
        
        Returns:
            predictions: (B, num_samples, 2) - 각 샘플에 대한 예측 좌표
        """
        self.eval()
        with torch.no_grad():
            h_X = self.encoder_x(x_seq, lengths)  # (B, hidden_dim)
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
        aggregation: str = "mean"
    ) -> torch.Tensor:
        """
        다중 샘플의 평균 또는 중앙값을 최종 예측으로 사용
        
        Args:
            aggregation: "mean" or "median"
        
        Returns:
            y_pred: (B, 2)
        """
        predictions = self.sample_multiple(x_seq, lengths, num_samples)
        
        if aggregation == "mean":
            return predictions.mean(dim=1)
        elif aggregation == "median":
            return predictions.median(dim=1).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
