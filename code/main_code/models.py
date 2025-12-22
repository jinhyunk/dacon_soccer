"""
모델 정의
"""
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from config import DEVICE, HIDDEN_DIM, LR


class LSTMBaseline(nn.Module):
    """LSTM 기반 좌표 예측 모델"""

    def __init__(self, input_dim: int = 2, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 2)  # (x_norm, y_norm)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 2] - 패딩된 시퀀스
            lengths: [B] - 각 시퀀스의 실제 길이
        Returns:
            [B, 2] - 예측 좌표 (정규화됨)
        """
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]  # [B, H] 마지막 layer의 hidden state
        out = self.fc(h_last)  # [B, 2]
        return out


def create_model(
    pretrain_state: Dict = None,
    hidden_dim: int = HIDDEN_DIM,
    lr: float = LR,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """
    모델, 손실함수, 옵티마이저 생성.
    pretrain_state가 주어지면 해당 가중치로 초기화.
    """
    model = LSTMBaseline(input_dim=2, hidden_dim=hidden_dim).to(DEVICE)

    if pretrain_state is not None:
        model.load_state_dict(pretrain_state)
        print("  (loaded pretrain weights)")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, criterion, optimizer

