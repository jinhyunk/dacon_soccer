"""
Transformer 모델 정의

모델 타입:
- encoder: 처음부터 학습하는 Transformer Encoder
- pooling: 평균 풀링을 사용하는 Transformer
- pretrained_gpt2: GPT-2 pretrained 가중치 사용
- pretrained_bert: BERT pretrained 가중치 사용
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
    좌표 예측을 위한 Transformer Encoder 모델 (처음부터 학습)
    
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


# ==========================================
# Pretrained Transformer Models
# ==========================================

class PretrainedGPT2Encoder(nn.Module):
    """
    GPT-2 Pretrained 가중치를 사용하는 좌표 예측 모델
    
    - GPT-2의 Transformer 레이어 사용 (pretrained)
    - 입력 임베딩: 2D 좌표 -> GPT-2 hidden size
    - 출력 레이어: hidden state -> 2D 좌표
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        model_name: str = "gpt2",  # "gpt2", "gpt2-medium", "gpt2-large"
        dropout: float = DROPOUT,
        freeze_pretrained: bool = False,  # pretrained 레이어 freeze 여부
    ):
        super().__init__()
        
        try:
            from transformers import GPT2Model, GPT2Config
        except ImportError:
            raise ImportError(
                "transformers 라이브러리가 필요합니다. "
                "pip install transformers 를 실행해주세요."
            )
        
        # GPT-2 모델 로드
        print(f"Loading pretrained GPT-2 model: {model_name}")
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.d_model = self.gpt2.config.hidden_size  # 768 for gpt2
        
        # Pretrained 레이어 freeze 옵션
        if freeze_pretrained:
            for param in self.gpt2.parameters():
                param.requires_grad = False
            print("  (pretrained layers frozen)")
        
        # 입력 임베딩: 2D 좌표 -> GPT-2 hidden size
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )
        
        # 출력 레이어: hidden state -> 2D 좌표
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 2),
        )
        
        # 입력/출력 레이어 초기화
        self._init_new_weights()
    
    def _init_new_weights(self):
        """새로 추가한 레이어만 초기화"""
        for module in [self.input_embed, self.output_layer]:
            for p in module.parameters():
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
        """
        batch_size = x.size(0)
        
        # 입력 임베딩
        x = self.input_embed(x)  # [B, T, d_model]
        
        # GPT-2 attention mask (1 = 유효, 0 = 패딩) - src_key_padding_mask와 반대
        if src_key_padding_mask is not None:
            attention_mask = (~src_key_padding_mask).long()
        else:
            attention_mask = torch.ones(batch_size, x.size(1), dtype=torch.long, device=x.device)
        
        # GPT-2 forward (inputs_embeds 사용)
        outputs = self.gpt2(
            inputs_embeds=x,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, T, d_model]
        
        # 마지막 유효 위치의 hidden state 추출
        last_indices = (lengths - 1).view(batch_size, 1, 1).expand(-1, -1, self.d_model)
        last_hidden = hidden_states.gather(1, last_indices).squeeze(1)  # [B, d_model]
        
        # 출력 레이어
        out = self.output_layer(last_hidden)  # [B, 2]
        
        return out


class PretrainedBERTEncoder(nn.Module):
    """
    BERT Pretrained 가중치를 사용하는 좌표 예측 모델
    
    - BERT의 Transformer 레이어 사용 (pretrained)
    - 입력 임베딩: 2D 좌표 -> BERT hidden size
    - 출력 레이어: [CLS] hidden state -> 2D 좌표
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        model_name: str = "bert-base-uncased",  # "bert-base-uncased", "bert-large-uncased"
        dropout: float = DROPOUT,
        freeze_pretrained: bool = False,
    ):
        super().__init__()
        
        try:
            from transformers import BertModel
        except ImportError:
            raise ImportError(
                "transformers 라이브러리가 필요합니다. "
                "pip install transformers 를 실행해주세요."
            )
        
        # BERT 모델 로드
        print(f"Loading pretrained BERT model: {model_name}")
        self.bert = BertModel.from_pretrained(model_name)
        self.d_model = self.bert.config.hidden_size  # 768 for bert-base
        
        # Pretrained 레이어 freeze 옵션
        if freeze_pretrained:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("  (pretrained layers frozen)")
        
        # 입력 임베딩: 2D 좌표 -> BERT hidden size
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )
        
        # 출력 레이어
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 2),
        )
        
        self._init_new_weights()
    
    def _init_new_weights(self):
        for module in [self.input_embed, self.output_layer]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 입력 임베딩
        x = self.input_embed(x)  # [B, T, d_model]
        
        # BERT attention mask
        if src_key_padding_mask is not None:
            attention_mask = (~src_key_padding_mask).long()
        else:
            attention_mask = torch.ones(batch_size, x.size(1), dtype=torch.long, device=x.device)
        
        # BERT forward
        outputs = self.bert(
            inputs_embeds=x,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state  # [B, T, d_model]
        
        # 마지막 유효 위치의 hidden state 사용 (BERT는 보통 [CLS] 사용하지만 여기서는 마지막 위치)
        last_indices = (lengths - 1).view(batch_size, 1, 1).expand(-1, -1, self.d_model)
        last_hidden = hidden_states.gather(1, last_indices).squeeze(1)
        
        out = self.output_layer(last_hidden)
        
        return out


class PretrainedDistilBERTEncoder(nn.Module):
    """
    DistilBERT Pretrained 가중치를 사용하는 좌표 예측 모델 (경량 버전)
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        model_name: str = "distilbert-base-uncased",
        dropout: float = DROPOUT,
        freeze_pretrained: bool = False,
    ):
        super().__init__()
        
        try:
            from transformers import DistilBertModel
        except ImportError:
            raise ImportError(
                "transformers 라이브러리가 필요합니다. "
                "pip install transformers 를 실행해주세요."
            )
        
        print(f"Loading pretrained DistilBERT model: {model_name}")
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        self.d_model = self.distilbert.config.hidden_size
        
        if freeze_pretrained:
            for param in self.distilbert.parameters():
                param.requires_grad = False
            print("  (pretrained layers frozen)")
        
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(dropout),
        )
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model // 2, 2),
        )
        
        self._init_new_weights()
    
    def _init_new_weights(self):
        for module in [self.input_embed, self.output_layer]:
            for p in module.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        
        x = self.input_embed(x)
        
        if src_key_padding_mask is not None:
            attention_mask = (~src_key_padding_mask).long()
        else:
            attention_mask = torch.ones(batch_size, x.size(1), dtype=torch.long, device=x.device)
        
        outputs = self.distilbert(
            inputs_embeds=x,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state
        
        last_indices = (lengths - 1).view(batch_size, 1, 1).expand(-1, -1, self.d_model)
        last_hidden = hidden_states.gather(1, last_indices).squeeze(1)
        
        out = self.output_layer(last_hidden)
        
        return out


# ==========================================
# Model Factory
# ==========================================

def create_model(
    model_type: str = "encoder",
    pretrain_state: Optional[Dict] = None,
    d_model: int = D_MODEL,
    nhead: int = NHEAD,
    num_layers: int = NUM_LAYERS,
    dim_feedforward: int = DIM_FEEDFORWARD,
    dropout: float = DROPOUT,
    lr: float = LR,
    freeze_pretrained: bool = False,
    pretrained_model_name: str = None,
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer]:
    """
    모델, 손실함수, 옵티마이저 생성
    
    Args:
        model_type: 
            - "encoder": 처음부터 학습하는 Transformer Encoder
            - "pooling": 평균 풀링을 사용하는 Transformer
            - "pretrained_gpt2": GPT-2 pretrained 가중치 사용
            - "pretrained_bert": BERT pretrained 가중치 사용
            - "pretrained_distilbert": DistilBERT pretrained 가중치 사용 (경량)
        pretrain_state: 사전학습된 가중치 (선택적, 자체 pretrain용)
        freeze_pretrained: pretrained 모델 레이어 freeze 여부
        pretrained_model_name: pretrained 모델 이름 (예: "gpt2", "bert-base-uncased")
        
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
    elif model_type == "pretrained_gpt2":
        model_name = pretrained_model_name or "gpt2"
        model = PretrainedGPT2Encoder(
            input_dim=2,
            model_name=model_name,
            dropout=dropout,
            freeze_pretrained=freeze_pretrained,
        ).to(DEVICE)
    elif model_type == "pretrained_bert":
        model_name = pretrained_model_name or "bert-base-uncased"
        model = PretrainedBERTEncoder(
            input_dim=2,
            model_name=model_name,
            dropout=dropout,
            freeze_pretrained=freeze_pretrained,
        ).to(DEVICE)
    elif model_type == "pretrained_distilbert":
        model_name = pretrained_model_name or "distilbert-base-uncased"
        model = PretrainedDistilBERTEncoder(
            input_dim=2,
            model_name=model_name,
            dropout=dropout,
            freeze_pretrained=freeze_pretrained,
        ).to(DEVICE)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            "Use 'encoder', 'pooling', 'pretrained_gpt2', 'pretrained_bert', or 'pretrained_distilbert'."
        )
    
    if pretrain_state is not None:
        model.load_state_dict(pretrain_state)
        print("  (loaded pretrain weights from state_dict)")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 모델 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, criterion, optimizer
