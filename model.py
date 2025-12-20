import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class BaselineLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=256, num_layers=3, output_size=2, dropout_rate=0.3):
        super(BaselineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. LSTM Layer 강화
        # dropout: 레이어 사이(Layer 1->2, 2->3)에 드롭아웃 적용
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout_rate)
        
        # 2. Prediction Head (MLP 구조로 변경)
        # 단순히 바로 좌표를 뽑는 것보다, 한 번 더 가공(Non-linearity)하는 것이 정확도에 유리
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), # 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout_rate),                 # 과적합 방지
            nn.Linear(hidden_size // 2, output_size)  # 128 -> 2
        )

    def forward(self, x, lengths):
        # Pack padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM
        # output: (Batch, Seq, Hidden) - 모든 시점의 출력
        # h_n: (Num_Layers, Batch, Hidden) - 마지막 시점의 Hidden State
        packed_out, (h_n, c_n) = self.lstm(packed_x)
        
        # 마지막 레이어의 Hidden State 추출
        # h_n은 [Layer1_Last, Layer2_Last, Layer3_Last] 순서로 쌓여있음
        last_hidden = h_n[-1] # (Batch, Hidden_Size) -> (Batch, 256)
        
        # Prediction Head 통과
        prediction = self.head(last_hidden) # (Batch, 2)
        
        return prediction
    

class HierarchicalLSTM(nn.Module):
    def __init__(self, input_size=5, phase_hidden_size=64, episode_hidden_size=128, output_size=2, dropout=0.0):
        super(HierarchicalLSTM, self).__init__()
        
        # --- Level 1: Phase Encoder ---
        self.phase_lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=phase_hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # --- Level 2: Episode Encoder ---
        self.episode_lstm = nn.LSTM(
            input_size=phase_hidden_size,
            hidden_size=episode_hidden_size,
            num_layers=1, 
            batch_first=True,
            dropout=dropout if dropout > 0 else 0 # layer가 1개일 땐 보통 무시되지만 명시
        )
        
        # --- Level 3: Prediction Head ---
        # Dropout 추가로 과적합 방지 및 main.py 설정 반영
        self.regressor = nn.Sequential(
            nn.Linear(episode_hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),  # Dropout 적용
            nn.Linear(64, output_size) 
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths):
        # 1. Phase Level Encoding
        packed_phases = pack_padded_sequence(
            padded_phases, phase_lengths.cpu(), 
            batch_first=True, enforce_sorted=False
        )
        
        _, (phase_h_n, _) = self.phase_lstm(packed_phases)
        phase_embeddings = phase_h_n[-1] 
        
        # 2. Reshape (Flat Phases -> Episode Sequence)
        phases_per_episode = torch.split(phase_embeddings, episode_lengths.tolist())
        padded_episodes = pad_sequence(phases_per_episode, batch_first=True, padding_value=0)
        
        # 3. Episode Level Encoding
        packed_episodes = pack_padded_sequence(
            padded_episodes, episode_lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        
        _, (episode_h_n, _) = self.episode_lstm(packed_episodes)
        episode_representation = episode_h_n[-1]
        
        # 4. Final Prediction
        prediction = self.regressor(episode_representation)
        
        return prediction
    
class ContextAwareHierarchicalLSTM(nn.Module):
    def __init__(self, 
                 input_size=5, 
                 phase_hidden=64, 
                 episode_hidden=256, 
                 output_size=2, 
                 dropout=0.3,
                 num_actions=33,        # Action 종류 개수
                 max_phase_len=30,      # Phase 길이 최대값
                 action_emb_dim=4,      # Action 임베딩 차원
                 len_emb_dim=4          # Length 임베딩 차원
                 ):
        super(ContextAwareHierarchicalLSTM, self).__init__()
        
        # 1. Embeddings (Context 정보)
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.length_embedding = nn.Embedding(max_phase_len, len_emb_dim)
        
        # 2. Phase LSTM Input Size 계산
        # 기존 좌표(5) + 시작액션정보 + 길이정보
        self.phase_input_dim = input_size + action_emb_dim + len_emb_dim
        
        # 3. Phase LSTM
        self.phase_lstm = nn.LSTM(self.phase_input_dim, phase_hidden, num_layers=1, batch_first=True)
        
        # 4. Episode LSTM & Head (기존과 동일)
        self.episode_lstm = nn.LSTM(phase_hidden, episode_hidden, num_layers=2, 
                                    batch_first=True, dropout=dropout)
        
        self.regressor = nn.Sequential(
            nn.Linear(episode_hidden, episode_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(episode_hidden // 2, output_size)
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths, start_action_ids, phase_len_ids):
        """
        start_action_ids: (Total_Phases,)
        phase_len_ids: (Total_Phases,)
        """
        
        # ==========================================
        # 1. Context Embedding & Concatenation
        # ==========================================
        # (Total_Phases, Emb_Dim)
        action_emb = self.action_embedding(start_action_ids) 
        len_emb = self.length_embedding(phase_len_ids)
        
        # 두 임베딩 결합 -> Context Vector (Total_Phases, Total_Emb_Dim)
        context_vector = torch.cat([action_emb, len_emb], dim=1)
        
        # 시퀀스 길이(Seq_Len)만큼 Context Vector 복사 (Broadcasting 준비)
        # padded_phases shape: (Total_Phases, Seq_Len, 5)
        seq_len = padded_phases.size(1)
        
        # (Total_Phases, 1, Total_Emb_Dim) -> (Total_Phases, Seq_Len, Total_Emb_Dim)
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 입력 데이터와 Context 결합
        # Result: (Total_Phases, Seq_Len, 5 + Action_Emb + Len_Emb)
        phase_inputs = torch.cat([padded_phases, context_expanded], dim=2)
        
        
        # ==========================================
        # 2. Phase Level Encoding (Context가 포함된 입력 사용)
        # ==========================================
        packed_phases = pack_padded_sequence(phase_inputs, phase_lengths.cpu(), 
                                             batch_first=True, enforce_sorted=False)
        _, (phase_h_n, _) = self.phase_lstm(packed_phases)
        phase_embeddings = phase_h_n[-1] 
        
        
        # ==========================================
        # 3. Reshape & Episode Level Encoding (기존 동일)
        # ==========================================
        phases_per_episode = torch.split(phase_embeddings, episode_lengths.tolist())
        padded_episodes = pad_sequence(phases_per_episode, batch_first=True, padding_value=0)
        
        packed_episodes = pack_padded_sequence(padded_episodes, episode_lengths.cpu(),
                                               batch_first=True, enforce_sorted=False)
        _, (episode_h_n, _) = self.episode_lstm(packed_episodes)
        episode_representation = episode_h_n[-1]
        
        
        # ==========================================
        # 4. Prediction
        # ==========================================
        prediction = self.regressor(episode_representation)
        
        return prediction