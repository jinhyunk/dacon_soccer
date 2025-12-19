import torch.nn as nn

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