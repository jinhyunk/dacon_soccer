import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
import glob
import os
from tqdm import tqdm
from model import * 

# --- 1. 설정 (Hyperparameters) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 256        # RTX 3060 활용을 위해 배치 크기 증가
LR = 0.001
EPOCHS = 50             # 충분한 학습을 위해 설정
MAX_X = 105.0
MAX_Y = 68.0
MAX_TIME = 5700.0

ACTION_TO_IDX = {
    'Aerial Clearance': 0,
    'Block': 1,
    'Carry': 2,
    'Catch': 3,
    'Clearance': 4,
    'Cross': 5,
    'Deflection': 6,
    'Duel': 7,
    'Error': 8,
    'Foul': 9,
    'Foul_Throw': 10,
    'Goal': 11,
    'Goal Kick': 12,
    'Handball_Foul': 13,
    'Hit': 14,
    'Interception': 15,
    'Intervention': 16,
    'Offside': 17,
    'Out': 18,
    'Own Goal': 19,
    'Parry': 20,
    'Pass': 21,
    'Pass_Corner': 22,
    'Pass_Freekick': 23,
    'Penalty Kick': 24,
    'Recovery': 25,
    'Shot': 26,
    'Shot_Corner': 27,
    'Shot_Freekick': 28,
    'Tackle': 29,
    'Take-On': 30,
    'Throw-In': 31,
    'Other': 32  # 매핑되지 않은 값이나 예외 처리를 위한 클래스
}


class SoccerActionAwareBaselineDataset(Dataset):
    def __init__(self, data_dir, action_map, max_len_embed=30):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        self.action_map = action_map
        self.MAX_X = 105.0
        self.MAX_Y = 68.0
        self.MAX_TIME = 5700.0
        self.max_len_embed = max_len_embed

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            df = pd.read_csv(self.file_paths[idx])
            if len(df) < 2: return None
            
            # --- 1. Context 추출 ---
            # Start Action
            first_action = df.iloc[0]['type_name']
            start_action_idx = self.action_map.get(first_action, self.action_map['Other'])
            
            # Length (Embedding용, 최대값 클리핑)
            # Baseline은 에피소드 전체 길이를 씁니다.
            ep_len = len(df) - 1 # 마지막 타겟 제외 길이
            ep_len_idx = min(ep_len, self.max_len_embed - 1)
            
            # --- 2. Feature 정규화 ---
            sx = df['start_x'].values / self.MAX_X
            sy = df['start_y'].values / self.MAX_Y
            ex = df['end_x'].values / self.MAX_X
            ey = df['end_y'].values / self.MAX_Y
            t  = df['time_seconds'].values / self.MAX_TIME
            
            features = np.stack([sx, sy, ex, ey, t], axis=1)
            
            # Input / Target
            target = features[-1, 2:4]
            input_seq = features[:-1]
            
            return (torch.FloatTensor(input_seq), 
                    torch.FloatTensor(target), 
                    start_action_idx, 
                    ep_len_idx)

        except Exception as e:
            return None

def action_aware_baseline_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None, None
    
    inputs, targets, start_acts, len_idxs = zip(*batch)
    
    lengths = torch.LongTensor([len(x) for x in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    start_acts = torch.LongTensor(start_acts)
    len_idxs = torch.LongTensor(len_idxs)
    
    return padded_inputs, targets, lengths, start_acts, len_idxs

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs):
    # Loss가 아닌 '거리 오차'를 기준으로 Best 모델을 판단합니다.
    best_dist_error = float('inf') 
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # --- Train Loop ---
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets, lengths, start_acts, len_idxs = batch
            
            if inputs is None: continue
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            start_acts = start_acts.to(device) # 추가됨
            len_idxs = len_idxs.to(device)     # 추가됨
            
            # 모델 입력
            optimizer.zero_grad()
            outputs = model(inputs, lengths, start_acts, len_idxs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        total_dist_error = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                inputs, targets, lengths = batch
                if inputs is None: continue
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                start_acts = start_acts.to(device) # 추가됨
                len_idxs = len_idxs.to(device) 
                
                outputs = model(inputs, lengths, start_acts, len_idxs)
                
                # Loss 계산 (MSE)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # 거리 오차 계산 (Meter 단위 복원)
                pred_real_x = outputs[:, 0] * MAX_X
                pred_real_y = outputs[:, 1] * MAX_Y
                true_real_x = targets[:, 0] * MAX_X
                true_real_y = targets[:, 1] * MAX_Y
                
                # 유클리드 거리
                dist = torch.sqrt((pred_real_x - true_real_x)**2 + (pred_real_y - true_real_y)**2)
                total_dist_error += dist.sum().item()
                count += inputs.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dist_error = total_dist_error / count if count > 0 else 0.0
        
        print(f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        print(f"Val Avg Distance Error: {avg_dist_error:.4f} meters") # 소수점 4자리까지 표시
        
        # --- Best Model 저장 로직 수정 ---
        if avg_dist_error < best_dist_error:
            best_dist_error = avg_dist_error
            
            # 저장 폴더 생성
            os.makedirs('./weight', exist_ok=True)
            
            # 파일명에 LR과 거리 오차(m)를 포함
            save_path = f'./weight/baseline_lr{LR}_dist{best_dist_error:.4f}m.pth'
            
            torch.save(model.state_dict(), save_path)
            print(f">> 🚀 Best model saved! (Error: {best_dist_error:.4f}m)")

# --- 5. 실행 ---
if __name__ == "__main__":
    # 데이터 경로 확인 필요
    train_dataset = SoccerActionAwareBaselineDataset('./data/train',ACTION_TO_IDX)
    val_dataset = SoccerActionAwareBaselineDataset('./data/val',ACTION_TO_IDX)
    
    # 3060 GPU 사용 시 num_workers를 높여 데이터 로딩 병목 해결 (4~8 추천)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, collate_fn=action_aware_baseline_collate_fn, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, collate_fn=action_aware_baseline_collate_fn, num_workers=4, pin_memory=True)
    
    # 모델 초기화 (RTX 3060용 설정)
    model = ActionAwareBaselineLSTM(input_size=5, 
                 hidden_size=256, 
                 num_layers=3, 
                 output_size=2, 
                 dropout_rate=0.3,
                 # --- 추가된 파라미터 ---
                 num_actions=33,       # Action 종류 개수
                 max_len=30,           # 길이 임베딩 최대값 (Baseline은 Sequence가 기므로 적절히 조절 필요, 여기선 phase와 맞춤)
                 action_emb_dim=4,     # Action 임베딩 차원
                 len_emb_dim=4         # Length 임베딩 차원
                 ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    train_and_validate(model, train_loader, val_loader, optimizer, criterion, EPOCHS)