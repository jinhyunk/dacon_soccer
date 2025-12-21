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
from model import * # --- 1. 설정 (Hyperparameters) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 256        
LR = 0.001
EPOCHS = 50             
MAX_X = 105.0
MAX_Y = 68.0
MAX_TIME = 5700.0

# [NEW] RealDistanceLoss 정의
# ==========================================
class RealDistanceLoss(nn.Module):
    def __init__(self, max_x=105.0, max_y=68.0):
        super(RealDistanceLoss, self).__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.epsilon = 1e-6

    def forward(self, pred, target):
        """
        pred, target: (Batch, 2) - Normalized [0, 1]
        """
        # 정규화된 좌표 차이를 실제 미터 단위 거리로 환산
        diff_x = (pred[:, 0] - target[:, 0]) * self.max_x
        diff_y = (pred[:, 1] - target[:, 1]) * self.max_y
        
        # 유클리드 거리 계산
        distance = torch.sqrt(diff_x**2 + diff_y**2 + self.epsilon)
        
        # 평균 거리 반환 (Scalar)
        return distance.mean()

class SoccerBaselineDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        self.MAX_X = 105.0
        self.MAX_Y = 68.0
        self.MAX_TIME = 5700.0 

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            df = pd.read_csv(file_path)
            if len(df) < 2: return None
            
            # 정규화
            sx = df['start_x'].values / self.MAX_X
            sy = df['start_y'].values / self.MAX_Y
            ex = df['end_x'].values / self.MAX_X
            ey = df['end_y'].values / self.MAX_Y
            t  = df['time_seconds'].values / self.MAX_TIME
            
            features = np.stack([sx, sy, ex, ey, t], axis=1)
            target = features[-1, 2:4] # [end_x, end_y]
            input_seq = features[:-1]
            
            return torch.FloatTensor(input_seq), torch.FloatTensor(target)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

def baseline_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None, None, None
    
    inputs, targets = zip(*batch)
    lengths = torch.LongTensor([len(x) for x in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    
    return padded_inputs, targets, lengths

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs):
    best_dist_error = float('inf') 
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # --- Train Loop ---
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            inputs, targets, lengths = batch
            if inputs is None: continue
            
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            
            # Loss 계산 (이제 이 값이 곧 미터 단위 거리 오차입니다)
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
                
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, lengths)
                
                # Val Loss (RealDistanceLoss)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                # 거리 오차 계산 (검증용 - Loss와 동일하지만 지표 확인용으로 유지)
                pred_real_x = outputs[:, 0] * MAX_X
                pred_real_y = outputs[:, 1] * MAX_Y
                true_real_x = targets[:, 0] * MAX_X
                true_real_y = targets[:, 1] * MAX_Y
                
                dist = torch.sqrt((pred_real_x - true_real_x)**2 + (pred_real_y - true_real_y)**2)
                total_dist_error += dist.sum().item()
                count += inputs.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_dist_error = total_dist_error / count if count > 0 else 0.0
        
        # 이제 Train/Val Loss도 미터(m) 단위입니다.
        print(f"Train Loss: {avg_train_loss:.4f}m | Val Loss: {avg_val_loss:.4f}m")
        print(f"Val Avg Distance Error: {avg_dist_error:.4f} meters")
        
        # --- Best Model 저장 ---
        if avg_dist_error < best_dist_error:
            best_dist_error = avg_dist_error
            os.makedirs('./weight', exist_ok=True)
            save_path = f'./weight/baseline_lr{LR}_dist{best_dist_error:.4f}m.pth'
            torch.save(model.state_dict(), save_path)
            print(f">> 🚀 Best model saved! (Error: {best_dist_error:.4f}m)")

# --- 5. 실행 ---
if __name__ == "__main__":
    train_dataset = SoccerBaselineDataset('./data/train')
    val_dataset = SoccerBaselineDataset('./data/val')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, collate_fn=baseline_collate_fn, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, collate_fn=baseline_collate_fn, num_workers=4, pin_memory=True)
    
    # 모델 초기화 (기본 LSTM)
    model = BaselineLSTM(
        input_size=5, 
        hidden_size=256, 
        num_layers=3, 
        output_size=2, 
        dropout_rate=0.3
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # [변경] MSE -> RealDistanceLoss
    criterion = RealDistanceLoss(max_x=MAX_X, max_y=MAX_Y)
    
    train_and_validate(model, train_loader, val_loader, optimizer, criterion, EPOCHS)