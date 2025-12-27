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
from trashcan.model import * 

# --- 1. 설정 (Hyperparameters) ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BATCH_SIZE = 256        # RTX 3060 활용을 위해 배치 크기 증가
LR = 0.001
EPOCHS = 50             # 충분한 학습을 위해 설정
MAX_X = 105.0
MAX_Y = 68.0
MAX_TIME = 5700.0

class SoccerBaselineDataset(Dataset):
    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): 데이터 파일들이 있는 폴더 경로 (예: './data/train')
        """
        # 폴더 내의 모든 csv 파일 경로를 리스트로 저장
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        
        # 정규화 상수
        self.MAX_X = 105.0
        self.MAX_Y = 68.0
        self.MAX_TIME = 5700.0 # 95분

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            df = pd.read_csv(file_path)
            
            # 데이터가 너무 짧으면(이벤트 1개 이하) 예측 불가 -> None 반환 후 collate_fn에서 처리
            if len(df) < 2:
                return None
            
            # --- 1. 정규화 (Normalization) ---
            # 모든 시점의 데이터를 정규화
            sx = df['start_x'].values / self.MAX_X
            sy = df['start_y'].values / self.MAX_Y
            ex = df['end_x'].values / self.MAX_X
            ey = df['end_y'].values / self.MAX_Y
            t  = df['time_seconds'].values / self.MAX_TIME
            
            # (Seq_Len, 5) 형태로 합치기
            features = np.stack([sx, sy, ex, ey, t], axis=1)
            
            # --- 2. Input / Target 분리 ---
            # Target: 이 에피소드의 '마지막' 이벤트의 도착 위치 (end_x, end_y)
            target = features[-1, 2:4] # [end_x, end_y]
            
            # Input: 마지막 이벤트가 발생하기 전까지의 모든 상황
            # (마지막 행 제외)
            input_seq = features[:-1]
            
            return torch.FloatTensor(input_seq), torch.FloatTensor(target)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

def baseline_collate_fn(batch):
    """
    배치 내의 None 값(에러/짧은 데이터)을 걸러내고 패딩 처리
    """
    # None 제거
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None
    
    inputs, targets = zip(*batch)
    
    # 입력 시퀀스 길이 (pack_padded_sequence용)
    lengths = torch.LongTensor([len(x) for x in inputs])
    
    # Padding (Batch, Max_Len, 5)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    
    # Targets (Batch, 2)
    targets = torch.stack(targets)
    
    return padded_inputs, targets, lengths

def train_and_validate(model, train_loader, val_loader, optimizer, criterion, epochs):
    # Loss가 아닌 '거리 오차'를 기준으로 Best 모델을 판단합니다.
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
    train_dataset = SoccerBaselineDataset('./data/train')
    val_dataset = SoccerBaselineDataset('./data/val')
    
    # 3060 GPU 사용 시 num_workers를 높여 데이터 로딩 병목 해결 (4~8 추천)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, collate_fn=baseline_collate_fn, num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, collate_fn=baseline_collate_fn, num_workers=4, pin_memory=True)
    
    # 모델 초기화 (RTX 3060용 설정)
    model = BaselineLSTM(
        input_size=5, 
        hidden_size=256, 
        num_layers=3, 
        output_size=2, 
        dropout_rate=0.3
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    train_and_validate(model, train_loader, val_loader, optimizer, criterion, EPOCHS)