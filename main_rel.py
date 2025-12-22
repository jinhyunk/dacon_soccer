import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 256        
    LR = 0.001
    MAX_LR = 0.01
    EPOCHS = 50
    NUM_WORKERS = 4
    
    # 데이터 상수
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    EOS_VALUE = 0.0 
    
    # 모델 파라미터
    NUM_ACTIONS = 33
    MAX_PHASE_LEN_EMBED = 30
    ACTION_EMB_DIM = 4
    LEN_EMB_DIM = 4
    
    INPUT_SIZE = 5       # Phase LSTM Input
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 512
    DROPOUT = 0.3        
    
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    WEIGHT_DIR = './weight'

ACTION_TO_IDX = {
    'Aerial Clearance': 0, 'Block': 1, 'Carry': 2, 'Catch': 3, 'Clearance': 4,
    'Cross': 5, 'Deflection': 6, 'Duel': 7, 'Error': 8, 'Foul': 9,
    'Foul_Throw': 10, 'Goal': 11, 'Goal Kick': 12, 'Handball_Foul': 13,
    'Hit': 14, 'Interception': 15, 'Intervention': 16, 'Offside': 17,
    'Out': 18, 'Own Goal': 19, 'Parry': 20, 'Pass': 21, 'Pass_Corner': 22,
    'Pass_Freekick': 23, 'Penalty Kick': 24, 'Recovery': 25, 'Shot': 26,
    'Shot_Corner': 27, 'Shot_Freekick': 28, 'Tackle': 29, 'Take-On': 30,
    'Throw-In': 31, 'Other': 32
}

# ==========================================
# 2. Dataset (Location Aware)
# ==========================================
class LocationAwareDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        self.action_map = ACTION_TO_IDX
    
    def __len__(self): return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            df = pd.read_csv(self.file_paths[idx])
            if len(df) < 2: return None
            if 'phase' not in df.columns:
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

            # 정규화
            sx = df['start_x'].values / Config.MAX_X
            sy = df['start_y'].values / Config.MAX_Y
            ex = df['end_x'].values / Config.MAX_X
            ey = df['end_y'].values / Config.MAX_Y
            t  = df['time_seconds'].values / Config.MAX_TIME
            
            # 상대 좌표 (Delta)
            dx = ex - sx
            dy = ey - sy
            
            # Phase Input Features
            features = np.stack([sx, sy, dx, dy, t], axis=1)
            target = np.array([ex[-1], ey[-1]]) 
            
            input_features = features[:-1]
            input_df = df.iloc[:-1].copy()
            
            phases_data, start_actions, phase_lens = [], [], []
            phase_end_coords = [] # [NEW] 각 Phase가 끝난 위치 저장
            
            for _, group in input_df.groupby('phase', sort=False):
                p_feats = input_features[group.index]
                eos = np.full((1, 5), Config.EOS_VALUE)
                phases_data.append(torch.FloatTensor(np.vstack([p_feats, eos])))
                
                act_name = group.iloc[0]['type_name']
                start_actions.append(self.action_map.get(act_name, 32))
                phase_lens.append(min(len(group), Config.MAX_PHASE_LEN_EMBED - 1))
                
                # [NEW] 이 Phase의 마지막 종료 위치 (Normalized)
                last_x = group.iloc[-1]['end_x'] / Config.MAX_X
                last_y = group.iloc[-1]['end_y'] / Config.MAX_Y
                phase_end_coords.append([last_x, last_y])
                
            if not phases_data: return None
            
            return (phases_data, torch.FloatTensor(target), start_actions, phase_lens, torch.FloatTensor(phase_end_coords))
        except: return None

def location_aware_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return (None,)*6
    
    b_phases, b_targets, b_acts, b_lens, b_coords = zip(*batch)
    
    all_phases, all_acts, all_lens_ids, ep_lens = [], [], [], []
    for i in range(len(b_phases)):
        all_phases.extend(b_phases[i])
        all_acts.extend(b_acts[i])
        all_lens_ids.extend(b_lens[i])
        ep_lens.append(len(b_phases[i]))
        
    pad_phases = pad_sequence(all_phases, batch_first=True, padding_value=Config.EOS_VALUE)
    phase_lengths = torch.LongTensor([len(p) for p in all_phases])
    episode_lengths = torch.LongTensor(ep_lens)
    targets = torch.stack(b_targets)
    start_action_ids = torch.LongTensor(all_acts)
    phase_len_ids = torch.LongTensor(all_lens_ids)
    
    # [NEW] Phase End Coords Padding (Batch, Max_Ep_Len, 2)
    # Episode LSTM 입력용이므로 Episode 길이만큼 패딩해야 함
    coords_list = [torch.FloatTensor(c) for c in b_coords]
    padded_coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    
    return pad_phases, phase_lengths, episode_lengths, targets, start_action_ids, phase_len_ids, padded_coords

# ==========================================
# 3. Model (Location Aware Hierarchical LSTM)
# ==========================================
class LocationAwareHierarchicalLSTM(nn.Module):
    def __init__(self, input_size=5, phase_hidden=64, episode_hidden=256, output_size=2, dropout=0.3,
                 num_actions=33, max_phase_len=30, action_emb_dim=4, len_emb_dim=4):
        super(LocationAwareHierarchicalLSTM, self).__init__()
        
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.length_embedding = nn.Embedding(max_phase_len, len_emb_dim)
        
        # 1. Phase LSTM
        self.phase_input_dim = input_size + action_emb_dim + len_emb_dim
        self.phase_lstm = nn.LSTM(self.phase_input_dim, phase_hidden, num_layers=1, batch_first=True)
        
        # 2. Episode LSTM (Input Size 증가!)
        # 입력: [Phase_Summary(64) + Phase_End_Coord(2)]
        self.episode_input_dim = phase_hidden + 2 
        self.episode_lstm = nn.LSTM(self.episode_input_dim, episode_hidden, num_layers=2, batch_first=True, dropout=dropout)
        
        self.regressor = nn.Sequential(
            nn.Linear(episode_hidden, episode_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(episode_hidden // 2, output_size)
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths, start_action_ids, phase_len_ids, padded_coords):
        """
        padded_coords: (Batch, Max_Ep_Len, 2) - 각 Phase가 끝난 실제 좌표
        """
        # --- A. Phase Level ---
        action_emb = self.action_embedding(start_action_ids)
        len_emb = self.length_embedding(phase_len_ids)
        context_vector = torch.cat([action_emb, len_emb], dim=1)
        
        seq_len = padded_phases.size(1)
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        phase_inputs = torch.cat([padded_phases, context_expanded], dim=2)
        
        packed_phases = pack_padded_sequence(phase_inputs, phase_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (phase_h_n, _) = self.phase_lstm(packed_phases)
        phase_embeddings = phase_h_n[-1] # (Total_Phases, Phase_Hidden)
        
        # --- B. Episode Level Preparation ---
        # 1. Phase Embedding을 Episode 단위로 다시 묶음
        phases_per_episode = torch.split(phase_embeddings, episode_lengths.tolist())
        padded_phase_embs = pad_sequence(phases_per_episode, batch_first=True, padding_value=0)
        
        # 2. [핵심] Phase Summary + 실제 좌표 결합
        # padded_phase_embs: (Batch, Ep_Len, 64)
        # padded_coords:     (Batch, Ep_Len, 2)
        # -> episode_inputs: (Batch, Ep_Len, 66)
        episode_inputs = torch.cat([padded_phase_embs, padded_coords], dim=2)
        
        # --- C. Episode LSTM ---
        packed_episodes = pack_padded_sequence(episode_inputs, episode_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (episode_h_n, _) = self.episode_lstm(packed_episodes)
        
        # --- D. Residual Prediction ---
        # 모델은 "마지막 Phase가 끝난 지점"에서 "얼마나 더 가는지"를 예측
        predicted_remaining_delta = self.regressor(episode_h_n[-1])
        
        # 마지막 Phase의 실제 끝 위치 추출 (Batch, 2)
        # padded_coords에서 각 배치의 마지막 유효한 값 가져오기
        batch_size = padded_coords.size(0)
        last_coords = []
        for i in range(batch_size):
            length = episode_lengths[i]
            last_coords.append(padded_coords[i, length-1, :])
        last_known_pos = torch.stack(last_coords)
        
        final_prediction = last_known_pos + predicted_remaining_delta
        
        return final_prediction

# ==========================================
# 4. Training Engine
# ==========================================
def run_training():
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    print(f"✅ Device: {Config.DEVICE}")
    print("📂 데이터 로드 중 (Location Aware)...")
    
    train_dataset = LocationAwareDataset(Config.TRAIN_DIR)
    val_dataset = LocationAwareDataset(Config.VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, collate_fn=location_aware_collate_fn, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, collate_fn=location_aware_collate_fn, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    model = LocationAwareHierarchicalLSTM(
        input_size=Config.INPUT_SIZE,
        phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # [안정화 1] AdamW
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    
    # [안정화 2] Scheduler (CosineAnnealingWarmRestarts 추천)
    # 전체 학습에서는 Local Minima 탈출을 위해 Cosine Annealing이 유리합니다.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=Config.EPOCHS,
        pct_start=0.3, # 30% 시점까지 LR 상승
        div_factor=25.0
    )
    
    criterion_train = nn.MSELoss()

    def dist_loss_fn(pred, target):
        diff_x = (pred[:, 0] - target[:, 0]) * Config.MAX_X
        diff_y = (pred[:, 1] - target[:, 1]) * Config.MAX_Y
        return torch.sqrt(diff_x**2 + diff_y**2 + 1e-6).mean()
    
    best_dist_error = float('inf')
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}"):
            batch = [b.to(Config.DEVICE) for b in batch]
            if batch[0] is None: continue
            
            optimizer.zero_grad()
            # Input index: 6 is padded_coords
            preds = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6])
            
            loss = criterion_train(preds, batch[3])
            loss.backward()

            optimizer.step()
            scheduler.step() # Batch마다 호출
            train_loss += loss.item()
        

        avg_train = train_loss / len(train_loader)
        
        model.eval()
        val_dist_error = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) for b in batch]
                if batch[0] is None: continue
                preds = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6])
                val_dist_error += dist_loss_fn(preds, batch[3]).item()
        
        avg_train_mse = train_loss / len(train_loader)
        avg_val_dist = val_dist_error / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   Train(MSE): {avg_train_mse:.5f} | Val(Dist): {avg_val_dist:.4f}m | LR: {current_lr:.6f}")

        # if avg_val < best_dist_error:
        #     best_dist_error = avg_val
        #     save_name = f"location_aware_dist{best_dist_error:.4f}m.pth"
        #     torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, save_name))
        #     print(f"   💾 Best Model Saved: {save_name}")

if __name__ == "__main__":
    run_training()