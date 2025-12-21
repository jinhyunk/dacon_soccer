import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 256        
    LR = 0.001
    EPOCHS = 50
    NUM_WORKERS = 0 # 멈춤 현상 방지
    
    # 데이터 정규화 상수
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    EOS_VALUE = 0.0 # Delta 값의 패딩은 0이 자연스러움 (이동 없음)
    
    # 임베딩 & 모델 파라미터
    NUM_ACTIONS = 33
    MAX_PHASE_LEN_EMBED = 30
    ACTION_EMB_DIM = 4
    LEN_EMB_DIM = 4
    
    INPUT_SIZE = 5       # [sx, sy, dx, dy, t]
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 256
    DROPOUT = 0.3        # 데이터가 적으므로 규제 유지
    
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    WEIGHT_DIR = './weight'

# Action Dictionary
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
# 1. Custom Loss (Real Distance)
# ==========================================
class RealDistanceLoss(nn.Module):
    def __init__(self, max_x=105.0, max_y=68.0):
        super(RealDistanceLoss, self).__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.epsilon = 1e-6

    def forward(self, pred, target):
        # pred, target: Normalized absolute coordinates [0, 1]
        diff_x = (pred[:, 0] - target[:, 0]) * self.max_x
        diff_y = (pred[:, 1] - target[:, 1]) * self.max_y
        distance = torch.sqrt(diff_x**2 + diff_y**2 + self.epsilon)
        return distance.mean()

# ==========================================
# 2. Dataset (Relative Coordinates)
# ==========================================
class SoccerRelativeDataset(Dataset):
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

            # 1. 정규화 (Normalization)
            sx = df['start_x'].values / Config.MAX_X
            sy = df['start_y'].values / Config.MAX_Y
            ex = df['end_x'].values / Config.MAX_X
            ey = df['end_y'].values / Config.MAX_Y
            t  = df['time_seconds'].values / Config.MAX_TIME
            
            # 2. [핵심] 상대 좌표 (Delta) 계산
            # 절대 좌표(sx, sy)는 현재 위치 정보를 위해 유지
            # 도착 좌표(ex, ey) 대신 이동 벡터(dx, dy) 사용
            dx = ex - sx
            dy = ey - sy
            
            # Input Features: [start_x, start_y, delta_x, delta_y, time]
            features = np.stack([sx, sy, dx, dy, t], axis=1)
            
            # Target: 에피소드 마지막의 '절대 좌표' (Loss 계산용)
            # (모델은 내부적으로 이동 흐름을 배우고, 최종적으로 절대 위치를 맞추도록 학습)
            target = np.array([ex[-1], ey[-1]]) 
            
            # 3. Phase 분리
            input_features = features[:-1]
            input_df = df.iloc[:-1].copy()
            
            phases_data, start_actions, phase_lens = [], [], []
            
            for _, group in input_df.groupby('phase', sort=False):
                p_feats = input_features[group.index]
                
                # Padding (EOS)
                eos = np.full((1, 5), Config.EOS_VALUE)
                phases_data.append(torch.FloatTensor(np.vstack([p_feats, eos])))
                
                # Context Info
                act_name = group.iloc[0]['type_name']
                start_actions.append(self.action_map.get(act_name, 32))
                phase_lens.append(min(len(group), Config.MAX_PHASE_LEN_EMBED - 1))
                
            if not phases_data: return None
            
            return (phases_data, torch.FloatTensor(target), start_actions, phase_lens)
        except: return None

def relative_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return (None,)*6
    
    b_phases, b_targets, b_acts, b_lens = zip(*batch)
    
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
    
    return pad_phases, phase_lengths, episode_lengths, targets, start_action_ids, phase_len_ids

# ==========================================
# 3. Model (ContextAwareHierarchicalLSTM)
# ==========================================
class ContextAwareHierarchicalLSTM(nn.Module):
    def __init__(self, input_size=5, phase_hidden=64, episode_hidden=256, output_size=2, dropout=0.3,
                 num_actions=33, max_phase_len=30, action_emb_dim=4, len_emb_dim=4):
        super(ContextAwareHierarchicalLSTM, self).__init__()
        
        # Context Embeddings
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.length_embedding = nn.Embedding(max_phase_len, len_emb_dim)
        
        # Phase LSTM
        self.phase_input_dim = input_size + action_emb_dim + len_emb_dim
        self.phase_lstm = nn.LSTM(self.phase_input_dim, phase_hidden, num_layers=1, batch_first=True)
        
        # Episode LSTM
        self.episode_lstm = nn.LSTM(phase_hidden, episode_hidden, num_layers=2, batch_first=True, dropout=dropout)
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(episode_hidden, episode_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(episode_hidden // 2, output_size)
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths, start_action_ids, phase_len_ids):
        # 1. Context Injection
        action_emb = self.action_embedding(start_action_ids)
        len_emb = self.length_embedding(phase_len_ids)
        context_vector = torch.cat([action_emb, len_emb], dim=1)
        
        seq_len = padded_phases.size(1)
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        phase_inputs = torch.cat([padded_phases, context_expanded], dim=2)
        
        # 2. Phase Encoding
        packed_phases = pack_padded_sequence(phase_inputs, phase_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (phase_h_n, _) = self.phase_lstm(packed_phases)
        phase_embeddings = phase_h_n[-1] 
        
        # 3. Episode Encoding
        phases_per_episode = torch.split(phase_embeddings, episode_lengths.tolist())
        padded_episodes = pad_sequence(phases_per_episode, batch_first=True, padding_value=0)
        
        packed_episodes = pack_padded_sequence(padded_episodes, episode_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (episode_h_n, _) = self.episode_lstm(packed_episodes)
        
        # 4. Prediction
        return self.regressor(episode_h_n[-1])

# ==========================================
# 4. Training Engine
# ==========================================
def run_training():
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    print(f"✅ Device: {Config.DEVICE}")
    print("📂 데이터 로드 중 (Relative Coordinates)...")
    
    train_dataset = SoccerRelativeDataset(Config.TRAIN_DIR)
    val_dataset = SoccerRelativeDataset(Config.VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, collate_fn=relative_collate_fn, 
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, collate_fn=relative_collate_fn, 
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # 모델 초기화
    model = ContextAwareHierarchicalLSTM(
        input_size=Config.INPUT_SIZE,
        phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = RealDistanceLoss(max_x=Config.MAX_X, max_y=Config.MAX_Y)
    
    best_dist_error = float('inf')
    
    for epoch in range(Config.EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]"):
            batch = [b.to(Config.DEVICE) for b in batch]
            if batch[0] is None: continue
            
            optimizer.zero_grad()
            # padded_phases, phase_lengths, episode_lengths, targets(index 3), start_action_ids, phase_len_ids
            # model input: phases, p_lens, ep_lens, act_ids, l_ids
            preds = model(batch[0], batch[1], batch[2], batch[4], batch[5])
            
            loss = criterion(preds, batch[3]) # targets
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train = train_loss / len(train_loader)
        
        # --- Valid ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) for b in batch]
                if batch[0] is None: continue
                
                preds = model(batch[0], batch[1], batch[2], batch[4], batch[5])
                loss = criterion(preds, batch[3])
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        
        print(f"   Train Loss: {avg_train:.4f}m | Val Loss: {avg_val:.4f}m")
        
        if avg_val < best_dist_error:
            best_dist_error = avg_val
            save_name = f"relative_action_dist{best_dist_error:.4f}m.pth"
            save_path = os.path.join(Config.WEIGHT_DIR, save_name)
            torch.save(model.state_dict(), save_path)
            print(f"   💾 Best Model Saved: {save_name}")

if __name__ == "__main__":
    run_training()