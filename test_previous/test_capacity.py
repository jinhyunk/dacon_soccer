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
    DEVICE = torch.device('cpu') 
    
    BATCH_SIZE = 64         
    LR = 0.01               # 시작은 높게 (빠른 수렴 유도)
    EPOCHS = 300            # 충분히 줌
    NUM_WORKERS = 0
    
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    EOS_VALUE = 0.0 
    
    NUM_ACTIONS = 33
    MAX_PHASE_LEN_EMBED = 30
    ACTION_EMB_DIM = 4
    LEN_EMB_DIM = 4
    INPUT_SIZE = 7       
    PHASE_HIDDEN = 64       # 64로 다시 복구 (안정성 우선)
    EPISODE_HIDDEN = 256
    DROPOUT = 0.0           
    
    TRAIN_DIR = './data/train' 

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
# 1. Dataset & Collate (기존 동일)
# ==========================================
class OverfitDataset(Dataset):
    def __init__(self, data_dir, limit=64):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))[:limit] 
        self.action_map = ACTION_TO_IDX
        print(f"🧪 [Overfit Test] Loading only {len(self.file_paths)} samples...")
    
    def __len__(self): return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            df = pd.read_csv(self.file_paths[idx])
            if len(df) < 2: return None
            if 'phase' not in df.columns:
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

            sx = df['start_x'].values / Config.MAX_X
            sy = df['start_y'].values / Config.MAX_Y
            ex = df['end_x'].values / Config.MAX_X
            ey = df['end_y'].values / Config.MAX_Y
            t  = df['time_seconds'].values / Config.MAX_TIME
            dx, dy = ex - sx, ey - sy
            
            feature_list, start_actions, phase_lens, phase_end_coords = [], [], [], []
            
            # 마지막 이벤트를 제외한 나머지로 입력 구성
            for _, group in df.iloc[:-1].groupby('phase', sort=False):
                p_start_x = group.iloc[0]['start_x'] / Config.MAX_X
                p_start_y = group.iloc[0]['start_y'] / Config.MAX_Y
                
                g_sx = group['start_x'].values / Config.MAX_X
                g_sy = group['start_y'].values / Config.MAX_Y
                g_ex = group['end_x'].values / Config.MAX_X
                g_ey = group['end_y'].values / Config.MAX_Y
                g_t  = group['time_seconds'].values / Config.MAX_TIME
                g_dx, g_dy = g_ex - g_sx, g_ey - g_sy
                g_rel_x, g_rel_y = g_sx - p_start_x, g_sy - p_start_y
                
                g_feats = np.stack([g_sx, g_sy, g_dx, g_dy, g_t, g_rel_x, g_rel_y], axis=1)
                eos = np.full((1, 7), Config.EOS_VALUE)
                feature_list.append(torch.FloatTensor(np.vstack([g_feats, eos])))
                
                act_name = group.iloc[0]['type_name']
                start_actions.append(self.action_map.get(act_name, 32))
                phase_lens.append(min(len(group), Config.MAX_PHASE_LEN_EMBED - 1))
                
                last_x = group.iloc[-1]['end_x'] / Config.MAX_X
                last_y = group.iloc[-1]['end_y'] / Config.MAX_Y
                phase_end_coords.append([last_x, last_y])

            if not feature_list: return None
            target = np.array([ex[-1], ey[-1]]) 
            return (feature_list, torch.FloatTensor(target), start_actions, phase_lens, torch.FloatTensor(phase_end_coords))
        except: return None

def collate_fn(batch):
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
    coords_list = [torch.FloatTensor(c) for c in b_coords]
    padded_coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    
    return pad_phases, phase_lengths, episode_lengths, targets, start_action_ids, phase_len_ids, padded_coords

# ==========================================
# 2. Model (Location Aware Hierarchical LSTM)
# ==========================================
class LocationAwareHierarchicalLSTM(nn.Module):
    def __init__(self, input_size=7, phase_hidden=64, episode_hidden=256, output_size=2, dropout=0.0,
                 num_actions=33, max_phase_len=30, action_emb_dim=4, len_emb_dim=4):
        super(LocationAwareHierarchicalLSTM, self).__init__()
        
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.length_embedding = nn.Embedding(max_phase_len, len_emb_dim)
        
        self.phase_input_dim = input_size + action_emb_dim + len_emb_dim
        self.phase_lstm = nn.LSTM(self.phase_input_dim, phase_hidden, num_layers=1, batch_first=True)
        
        self.episode_input_dim = phase_hidden + 2 
        self.episode_lstm = nn.LSTM(self.episode_input_dim, episode_hidden, num_layers=2, batch_first=True, dropout=dropout)
        
        self.regressor = nn.Sequential(
            nn.Linear(episode_hidden, episode_hidden // 2),
            nn.ReLU(),
            nn.Linear(episode_hidden // 2, output_size)
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths, start_action_ids, phase_len_ids, padded_coords):
        action_emb = self.action_embedding(start_action_ids)
        len_emb = self.length_embedding(phase_len_ids)
        context_vector = torch.cat([action_emb, len_emb], dim=1)
        
        seq_len = padded_phases.size(1)
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        phase_inputs = torch.cat([padded_phases, context_expanded], dim=2)
        
        packed_phases = pack_padded_sequence(phase_inputs, phase_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (phase_h_n, _) = self.phase_lstm(packed_phases)
        phase_embeddings = phase_h_n[-1] 
        
        phases_per_episode = torch.split(phase_embeddings, episode_lengths.tolist())
        padded_phase_embs = pad_sequence(phases_per_episode, batch_first=True, padding_value=0)
        
        episode_inputs = torch.cat([padded_phase_embs, padded_coords], dim=2)
        
        packed_episodes = pack_padded_sequence(episode_inputs, episode_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (episode_h_n, _) = self.episode_lstm(packed_episodes)
        
        predicted_remaining_delta = self.regressor(episode_h_n[-1])
        
        batch_size = padded_coords.size(0)
        # 각 에피소드의 '마지막 Phase 종료 좌표'를 추출
        last_coords = [padded_coords[i, episode_lengths[i]-1, :] for i in range(batch_size)]
        last_known_pos = torch.stack(last_coords)
        
        return last_known_pos + predicted_remaining_delta

# ==========================================
# 3. Test Runner
# ==========================================
class RealDistanceLoss(nn.Module):
    def __init__(self, max_x=105.0, max_y=68.0):
        super(RealDistanceLoss, self).__init__()
        self.max_x, self.max_y, self.epsilon = max_x, max_y, 1e-6
    def forward(self, pred, target):
        diff_x = (pred[:, 0] - target[:, 0]) * self.max_x
        diff_y = (pred[:, 1] - target[:, 1]) * self.max_y
        return torch.sqrt(diff_x**2 + diff_y**2 + self.epsilon).mean()

def run_overfit_test():
    print(f"💻 Running Overfit Test on {Config.DEVICE}...")
    
    # 1. Dataset
    dataset = OverfitDataset(Config.TRAIN_DIR, limit=Config.BATCH_SIZE)
    if len(dataset) == 0: return
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # 2. Model
    model = LocationAwareHierarchicalLSTM(
        input_size=Config.INPUT_SIZE,
        phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    # [안정화 1] Scheduler 추가 (Loss가 안 줄면 LR을 팍팍 줄임)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    criterion = RealDistanceLoss(max_x=Config.MAX_X, max_y=Config.MAX_Y)
    
    model.train()
    single_batch = next(iter(loader))
    single_batch = [b.to(Config.DEVICE) for b in single_batch]
    
    print("\n🚀 Start Stabilized Overfitting (Goal: Loss < 0.1m)")
    pbar = tqdm(range(Config.EPOCHS))
    
    for epoch in pbar:
        optimizer.zero_grad()
        
        preds = model(single_batch[0], single_batch[1], single_batch[2], 
                      single_batch[4], single_batch[5], single_batch[6])
        
        loss = criterion(preds, single_batch[3])
        loss.backward()
        
        # [안정화 2] Gradient Clipping (기울기 폭발 방지)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        scheduler.step(loss.item()) # Scheduler Update
        
        pbar.set_description(f"Loss: {loss.item():.4f}m | LR: {optimizer.param_groups[0]['lr']:.5f}")
        
        if loss.item() < 0.1:
            print(f"\n✅ [SUCCESS] Loss reached {loss.item():.4f}m")
            break
            
    if loss.item() >= 0.1:
        print(f"\n❌ [FAIL] Loss stuck at {loss.item():.4f}m.")

if __name__ == "__main__":
    run_overfit_test()