import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # [경로 설정] Adversarial Validation으로 나눈 데이터 경로
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    WEIGHT_DIR = './weights'
    
    BATCH_SIZE = 256        
    LR = 0.001
    EPOCHS = 50
    NUM_WORKERS = 4
    
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    EOS_VALUE = 0.0 
    
    NUM_ACTIONS = 33
    MAX_PHASE_LEN_EMBED = 30
    ACTION_EMB_DIM = 4
    LEN_EMB_DIM = 4
    
    INPUT_SIZE = 5       
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 256
    DROPOUT = 0.3

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
DEFAULT_ACTION_IDX = 32

# ==========================================
# 1. Uncertainty Model
# ==========================================
class UncertaintyLocationAwareLSTM(nn.Module):
    def __init__(self, input_size=5, phase_hidden=64, episode_hidden=256, output_size=2, dropout=0.3,
                 num_actions=33, max_phase_len=30, action_emb_dim=4, len_emb_dim=4):
        super(UncertaintyLocationAwareLSTM, self).__init__()
        
        # Embeddings
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.length_embedding = nn.Embedding(max_phase_len, len_emb_dim)
        
        # Phase LSTM
        self.phase_input_dim = input_size + action_emb_dim + len_emb_dim
        self.phase_lstm = nn.LSTM(self.phase_input_dim, phase_hidden, num_layers=1, batch_first=True)
        
        # Episode LSTM
        self.episode_input_dim = phase_hidden + 2 
        self.episode_lstm = nn.LSTM(self.episode_input_dim, episode_hidden, num_layers=2, batch_first=True, dropout=dropout)
        
        # [MODIFIED] Regressor Head
        # Output: 4 (mu_x, mu_y, log_var_x, log_var_y)
        self.regressor = nn.Sequential(
            nn.Linear(episode_hidden, episode_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(episode_hidden // 2, output_size * 2) # 2 -> 4
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths, start_action_ids, phase_len_ids, padded_coords):
        # 1. Context Embedding
        action_emb = self.action_embedding(start_action_ids)
        len_emb = self.length_embedding(phase_len_ids)
        context_vector = torch.cat([action_emb, len_emb], dim=1)
        
        seq_len = padded_phases.size(1)
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        phase_inputs = torch.cat([padded_phases, context_expanded], dim=2)
        
        # 2. Phase LSTM
        packed_phases = pack_padded_sequence(phase_inputs, phase_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (phase_h_n, _) = self.phase_lstm(packed_phases)
        phase_embeddings = phase_h_n[-1] 
        
        # 3. Episode LSTM
        phases_per_episode = torch.split(phase_embeddings, episode_lengths.tolist())
        padded_phase_embs = pad_sequence(phases_per_episode, batch_first=True, padding_value=0)
        episode_inputs = torch.cat([padded_phase_embs, padded_coords], dim=2)
        
        packed_episodes = pack_padded_sequence(episode_inputs, episode_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (episode_h_n, _) = self.episode_lstm(packed_episodes)
        
        # 4. Uncertainty Prediction
        # output shape: (Batch, 4) -> [mu_x, mu_y, log_var_x, log_var_y]
        raw_output = self.regressor(episode_h_n[-1])
        
        mu_delta = raw_output[:, :2]      # 예측된 이동량 (Delta)
        log_var = raw_output[:, 2:]       # 불확실성 (Log Variance)
        
        # Log Variance -> Standard Deviation (Sigma) 변환
        # exp(0.5 * log_var) = sqrt(var) = sigma
        pred_sigma = torch.exp(0.5 * log_var)
        
        # 5. Residual Logic (좌표에만 적용)
        batch_size = padded_coords.size(0)
        last_coords = []
        for i in range(batch_size):
            length = episode_lengths[i]
            last_coords.append(padded_coords[i, length-1, :])
        last_known_pos = torch.stack(last_coords)
        
        final_mu = last_known_pos + mu_delta
        
        return final_mu, pred_sigma

# ==========================================
# 2. Uncertainty Loss (Gaussian NLL)
# ==========================================
class UncertaintyLoss(nn.Module):
    def __init__(self, max_x=105.0, max_y=68.0):
        super(UncertaintyLoss, self).__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.epsilon = 1e-6

    def forward(self, pred_mu, pred_sigma, target):
        """
        Calculates Gaussian Negative Log Likelihood.
        Loss = 0.5 * (log(sigma^2) + (target - mu)^2 / sigma^2)
        """
        # 실제 거리(m) 기준으로 Loss를 계산하여 물리적 의미를 부여합니다.
        # (Normalized 상태에서 계산해도 되지만, Sigma 해석을 위해 미터 단위 변환)
        
        diff_x = (pred_mu[:, 0] - target[:, 0]) * self.max_x
        diff_y = (pred_mu[:, 1] - target[:, 1]) * self.max_y
        
        # Sigma도 미터 단위로 스케일링한다고 가정 (Network가 Normalized Sigma를 뱉으므로)
        sigma_x = pred_sigma[:, 0] * self.max_x
        sigma_y = pred_sigma[:, 1] * self.max_y
        
        # Variance
        var_x = sigma_x ** 2
        var_y = sigma_y ** 2
        
        # Gaussian NLL Formula
        loss_x = 0.5 * (torch.log(var_x + self.epsilon) + (diff_x**2 / (var_x + self.epsilon)))
        loss_y = 0.5 * (torch.log(var_y + self.epsilon) + (diff_y**2 / (var_y + self.epsilon)))
        
        return (loss_x + loss_y).mean()

# ==========================================
# 3. Dataset (Original Location Aware)
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
            
            # Ground Truth Target
            target_ex = df['end_x'].values[-1] / Config.MAX_X
            target_ey = df['end_y'].values[-1] / Config.MAX_Y
            
            if 'phase' not in df.columns:
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

            sx = df['start_x'].values / Config.MAX_X
            sy = df['start_y'].values / Config.MAX_Y
            ex = df['end_x'].values / Config.MAX_X
            ey = df['end_y'].values / Config.MAX_Y
            t  = df['time_seconds'].values / Config.MAX_TIME
            
            target_start_x = sx[-1]
            target_start_y = sy[-1]
            
            dx = ex - sx; dy = ey - sy
            features = np.stack([sx, sy, dx, dy, t], axis=1)
            target = np.array([target_ex, target_ey])
            
            input_features = features[:-1]
            input_df = df.iloc[:-1].copy()
            
            phases_data, start_actions, phase_lens = [], [], []
            phase_end_coords = []
            
            for _, group in input_df.groupby('phase', sort=False):
                p_feats = input_features[group.index]
                eos = np.full((1, 5), Config.EOS_VALUE)
                phases_data.append(torch.FloatTensor(np.vstack([p_feats, eos])))
                
                act_name = group.iloc[0]['type_name']
                start_actions.append(self.action_map.get(act_name, DEFAULT_ACTION_IDX))
                phase_lens.append(min(len(group), Config.MAX_PHASE_LEN_EMBED - 1))
                
                last_x = group.iloc[-1]['end_x'] / Config.MAX_X
                last_y = group.iloc[-1]['end_y'] / Config.MAX_Y
                phase_end_coords.append([last_x, last_y])
            
            if len(phase_end_coords) > 0:
                phase_end_coords[-1] = [target_start_x, target_start_y]
            else:
                 phase_end_coords.append([target_start_x, target_start_y])
                 
            if not phases_data: return None
            
            return (phases_data, torch.FloatTensor(target), start_actions, phase_lens, torch.FloatTensor(phase_end_coords))
        except: return None

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return (None,)*5
    
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
# 4. Training Engine
# ==========================================
def run_training():
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    print(f"✅ Device: {Config.DEVICE}")
    print("📂 Uncertainty Loss 학습 시작...")
    
    train_dataset = LocationAwareDataset(Config.TRAIN_DIR)
    val_dataset = LocationAwareDataset(Config.VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_fn, num_workers=Config.NUM_WORKERS)
    
    model = UncertaintyLocationAwareLSTM(
        input_size=Config.INPUT_SIZE, phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN, dropout=Config.DROPOUT,
        num_actions=Config.NUM_ACTIONS
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = UncertaintyLoss(max_x=Config.MAX_X, max_y=Config.MAX_Y)
    
    best_dist = float('inf')
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_accum = 0.0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            batch = [b.to(Config.DEVICE) if b is not None else None for b in batch]
            if batch[0] is None: continue
            
            optimizer.zero_grad()
            
            # Forward: Return (mu, sigma)
            pred_mu, pred_sigma = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6])
            
            # Loss Calculation (Uncertainty)
            loss = criterion(pred_mu, pred_sigma, batch[3])
            
            loss.backward()
            optimizer.step()
            
            # 기록은 보기 편하게 Loss 값 자체로 (또는 별도로 거리 오차 계산 가능)
            train_loss_accum += loss.item()
            
        avg_train_loss = train_loss_accum / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_accum = 0.0
        val_dist_sum = 0.0 # 실제 거리 오차(m)도 확인
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) if b is not None else None for b in batch]
                if batch[0] is None: continue
                
                pred_mu, pred_sigma = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6])
                
                loss = criterion(pred_mu, pred_sigma, batch[3])
                val_loss_accum += loss.item()
                
                # 순수 거리 오차 (Metric 확인용)
                diff_x = (pred_mu[:, 0] - batch[3][:, 0]) * Config.MAX_X
                diff_y = (pred_mu[:, 1] - batch[3][:, 1]) * Config.MAX_Y
                dist = torch.sqrt(diff_x**2 + diff_y**2).mean()
                val_dist_sum += dist.item()
                count += 1
                
        avg_val_loss = val_loss_accum / len(val_loader)
        avg_val_dist = val_dist_sum / count
        
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dist: {avg_val_dist:.4f}m")
        
        # 모델 저장은 '거리 오차' 기준으로 하는 것이 좋습니다.
        if avg_val_dist < best_dist:
            best_dist = avg_val_dist
            save_name = "Gaussian_best.pth"
            torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, save_name))
            print(f"   💾 Best Saved: {best_dist:.4f}m")

    last_save_name = "Gaussian_last.pth"
    torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, last_save_name))
    print(f"   🏁 Last Model Saved: {last_save_name}")


if __name__ == '__main__':
    run_training()