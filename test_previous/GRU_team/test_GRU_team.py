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
    
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    WEIGHT_DIR = './weights' # 경로 구분
    
    BATCH_SIZE = 256        
    LR = 0.001
    EPOCHS = 50
    NUM_WORKERS = 4
    
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0 # [중요] 누락 방지
    EOS_VALUE = 0.0 
    
    # 모델 파라미터
    NUM_ACTIONS = 33
    MAX_PHASE_LEN = 50 
    ACTION_EMB_DIM = 4
    
    # Team Embedding
    NUM_TEAMS = 35     
    TEAM_EMB_DIM = 4   
    
    INPUT_SIZE = 5       
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 128
    NUM_LAYERS = 1
    
    # Loss 가중치
    LAMBDA_ANGLE = 5.0 

# Team ID 매핑 전역 변수
TEAM_TO_IDX = {}

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
# 1. Model: Team + Functional Length (Deterministic)
# ==========================================
class TeamGRU(nn.Module):
    def __init__(self, 
                 input_size=5, 
                 phase_hidden=64, 
                 episode_hidden=128, 
                 num_layers=1,
                 num_actions=33, 
                 action_emb_dim=4,
                 num_teams=35,
                 team_emb_dim=4):
        super(TeamGRU, self).__init__()
        
        # 1. Embeddings
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.team_embedding = nn.Embedding(num_teams, team_emb_dim)
        
        # Functional Length Encoding (Tanh Scaling)
        self.length_feature_dim = 4
        self.length_encoder = nn.Sequential(
            nn.Linear(1, self.length_feature_dim),
            nn.Tanh() 
        )
        
        # 2. Phase GRU
        self.phase_input_dim = input_size + action_emb_dim + self.length_feature_dim + team_emb_dim
        self.phase_gru = nn.GRU(self.phase_input_dim, phase_hidden, num_layers=num_layers, batch_first=True)
        
        # 3. Episode GRU
        self.episode_input_dim = phase_hidden + 2 
        self.episode_gru = nn.GRU(self.episode_input_dim, episode_hidden, num_layers=num_layers, batch_first=True)
        
        # 4. Regressor Head (Output: 2 -> dx, dy) - Gaussian 제거됨
        self.head = nn.Sequential(
            nn.Linear(episode_hidden, episode_hidden // 2),
            nn.ReLU(),
            nn.Linear(episode_hidden // 2, 2) 
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths, start_action_ids, phase_raw_lens, padded_coords, phase_team_ids):
        # --- (1) Context Encoding ---
        action_emb = self.action_embedding(start_action_ids) 
        team_emb = self.team_embedding(phase_team_ids)       
        
        # Functional Length
        normalized_len = torch.tanh(phase_raw_lens.float().unsqueeze(1) / 20.0) 
        len_feat = self.length_encoder(normalized_len)       
        
        context_vector = torch.cat([action_emb, len_feat, team_emb], dim=1) 
        
        seq_len = padded_phases.size(1)
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        phase_inputs = torch.cat([padded_phases, context_expanded], dim=2)
        
        # --- (2) Phase GRU ---
        packed_phases = pack_padded_sequence(phase_inputs, phase_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, phase_h_n = self.phase_gru(packed_phases)
        phase_embeddings = phase_h_n[-1] 
        
        # --- (3) Episode GRU ---
        phases_per_episode = torch.split(phase_embeddings, episode_lengths.tolist())
        padded_phase_embs = pad_sequence(phases_per_episode, batch_first=True, padding_value=0)
        episode_inputs = torch.cat([padded_phase_embs, padded_coords], dim=2)
        
        packed_episodes = pack_padded_sequence(episode_inputs, episode_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, episode_h_n = self.episode_gru(packed_episodes)
        
        # --- (4) Prediction ---
        pred_delta = self.head(episode_h_n[-1]) # (dx, dy)
        
        # Residual Connection
        batch_size = padded_coords.size(0)
        last_coords = []
        for i in range(batch_size):
            length = episode_lengths[i]
            last_coords.append(padded_coords[i, length-1, :])
        last_known_pos = torch.stack(last_coords)
        
        final_prediction = last_known_pos + pred_delta
        
        return final_prediction, last_known_pos

# ==========================================
# 2. Original Loss (Directional + Distance)
# ==========================================
class DirectionalRealDistanceLoss(nn.Module):
    def __init__(self, max_x=105.0, max_y=68.0, lambda_angle=5.0):
        super(DirectionalRealDistanceLoss, self).__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.epsilon = 1e-6
        self.lambda_angle = lambda_angle

    def forward(self, pred, target, start_pos):
        # 거리 Loss (미터 단위)
        diff_x = (pred[:, 0] - target[:, 0]) * self.max_x
        diff_y = (pred[:, 1] - target[:, 1]) * self.max_y
        distance_loss = torch.sqrt(diff_x**2 + diff_y**2 + self.epsilon).mean()

        # 각도 Loss
        pred_vec = pred - start_pos
        target_vec = target - start_pos
        target_norm = target_vec.norm(dim=1)
        valid_mask = target_norm > (0.1 / self.max_x) 
        
        if valid_mask.sum() > 0:
            cosine_loss = 1 - F.cosine_similarity(pred_vec[valid_mask], target_vec[valid_mask], dim=1)
            angle_loss = cosine_loss.mean()
        else:
            angle_loss = torch.tensor(0.0).to(pred.device)

        total_loss = distance_loss + (self.lambda_angle * angle_loss)
        return total_loss, distance_loss.item()

# ==========================================
# 3. Dataset (동일)
# ==========================================
class SoccerDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        self.action_map = ACTION_TO_IDX
        print(f"📂 Dataset Loaded from {data_dir}: Found {len(self.file_paths)} files.")
    
    def __len__(self): return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            fpath = self.file_paths[idx]
            df = pd.read_csv(fpath)
            if len(df) < 2: return None
            
            target_ex = df['end_x'].values[-1] / Config.MAX_X
            target_ey = df['end_y'].values[-1] / Config.MAX_Y
            target = np.array([target_ex, target_ey])
            
            if 'phase' not in df.columns:
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

            sx = df['start_x'].values / Config.MAX_X
            sy = df['start_y'].values / Config.MAX_Y
            ex = df['end_x'].values / Config.MAX_X
            ey = df['end_y'].values / Config.MAX_Y
            t  = df['time_seconds'].values / Config.MAX_TIME
            dx, dy = ex - sx, ey - sy
            
            features = np.stack([sx, sy, dx, dy, t], axis=1)
            input_features = features[:-1]
            input_df = df.iloc[:-1].copy()
            
            phases_data, start_actions, phase_lens_raw = [], [], []
            phase_end_coords = []
            phase_teams = [] 
            
            for _, group in input_df.groupby('phase', sort=False):
                p_feats = input_features[group.index]
                eos = np.full((1, 5), Config.EOS_VALUE)
                phases_data.append(torch.FloatTensor(np.vstack([p_feats, eos])))
                
                act_name = group.iloc[0]['type_name']
                start_actions.append(self.action_map.get(act_name, DEFAULT_ACTION_IDX))
                phase_lens_raw.append(len(group))
                
                raw_team = group.iloc[0]['team_id']
                team_idx = TEAM_TO_IDX.get(raw_team, 0)
                phase_teams.append(team_idx)
                
                last_x = group.iloc[-1]['end_x'] / Config.MAX_X
                last_y = group.iloc[-1]['end_y'] / Config.MAX_Y
                phase_end_coords.append([last_x, last_y])
            
            target_start_x = sx[-1]
            target_start_y = sy[-1]
            if len(phase_end_coords) > 0:
                phase_end_coords[-1] = [target_start_x, target_start_y]
            else:
                 phase_end_coords.append([target_start_x, target_start_y])
                 
            if not phases_data: return None
            
            return (phases_data, torch.FloatTensor(target), start_actions, 
                    torch.FloatTensor(phase_lens_raw), 
                    torch.FloatTensor(phase_end_coords),
                    torch.LongTensor(phase_teams))
        except: return None

def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return (None,)*7
    
    b_phases, b_targets, b_acts, b_raw_lens, b_coords, b_teams = zip(*batch)
    
    all_phases, all_acts, all_raw_lens, ep_lens, all_teams = [], [], [], [], []
    for i in range(len(b_phases)):
        all_phases.extend(b_phases[i])
        all_acts.extend(b_acts[i])
        all_raw_lens.extend(b_raw_lens[i])
        all_teams.extend(b_teams[i])
        ep_lens.append(len(b_phases[i]))
        
    pad_phases = pad_sequence(all_phases, batch_first=True, padding_value=Config.EOS_VALUE)
    phase_lengths = torch.LongTensor([len(p) for p in all_phases])
    episode_lengths = torch.LongTensor(ep_lens)
    targets = torch.stack(b_targets)
    
    start_action_ids = torch.LongTensor(all_acts)
    phase_raw_lens = torch.FloatTensor(all_raw_lens)
    phase_team_ids = torch.LongTensor(all_teams)
    
    coords_list = [torch.FloatTensor(c) for c in b_coords]
    padded_coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    
    return pad_phases, phase_lengths, episode_lengths, targets, start_action_ids, phase_raw_lens, padded_coords, phase_team_ids

# ==========================================
# 4. Build Team Mapping
# ==========================================
def build_team_mapping(data_dir):
    print("🔄 Scanning Data for Team IDs...")
    files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    unique_teams = set()
    
    for fpath in tqdm(files[:2000], desc="Building Map"): 
        try:
            df = pd.read_csv(fpath)
            if 'team_id' in df.columns:
                unique_teams.update(df['team_id'].unique())
        except: continue
        
    for idx, team_id in enumerate(sorted(list(unique_teams))):
        TEAM_TO_IDX[team_id] = idx
        
    Config.NUM_TEAMS = len(TEAM_TO_IDX) + 1 
    print(f"✅ Found {len(unique_teams)} teams. Map Size: {Config.NUM_TEAMS}")

# ==========================================
# 5. Training Loop
# ==========================================
def run_training():
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    
    build_team_mapping(Config.TRAIN_DIR)
    
    train_dataset = SoccerDataset(Config.TRAIN_DIR)
    val_dataset = SoccerDataset(Config.VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=Config.NUM_WORKERS)
    
    model = TeamGRU(
        input_size=Config.INPUT_SIZE,
        phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        num_layers=Config.NUM_LAYERS,
        num_actions=Config.NUM_ACTIONS,
        num_teams=Config.NUM_TEAMS
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    criterion = DirectionalRealDistanceLoss(max_x=Config.MAX_X, max_y=Config.MAX_Y, lambda_angle=Config.LAMBDA_ANGLE)
    
    best_dist = float('inf')
    
    print("🚀 Training Started (Team + Functional Length + Original Loss)...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_dist_sum = 0.0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            batch = [b.to(Config.DEVICE) if b is not None else None for b in batch]
            if batch[0] is None: continue
            
            optimizer.zero_grad()
            
            # Forward: (pred, start_pos)
            preds, start_pos = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6], batch[7])
            
            # Loss: (total, dist_item)
            loss, dist_v = criterion(preds, batch[3], start_pos)
            
            loss.backward()
            optimizer.step()
            
            train_dist_sum += dist_v
            
        avg_train_dist = train_dist_sum / len(train_loader)
        
        # Validation
        model.eval()
        val_dist_sum = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) if b is not None else None for b in batch]
                if batch[0] is None: continue
                
                preds, start_pos = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6], batch[7])
                _, dist_v = criterion(preds, batch[3], start_pos)
                
                val_dist_sum += dist_v
                count += 1
        
        if count > 0:
            avg_val_dist = val_dist_sum / count
            print(f"   Train Dist: {avg_train_dist:.4f}m | Val Dist: {avg_val_dist:.4f}m")
            
            if avg_val_dist < best_dist:
                best_dist = avg_val_dist
                save_name = "best_team_det.pth"
                torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, save_name))
                print(f"   💾 Best Model Saved ({best_dist:.4f}m)")
        else:
             print("⚠️ No valid validation batches found.")

    torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, "last_team_det.pth"))
    print("🏁 Training Finished.")

if __name__ == '__main__':
    run_training()