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
    WEIGHT_DIR = './weights'
    
    BATCH_SIZE = 256        
    LR = 0.001
    EPOCHS = 50
    NUM_WORKERS = 4
    
    MAX_X = 105.0
    MAX_Y = 68.0
    EOS_VALUE = 0.0 
    
    # 모델 파라미터
    NUM_ACTIONS = 33
    MAX_PHASE_LEN = 50 # 함수 스케일링 기준값
    ACTION_EMB_DIM = 4
    
    # [NEW] Team Embedding
    NUM_TEAMS = 35     # 데이터에 따라 자동 조정됨
    TEAM_EMB_DIM = 4   
    
    INPUT_SIZE = 5       
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 128
    NUM_LAYERS = 1
    
    # [Loss 가중치]
    LAMBDA_ANGLE = 10.0 # NLL Loss가 작으므로 각도 Loss 가중치를 높임

# Team ID 매핑을 위한 전역 변수
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
# 1. Model: Team + Functional Length + Gaussian
# ==========================================
class GaussianTeamGRU(nn.Module):
    def __init__(self, 
                 input_size=5, 
                 phase_hidden=64, 
                 episode_hidden=128, 
                 num_layers=1,
                 num_actions=33, 
                 action_emb_dim=4,
                 num_teams=35,
                 team_emb_dim=4):
        super(GaussianTeamGRU, self).__init__()
        
        # 1. Embeddings
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.team_embedding = nn.Embedding(num_teams, team_emb_dim)
        
        # [NEW] Length는 별도 Embedding 없이 함수형으로 처리하므로 Linear Layer로 차원만 맞춤
        # 스칼라 값(Length) 하나를 받아서 벡터로 변환
        self.length_feature_dim = 4
        self.length_encoder = nn.Sequential(
            nn.Linear(1, self.length_feature_dim),
            nn.Tanh() # Activation을 통해 비선형성 추가
        )
        
        # 2. Phase GRU
        # Input: Coords(5) + Action(4) + LengthFeature(4) + Team(4)
        self.phase_input_dim = input_size + action_emb_dim + self.length_feature_dim + team_emb_dim
        self.phase_gru = nn.GRU(self.phase_input_dim, phase_hidden, num_layers=num_layers, batch_first=True)
        
        # 3. Episode GRU
        self.episode_input_dim = phase_hidden + 2 
        self.episode_gru = nn.GRU(self.episode_input_dim, episode_hidden, num_layers=num_layers, batch_first=True)
        
        # 4. Gaussian Head (Output: 4 -> mu_x, mu_y, log_var_x, log_var_y)
        self.head = nn.Sequential(
            nn.Linear(episode_hidden, episode_hidden // 2),
            nn.ReLU(),
            nn.Linear(episode_hidden // 2, 4) 
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths, start_action_ids, phase_raw_lens, padded_coords, phase_team_ids):
        # --- (1) Phase Context Encoding ---
        action_emb = self.action_embedding(start_action_ids) # (Total, 4)
        team_emb = self.team_embedding(phase_team_ids)       # (Total, 4)
        
        # [NEW] Functional Length Encoding
        # Tanh Scaling: 길이 20 정도를 기준으로 비선형 변환 (사용자 요청 반영)
        # raw_len / 20.0 -> tanh -> Feature vector
        normalized_len = torch.tanh(phase_raw_lens.float().unsqueeze(1) / 20.0) 
        len_feat = self.length_encoder(normalized_len)       # (Total, 4)
        
        # Context 결합
        context_vector = torch.cat([action_emb, len_feat, team_emb], dim=1) # (Total, 12)
        
        # Sequence 길이만큼 복사 (Broadcasting)
        seq_len = padded_phases.size(1)
        context_expanded = context_vector.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 입력 데이터와 결합
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
        
        # --- (4) Gaussian Prediction ---
        output = self.head(episode_h_n[-1])
        
        pred_mu_delta = output[:, :2]  # (dx, dy)
        pred_log_var = output[:, 2:]   # log(sigma^2)
        
        # Residual Connection for Mu
        batch_size = padded_coords.size(0)
        last_coords = []
        for i in range(batch_size):
            length = episode_lengths[i]
            last_coords.append(padded_coords[i, length-1, :])
        last_known_pos = torch.stack(last_coords)
        
        final_mu = last_known_pos + pred_mu_delta
        
        # Log Var -> Sigma (Standard Deviation)
        # exp(0.5 * log_var)
        pred_sigma = torch.exp(0.5 * pred_log_var)
        
        return final_mu, pred_sigma, last_known_pos

# ==========================================
# 2. Unified Loss (Gaussian NLL + Angle)
# ==========================================
class GaussianAngleLoss(nn.Module):
    def __init__(self, max_x=105.0, max_y=68.0, lambda_angle=10.0):
        super(GaussianAngleLoss, self).__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.lambda_angle = lambda_angle
        # GaussianNLLLoss는 분산을 고려한 거리 오차를 계산해줌 (MSE 대체)
        self.nll_loss = nn.GaussianNLLLoss(reduction='mean', eps=1e-6)

    def forward(self, pred_mu, pred_sigma, target, start_pos):
        # 1. Gaussian NLL Loss (Distance & Uncertainty)
        # Normalized 좌표계(0~1)에서 계산하는 것이 학습 안정성에 좋습니다.
        loss_nll = self.nll_loss(pred_mu, target, pred_sigma**2) # 인자는 Variance(sigma^2)
        
        # 2. Angle Loss (Direction)
        pred_vec = pred_mu - start_pos
        target_vec = target - start_pos
        
        # 벡터 길이가 너무 짧으면 각도가 의미 없으므로 마스킹
        target_norm = target_vec.norm(dim=1)
        valid_mask = target_norm > (0.1 / self.max_x) # 10cm 이상 움직인 경우만
        
        if valid_mask.sum() > 0:
            cosine_sim = F.cosine_similarity(pred_vec[valid_mask], target_vec[valid_mask], dim=1)
            loss_angle = 1 - cosine_sim.mean()
        else:
            loss_angle = torch.tensor(0.0).to(pred_mu.device)
            
        # Total Loss
        total_loss = loss_nll + (self.lambda_angle * loss_angle)
        
        return total_loss, loss_nll.item(), loss_angle.item()

# ==========================================
# 3. Dataset (Updated)
# ==========================================
class SoccerDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        self.action_map = ACTION_TO_IDX
    
    def __len__(self): return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            df = pd.read_csv(self.file_paths[idx])
            if len(df) < 2: return None
            
            # Target
            target_ex = df['end_x'].values[-1] / Config.MAX_X
            target_ey = df['end_y'].values[-1] / Config.MAX_Y
            target = np.array([target_ex, target_ey])
            
            # Phase Grouping
            if 'phase' not in df.columns:
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

            # Features
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
            phase_teams = [] # [NEW] Team ID List
            
            for _, group in input_df.groupby('phase', sort=False):
                p_feats = input_features[group.index]
                eos = np.full((1, 5), Config.EOS_VALUE)
                phases_data.append(torch.FloatTensor(np.vstack([p_feats, eos])))
                
                # Context Info
                act_name = group.iloc[0]['type_name']
                start_actions.append(self.action_map.get(act_name, DEFAULT_ACTION_IDX))
                
                # [NEW] Raw Length 저장 (함수형 입력용)
                phase_lens_raw.append(len(group))
                
                # [NEW] Team ID Mapping
                raw_team = group.iloc[0]['team_id']
                team_idx = TEAM_TO_IDX.get(raw_team, 0) # 매핑 없으면 0
                phase_teams.append(team_idx)
                
                last_x = group.iloc[-1]['end_x'] / Config.MAX_X
                last_y = group.iloc[-1]['end_y'] / Config.MAX_Y
                phase_end_coords.append([last_x, last_y])
            
            # Residual 연결을 위한 마지막 위치 (Target Start)
            target_start_x = sx[-1]
            target_start_y = sy[-1]
            if len(phase_end_coords) > 0:
                phase_end_coords[-1] = [target_start_x, target_start_y]
            else:
                 phase_end_coords.append([target_start_x, target_start_y])
                 
            if not phases_data: return None
            
            return (phases_data, torch.FloatTensor(target), start_actions, 
                    torch.FloatTensor(phase_lens_raw), # 스칼라 길이 반환
                    torch.FloatTensor(phase_end_coords),
                    torch.LongTensor(phase_teams))     # 팀 ID 반환
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
    phase_raw_lens = torch.FloatTensor(all_raw_lens) # Float for Function input
    phase_team_ids = torch.LongTensor(all_teams)
    
    coords_list = [torch.FloatTensor(c) for c in b_coords]
    padded_coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    
    return pad_phases, phase_lengths, episode_lengths, targets, start_action_ids, phase_raw_lens, padded_coords, phase_team_ids

# ==========================================
# 4. Build Team Mapping
# ==========================================
def build_team_mapping(data_dir):
    print("🔄 Scanning Data for Team IDs...")
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    unique_teams = set()
    
    # 1000개 정도만 샘플링해서 스캔 (속도 최적화)
    # 실제로는 전체를 다 훑는게 안전하지만, 팀 ID는 보통 초반에 다 등장합니다.
    for fpath in tqdm(files[:2000], desc="Building Map"): 
        try:
            df = pd.read_csv(fpath)
            if 'team_id' in df.columns:
                unique_teams.update(df['team_id'].unique())
        except: continue
        
    # Mapping 생성
    for idx, team_id in enumerate(sorted(list(unique_teams))):
        TEAM_TO_IDX[team_id] = idx
        
    Config.NUM_TEAMS = len(TEAM_TO_IDX) + 1 # 미확인 팀 대비 +1
    print(f"✅ Found {len(unique_teams)} teams. Map Size: {Config.NUM_TEAMS}")

# ==========================================
# 5. Training Loop
# ==========================================
def run_training():
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    
    # 1. Team Mapping
    build_team_mapping(Config.TRAIN_DIR)
    
    # 2. Dataset
    train_dataset = SoccerDataset(Config.TRAIN_DIR)
    val_dataset = SoccerDataset(Config.VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=Config.NUM_WORKERS)
    
    # 3. Model
    model = GaussianTeamGRU(
        input_size=Config.INPUT_SIZE,
        phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        num_layers=Config.NUM_LAYERS,
        num_actions=Config.NUM_ACTIONS,
        num_teams=Config.NUM_TEAMS
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    criterion = GaussianAngleLoss(max_x=Config.MAX_X, max_y=Config.MAX_Y, lambda_angle=Config.LAMBDA_ANGLE)
    
    best_metric = float('inf')
    
    print("🚀 Training Started with Gaussian + Team Embedding...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_sum = 0.0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            batch = [b.to(Config.DEVICE) if b is not None else None for b in batch]
            if batch[0] is None: continue
            
            optimizer.zero_grad()
            
            # Forward: (mu, sigma, start_pos)
            # Args: phases, p_lens, e_lens, actions, RAW_LENS, coords, TEAMS
            mu, sigma, start_pos = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6], batch[7])
            
            # Loss: target is batch[3]
            loss, nll_v, ang_v = criterion(mu, sigma, batch[3], start_pos)
            
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_dist_metric = 0.0 # 실제 거리 오차 (m)
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) if b is not None else None for b in batch]
                if batch[0] is None: continue
                
                mu, sigma, start_pos = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6], batch[7])
                loss, nll_v, ang_v = criterion(mu, sigma, batch[3], start_pos)
                val_loss_sum += loss.item()
                
                # Metric Calculation (Real Distance in Meters)
                # mu와 target은 normalized 상태이므로 Max 거리 곱해서 계산
                diff_x = (mu[:, 0] - batch[3][:, 0]) * Config.MAX_X
                diff_y = (mu[:, 1] - batch[3][:, 1]) * Config.MAX_Y
                dist = torch.sqrt(diff_x**2 + diff_y**2).mean().item()
                
                val_dist_metric += dist
                count += 1
        if count > 0:
            avg_val_loss = val_loss_sum / count
            avg_val_dist = val_dist_metric / count
        else:
            print("⚠️ Warning: Validation set yielded no valid batches.")
            avg_val_loss = 0.0
            avg_val_dist = float('inf')        
        
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   >>> Val Metric (Distance): {avg_val_dist:.4f}m") # 14~18m 나오는 그 지표
        
        # Best Model Save (거리 기준)
        if avg_val_dist < best_metric:
            best_metric = avg_val_dist
            torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, "best_gaussian_team.pth"))
            print(f"   💾 Best Model Saved ({best_metric:.4f}m)")
            
    # Last Model Save
    torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, "last_gaussian_team.pth"))
    print("🏁 Training Finished.")

if __name__ == '__main__':
    run_training()