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
    
    # [경로 설정] Adversarial Validation으로 나눈 데이터 경로 권장
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    WEIGHT_DIR = './weights'
    
    BATCH_SIZE = 256        
    LR = 0.001
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
    
    # [NEW] Team Embedding 관련
    NUM_TEAMS = 30     # 자동 감지되지만 초기값 설정
    TEAM_EMB_DIM = 4   # 팀 임베딩 차원
    
    INPUT_SIZE = 5       
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 256
    DROPOUT = 0.3
    
    # Multi-Task Classification Class 개수
    NUM_CLASSES = 3  # 0:Goal/Attack, 1:Side/Out, 2:Inner

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

# 전역 변수로 팀 매핑 딕셔너리 선언 (run_training에서 초기화)
TEAM_TO_IDX = {}

# ==========================================
# 1. Multi-Task Model (With Team Embedding)
# ==========================================
class LocationAwareHierarchicalLSTM(nn.Module):
    def __init__(self, 
                 input_size=5, 
                 phase_hidden=64, 
                 episode_hidden=256, 
                 output_size=2, 
                 dropout=0.3,
                 num_actions=33, 
                 max_phase_len=30, 
                 action_emb_dim=4, 
                 len_emb_dim=4,
                 num_teams=30,      # [NEW]
                 team_emb_dim=4,    # [NEW]
                 num_classes=3      # [NEW]
                 ):
        super(LocationAwareHierarchicalLSTM, self).__init__()
        
        # Embeddings
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.length_embedding = nn.Embedding(max_phase_len, len_emb_dim)
        self.team_embedding = nn.Embedding(num_teams, team_emb_dim) # [NEW]
        
        # Phase LSTM Input: 좌표(5) + Action(4) + Length(4) + Team(4)
        self.phase_input_dim = input_size + action_emb_dim + len_emb_dim + team_emb_dim
        self.phase_lstm = nn.LSTM(self.phase_input_dim, phase_hidden, num_layers=1, batch_first=True)
        
        # Episode LSTM: Phase_Summary(64) + Last_Coord(2)
        self.episode_input_dim = phase_hidden + 2 
        self.episode_lstm = nn.LSTM(self.episode_input_dim, episode_hidden, num_layers=2, batch_first=True, dropout=dropout)
        
        # [Task 1] Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(episode_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes) # Logits
        )
        
        # [Task 2] Regressor Head (With Class Probabilities Hint)
        self.regressor = nn.Sequential(
            nn.Linear(episode_hidden + num_classes, 128), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )

    def forward(self, padded_phases, phase_lengths, episode_lengths, start_action_ids, phase_len_ids, padded_coords, phase_team_ids):
        # 1. Context Embeddings
        action_emb = self.action_embedding(start_action_ids)
        len_emb = self.length_embedding(phase_len_ids)
        team_emb = self.team_embedding(phase_team_ids) # [NEW]
        
        # Context Vector 결합
        context_vector = torch.cat([action_emb, len_emb, team_emb], dim=1)
        
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
        
        episode_context = episode_h_n[-1]
        
        # --- Multi-Task Heads ---
        # A. Classification
        class_logits = self.classifier(episode_context)
        class_probs = F.softmax(class_logits, dim=1)
        
        # B. Regression (Context + Probabilities)
        reg_input = torch.cat([episode_context, class_probs], dim=1)
        predicted_remaining_delta = self.regressor(reg_input)
        
        # Last Known Position Logic
        batch_size = padded_coords.size(0)
        last_coords = []
        for i in range(batch_size):
            length = episode_lengths[i]
            last_coords.append(padded_coords[i, length-1, :])
        last_known_pos = torch.stack(last_coords)
        
        final_prediction = last_known_pos + predicted_remaining_delta
        
        return final_prediction, class_logits, last_known_pos

# ==========================================
# 2. Loss & Labeling
# ==========================================
def get_zone_label(ex, ey):
    """
    EDA 기반 구역 라벨링 (Normalized Input 0~1)
    """
    abs_x = ex * 105.0
    abs_y = ey * 68.0
    
    # Zone 0: Shooting / Goal Threat (중앙 공격)
    # X > 85m (Deep), 20m < Y < 48m (Central)
    if (abs_x > 85.0) and (20.0 < abs_y < 48.0):
        return 0 

    # Zone 1: Sideline / Out (위/아래 띠 + 골킥 지역)
    # 위쪽 띠 (Y > 61), 아래쪽 띠 (Y < 7)
    if (abs_y > 61.0) or (abs_y < 7.0):
        return 1
    # 너무 깊은 엔드라인 근처 (Goal Kick)
    if abs_x > 102.0:
        return 1

    # Zone 2: Inner Field (나머지)
    return 2

class MultiTaskLoss(nn.Module):
    def __init__(self, max_x=105.0, max_y=68.0, lambda_angle=5.0, lambda_cls=0.5):
        super(MultiTaskLoss, self).__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.lambda_angle = lambda_angle
        self.lambda_cls = lambda_cls 
        self.cls_criterion = nn.CrossEntropyLoss()
        self.epsilon = 1e-6

    def forward(self, pred_coords, pred_logits, target_coords, target_labels, start_pos):
        # 1. Regression Loss
        diff_x = (pred_coords[:, 0] - target_coords[:, 0]) * self.max_x
        diff_y = (pred_coords[:, 1] - target_coords[:, 1]) * self.max_y
        dist_loss = torch.sqrt(diff_x**2 + diff_y**2 + self.epsilon).mean()
        
        pred_vec = pred_coords - start_pos
        target_vec = target_coords - start_pos
        target_norm = target_vec.norm(dim=1)
        valid_mask = target_norm > (0.1 / self.max_x)
        
        if valid_mask.sum() > 0:
            cosine_loss = 1 - F.cosine_similarity(pred_vec[valid_mask], target_vec[valid_mask], dim=1)
            angle_loss = cosine_loss.mean()
        else:
            angle_loss = torch.tensor(0.0).to(pred_coords.device)
            
        reg_loss = dist_loss + (self.lambda_angle * angle_loss)
        
        # 2. Classification Loss
        cls_loss = self.cls_criterion(pred_logits, target_labels)
        
        return reg_loss + (self.lambda_cls * cls_loss), dist_loss.item(), cls_loss.item()

# ==========================================
# 3. Dataset (With Team ID)
# ==========================================
class MultiTaskTeamDataset(Dataset):
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
            target_label = get_zone_label(target_ex, target_ey)
            
            # Preprocessing
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
            phase_teams = [] # [NEW]
            
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
                
                # [NEW] Team ID Mapping
                raw_team_id = group.iloc[0]['team_id']
                team_idx = TEAM_TO_IDX.get(raw_team_id, 0)
                phase_teams.append(team_idx)
            
            if len(phase_end_coords) > 0:
                phase_end_coords[-1] = [target_start_x, target_start_y]
            else:
                 phase_end_coords.append([target_start_x, target_start_y])
                 
            if not phases_data: return None
            
            return (phases_data, torch.FloatTensor(target), start_actions, phase_lens, 
                    torch.FloatTensor(phase_end_coords), torch.LongTensor([target_label]),
                    torch.LongTensor(phase_teams)) # [NEW] Return Team IDs
        except: return None

def multitask_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return (None,)*8
    
    b_phases, b_targets, b_acts, b_lens, b_coords, b_labels, b_teams = zip(*batch)
    
    all_phases, all_acts, all_lens_ids, ep_lens, all_teams = [], [], [], [], []
    for i in range(len(b_phases)):
        all_phases.extend(b_phases[i])
        all_acts.extend(b_acts[i])
        all_lens_ids.extend(b_lens[i])
        all_teams.extend(b_teams[i]) # [NEW]
        ep_lens.append(len(b_phases[i]))
        
    pad_phases = pad_sequence(all_phases, batch_first=True, padding_value=Config.EOS_VALUE)
    phase_lengths = torch.LongTensor([len(p) for p in all_phases])
    episode_lengths = torch.LongTensor(ep_lens)
    targets = torch.stack(b_targets)
    start_action_ids = torch.LongTensor(all_acts)
    phase_len_ids = torch.LongTensor(all_lens_ids)
    
    coords_list = [torch.FloatTensor(c) for c in b_coords]
    padded_coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    
    labels = torch.cat(b_labels)
    phase_team_ids = torch.LongTensor(all_teams) # [NEW]
    
    return pad_phases, phase_lengths, episode_lengths, targets, start_action_ids, phase_len_ids, padded_coords, labels, phase_team_ids

# ==========================================
# 4. Training Engine
# ==========================================
def build_team_mapping(data_dir):
    """ 데이터 폴더를 스캔하여 Unique Team ID를 찾고 매핑을 생성 """
    print("🔄 Building Team ID Mapping...")
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    unique_teams = set()
    
    # 너무 느릴 경우, 샘플링해서 읽거나 train.csv 원본에서 한 번에 읽는 것 권장
    # 여기서는 안전하게 앞부분 1000개 파일만 스캔 (팀이 보통 초반에 다 나오므로)
    for fpath in tqdm(files[:2000], desc="Scanning Teams"): 
        try:
            df = pd.read_csv(fpath)
            if 'team_id' in df.columns:
                unique_teams.update(df['team_id'].unique())
        except: continue
        
    # Mapping 생성
    for idx, team_id in enumerate(sorted(list(unique_teams))):
        TEAM_TO_IDX[team_id] = idx
        
    Config.NUM_TEAMS = len(TEAM_TO_IDX) + 1 # 미확인 팀 대비 +1
    print(f"✅ Found {len(unique_teams)} teams. Map: {TEAM_TO_IDX}")

def run_training():
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    print(f"✅ Device: {Config.DEVICE}")
    
    # 0. 팀 매핑 생성 (Train 데이터 기준)
    build_team_mapping(Config.TRAIN_DIR)
    
    print("📂 Multi-Task + Team Aware 학습 시작...")
    
    train_dataset = MultiTaskTeamDataset(Config.TRAIN_DIR)
    val_dataset = MultiTaskTeamDataset(Config.VAL_DIR)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                              collate_fn=multitask_collate_fn, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                            collate_fn=multitask_collate_fn, num_workers=Config.NUM_WORKERS)
    
    model = LocationAwareHierarchicalLSTM(
        input_size=Config.INPUT_SIZE, phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN, dropout=Config.DROPOUT,
        num_actions=Config.NUM_ACTIONS, num_classes=Config.NUM_CLASSES,
        num_teams=Config.NUM_TEAMS, team_emb_dim=Config.TEAM_EMB_DIM
    ).to(Config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    criterion = MultiTaskLoss(max_x=Config.MAX_X, max_y=Config.MAX_Y)
    
    best_dist = float('inf')
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_dist_sum = 0.0
        train_cls_sum = 0.0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            batch = [b.to(Config.DEVICE) if b is not None else None for b in batch]
            if batch[0] is None: continue
            
            optimizer.zero_grad()
            # preds, logits, start_pos
            # Forward 인자: phases, lens, ep_lens, actions, len_ids, coords, TEAMS
            preds, logits, start_pos = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6], batch[8])
            
            # Loss
            loss, dist_v, cls_v = criterion(preds, logits, batch[3], batch[7], start_pos)
            
            loss.backward()
            optimizer.step()
            
            train_dist_sum += dist_v
            train_cls_sum += cls_v
            
        avg_dist = train_dist_sum / len(train_loader)
        avg_cls = train_cls_sum / len(train_loader)
        
        # Validation
        model.eval()
        val_dist_sum = 0.0
        val_acc_sum = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) if b is not None else None for b in batch]
                if batch[0] is None: continue
                
                preds, logits, start_pos = model(batch[0], batch[1], batch[2], batch[4], batch[5], batch[6], batch[8])
                _, dist_v, _ = criterion(preds, logits, batch[3], batch[7], start_pos)
                
                val_dist_sum += dist_v
                
                pred_cls = torch.argmax(logits, dim=1)
                acc = (pred_cls == batch[7]).float().mean().item()
                val_acc_sum += acc
                count += 1
                
        val_avg_dist = val_dist_sum / count
        val_avg_acc = val_acc_sum / count
        
        print(f"   Train Dist: {avg_dist:.2f}m, Cls Loss: {avg_cls:.4f} | Val Dist: {val_avg_dist:.2f}m, Acc: {val_avg_acc*100:.1f}%")
        
        if val_avg_dist < best_dist:
            best_dist = val_avg_dist
            save_name = "multitask_team_best.pth"
            torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, save_name))
            print(f"   💾 Best Saved: {best_dist:.4f}m")
    
    last_save_name = "multitask_team_last.pth"
    torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, last_save_name))
    print(f"   🏁 Last Model Saved: {last_save_name}")


if __name__ == '__main__':
    run_training()