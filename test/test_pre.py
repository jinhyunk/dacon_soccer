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
# 1. Configuration (Light & Optimized)
# ==========================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pre-train 설정
    PRETRAIN_EPOCHS = 10     # Phase 패턴 학습은 금방 됨
    PRETRAIN_BATCH = 512     # Phase는 짧아서 배치를 키워도 됨
    PRETRAIN_LR = 0.001
    
    # Fine-tune 설정
    FINETUNE_EPOCHS = 50
    FINETUNE_BATCH = 128     # 일반화 성능을 위해 배치 줄임
    FINETUNE_LR = 0.001
    
    # 공통 설정
    NUM_WORKERS = 4
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    EOS_VALUE = -1.0
    EOS_ACTION_ID = 32
    
    # 임베딩 & 차원
    NUM_ACTIONS = 33
    MAX_PHASE_LEN_EMBED = 30
    ACTION_EMB_DIM = 4
    LEN_EMB_DIM = 4
    DOMAIN_INPUT_DIM = 2
    
    # 모델 하이퍼파라미터 (최적화된 Light 버전)
    INPUT_SIZE = 5
    PHASE_HIDDEN = 64        # Phase 정보를 담기에 적절
    EPISODE_HIDDEN = 128     # 과적합 방지
    DROPOUT = 0.1            # 규제 완화 (학습 촉진)
    EPISODE_LAYERS = 1       # 레이어 깊이 축소
    
    # 경로
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    WEIGHT_DIR = './weight'

# Action Map
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
# 2. Datasets
# ==========================================

# [Dataset 1] Pre-training용 (Phase 단위 Flat 데이터)
# ==========================================
# 2. Datasets (Fix: Index Out of Bounds)
# ==========================================

# [Dataset 1] Pre-training용
class PhasePretrainDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        self.action_map = ACTION_TO_IDX
        self.all_phases = []

        print(f"🔄 [Pre-train] Loading Phase Data from {data_dir}...")
        for fpath in tqdm(self.file_paths):
            try:
                df = pd.read_csv(fpath)
                if len(df) < 2: continue
                if 'phase' not in df.columns:
                     df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()
                
                # 정규화
                df['start_x'] /= Config.MAX_X
                df['start_y'] /= Config.MAX_Y
                df['end_x'] /= Config.MAX_X
                df['end_y'] /= Config.MAX_Y
                df['time_seconds'] /= Config.MAX_TIME
                
                features = df[['start_x', 'start_y', 'end_x', 'end_y', 'time_seconds']].values
                
                for _, group in df.groupby('phase', sort=False):
                    phase_data = features[group.index]
                    actions = [self.action_map.get(n, 32) for n in group['type_name']]
                    
                    target = phase_data[-1, 2:4]
                    
                    # [수정됨] 30 -> Config.MAX_PHASE_LEN_EMBED - 1 (즉, 29)
                    # 인덱스는 0부터 시작하므로 크기가 30이면 최대 인덱스는 29여야 함
                    len_idx = min(len(group), Config.MAX_PHASE_LEN_EMBED - 1)
                    
                    self.all_phases.append({
                        'data': torch.FloatTensor(phase_data),
                        'actions': torch.LongTensor(actions),
                        'target': torch.FloatTensor(target),
                        'len_id': len_idx 
                    })
            except: continue
            
    def __len__(self): return len(self.all_phases)
    def __getitem__(self, idx):
        item = self.all_phases[idx]
        return item['data'], item['actions'], item['target'], item['len_id']

# [Dataset 2] Fine-tuning용
class SoccerCompleteDataset(Dataset):
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

            sx = df['start_x'].values / Config.MAX_X
            sy = df['start_y'].values / Config.MAX_Y
            ex = df['end_x'].values / Config.MAX_X
            ey = df['end_y'].values / Config.MAX_Y
            t  = df['time_seconds'].values / Config.MAX_TIME
            
            features = np.stack([sx, sy, ex, ey, t], axis=1)
            target = features[-1, 2:4]
            
            input_features = features[:-1]
            input_df = df.iloc[:-1].copy()
            
            phases_data, phases_actions, phase_lens, domain_feats = [], [], [], []
            
            for _, group in input_df.groupby('phase', sort=False):
                p_feats = input_features[group.index]
                eos = np.full((1, 5), Config.EOS_VALUE)
                phases_data.append(torch.FloatTensor(np.vstack([p_feats, eos])))
                
                act = [self.action_map.get(n, 32) for n in group['type_name']]
                act.append(Config.EOS_ACTION_ID)
                phases_actions.append(torch.LongTensor(act))
                
                # [수정됨] 여기도 안전하게 -1 적용
                phase_lens.append(min(len(group), Config.MAX_PHASE_LEN_EMBED - 1))
                
                p_start_x, p_end_x = group.iloc[0]['start_x'], group.iloc[-1]['end_x']
                dur = group.iloc[-1]['time_seconds'] - group.iloc[0]['time_seconds']
                
                dx = (p_end_x - p_start_x) / Config.MAX_X
                norm_dur = dur / 30.0
                domain_feats.append(np.array([dx, norm_dur]))
                
            if not phases_data: return None
            return (phases_data, phases_actions, torch.FloatTensor(target), 
                    phase_lens, torch.FloatTensor(np.array(domain_feats)))
        except: return None

def pretrain_collate_fn(batch):
    data, actions, targets, lens = zip(*batch)
    padded_data = pad_sequence(data, batch_first=True, padding_value=0)
    padded_actions = pad_sequence(actions, batch_first=True, padding_value=Config.EOS_ACTION_ID)
    lengths = torch.LongTensor([len(x) for x in data])
    targets = torch.stack(targets)
    len_ids = torch.LongTensor(lens)
    return padded_data, padded_actions, lengths, targets, len_ids

# [Dataset 2] Fine-tuning용 (Episode 단위 Hierarchical 데이터)
class SoccerCompleteDataset(Dataset):
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

            sx = df['start_x'].values / Config.MAX_X
            sy = df['start_y'].values / Config.MAX_Y
            ex = df['end_x'].values / Config.MAX_X
            ey = df['end_y'].values / Config.MAX_Y
            t  = df['time_seconds'].values / Config.MAX_TIME
            
            features = np.stack([sx, sy, ex, ey, t], axis=1)
            target = features[-1, 2:4]
            
            input_features = features[:-1]
            input_df = df.iloc[:-1].copy()
            
            phases_data, phases_actions, phase_lens, domain_feats = [], [], [], []
            
            for _, group in input_df.groupby('phase', sort=False):
                p_feats = input_features[group.index]
                eos = np.full((1, 5), Config.EOS_VALUE)
                phases_data.append(torch.FloatTensor(np.vstack([p_feats, eos])))
                
                act = [self.action_map.get(n, 32) for n in group['type_name']]
                act.append(Config.EOS_ACTION_ID)
                phases_actions.append(torch.LongTensor(act))
                
                # [수정됨] 여기도 안전하게 -1 적용
                phase_lens.append(min(len(group), Config.MAX_PHASE_LEN_EMBED - 1))
                
                p_start_x, p_end_x = group.iloc[0]['start_x'], group.iloc[-1]['end_x']
                dur = group.iloc[-1]['time_seconds'] - group.iloc[0]['time_seconds']
                
                dx = (p_end_x - p_start_x) / Config.MAX_X
                norm_dur = dur / 30.0
                domain_feats.append(np.array([dx, norm_dur]))
                
            if not phases_data: return None
            return (phases_data, phases_actions, torch.FloatTensor(target), 
                    phase_lens, torch.FloatTensor(np.array(domain_feats)))
        except: return None

def complete_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return (None,)*7
    
    b_phases, b_actions, b_targets, b_lens, b_domain = zip(*batch)
    
    all_phases, all_actions, ep_lens, all_lens_ids, all_domain = [], [], [], [], []
    for i in range(len(b_phases)):
        all_phases.extend(b_phases[i])
        all_actions.extend(b_actions[i])
        ep_lens.append(len(b_phases[i]))
        all_lens_ids.extend(b_lens[i])
        all_domain.extend(b_domain[i])
        
    pad_phases = pad_sequence(all_phases, batch_first=True, padding_value=0)
    pad_actions = pad_sequence(all_actions, batch_first=True, padding_value=Config.EOS_ACTION_ID)
    phase_lens = torch.LongTensor([len(p) for p in all_phases])
    ep_lens = torch.LongTensor(ep_lens)
    targets = torch.stack(b_targets)
    lens_ids = torch.LongTensor(all_lens_ids)
    domain = torch.stack([torch.FloatTensor(d) for d in all_domain])
    
    return pad_phases, pad_actions, phase_lens, ep_lens, targets, lens_ids, domain

# ==========================================
# 3. Model Definition
# ==========================================
class CompleteHierarchicalLSTM(nn.Module):
    def __init__(self, input_size, phase_hidden, episode_hidden, output_size=2, dropout=0.3,
                 num_actions=33, action_emb_dim=4, max_phase_len=30, len_emb_dim=4, 
                 domain_input_dim=2, episode_layers=1):
        super(CompleteHierarchicalLSTM, self).__init__()
        
        # Embeddings
        self.action_embedding = nn.Embedding(num_actions, action_emb_dim)
        self.length_embedding = nn.Embedding(max_phase_len, len_emb_dim)
        
        # Phase LSTM
        self.phase_input_dim = input_size + action_emb_dim + len_emb_dim
        self.phase_lstm = nn.LSTM(self.phase_input_dim, phase_hidden, num_layers=1, batch_first=True)
        
        # Episode LSTM
        self.episode_input_dim = phase_hidden + domain_input_dim
        self.episode_lstm = nn.LSTM(self.episode_input_dim, episode_hidden, num_layers=episode_layers, 
                                    batch_first=True, dropout=dropout if episode_layers > 1 else 0)
        
        # Head
        self.regressor = nn.Sequential(
            nn.Linear(episode_hidden, episode_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(episode_hidden // 2, output_size)
        )

    def forward(self, padded_phases, padded_actions, phase_lengths, episode_lengths, phase_len_ids, domain_features):
        # 1. Phase Encoding
        action_emb = self.action_embedding(padded_actions)
        len_emb = self.length_embedding(phase_len_ids).unsqueeze(1).expand(-1, padded_phases.size(1), -1)
        phase_inputs = torch.cat([padded_phases, action_emb, len_emb], dim=2)
        
        packed_phases = pack_padded_sequence(phase_inputs, phase_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (phase_h_n, _) = self.phase_lstm(packed_phases)
        phase_embeddings = phase_h_n[-1] 
        
        # 2. Episode Encoding
        augmented_features = torch.cat([phase_embeddings, domain_features], dim=1)
        features_per_episode = torch.split(augmented_features, episode_lengths.tolist())
        padded_episodes = pad_sequence(features_per_episode, batch_first=True, padding_value=0)
        
        packed_episodes = pack_padded_sequence(padded_episodes, episode_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (episode_h_n, _) = self.episode_lstm(packed_episodes)
        
        return self.regressor(episode_h_n[-1])

# ==========================================
# 4. Execution Functions
# ==========================================

class RealDistanceLoss(nn.Module):
    def __init__(self, max_x=105.0, max_y=68.0):
        super(RealDistanceLoss, self).__init__()
        self.max_x = max_x
        self.max_y = max_y
        self.epsilon = 1e-6 # 0으로 나누기 방지용

    def forward(self, pred, target):
        """
        pred, target: (Batch, 2) - Normalized [0, 1]
        """
        # 1. 실제 미터 단위로 변환 (역정규화 아님, 차이만 계산)
        diff_x = (pred[:, 0] - target[:, 0]) * self.max_x
        diff_y = (pred[:, 1] - target[:, 1]) * self.max_y
        
        # 2. 유클리드 거리 계산 (Distance)
        # sqrt(x^2 + y^2)
        distance = torch.sqrt(diff_x**2 + diff_y**2 + self.epsilon)
        
        # 3. 평균 거리 반환 (Avg Distance)
        return distance.mean()
    

def run_pretraining(model, train_loader):
    print("\n🚀 [Step 1] Starting Pre-training (Phase Encoder)...")
    
    # Pre-training용 임시 Head (Phase Hidden -> Coordinate)
    # 모델의 phase_hidden 사이즈를 가져옴
    hidden_dim = model.phase_lstm.hidden_size
    temp_head = nn.Linear(hidden_dim, 2).to(Config.DEVICE)
    
    # Optimizer: Phase LSTM, Embeddings, Temp Head만 학습
    optimizer = optim.Adam(
        list(model.phase_lstm.parameters()) + 
        list(model.action_embedding.parameters()) + 
        list(model.length_embedding.parameters()) +
        list(temp_head.parameters()),
        lr=Config.PRETRAIN_LR
    )
    criterion = RealDistanceLoss(max_x=Config.MAX_X, max_y=Config.MAX_Y)
    
    model.train()
    temp_head.train()
    
    for epoch in range(Config.PRETRAIN_EPOCHS):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{Config.PRETRAIN_EPOCHS}"):
            # phases, actions, lengths, targets, len_ids
            phases, actions, lens, targets, len_ids = [b.to(Config.DEVICE) for b in batch]
            
            optimizer.zero_grad()
            
            # --- Manual Forward Pass (Phase Only) ---
            act_emb = model.action_embedding(actions)
            l_emb = model.length_embedding(len_ids).unsqueeze(1).expand(-1, phases.size(1), -1)
            inputs = torch.cat([phases, act_emb, l_emb], dim=2)
            
            packed = pack_padded_sequence(inputs, lens.cpu(), batch_first=True, enforce_sorted=False)
            _, (hn, _) = model.phase_lstm(packed)
            phase_vec = hn[-1]
            
            preds = temp_head(phase_vec)
            # ----------------------------------------
            
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"   Pretrain Loss: {total_loss / len(train_loader):.5f}")
    
    print("✅ Pre-training Completed.")
    return model

def run_finetuning(model, train_loader, val_loader):
    print("\n❄️ [Step 2] Freezing Phase Encoder & Starting Fine-tuning...")
    
    # 1. Freeze Phase Modules
    for name, param in model.named_parameters():
        if 'phase' in name or 'embedding' in name:
            param.requires_grad = False
    
    # 2. Optimizer (Only Unfrozen Parameters)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=Config.FINETUNE_LR, weight_decay=1e-5)
    criterion = RealDistanceLoss(max_x=Config.MAX_X, max_y=Config.MAX_Y)

    best_dist = float('inf')
    
    for epoch in range(Config.FINETUNE_EPOCHS):
        # --- Train ---
        model.train() 
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{Config.FINETUNE_EPOCHS}"):
            batch = [b.to(Config.DEVICE) for b in batch]
            if batch[0] is None: continue
            
            optimizer.zero_grad()
            
            # [수정] *batch 대신 인덱스로 입력 지정 (targets인 batch[4] 제외)
            # 순서: phases(0), actions(1), p_lens(2), ep_lens(3), len_ids(5), domain(6)
            preds = model(batch[0], batch[1], batch[2], batch[3], batch[5], batch[6])
            
            loss = criterion(preds, batch[4]) # batch[4] is targets
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # --- Validation ---
        model.eval()
        val_loss = 0
        total_dist = 0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) for b in batch]
                if batch[0] is None: continue
                
                # [수정] 검증 때도 동일하게 입력
                preds = model(batch[0], batch[1], batch[2], batch[3], batch[5], batch[6])
                
                loss = criterion(preds, batch[4])
                val_loss += loss.item()
                
                # Metric
                px, py = preds[:,0]*Config.MAX_X, preds[:,1]*Config.MAX_Y
                tx, ty = batch[4][:,0]*Config.MAX_X, batch[4][:,1]*Config.MAX_Y
                dist = torch.sqrt((px-tx)**2 + (py-ty)**2).sum().item()
                total_dist += dist
                count += batch[4].size(0)
                
        avg_dist = total_dist / count if count > 0 else 0
        print(f"   Train Loss: {train_loss/len(train_loader):.5f} | Val Loss: {val_loss/len(val_loader):.5f}")
        print(f"   📏 Avg Distance: {avg_dist:.4f} meters")
        
        # if avg_dist < best_dist:
        #     best_dist = avg_dist
        #     os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
        #     torch.save(model.state_dict(), f"{Config.WEIGHT_DIR}/best_unified_model.pth")
        #     print(f"   💾 Best Model Saved ({best_dist:.4f}m)")

# ==========================================
# 5. Main Execution Block
# ==========================================
if __name__ == "__main__":
    print(f"✅ Device: {Config.DEVICE}")
    
    # 1. Dataset Init
    pretrain_ds = PhasePretrainDataset(Config.TRAIN_DIR)
    pretrain_loader = DataLoader(pretrain_ds, batch_size=Config.PRETRAIN_BATCH, 
                                 shuffle=True, collate_fn=pretrain_collate_fn, num_workers=Config.NUM_WORKERS)
    
    train_ds = SoccerCompleteDataset(Config.TRAIN_DIR)
    val_ds = SoccerCompleteDataset(Config.VAL_DIR)
    train_loader = DataLoader(train_ds, batch_size=Config.FINETUNE_BATCH, 
                              shuffle=True, collate_fn=complete_collate_fn, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=Config.FINETUNE_BATCH, 
                            shuffle=False, collate_fn=complete_collate_fn, num_workers=Config.NUM_WORKERS)
    
    # 2. Model Init
    model = CompleteHierarchicalLSTM(
        input_size=Config.INPUT_SIZE,
        phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        dropout=Config.DROPOUT,
        episode_layers=Config.EPISODE_LAYERS,
        domain_input_dim=Config.DOMAIN_INPUT_DIM
    ).to(Config.DEVICE)
    
    # 3. Execute
    # Phase 1: Pre-training (In-place update)
    model = run_pretraining(model, pretrain_loader)
    
    # Phase 2: Fine-tuning (With Freeze)
    run_finetuning(model, train_loader, val_loader)