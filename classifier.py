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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    TRAIN_DIR = './open_track1' # train.csv 경로
    MATCH_INFO_PATH = './open_track1/match_info.csv'
    WEIGHT_DIR = './weights_cls'
    
    BATCH_SIZE = 256
    LR = 0.001
    EPOCHS = 20 # 분류 문제는 금방 수렴함
    NUM_WORKERS = 4
    
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    
    # 모델 파라미터 (TeamGRU와 동일한 Encoder 구조 권장)
    NUM_ACTIONS = 33
    ACTION_EMB_DIM = 4
    NUM_TEAMS = 35
    TEAM_EMB_DIM = 4
    
    INPUT_SIZE = 5
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 128
    NUM_LAYERS = 1
    
    # Output Class (Lose=0, Draw=1, Win=2)
    NUM_CLASSES = 3

# 전역 변수 (Team ID 매핑)
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
# 1. Logic: Score Reconstruction from Kick-off
# ==========================================
def label_game_state(train_df, match_df):
    print("🔄 Reconstructing Scores based on Kick-off Logic...")
    
    # 1. Match Info 병합 (Home/Away ID 확인용)
    train_df = train_df.merge(match_df[['game_id', 'home_team_id', 'away_team_id']], on='game_id', how='left')
    
    # 2. 정렬 (매우 중요)
    train_df = train_df.sort_values(['game_id', 'period_id', 'time_seconds'])
    
    # 3. Phase 단위로 첫 번째 이벤트만 추출하여 '킥오프 여부' 판단
    # (Phase가 없으면 생성)
    if 'phase' not in train_df.columns:
        train_df['phase'] = (train_df['team_id'] != train_df['team_id'].shift(1)).fillna(0).cumsum()
    
    # 각 Phase의 첫 번째 행 추출
    phase_starts = train_df.groupby(['game_id', 'phase']).first().reset_index()
    
    # 4. 킥오프 탐지 조건
    # 조건: Action == Pass AND 위치가 중앙(52.5, 34) 근처
    # 관용 범위(Tolerance): 중앙에서 2m 이내라고 가정
    is_pass = phase_starts['type_name'] == 'Pass'
    center_x, center_y = 52.5, 34.0
    dist_from_center = np.sqrt((phase_starts['start_x'] - center_x)**2 + (phase_starts['start_y'] - center_y)**2)
    is_center = dist_from_center < 3.0 # 3미터 이내 오차 허용
    
    phase_starts['is_kickoff'] = is_pass & is_center
    
    # 5. 스코어 추적 루프
    # 벡터 연산이 어렵기 때문에 게임별로 순회 (데이터 크기가 크지 않아 가능)
    
    phase_starts['home_score'] = 0
    phase_starts['away_score'] = 0
    
    # tqdm을 위해 game_id 별로 그룹핑
    game_groups = phase_starts.groupby('game_id')
    
    results = []
    
    for g_id, group in tqdm(game_groups, desc="Labeling Scores"):
        group = group.sort_values('time_seconds')
        
        curr_h = 0
        curr_a = 0
        
        # 이전 period 추적 (하프타임 구분용)
        prev_period = -1
        
        for idx, row in group.iterrows():
            # 기간(전반/후반)이 바뀌면 -> 하프타임 킥오프 (골 아님)
            if row['period_id'] != prev_period:
                prev_period = row['period_id']
                # 점수 유지, 다음 루프로
                pass
            
            # 기간 중인데 킥오프다? -> 직전에 골이 터짐
            elif row['is_kickoff']:
                # 누가 킥오프? -> 실점한 팀이 킥오프
                kicker_team = row['team_id']
                home_team = row['home_team_id']
                
                if kicker_team == home_team:
                    # 홈팀이 킥오프 = 어웨이팀이 득점
                    curr_a += 1
                else:
                    # 어웨이팀이 킥오프 = 홈팀이 득점
                    curr_h += 1
            
            # 현재 스코어 기록
            results.append({
                'phase': row['phase'],
                'current_home_score': curr_h,
                'current_away_score': curr_a
            })
            
    score_df = pd.DataFrame(results)
    
    # 6. 원본 데이터에 스코어 병합
    train_df = train_df.merge(score_df, on='phase', how='left')
    train_df['current_home_score'] = train_df['current_home_score'].fillna(method='ffill').fillna(0)
    train_df['current_away_score'] = train_df['current_away_score'].fillna(method='ffill').fillna(0)
    
    # 7. Win/Draw/Lose 라벨링 (내 팀 기준)
    # 0: Lose, 1: Draw, 2: Win
    
    is_home_team = (train_df['team_id'] == train_df['home_team_id'])
    
    my_score = np.where(is_home_team, train_df['current_home_score'], train_df['current_away_score'])
    opp_score = np.where(is_home_team, train_df['current_away_score'], train_df['current_home_score'])
    
    score_diff = my_score - opp_score
    
    conditions = [
        (score_diff < 0), # Lose
        (score_diff == 0), # Draw
        (score_diff > 0)  # Win
    ]
    choices = [0, 1, 2]
    
    train_df['game_state'] = np.select(conditions, choices, default=1)
    
    print("\n📊 Game State Label Distribution:")
    print(train_df.groupby('phase')['game_state'].first().value_counts().rename({0:'Lose', 1:'Draw', 2:'Win'}))
    
    return train_df

# ==========================================
# 2. Dataset
# ==========================================
class GameStateDataset(Dataset):
    def __init__(self, df):
        self.data = df
        self.action_map = ACTION_TO_IDX
        
        # Phase 별로 그룹핑하여 리스트로 변환 (메모리 효율을 위해 인덱싱 사용 권장하지만 여기선 직관적으로)
        self.phases = []
        
        # 필요한 컬럼만 추출하여 그룹핑 (속도 최적화)
        cols = ['phase', 'start_x', 'start_y', 'end_x', 'end_y', 'time_seconds', 
                'type_name', 'team_id', 'game_state']
        
        grouped = self.data[cols].groupby('phase', sort=False)
        
        for _, group in tqdm(grouped, desc="Building Dataset"):
            if len(group) < 1: continue
            
            # Features
            sx = group['start_x'].values / Config.MAX_X
            sy = group['start_y'].values / Config.MAX_Y
            ex = group['end_x'].values / Config.MAX_X
            ey = group['end_y'].values / Config.MAX_Y
            t  = group['time_seconds'].values / Config.MAX_TIME
            
            dx = ex - sx
            dy = ey - sy
            
            feats = np.stack([sx, sy, dx, dy, t], axis=1)
            
            # Actions
            actions = [self.action_map.get(a, DEFAULT_ACTION_IDX) for a in group['type_name']]
            
            # Team ID
            raw_team = group.iloc[0]['team_id']
            team_idx = TEAM_TO_IDX.get(raw_team, 0)
            
            # Length
            raw_len = len(group)
            
            # Target (Label): Phase의 상태는 동일하므로 첫 번째 값 사용
            label = group.iloc[0]['game_state']
            
            self.phases.append({
                'features': torch.FloatTensor(feats),
                'actions': torch.LongTensor(actions),
                'team_idx': torch.LongTensor([team_idx]),
                'raw_len': torch.FloatTensor([raw_len]),
                'label': torch.LongTensor([label])
            })
            
    def __len__(self): return len(self.phases)
    
    def __getitem__(self, idx):
        return self.phases[idx]

def collate_fn(batch):
    # Padding logic
    features = [b['features'] for b in batch]
    actions = [b['actions'] for b in batch]
    teams = [b['team_idx'] for b in batch]
    raw_lens = [b['raw_len'] for b in batch]
    labels = [b['label'] for b in batch]
    
    lengths = torch.LongTensor([len(f) for f in features])
    
    padded_feats = pad_sequence(features, batch_first=True, padding_value=0.0)
    padded_actions = pad_sequence(actions, batch_first=True, padding_value=DEFAULT_ACTION_IDX)
    
    return (padded_feats, lengths, padded_actions, 
            torch.cat(teams), torch.cat(raw_lens), torch.cat(labels))

# ==========================================
# 3. Model: GameStateRNN (Classifier)
# ==========================================
class GameStateRNN(nn.Module):
    def __init__(self):
        super(GameStateRNN, self).__init__()
        
        # 1. Embeddings
        self.action_embedding = nn.Embedding(Config.NUM_ACTIONS, Config.ACTION_EMB_DIM)
        self.team_embedding = nn.Embedding(Config.NUM_TEAMS, Config.TEAM_EMB_DIM)
        
        # Functional Length Feature
        self.len_encoder = nn.Sequential(nn.Linear(1, 4), nn.Tanh())
        
        # 2. Phase GRU (Encoder)
        # Input: Coords(5) + Action(4) + Team(4) + Len(4) = 17
        input_dim = Config.INPUT_SIZE + Config.ACTION_EMB_DIM + Config.TEAM_EMB_DIM + 4
        self.gru = nn.GRU(input_dim, Config.PHASE_HIDDEN, num_layers=1, batch_first=True)
        
        # 3. Classifier Head (Win/Draw/Lose)
        # GRU의 마지막 Hidden State를 사용하여 분류
        self.classifier = nn.Sequential(
            nn.Linear(Config.PHASE_HIDDEN, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, Config.NUM_CLASSES) # Output: 3 (Logits)
        )

    def forward(self, padded_feats, lengths, padded_actions, team_ids, raw_lens):
        # Embeddings
        act_emb = self.action_embedding(padded_actions)
        team_emb = self.team_embedding(team_ids).unsqueeze(1).expand(-1, padded_feats.size(1), -1)
        
        # Length Feature
        len_feat = self.len_encoder(raw_lens.unsqueeze(1)).unsqueeze(1).expand(-1, padded_feats.size(1), -1)
        
        # Concat
        inputs = torch.cat([padded_feats, act_emb, team_emb, len_feat], dim=2)
        
        # Pack & GRU
        packed = pack_padded_sequence(inputs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        
        last_hidden = h_n[-1] # (Batch, Hidden)
        
        # Classification
        logits = self.classifier(last_hidden)
        return logits

# ==========================================
# 4. Training Loop
# ==========================================
def build_team_mapping(df):
    unique_teams = df['team_id'].unique()
    for idx, team_id in enumerate(sorted(unique_teams)):
        TEAM_TO_IDX[team_id] = idx
    Config.NUM_TEAMS = len(TEAM_TO_IDX) + 1
    print(f"✅ Team Mapping Built: {len(TEAM_TO_IDX)} teams")

def run_training():
    os.makedirs(Config.WEIGHT_DIR, exist_ok=True)
    
    # 1. Load & Label Data
    print("📂 Loading CSVs...")
    # train.csv가 여러 개라면 합쳐야 함. 여기선 하나라고 가정
    if os.path.isdir(Config.TRAIN_DIR):
        files = glob.glob(os.path.join(Config.TRAIN_DIR, '*.csv'))
        train_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        train_df = pd.read_csv(os.path.join(Config.TRAIN_DIR, 'train.csv'))
        
    match_df = pd.read_csv(Config.MATCH_INFO_PATH)
    
    # Labeling
    labeled_df = label_game_state(train_df, match_df)
    
    # Team Mapping
    build_team_mapping(labeled_df)
    
    # 2. Dataset Split (Phase 단위)
    # Train/Val Split (게임 단위로 나누는게 좋지만 편의상 Random Split)
    # DataFrame의 'phase' 컬럼 기준으로 unique phase 추출 후 split
    phases = labeled_df['phase'].unique()
    train_phases, val_phases = train_test_split(phases, test_size=0.2, random_state=42)
    
    train_data = labeled_df[labeled_df['phase'].isin(train_phases)]
    val_data = labeled_df[labeled_df['phase'].isin(val_phases)]
    
    train_ds = GameStateDataset(train_data)
    val_ds = GameStateDataset(val_data)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 3. Model Setup
    model = GameStateRNN().to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss() # Multi-class Classification
    optimizer = optim.Adam(model.parameters(), lr=Config.LR)
    
    best_acc = 0.0
    
    print("🚀 Training Game State Classifier (RNN)...")
    
    for epoch in range(Config.EPOCHS):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0
        
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            batch = [b.to(Config.DEVICE) for b in batch]
            feats, lens, actions, teams, raw_lens, labels = batch
            
            optimizer.zero_grad()
            
            logits = model(feats, lens, actions, teams, raw_lens)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = correct / total
        avg_loss = loss_sum / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = [b.to(Config.DEVICE) for b in batch]
                feats, lens, actions, teams, raw_lens, labels = batch
                
                logits = model(feats, lens, actions, teams, raw_lens)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        
        print(f"   Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(Config.WEIGHT_DIR, "state_classifier.pth"))
            print(f"   💾 Best Model Saved (Acc: {best_acc:.4f})")

if __name__ == '__main__':
    run_training()