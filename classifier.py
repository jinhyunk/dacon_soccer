import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os

# ==========================================
# 1. Configuration & Data Preparation
# ==========================================
CONFIG = {
    'seq_len': 30,        # Phase당 최대 이벤트 길이 (이보다 길면 자르고, 짧으면 0 채움)
    'embed_dim': 32,      # 임베딩 차원
    'hidden_dim': 64,     # LSTM 히든 레이어 크기
    'batch_size': 64,
    'epochs': 10,
    'lr': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

def load_and_preprocess(file_path, match_info_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    match_info = pd.read_csv(match_info_path)
    
    # 1. Phase ID 생성 (Episode가 바뀌거나 Team이 바뀌면 새로운 Phase)
    # 데이터는 이미 시간순 정렬되어 있다고 가정 (score_good.csv 생성 시 정렬함)
    df['prev_episode'] = df['game_episode'].shift(1)
    df['prev_team'] = df['team_id'].shift(1)
    
    # 첫 행 처리 및 변경 지점 확인
    df['is_new_phase'] = (df['game_episode'] != df['prev_episode']) | (df['team_id'] != df['prev_team'])
    df.loc[0, 'is_new_phase'] = True
    
    # Phase ID 부여
    df['phase_id'] = df.groupby('game_id')['is_new_phase'].cumsum()
    df['phase_uid'] = df['game_id'].astype(str) + '_' + df['phase_id'].astype(str)
    
    print("Grouping data by Phase...")
    # Phase별 데이터 압축
    phase_data = df.groupby(['phase_uid', 'game_id', 'team_id']).agg({
        'type_name': list,          # Event Sequence
        'home_score': 'first',      # Phase 시작 시점 스코어
        'away_score': 'first'
    }).reset_index()
    
    # 2. Target Labeling (Winning/Drawing/Losing)
    # 팀 ID 매핑을 위해 match_info 병합
    match_teams = match_info[['game_id', 'home_team_id', 'away_team_id']].drop_duplicates()
    match_teams['home_team_id'] = match_teams['home_team_id'].astype(str)
    match_teams['away_team_id'] = match_teams['away_team_id'].astype(str)
    phase_data['team_id_str'] = phase_data['team_id'].astype(int).astype(str)
    
    phase_data = phase_data.merge(match_teams, on='game_id', how='left')
    
    def get_state(row):
        is_home = row['team_id_str'] == row['home_team_id']
        my_score = row['home_score'] if is_home else row['away_score']
        op_score = row['away_score'] if is_home else row['home_score']
        
        if my_score > op_score: return 'Winning'
        elif my_score < op_score: return 'Losing'
        else: return 'Drawing'
        
    phase_data['match_state'] = phase_data.apply(get_state, axis=1)
    
    return phase_data

# ==========================================
# 2. Dataset & Encoding
# ==========================================
class PhaseDataset(Dataset):
    def __init__(self, phases, teams, labels, event_le, max_len):
        self.phases = phases
        self.teams = teams
        self.labels = labels
        self.event_le = event_le
        self.max_len = max_len
        
    def __len__(self):
        return len(self.phases)
    
    def __getitem__(self, idx):
        # Event Sequence Encoding
        seq_strs = self.phases[idx]
        # Unknown event handling (if any new event types appear)
        seq_idx = [self.event_le.transform([e])[0] if e in self.event_le.classes_ else 0 for e in seq_strs]
        
        # Padding / Truncating
        if len(seq_idx) > self.max_len:
            seq_idx = seq_idx[:self.max_len]
        
        # 0 is reserved for padding, so shift indices by +1
        padded = np.zeros(self.max_len, dtype=int)
        padded[:len(seq_idx)] = np.array(seq_idx) + 1
        
        return {
            'sequence': torch.tensor(padded, dtype=torch.long),
            'team': torch.tensor(self.teams[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ==========================================
# 3. Model Definition
# ==========================================
class SoccerLSTM(nn.Module):
    def __init__(self, vocab_size, num_teams, embed_dim, hidden_dim, num_classes):
        super(SoccerLSTM, self).__init__()
        
        # Embeddings
        self.event_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.team_embed = nn.Embedding(num_teams, embed_dim)
        
        # LSTM
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        
        # Classifier Head (Context Vector + Team Vector -> Probabilities)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, sequence, team):
        # 1. Sequence Feature
        x = self.event_embed(sequence)     # [Batch, Seq, Embed]
        _, (h_n, _) = self.lstm(x)         # h_n: [1, Batch, Hidden]
        seq_feat = h_n[-1]                 # [Batch, Hidden]
        
        # 2. Team Feature
        team_feat = self.team_embed(team)  # [Batch, Embed]
        
        # 3. Concatenate & Predict
        combined = torch.cat((seq_feat, team_feat), dim=1)
        logits = self.fc(combined)
        return logits

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # --- A. Data Loading ---
    if not os.path.exists('./score_good.csv'):
        print("Error: 'score_good.csv' not found. Please run the previous data generation step.")
        exit()
        
    df_phase = load_and_preprocess('score_good.csv', './open_track1/match_info.csv')
    print(f"Total Phases: {len(df_phase)}")
    
    # --- B. Encoders ---
    # Event Type Encoder
    all_events = [e for seq in df_phase['type_name'] for e in seq]
    event_le = LabelEncoder()
    event_le.fit(all_events)
    vocab_size = len(event_le.classes_) + 1 # +1 for padding
    
    # Team ID Encoder
    team_le = LabelEncoder()
    df_phase['team_encoded'] = team_le.fit_transform(df_phase['team_id'])
    num_teams = len(team_le.classes_)
    
    # Target Label Encoder
    target_le = LabelEncoder()
    df_phase['label'] = target_le.fit_transform(df_phase['match_state'])
    num_classes = len(target_le.classes_)
    print(f"Classes: {target_le.classes_}") # ['Drawing', 'Losing', 'Winning'] expected
    
    # --- C. Train/Val Split ---
    train_df, val_df = train_test_split(
        df_phase, test_size=0.2, random_state=42, stratify=df_phase['label']
    )
    
    train_dataset = PhaseDataset(
        train_df['type_name'].values, train_df['team_encoded'].values, 
        train_df['label'].values, event_le, CONFIG['seq_len']
    )
    val_dataset = PhaseDataset(
        val_df['type_name'].values, val_df['team_encoded'].values, 
        val_df['label'].values, event_le, CONFIG['seq_len']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # --- D. Initialize Model ---
    model = SoccerLSTM(vocab_size, num_teams, CONFIG['embed_dim'], CONFIG['hidden_dim'], num_classes)
    model = model.to(CONFIG['device'])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    
    # --- E. Training Loop ---
    print(f"\nStart Training on {CONFIG['device']}...")
    best_acc = 0.0
    
    for epoch in range(CONFIG['epochs']):