import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    BATCH_SIZE = 256
    NUM_WORKERS = 0  # 윈도우 환경이라면 0 추천
    
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    EOS_VALUE = 0.0 
    
    # 모델 파라미터 (학습 설정과 동일해야 함)
    NUM_ACTIONS = 33
    MAX_PHASE_LEN_EMBED = 30
    ACTION_EMB_DIM = 4
    LEN_EMB_DIM = 4
    
    INPUT_SIZE = 5          
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 256    
    DROPOUT = 0.0           # Inference시에는 Dropout 안 씀 (eval 모드에서 자동 처리되지만 명시)
    
    # [경로 설정]
    TEST_DIR = './open_track1/test'   # 테스트 파일들이 들어있는 폴더
    WEIGHT_DIR = './weights'
    MODEL_NAME = 'main_best.pth'  # 'location_aware_last.pth' 도 가능
    OUTPUT_FILE = './submission.csv'

# 학습 때와 동일한 Action Map
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
# 1. Model Definition
# ==========================================
class LocationAwareHierarchicalLSTM(nn.Module):
    def __init__(self, input_size=5, phase_hidden=64, episode_hidden=256, output_size=2, dropout=0.3,
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
            nn.Dropout(dropout),
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
        last_coords = []
        for i in range(batch_size):
            length = episode_lengths[i]
            last_coords.append(padded_coords[i, length-1, :])
        last_known_pos = torch.stack(last_coords)
        
        # Inference에서는 last_known_pos를 반환할 필요 없음
        return last_known_pos + predicted_remaining_delta

# ==========================================
# 2. Test Dataset (Location Aware Logic Applied)
# ==========================================
class SoccerTestDataset(Dataset):
    def __init__(self, data_dir):
        # 하위 폴더까지 검색
        self.file_paths = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
        self.action_map = ACTION_TO_IDX
        if len(self.file_paths) == 0:
            print(f"⚠️  No csv files found in {data_dir}")
        else:
            print(f"📂 Found {len(self.file_paths)} test files.")
    
    def __len__(self): return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            fpath = self.file_paths[idx]
            file_name = os.path.basename(fpath)
            # 파일명에서 game_episode_id 추출 (예: 'test_123.csv' -> 'test_123')
            game_episode_id = os.path.splitext(file_name)[0]
            
            df = pd.read_csv(fpath)
            
            # 데이터가 최소 1줄은 있어야 함
            if len(df) < 1: return None 
            
            # [핵심 로직] 
            # 테스트 파일의 '마지막 행'이 우리가 예측해야 할 이벤트입니다.
            # 이 행의 'start_x, start_y'가 바로 'last_known_pos'가 되어야 합니다.
            last_row = df.iloc[-1]
            target_start_x = last_row['start_x'] / Config.MAX_X
            target_start_y = last_row['start_y'] / Config.MAX_Y

            # LSTM에 들어갈 History는 마지막 행(예측 대상)을 제외한 이전 기록들
            if len(df) > 1:
                df = df.iloc[:-1].copy()
            else:
                # 만약 데이터가 1줄밖에 없다면 히스토리가 없는 것임.
                # 이 경우 로직 처리가 애매하지만, 일단 자기 자신을 히스토리로 넣거나 비워야 함.
                # 여기서는 에러 방지를 위해 자기 자신을 그대로 둠 (단, 정답은 모름)
                pass 
            
            # NaN 처리
            df = df.fillna(0)
            
            if 'phase' not in df.columns:
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()
                 
            sx = df['start_x'].values / Config.MAX_X
            sy = df['start_y'].values / Config.MAX_Y
            ex = df['end_x'].values / Config.MAX_X
            ey = df['end_y'].values / Config.MAX_Y
            t  = df['time_seconds'].values / Config.MAX_TIME
            
            dx = ex - sx
            dy = ey - sy
            
            input_features = np.stack([sx, sy, dx, dy, t], axis=1)
            
            phases_data, start_actions, phase_lens = [], [], []
            phase_end_coords = []
            
            for _, group in df.groupby('phase', sort=False):
                p_feats = input_features[group.index]
                eos = np.full((1, 5), Config.EOS_VALUE)
                phases_data.append(torch.FloatTensor(np.vstack([p_feats, eos])))
                
                act_name = group.iloc[0]['type_name']
                start_actions.append(self.action_map.get(act_name, DEFAULT_ACTION_IDX))
                phase_lens.append(min(len(group), Config.MAX_PHASE_LEN_EMBED - 1))
                
                # 히스토리 상의 마지막 좌표
                last_x = group.iloc[-1]['end_x'] / Config.MAX_X
                last_y = group.iloc[-1]['end_y'] / Config.MAX_Y
                phase_end_coords.append([last_x, last_y])

            # [Location Aware 핵심]
            # 히스토리의 마지막 Phase가 끝난 지점을 -> '실제 예측할 이벤트의 시작점'으로 덮어씀
            # 이를 통해 모델은 정확한 시작 위치에서 예측을 수행함
            if len(phase_end_coords) > 0:
                phase_end_coords[-1] = [target_start_x, target_start_y]
            else:
                # 히스토리가 없어서 루프를 안 돈 경우 (예외처리)
                phase_end_coords.append([target_start_x, target_start_y])
                # 더미 데이터 추가 필요할 수 있음
                return None 

            return (phases_data, start_actions, phase_lens, torch.FloatTensor(phase_end_coords), game_episode_id)
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            return None

def test_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return (None,)*7
    
    b_phases, b_acts, b_lens, b_coords, b_ids = zip(*batch)
    
    all_phases, all_acts, all_lens_ids, ep_lens = [], [], [], []
    for i in range(len(b_phases)):
        all_phases.extend(b_phases[i])
        all_acts.extend(b_acts[i])
        all_lens_ids.extend(b_lens[i])
        ep_lens.append(len(b_phases[i]))
        
    pad_phases = pad_sequence(all_phases, batch_first=True, padding_value=Config.EOS_VALUE)
    phase_lengths = torch.LongTensor([len(p) for p in all_phases])
    episode_lengths = torch.LongTensor(ep_lens)
    start_action_ids = torch.LongTensor(all_acts)
    phase_len_ids = torch.LongTensor(all_lens_ids)
    
    coords_list = [torch.FloatTensor(c) for c in b_coords]
    padded_coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    
    return pad_phases, phase_lengths, episode_lengths, start_action_ids, phase_len_ids, padded_coords, b_ids

# ==========================================
# 3. Inference Engine
# ==========================================
def run_inference():
    print(f"✅ Device: {Config.DEVICE}")
    
    model_path = os.path.join(Config.WEIGHT_DIR, Config.MODEL_NAME)
    if not os.path.exists(model_path):
        print(f"❌ Error: Model {model_path} not found.")
        return

    # Init Model
    model = LocationAwareHierarchicalLSTM(
        input_size=Config.INPUT_SIZE,
        phase_hidden=Config.PHASE_HIDDEN,
        episode_hidden=Config.EPISODE_HIDDEN,
        dropout=Config.DROPOUT,
        num_actions=Config.NUM_ACTIONS
    ).to(Config.DEVICE)
    
    print(f"🔄 Loading model from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    except Exception as e:
        print(f"❌ Model Load Fail: {e}")
        return
        
    model.eval()
    
    test_ds = SoccerTestDataset(Config.TEST_DIR)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
                             collate_fn=test_collate_fn, num_workers=Config.NUM_WORKERS)
    
    results = []
    
    print("🚀 Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            if batch[0] is None: continue
            
            padded_phases = batch[0].to(Config.DEVICE)
            phase_lengths = batch[1].to(Config.DEVICE)
            episode_lengths = batch[2].to(Config.DEVICE)
            start_action_ids = batch[3].to(Config.DEVICE)
            phase_len_ids = batch[4].to(Config.DEVICE)
            padded_coords = batch[5].to(Config.DEVICE)
            ids = batch[6]
            
            # Forward
            preds = model(padded_phases, phase_lengths, episode_lengths, 
                          start_action_ids, phase_len_ids, padded_coords)
            
            # Denormalize
            pred_x = preds[:, 0].cpu().numpy() * Config.MAX_X
            pred_y = preds[:, 1].cpu().numpy() * Config.MAX_Y
            
            # NaN Handling
            if np.isnan(pred_x).any() or np.isnan(pred_y).any():
                pred_x = np.nan_to_num(pred_x, nan=52.5)
                pred_y = np.nan_to_num(pred_y, nan=34.0)

            # Clipping
            pred_x = np.clip(pred_x, 0, Config.MAX_X)
            pred_y = np.clip(pred_y, 0, Config.MAX_Y)
            
            for i, game_ep_id in enumerate(ids):
                results.append({
                    'game_episode': game_ep_id,
                    'end_x': float(pred_x[i]),
                    'end_y': float(pred_y[i])
                })
    
    # Save
    submission_df = pd.DataFrame(results)
    
    # Sort just in case (optional)
    submission_df = submission_df.sort_values('game_episode')
    
    print("\n🔍 [Check] First 5 predictions:")
    print(submission_df.head())
    
    submission_df.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✅ Saved submission to {Config.OUTPUT_FILE} (Rows: {len(submission_df)})")

if __name__ == "__main__":
    run_inference()