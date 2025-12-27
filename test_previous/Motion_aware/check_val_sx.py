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
    NUM_WORKERS = 4  # Windows users might need to set this to 0
    
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    EOS_VALUE = 0.0 
    
    # Model Parameters
    NUM_ACTIONS = 33
    MAX_PHASE_LEN_EMBED = 30
    ACTION_EMB_DIM = 4
    LEN_EMB_DIM = 4
    
    INPUT_SIZE = 5          
    PHASE_HIDDEN = 64
    EPISODE_HIDDEN = 256
    DROPOUT = 0.0           
    
    # [Target Paths]
    VAL_DIR = './data/val'        # Path to the adversarial validation CSVs
    WEIGHT_DIR = './weights'
    MODEL_NAME = 'location_aware_best.pth'       # Ensure this matches your saved model name

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
        
        return last_known_pos + predicted_remaining_delta

# ==========================================
# 2. Validation Dataset (Location Aware)
# ==========================================
class SoccerValDataset(Dataset):
    def __init__(self, data_dir):
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        self.action_map = ACTION_TO_IDX
        if len(self.file_paths) == 0:
            print(f"⚠️  No csv files found in {data_dir}")
        else:
            print(f"📂 Found {len(self.file_paths)} validation files.")
    
    def __len__(self): return len(self.file_paths)
    
    def __getitem__(self, idx):
        try:
            fpath = self.file_paths[idx]
            df = pd.read_csv(fpath)
            
            if len(df) < 2: return None 
            
            # [Core Logic] Validation files have Ground Truth in the last row
            target_row = df.iloc[-1]
            target_ex = target_row['end_x'] / Config.MAX_X
            target_ey = target_row['end_y'] / Config.MAX_Y
            target = np.array([target_ex, target_ey], dtype=np.float32)
            
            # The start position of the target pass is our 'last_known_pos'
            target_start_x = target_row['start_x'] / Config.MAX_X
            target_start_y = target_row['start_y'] / Config.MAX_Y
            
            # Input history excludes the last row (target)
            input_df = df.iloc[:-1].copy()
            input_df = input_df.fillna(0)
            
            if 'phase' not in input_df.columns:
                 input_df['phase'] = (input_df['team_id'] != input_df['team_id'].shift(1)).fillna(0).cumsum()
                 
            sx = input_df['start_x'].values / Config.MAX_X
            sy = input_df['start_y'].values / Config.MAX_Y
            ex = input_df['end_x'].values / Config.MAX_X
            ey = input_df['end_y'].values / Config.MAX_Y
            t  = input_df['time_seconds'].values / Config.MAX_TIME
            
            dx = ex - sx
            dy = ey - sy
            
            input_features = np.stack([sx, sy, dx, dy, t], axis=1)
            
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

            # [Important] Overwrite last known position with the actual start of the target pass
            if len(phase_end_coords) > 0:
                phase_end_coords[-1] = [target_start_x, target_start_y]
            else:
                phase_end_coords.append([target_start_x, target_start_y])
                return None

            return (phases_data, start_actions, phase_lens, torch.FloatTensor(phase_end_coords), torch.FloatTensor(target))
        except Exception as e:
            return None

def val_collate_fn(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return (None,)*7
    
    b_phases, b_acts, b_lens, b_coords, b_targets = zip(*batch)
    
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
    
    targets = torch.stack(b_targets)
    
    return pad_phases, phase_lengths, episode_lengths, start_action_ids, phase_len_ids, padded_coords, targets

# ==========================================
# 3. Validation Logic
# ==========================================
def run_validation():
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
    
    # Load Adversarial Validation Set
    val_ds = SoccerValDataset(Config.VAL_DIR)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, 
                             collate_fn=val_collate_fn, num_workers=Config.NUM_WORKERS)
    
    total_dist_error = 0.0
    count = 0
    
    print("🚀 Starting Validation on Adversarial Set...")
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if batch[0] is None: continue
            
            padded_phases = batch[0].to(Config.DEVICE)
            phase_lengths = batch[1].to(Config.DEVICE)
            episode_lengths = batch[2].to(Config.DEVICE)
            start_action_ids = batch[3].to(Config.DEVICE)
            phase_len_ids = batch[4].to(Config.DEVICE)
            padded_coords = batch[5].to(Config.DEVICE)
            targets = batch[6].to(Config.DEVICE) # (Batch, 2) Normalized Ground Truth
            
            # Forward
            preds = model(padded_phases, phase_lengths, episode_lengths, 
                          start_action_ids, phase_len_ids, padded_coords)
            
            # --- Metric Calculation (Real Meters) ---
            # Denormalize Predictions
            pred_x = preds[:, 0] * Config.MAX_X
            pred_y = preds[:, 1] * Config.MAX_Y
            
            # Denormalize Targets
            target_x = targets[:, 0] * Config.MAX_X
            target_y = targets[:, 1] * Config.MAX_Y
            
            # Euclidean Distance
            diff_x = pred_x - target_x
            diff_y = pred_y - target_y
            dist_error = torch.sqrt(diff_x**2 + diff_y**2)
            
            total_dist_error += dist_error.sum().item()
            count += preds.size(0)
            
    if count > 0:
        avg_dist_error = total_dist_error / count
        print(f"\n📊 Validation Result (Adversarial Set)")
        print(f"   - Total Samples: {count}")
        print(f"   - Mean Distance Error: {avg_dist_error:.4f} m")
    else:
        print("⚠️ No valid samples found.")

if __name__ == "__main__":
    run_validation()