import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

def ensure_phase_column(df):
    # 1. phase 컬럼이 이미 있는지 확인
    if 'phase' in df.columns:
        return df

    if 'team_id' not in df.columns:
        raise ValueError("Phase를 생성하기 위해서는 'team_id' 컬럼이 필요합니다.")

    
    if 'game_id' in df.columns and 'episode_id' in df.columns:
        # 그룹별로 shift를 적용하여 경계선 문제(다른 게임끼리 phase 이어짐) 방지
        df['phase'] = df.groupby(['game_id', 'episode_id'])['team_id'] \
                        .apply(lambda x: (x != x.shift(1)).fillna(0).cumsum()) \
                        .reset_index(level=[0, 1], drop=True) # 인덱스 정렬 유지
    else:
        # 단일 에피소드/게임 데이터인 경우 단순 shift 적용
        df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()
        
    return df

class SoccerDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        
        # 정규화 상수
        self.MAX_X = 105.0
        self.MAX_Y = 68.0
        self.MAX_TIME = 95 * 60.0 # 5700초
        self.EOS_VALUE = -1.0 # EOS 토큰 값

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            df = pd.read_csv(file_path)
            
            # 최소 2개 이상의 이벤트가 있어야 (과거 -> 미래) 예측 가능
            if len(df) < 2:
                return None
            
            # Phase 컬럼 방어 코드
            if 'phase' not in df.columns:
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

            # --- 1. 정규화 (Normalization) ---
            sx = df['start_x'].values / self.MAX_X
            sy = df['start_y'].values / self.MAX_Y
            ex = df['end_x'].values / self.MAX_X
            ey = df['end_y'].values / self.MAX_Y
            t  = df['time_seconds'].values / self.MAX_TIME
            
            # (N, 5) Feature Matrix
            features = np.stack([sx, sy, ex, ey, t], axis=1)
            
            # --- 2. Input / Target 분리 ---
            # Target: 에피소드의 '진짜' 마지막 이벤트의 도착 위치
            target = features[-1, 2:4] # [end_x, end_y]
            
            # Input Data: 마지막 이벤트를 제외한 모든 데이터
            input_features = features[:-1]
            input_df = df.iloc[:-1].copy() # Phase 그룹화를 위해 DF도 자름
            
            # --- 3. Phase 별 분할 및 EOS 추가 (Input Data에 대해서만) ---
            phases_data = []
            
            # 자른 데이터(input_df) 기준으로 phase 그룹화
            for _, group in input_df.groupby('phase', sort=False):
                phase_indices = group.index
                
                # 해당 phase의 feature 가져오기 (인덱스 주의: iloc과 numpy 인덱싱 매칭)
                # input_df는 0부터 다시 인덱싱 된게 아니라 원본 인덱스를 유지하거나,
                # 가장 안전하게는 numpy array에서 직접 슬라이싱하는 것이 좋습니다.
                # 여기서는 길이 기준으로 매칭하겠습니다.
                
                # 현재 그룹의 상대적인 위치 파악이 복잡할 수 있으므로, 
                # input_features를 직접 슬라이싱해서 가져오는 방식을 추천합니다.
                # 하지만 간단하게 group.index가 0부터 시작하는 reset_index 상태라면 아래 방식이 맞습니다.
                # 파일별로 읽으므로 df.index는 0, 1, 2... 순서입니다.
                
                phase_feats = input_features[group.index] # (Seq_Len, 5)
                
                # EOS Token Row 추가
                eos_row = np.full((1, 5), self.EOS_VALUE)
                phase_feats_with_eos = np.vstack([phase_feats, eos_row])
                
                phases_data.append(torch.FloatTensor(phase_feats_with_eos))
            
            if not phases_data: # 입력 데이터가 비어버린 경우 (ex: 이벤트가 1개뿐이었을 때)
                return None
                
            # 반환값: (Phase 리스트, Target 텐서)
            return phases_data, torch.FloatTensor(target)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
def hierarchical_collate_fn(batch):
    """
    Args:
        batch: [(phases_list_1, target_1), (phases_list_2, target_2), ...]
    """
    # 1. None 데이터 필터링
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None, None, None, None
        
    # Input(phases_list)과 Target 분리
    batch_phases_lists, batch_targets = zip(*batch)
    
    # --- Target 처리 ---
    targets = torch.stack(batch_targets) # (Batch_Size, 2)
    
    # --- Input 계층 구조 처리 ---
    all_phases = []          # Flatten된 모든 Phase
    episode_lengths = []     # 각 Episode가 몇 개의 Phase를 가지는지
    
    for phases_list in batch_phases_lists:
        all_phases.extend(phases_list)
        episode_lengths.append(len(phases_list))
        
    # Phase 단위 Padding (Phase LSTM 입력용)
    # (Total_Phases, Max_Phase_Len, 5)
    padded_phases = pad_sequence(all_phases, batch_first=True, padding_value=0)
    
    # 각 Phase의 실제 길이 (EOS 포함)
    phase_lengths = torch.LongTensor([len(p) for p in all_phases])
    
    # Episode 구조 정보
    episode_lengths = torch.LongTensor(episode_lengths)
    
    return padded_phases, phase_lengths, episode_lengths, targets