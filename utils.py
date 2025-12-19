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
        """
        Args:
            file_paths (list): 데이터 파일 경로 리스트 (train_files 등)
        """
        self.file_paths = file_paths
        
        # 정규화 상수 정의
        self.MAX_X = 105.0
        self.MAX_Y = 68.0
        self.MAX_TIME = 95 * 60.0  # 95분 * 60초 = 5700초
        
        # EOS Token 값 (정규화 범위 0~1 밖의 값 사용)
        self.EOS_VALUE = -1.0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            df = pd.read_csv(file_path)
            
            # Phase 컬럼 방어 코드 (혹시 없으면 생성)
            # (이전에 만든 ensure_phase_column 함수가 있다고 가정하거나 여기에 직접 구현)
            if 'phase' not in df.columns:
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

            # --- 1. 정규화 (Normalization) ---
            # 좌표는 0~1 사이로, 시간도 0~1 사이로 변환
            # time_seconds 컬럼이 있다고 가정
            x_norm = df['x'].values / self.MAX_X
            y_norm = df['y'].values / self.MAX_Y
            t_norm = df['time_seconds'].values / self.MAX_TIME
            
            # Feature 합치기 (N, 3) -> [x, y, time]
            # 필요하다면 여기에 action embedding을 위한 action index 등을 추가할 수 있음
            features = np.stack([x_norm, y_norm, t_norm], axis=1)
            
            # --- 2. Phase 별 분할 및 EOS 추가 ---
            phases_data = []
            
            # df['phase'] 기준으로 그룹화
            # np.unique 등을 써도 되지만, 순서를 유지하기 위해 dataframe groupby 활용
            for _, group in df.groupby('phase', sort=False):
                # 해당 phase의 feature 가져오기
                phase_indices = group.index
                phase_feats = features[phase_indices] # (seq_len, 3)
                
                # EOS Token Row 생성 ([-1, -1, -1])
                eos_row = np.full((1, phase_feats.shape[1]), self.EOS_VALUE)
                
                # 원본 데이터 뒤에 EOS 붙이기
                phase_feats_with_eos = np.vstack([phase_feats, eos_row])
                
                # Tensor로 변환하여 리스트에 추가
                phases_data.append(torch.FloatTensor(phase_feats_with_eos))
            
            # 반환값: 이 Episode에 속한 Phase들의 리스트 (List of Tensors)
            return phases_data

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
        
def hierarchical_collate_fn(batch):
    """
    batch: [ episode1_phases, episode2_phases, ... ]
      - episode1_phases: [phase1_tensor, phase2_tensor, ...]
    """
    
    # 1. 빈 데이터(로딩 에러 등) 필터링
    batch = [item for item in batch if len(item) > 0]
    
    # 2. 계층 구조를 위한 정보 수집
    all_phases = []          # 모든 phase 텐서를 일렬로 담을 리스트
    episode_lengths = []     # 각 에피소드가 몇 개의 phase로 구성되었는지 (예: [3, 5, 2])
    
    for episode_phases in batch:
        all_phases.extend(episode_phases)    # 리스트 확장
        episode_lengths.append(len(episode_phases))
        
    # 3. Phase 단위 Padding (Phase LSTM 입력용)
    # pad_sequence는 (Batch, Time, Feat) 형태를 만듦 (batch_first=True)
    # 길이가 다른 phase들을 가장 긴 phase 길이에 맞춰 0으로 padding
    # (이미 EOS 토큰이 끝에 있으므로 0 패딩과 구분 가능)
    padded_phases = pad_sequence(all_phases, batch_first=True, padding_value=0)
    
    # 4. 각 Phase의 실제 길이 (Pack_padded_sequence 사용 시 필요)
    phase_lengths = torch.LongTensor([len(p) for p in all_phases])
    
    # 5. Episode 구조 정보 Tensor 변환
    episode_lengths = torch.LongTensor(episode_lengths)
    
    return {
        'padded_phases': padded_phases,   # (Total_Phases, Max_Seq_Len, 3) -> Phase LSTM Input
        'phase_lengths': phase_lengths,   # (Total_Phases) -> 각 Phase의 실제 길이
        'episode_lengths': episode_lengths # (Batch_Size) -> 각 에피소드의 Phase 개수
    }