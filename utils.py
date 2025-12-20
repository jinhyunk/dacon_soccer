import torch
import glob 
import os 
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence

ACTION_TO_IDX = {
    'Aerial Clearance': 0,
    'Block': 1,
    'Carry': 2,
    'Catch': 3,
    'Clearance': 4,
    'Cross': 5,
    'Deflection': 6,
    'Duel': 7,
    'Error': 8,
    'Foul': 9,
    'Foul_Throw': 10,
    'Goal': 11,
    'Goal Kick': 12,
    'Handball_Foul': 13,
    'Hit': 14,
    'Interception': 15,
    'Intervention': 16,
    'Offside': 17,
    'Out': 18,
    'Own Goal': 19,
    'Parry': 20,
    'Pass': 21,
    'Pass_Corner': 22,
    'Pass_Freekick': 23,
    'Penalty Kick': 24,
    'Recovery': 25,
    'Shot': 26,
    'Shot_Corner': 27,
    'Shot_Freekick': 28,
    'Tackle': 29,
    'Take-On': 30,
    'Throw-In': 31,
    'Other': 32  # 매핑되지 않은 값이나 예외 처리를 위한 클래스
}

# 2. Config 클래스 (학습 코드와 공유하는 설정값)
# (이미 main.py에 있다면 중복 정의하지 않도록 주의하세요)
    
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

class SoccerHierarchicalDataset(Dataset):
    def __init__(self, data_dir):
        # 해당 경로의 모든 csv 파일 리스트 읽기
        self.file_paths = glob.glob(os.path.join(data_dir, '*.csv'))
        
        self.MAX_X = 105.0
        self.MAX_Y = 68.0
        self.MAX_TIME = 95 * 60.0 # 5700초
        self.EOS_VALUE = -1.0 # EOS 토큰 값
        self.MAX_PHASE_LEN_EMBED = 30
        # Action 매핑 로드
        self.action_map = ACTION_TO_IDX
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            # 1. 데이터 로드
            df = pd.read_csv(file_path)
            
            # 예외 처리: 데이터가 너무 짧아 예측(Target)을 만들 수 없는 경우
            if len(df) < 2: 
                return None
            
            # 2. 전처리: Phase 컬럼 확인 및 생성
            if 'phase' not in df.columns:
                 # team_id가 바뀔 때마다 phase 증가
                 df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

            # 3. 정규화 (Normalization)
            # 좌표와 시간을 0~1 근처 값으로 스케일링
            sx = df['start_x'].values / self.MAX_X
            sy = df['start_y'].values / self.MAX_Y
            ex = df['end_x'].values / self.MAX_X
            ey = df['end_y'].values / self.MAX_Y
            t  = df['time_seconds'].values / self.MAX_TIME
            
            # (N, 5) 형태의 Feature Matrix 생성
            features = np.stack([sx, sy, ex, ey, t], axis=1)
            
            # 4. Input / Target 분리
            # Target: 이 에피소드의 '마지막' 이벤트가 끝난 위치 [end_x, end_y]
            target = features[-1, 2:4] 
            
            # Input: 마지막 이벤트를 제외한 모든 데이터 (과거 정보)
            input_features = features[:-1]
            input_df = df.iloc[:-1].copy() # 그룹화를 위해 DF도 길이 맞춤
            
            # 5. Phase 별 분할 및 Context 추출
            phases_data = []        # Phase별 데이터 텐서 리스트
            start_actions = []      # Context: 시작 액션 ID
            phase_lengths_list = [] # Context: Phase 길이
            
            # Phase 단위로 그룹화하여 순회
            for _, group in input_df.groupby('phase', sort=False):
                # A. 해당 Phase의 Feature 가져오기
                # group.index를 사용하여 input_features에서 해당 행들을 슬라이싱
                phase_feats = input_features[group.index] 
                
                # EOS(End of Sequence) 행 추가 ([-1, -1, -1, -1, -1])
                eos_row = np.full((1, 5), self.EOS_VALUE)
                phase_feats_with_eos = np.vstack([phase_feats, eos_row])
                
                # 텐서로 변환하여 리스트에 추가
                phases_data.append(torch.FloatTensor(phase_feats_with_eos))
                
                # B. Start Action 추출 (Context)
                # 그룹의 첫 번째 이벤트 타입 가져오기
                first_action_name = group.iloc[0]['type_name']
                # 매핑 딕셔너리에서 ID 찾기 (없으면 'Other')
                action_idx = self.action_map.get(first_action_name, self.action_map['Other'])
                start_actions.append(action_idx)
                
                # C. Phase Length 추출 (Context)
                length = len(group)
                # 임베딩 테이블 크기(30)를 넘지 않도록 제한 (Clipping)
                length = min(length, self.MAX_PHASE_LEN_EMBED - 1)
                phase_lengths_list.append(length)
            
            # 데이터가 비어있으면 None 반환
            if not phases_data: 
                return None
            
            # 반환값: (입력 데이터 리스트, 정답, 시작 액션 리스트, 길이 리스트)
            return phases_data, torch.FloatTensor(target), start_actions, phase_lengths_list

        except Exception as e:
            # 파일 읽기 에러 시 출력하고 None 반환 (학습 시 건너뜀)
            print(f"Error loading {file_path}: {e}")
            return None
            
def hierarchical_collate_fn2(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None, None, None, None, None, None
        
    # Unpack (반환값이 늘어남)
    batch_phases, batch_targets, batch_start_acts, batch_lens = zip(*batch)
    
    targets = torch.stack(batch_targets)
    
    all_phases = []
    episode_lengths = []
    
    # Context 정보를 위한 리스트
    all_start_actions = [] 
    all_phase_lens_idx = [] # 임베딩 인덱스용
    
    for i, phases_list in enumerate(batch_phases):
        all_phases.extend(phases_list)
        episode_lengths.append(len(phases_list))
        
        # Context 정보도 Flatten (Total_Phases 차원으로)
        all_start_actions.extend(batch_start_acts[i])
        all_phase_lens_idx.extend(batch_lens[i])
        
    padded_phases = pad_sequence(all_phases, batch_first=True, padding_value=0)
    phase_lengths = torch.LongTensor([len(p) for p in all_phases])
    episode_lengths = torch.LongTensor(episode_lengths)
    
    # Context Tensor 변환
    start_action_ids = torch.LongTensor(all_start_actions) # (Total_Phases,)
    phase_len_ids = torch.LongTensor(all_phase_lens_idx)   # (Total_Phases,)
    
    # 반환값에 Context Tensor들 추가
    return padded_phases, phase_lengths, episode_lengths, targets, start_action_ids, phase_len_ids