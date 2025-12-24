import pandas as pd
import os
import random
from tqdm import tqdm

def process_and_save_by_game(df, output_dir='./data', val_ratio=0.10):
    """
    데이터프레임을 받아 Phase를 추가하고, 
    [Game 단위]로 Train/Val을 나누어 저장합니다.
    
    저장 경로: 
      - ./data_test/train/{game_id}_{episode_id}.csv
      - ./data_test/val/{game_id}_{episode_id}.csv
    """
    
    # 1. Phase 컬럼 추가
    if 'team_id' not in df.columns:
        raise ValueError("데이터에 'team_id' 컬럼이 필요합니다.")
        
    print("Phase 정보 생성 중...")
    df['phase'] = (df['team_id'] != df['team_id'].shift(1)).fillna(0).cumsum()

    # 2. 저장 폴더 생성
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 3. [수정됨] Game 단위로 Split
    # 전체 고유 game_id 추출
    unique_games = df['game_id'].unique()
    
    # 랜덤 셔플
    random.shuffle(unique_games)
    
    total_games = len(unique_games)
    val_count = int(total_games * val_ratio)
    
    # Validation으로 사용할 game_id 선정
    val_games_set = set(unique_games[:val_count])
    
    print(f"Total Games: {total_games}")
    print(f"Train Games: {total_games - val_count}, Val Games: {val_count}")

    # 4. Episode 단위로 그룹화 및 저장
    print("에피소드별 분할 저장 중...")
    grouped = df.groupby(['game_id', 'episode_id'])
    
    # 그룹의 키들 [(game_id, ep_id), ...]
    episode_keys = list(grouped.groups.keys())
    
    for key in tqdm(episode_keys, desc="Saving episodes"):
        game_id, episode_id = key
        
        # 해당 에피소드가 속한 game_id가 Validation Set인지 확인
        if game_id in val_games_set:
            save_path = os.path.join(val_dir, f"{game_id}_{episode_id}.csv")
        else:
            save_path = os.path.join(train_dir, f"{game_id}_{episode_id}.csv")
            
        # 데이터 저장
        episode_df = grouped.get_group(key)
        episode_df.to_csv(save_path, index=False)

    print("모든 데이터 처리가 완료되었습니다.")

# --- 사용 예시 ---
if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv('./open_track1/train.csv')
    
    # 함수 실행
    process_and_save_by_game(df)