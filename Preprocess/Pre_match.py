import pandas as pd
import pickle
import os
from tqdm import tqdm

# 1. match_info 데이터 로드
csv_file_path = './open_track1/match_info.csv' 

print("1. 데이터 파일을 읽어오는 중입니다...")
if os.path.exists(csv_file_path):
    df = pd.read_csv(csv_file_path)
    print(f"   -> 로드 완료! 총 {len(df)}개의 경기 데이터가 있습니다.")
else:
    print(f"Error: {csv_file_path} 파일을 찾을 수 없습니다.")
    exit() # 파일이 없으면 여기서 종료

# 2. 데이터 처리 (팀 정보 & 매치 정보)
# 진행률을 보여주기 위해 tqdm을 사용합니다.
team_dict = {}
match_dict = {}

print("\n2. 데이터를 순회하며 딕셔너리를 생성합니다...")

# iterrows()는 데이터가 많으면 느릴 수 있어, tqdm으로 진행상황을 봅니다.
# total=len(df)를 넣어줘야 진행률 바가 정확히 나옵니다.
for _, row in tqdm(df.iterrows(), total=len(df), desc="처리 중"):
    # --- [정보 추출] ---
    g_id = row['game_id']
    
    h_id = row['home_team_id']
    h_name = row['home_team_name_ko']
    venue = row['venue']
    
    a_id = row['away_team_id']
    a_name = row['away_team_name_ko']
    
    # --- [Team Dict 구축] ---
    # 홈팀 저장
    team_dict[h_id] = {
        'name': h_name,
        'venue': venue
    }
    
    # 어웨이팀 저장 (없을 경우에만)
    if a_id not in team_dict:
        team_dict[a_id] = {
            'name': a_name,
            'venue': None 
        }
        
    # --- [Match Dict 구축] ---
    # 두 번 돌 필요 없이 한 번 돌 때 같이 처리하면 훨씬 빠릅니다.
    match_dict[g_id] = {
        'home_team_id': h_id,
        'away_team_id': a_id
    }

# 3. 결과 확인 및 저장
print("\n3. 처리가 완료되었습니다.")
print(f"   - 추출된 팀 개수: {len(team_dict)}")
print(f"   - 추출된 경기 개수: {len(match_dict)}")

# 저장 단계
save_path = 'processed_match_data.pkl'
print(f"\n4. '{save_path}' 파일로 저장 중입니다...")

with open(save_path, 'wb') as f:
    pickle.dump({'match_dict': match_dict, 'team_dict': team_dict}, f)
    
print("   -> 저장 완료! 모든 작업이 끝났습니다.")