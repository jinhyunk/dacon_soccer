import pandas as pd
import numpy as np

# 1. 데이터 로드 및 정렬
train = pd.read_csv('./open_track1/train.csv')
match_info = pd.read_csv('./open_track1/match_info.csv')

# 시간 순서대로 정렬 (매우 중요)
train = train.sort_values(['game_id', 'period_id', 'time_seconds', 'action_id']).reset_index(drop=True)

# 2. 킥오프(Kickoff) 탐지 로직
# 이전 에피소드와 비교하여 새로운 에피소드의 시작인지 확인
train['prev_episode'] = train['game_episode'].shift(1)
train['is_new_episode'] = train['game_episode'] != train['prev_episode']

# 킥오프 조건: 패스(Pass) + 센터 서클 내부(x:48~57, y:30~38) + 새로운 에피소드의 시작
kickoff_condition = (
    (train['type_name'] == 'Pass') &
    (train['start_x'].between(48, 57)) & 
    (train['start_y'].between(30, 38)) &
    (train['is_new_episode'])
)
train['is_kickoff'] = kickoff_condition

# 3. 스코어 계산 (Score Calculation)
# 3-1. 득점 상황이 발생한 킥오프 식별
# (각 피리어드의 첫 번째 킥오프는 전/후반 시작이므로 제외, 두 번째부터가 득점 후 킥오프)
kickoffs = train[train['is_kickoff']].copy()
kickoffs['ko_rank'] = kickoffs.groupby(['game_id', 'period_id']).cumcount() + 1
goal_kickoffs = kickoffs[kickoffs['ko_rank'] > 1].copy()

# 3-2. 득점 팀 판별
# 킥오프를 하는 팀은 '실점'한 팀이므로, 상대 팀에게 득점을 부여
match_teams = match_info[['game_id', 'home_team_id', 'away_team_id']].drop_duplicates()
match_teams['home_team_id'] = match_teams['home_team_id'].astype(str)
match_teams['away_team_id'] = match_teams['away_team_id'].astype(str)

goal_kickoffs['team_id_str'] = goal_kickoffs['team_id'].astype(int).astype(str)
goal_kickoffs = goal_kickoffs.merge(match_teams, on='game_id', how='left')

def get_scorer(row):
    if row['team_id_str'] == row['home_team_id']: return 'Away' # 홈팀 킥오프 -> 어웨이 득점
    elif row['team_id_str'] == row['away_team_id']: return 'Home' # 어웨이팀 킥오프 -> 홈 득점
    return None

goal_kickoffs['scorer'] = goal_kickoffs.apply(get_scorer, axis=1)

# 3-3. 스코어 변동 사항을 메인 데이터에 반영
# 게임별, 에피소드별 스코어 변동량 집계 (중복 방지를 위해 그룹화)
score_changes = goal_kickoffs[['game_id', 'game_episode', 'scorer']].copy()
score_changes['home_change'] = (score_changes['scorer'] == 'Home').astype(int)
score_changes['away_change'] = (score_changes['scorer'] == 'Away').astype(int)

# 에피소드별 총 변동량 (보통 0 아니면 1)
score_changes_grp = score_changes.groupby(['game_id', 'game_episode'])[['home_change', 'away_change']].sum().reset_index()

# 메인 데이터와 병합
train = train.merge(score_changes_grp, on=['game_id', 'game_episode'], how='left')
train['home_change'] = train['home_change'].fillna(0)
train['away_change'] = train['away_change'].fillna(0)

# [중요] 에피소드 내의 모든 row에 점수가 더해지지 않도록 마스킹 처리
# 'is_new_episode'가 True인 행(에피소드 시작점)에만 점수 변동 적용
train['home_inc'] = np.where(train['is_new_episode'], train['home_change'], 0)
train['away_inc'] = np.where(train['is_new_episode'], train['away_change'], 0)

# 누적 합(Cumsum)으로 현재 스코어 계산
train['home_score'] = train.groupby('game_id')['home_inc'].cumsum()
train['away_score'] = train.groupby('game_id')['away_inc'].cumsum()

# 4. 검증 (Validation)
# 계산된 최종 스코어
calc_final = train.groupby('game_id')[['home_score', 'away_score']].max().reset_index()
# 실제 스코어
actual_final = match_info[['game_id', 'home_score', 'away_score']].drop_duplicates()

validation = actual_final.merge(calc_final, on='game_id', suffixes=('_actual', '_calc')).fillna(0)

# 유효성 조건: 실제 스코어가 -1(결측)이 아니고, 계산된 스코어와 실제 스코어가 정확히 일치해야 함
validation['is_valid_data'] = (validation['home_score_actual'] != -1) & (validation['away_score_actual'] != -1)
validation['is_perfect'] = (validation['home_score_actual'] == validation['home_score_calc']) & \
                           (validation['away_score_actual'] == validation['away_score_calc'])

good_ids = validation[validation['is_valid_data'] & validation['is_perfect']]['game_id'].unique()
bad_ids = validation[~validation['is_perfect'] | ~validation['is_valid_data']]['game_id'].unique()

print(f"Good Games Count: {len(good_ids)}") # 예상: 174
print(f"Bad Games Count: {len(bad_ids)}")   # 예상: 54 (결측 포함)

# 5. 파일 분리 및 저장
# Good Data: 학습용 (스코어 포함)
score_good_df = train[train['game_id'].isin(good_ids)].copy()
# 불필요한 임시 컬럼 제거
cols_to_drop = ['home_change', 'away_change', 'home_inc', 'away_inc', 'prev_episode', 'is_new_episode', 'is_kickoff']
score_good_df.drop(columns=[c for c in cols_to_drop if c in score_good_df.columns], inplace=True)
score_good_df.to_csv('score_good.csv', index=False)

# Bad Data: 디버깅용 (비교 정보 포함)
score_bad_df = validation[validation['game_id'].isin(bad_ids)].copy()
score_bad_df['diff_home'] = score_bad_df['home_score_actual'] - score_bad_df['home_score_calc']
score_bad_df['diff_away'] = score_bad_df['away_score_actual'] - score_bad_df['away_score_calc']
score_bad_df.to_csv('score_bad.csv', index=False)

print("Files saved: 'score_good.csv', 'score_bad.csv'")