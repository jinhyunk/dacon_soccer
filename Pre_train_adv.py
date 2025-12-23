import pandas as pd
import numpy as np
import os
import glob
import random
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. Feature Extraction (분포 비교용 특징 추출)
# ==========================================
def extract_features(df):
    """
    에피소드 하나에 대한 통계적 특징을 추출합니다.
    (Classifier가 Train vs Test를 구분하는 데 사용할 Input)
    """
    stats = {}
    
    # 위치 통계
    stats['x_mean'] = df['start_x'].mean()
    stats['y_mean'] = df['start_y'].mean()
    stats['x_std'] = df['start_x'].std()
    stats['y_std'] = df['start_y'].std()
    
    # 에피소드 길이 및 시간
    stats['event_count'] = len(df)
    if len(df) > 1:
        stats['duration'] = df['time_seconds'].values[-1] - df['time_seconds'].values[0]
    else:
        stats['duration'] = 0
        
    # 주요 액션 빈도 (스타일 차이 반영)
    action_types = ['Pass', 'Carry', 'Ball Recovery', 'Duel']
    type_counts = df['type_name'].value_counts()
    for act in action_types:
        # 전체 길이 대비 비율
        stats[f'ratio_{act}'] = type_counts.get(act, 0) / len(df)
        
    return pd.Series(stats)

# ==========================================
# 2. Main Processing Function
# ==========================================
def process_and_save_adversarial(train_csv_path='train.csv', test_dir='./open_track/test', output_dir='./data_test', val_ratio=0.10):
    """
    1. train.csv와 test 폴더의 데이터를 읽어 분포 차이를 학습합니다.
    2. Train 데이터 중 Test와 가장 비슷한(Adversarial Score가 높은) 에피소드를 Validation으로 선정합니다.
    3. 결과를 ./data_test/train과 ./data_test/val 폴더에 개별 CSV로 저장합니다.
    """
    
    # ---------------------------------------------------------
    # Step 1: 데이터 로드 및 전처리 (Phase 추가)
    # ---------------------------------------------------------
    print("📂 1. Train 데이터 로드 중...")
    train_df = pd.read_csv(train_csv_path)
    
    if 'phase' not in train_df.columns:
        print("   -> Phase 컬럼 생성 중...")
        train_df['phase'] = (train_df['team_id'] != train_df['team_id'].shift(1)).fillna(0).cumsum()

    # ---------------------------------------------------------
    # Step 2: Adversarial Validation을 위한 특징 추출
    # ---------------------------------------------------------
    print("\n📊 2. 에피소드별 특징 추출 중 (Adversarial Validation)...")
    
    # (A) Train 에피소드 특징 추출
    # game_id, episode_id를 인덱스로 사용
    train_groups = train_df.groupby(['game_id', 'episode_id'])
    train_features = train_groups.apply(extract_features)
    train_features['is_test'] = 0  # 레이블: Train
    
    # (B) Test 에피소드 특징 추출
    test_files = glob.glob(os.path.join(test_dir, '**', '*.csv'), recursive=True)
    print(f"   -> Test 파일 {len(test_files)}개 발견. 특징 추출 시작...")
    
    test_feature_list = []
    for fpath in tqdm(test_files, desc="Processing Test Files"):
        try:
            temp_df = pd.read_csv(fpath)
            if len(temp_df) < 1: continue
            # 분포를 정확히 맞추기 위해 Test 데이터도 전처리(NaN 등) 처리 후 특징 추출
            temp_df = temp_df.fillna(0) 
            feats = extract_features(temp_df)
            test_feature_list.append(feats)
        except:
            continue
            
    test_features = pd.DataFrame(test_feature_list)
    test_features['is_test'] = 1  # 레이블: Test
    
    # ---------------------------------------------------------
    # Step 3: Classifier 학습 (Train vs Test 구분)
    # ---------------------------------------------------------
    print("\n🤖 3. Train vs Test 분포 차이 학습 중...")
    
    # 데이터 합치기
    full_data = pd.concat([train_features, test_features], axis=0).fillna(0)
    X = full_data.drop('is_test', axis=1)
    y = full_data['is_test']
    
    # 분류기 학습
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    
    # Train 데이터에 대해 "Test 데이터일 확률" 예측
    # (이 확률이 높을수록 Test 데이터와 성질이 비슷함)
    train_X = train_features.drop('is_test', axis=1).fillna(0)
    similarity_scores = clf.predict_proba(train_X)[:, 1]
    
    train_features['similarity'] = similarity_scores
    
    # ---------------------------------------------------------
    # Step 4: Validation Set 선정 (Top Similarity)
    # ---------------------------------------------------------
    print("\n🎯 4. Validation Set 선정 중...")
    
    # 점수 높은 순 정렬
    sorted_episodes = train_features.sort_values('similarity', ascending=False)
    
    n_val = int(len(sorted_episodes) * val_ratio)
    val_indices = sorted_episodes.index[:n_val]  # (game_id, episode_id) 튜플 리스트
    
    # 빠른 조회를 위해 Set으로 변환
    val_keys = set(val_indices)
    
    print(f"   -> 전체 에피소드: {len(sorted_episodes)}")
    print(f"   -> Validation 선정: {len(val_keys)} (상위 {val_ratio*100:.1f}%)")
    
    # ---------------------------------------------------------
    # Step 5: 파일 저장 (폴더 분리)
    # ---------------------------------------------------------
    print("\n💾 5. 파일 저장 시작...")
    
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # groupby 객체를 다시 순회하며 저장
    for (game_id, episode_id), group in tqdm(train_groups, desc="Saving CSVs"):
        save_name = f"{game_id}_{episode_id}.csv"
        
        if (game_id, episode_id) in val_keys:
            save_path = os.path.join(val_dir, save_name)
        else:
            save_path = os.path.join(train_dir, save_name)
            
        group.to_csv(save_path, index=False)
        
    print("\n✅ 모든 작업 완료!")
    print(f"   Train saved to: {train_dir}")
    print(f"   Val saved to:   {val_dir}")

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    # 경로 설정 (사용자 환경에 맞게 수정 가능)
    TRAIN_CSV = './open_track1/train.csv'
    TEST_FOLDER = './open_track1/test'
    OUTPUT_FOLDER = './data'
    
    process_and_save_adversarial(
        train_csv_path=TRAIN_CSV,
        test_dir=TEST_FOLDER,
        output_dir=OUTPUT_FOLDER,
        val_ratio=0.1  # 10%를 Validation으로 사용
    )