import os
import glob
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    TEST_DIR = './open_track1/test' # 테스트 데이터 경로
    MODEL_DIR = './weights_lgbm'
    OUTPUT_FILE = './submission_lgbm.csv'
    
    MAX_X = 105.0
    MAX_Y = 68.0

# ==========================================
# 1. Feature Engineering (Train과 동일)
# ==========================================
def extract_features_for_test(df):
    if len(df) < 1: return None
    
    df = df.sort_values('time_seconds')
    last_row = df.iloc[-1]
    
    features = {
        'start_x': last_row['start_x'],
        'start_y': last_row['start_y'],
        'time_seconds': last_row['time_seconds'],
        'team_id': last_row['team_id'],
        'player_id': last_row['player_id'],
        'type_name': last_row['type_name'],
        'phase_duration': df['time_seconds'].max() - df['time_seconds'].min(),
        'phase_event_count': len(df),
        'total_dist_x': df['end_x'].iloc[-1] - df['start_x'].iloc[0],
        'total_dist_y': df['end_y'].iloc[-1] - df['start_y'].iloc[0],
    }
    
    if len(df) >= 2:
        prev_1 = df.iloc[-2]
        features.update({
            'prev1_start_x': prev_1['start_x'], 'prev1_start_y': prev_1['start_y'],
            'prev1_end_x': prev_1['end_x'], 'prev1_end_y': prev_1['end_y'],
            'prev1_type': prev_1['type_name'], 'prev1_team': prev_1['team_id']
        })
    else:
        features.update({
            'prev1_start_x': -1, 'prev1_start_y': -1,
            'prev1_end_x': -1, 'prev1_end_y': -1,
            'prev1_type': 'None', 'prev1_team': -1
        })
        
    if len(df) >= 3:
        prev_2 = df.iloc[-3]
        features.update({
            'prev2_start_x': prev_2['start_x'], 'prev2_start_y': prev_2['start_y'],
            'prev2_end_x': prev_2['end_x'], 'prev2_end_y': prev_2['end_y'],
            'prev2_type': prev_2['type_name']
        })
    else:
        features.update({
            'prev2_start_x': -1, 'prev2_start_y': -1,
            'prev2_end_x': -1, 'prev2_end_y': -1,
            'prev2_type': 'None'
        })

    return features

# ==========================================
# 2. Inference Engine
# ==========================================
def run_inference():
    # 모델 로드
    print("🔄 Loading Models...")
    try:
        model_x = joblib.load(os.path.join(Config.MODEL_DIR, 'lgbm_model_x.pkl'))
        model_y = joblib.load(os.path.join(Config.MODEL_DIR, 'lgbm_model_y.pkl'))
    except:
        print("❌ Model not found. Run training first.")
        return

    # 테스트 파일 스캔
    print("📂 Scanning Test Files...")
    files = glob.glob(os.path.join(Config.TEST_DIR, '**', '*.csv'), recursive=True)
    
    results = []
    
    for fpath in tqdm(files, desc="Predicting"):
        try:
            df = pd.read_csv(fpath)
            game_episode_id = os.path.splitext(os.path.basename(fpath))[0]
            
            # Feature Extraction
            feats = extract_features_for_test(df)
            
            if feats is None: continue
            
            # DataFrame 변환 (1 row)
            X_test = pd.DataFrame([feats])
            
            # 범주형 처리 (Train과 동일하게)
            cat_cols = ['type_name', 'prev1_type', 'prev2_type', 'team_id', 'prev1_team', 'player_id']
            for col in cat_cols:
                if col in X_test.columns:
                    X_test[col] = X_test[col].astype('category')
            
            # Predict
            pred_x = model_x.predict(X_test)[0]
            pred_y = model_y.predict(X_test)[0]
            
            # Clipping (경기장 밖으로 나가는 것 방지)
            pred_x = np.clip(pred_x, 0, Config.MAX_X)
            pred_y = np.clip(pred_y, 0, Config.MAX_Y)
            
            results.append({
                'game_episode': game_episode_id,
                'end_x': float(pred_x),
                'end_y': float(pred_y)
            })
            
        except Exception as e:
            print(f"Error in {fpath}: {e}")
            continue
            
    # Save Submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✅ Saved submission to {Config.OUTPUT_FILE} (Rows: {len(sub_df)})")

if __name__ == '__main__':
    run_inference()