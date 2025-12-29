import os
import glob
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. Configuration
# ==========================================
class Config:
    TRAIN_DIR = './data/train'
    VAL_DIR = './data/val'
    MODEL_DIR = './weights_lgbm'
    
    # 데이터 상수
    MAX_X = 105.0
    MAX_Y = 68.0
    MAX_TIME = 5700.0
    
    # LGBM 파라미터 (과적합 방지 위주 설정)
    LGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,         # 트리의 복잡도
        'max_depth': 7,           # 트리 깊이 제한 (과적합 방지)
        'min_child_samples': 20,  # 리프 노드 최소 데이터 수
        'subsample': 0.8,         # 데이터 샘플링 (Bagging)
        'colsample_bytree': 0.8,  # 컬럼 샘플링
        'reg_alpha': 0.1,         # L1 규제
        'reg_lambda': 0.1,        # L2 규제
        'random_state': 42,
        'n_jobs': -1
    }

# ==========================================
# 1. Feature Engineering (핵심)
# ==========================================
def extract_features(df):
    """
    시계열 데이터프레임을 받아서, 마지막 시점의 예측을 위한 
    1개의 행(Row)으로 요약된 Feature를 반환합니다.
    """
    # 데이터가 비어있으면 None 반환
    if len(df) < 1: return None
    
    # 정렬 (시간 순)
    df = df.sort_values('time_seconds')
    
    # --- [1] 타겟 (정답) 추출 ---
    # 우리가 맞춰야 할 것은 '마지막 동작'의 end_x, end_y 입니다.
    target_x = df.iloc[-1]['end_x']
    target_y = df.iloc[-1]['end_y']
    
    # --- [2] 기본 Features (마지막 상태) ---
    last_row = df.iloc[-1]
    
    features = {
        # 위치 정보 (Normalize 안 해도 Tree는 잘 찾지만, 스케일 맞춤)
        'start_x': last_row['start_x'],
        'start_y': last_row['start_y'],
        'time_seconds': last_row['time_seconds'],
        
        # 범주형 정보 (Category로 변환 예정)
        'team_id': last_row['team_id'],
        'player_id': last_row['player_id'], # 카디널리티가 높지만 일단 포함
        'type_name': last_row['type_name'], # Action Type
        
        # 통계 정보
        'phase_duration': df['time_seconds'].max() - df['time_seconds'].min(),
        'phase_event_count': len(df),
        'total_dist_x': df['end_x'].iloc[-1] - df['start_x'].iloc[0],
        'total_dist_y': df['end_y'].iloc[-1] - df['start_y'].iloc[0],
    }
    
    # --- [3] Lag Features (과거 이력) ---
    # 직전(t-1) 동작 정보
    if len(df) >= 2:
        prev_1 = df.iloc[-2]
        features.update({
            'prev1_start_x': prev_1['start_x'],
            'prev1_start_y': prev_1['start_y'],
            'prev1_end_x': prev_1['end_x'],
            'prev1_end_y': prev_1['end_y'],
            'prev1_type': prev_1['type_name'],
            'prev1_team': prev_1['team_id']
        })
    else:
        # 역사가 없으면 -1 또는 0으로 채움
        features.update({
            'prev1_start_x': -1, 'prev1_start_y': -1,
            'prev1_end_x': -1, 'prev1_end_y': -1,
            'prev1_type': 'None', 'prev1_team': -1
        })
        
    # 전전(t-2) 동작 정보
    if len(df) >= 3:
        prev_2 = df.iloc[-3]
        features.update({
            'prev2_start_x': prev_2['start_x'],
            'prev2_start_y': prev_2['start_y'],
            'prev2_end_x': prev_2['end_x'],
            'prev2_end_y': prev_2['end_y'],
            'prev2_type': prev_2['type_name']
        })
    else:
        features.update({
            'prev2_start_x': -1, 'prev2_start_y': -1,
            'prev2_end_x': -1, 'prev2_end_y': -1,
            'prev2_type': 'None'
        })

    return features, target_x, target_y

def load_and_preprocess(data_dir, mode='train'):
    print(f"🔄 Loading CSVs from {data_dir}...")
    files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    feature_list = []
    target_x_list = []
    target_y_list = []
    
    # 진행상황 표시
    for fpath in tqdm(files):
        try:
            df = pd.read_csv(fpath)
            # Feature Engineering
            feats, tx, ty = extract_features(df)
            
            if feats is not None:
                feature_list.append(feats)
                target_x_list.append(tx)
                target_y_list.append(ty)
        except Exception as e:
            continue
            
    # DataFrame 변환
    X = pd.DataFrame(feature_list)
    y_x = np.array(target_x_list)
    y_y = np.array(target_y_list)
    
    # 범주형 데이터 처리
    cat_cols = ['type_name', 'prev1_type', 'prev2_type', 'team_id', 'prev1_team', 'player_id']
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
            
    return X, y_x, y_y

# ==========================================
# 2. Training Engine
# ==========================================
def run_training():
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # 1. 데이터 로드 및 전처리
    print("📂 Preparing Train Data...")
    X_train, y_x_train, y_y_train = load_and_preprocess(Config.TRAIN_DIR)
    
    print("📂 Preparing Val Data...")
    X_val, y_x_val, y_y_val = load_and_preprocess(Config.VAL_DIR)
    
    print(f"✅ Data Shape: Train {X_train.shape}, Val {X_val.shape}")
    
    # 2. 모델 학습 (X 좌표용, Y 좌표용 따로 학습)
    print("\n🚀 Training Model for X Coordinate...")
    model_x = lgb.LGBMRegressor(**Config.LGBM_PARAMS)
    model_x.fit(
        X_train, y_x_train,
        eval_set=[(X_val, y_x_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    print("\n🚀 Training Model for Y Coordinate...")
    model_y = lgb.LGBMRegressor(**Config.LGBM_PARAMS)
    model_y.fit(
        X_train, y_y_train,
        eval_set=[(X_val, y_y_val)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
    )
    
    # 3. 평가 (Euclidean Distance)
    print("\n📊 Evaluating...")
    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)
    
    # 거리 오차 계산
    diff_x = pred_x - y_x_val
    diff_y = pred_y - y_y_val
    dist = np.sqrt(diff_x**2 + diff_y**2)
    avg_dist = np.mean(dist)
    
    print(f"   >>> Validation Mean Distance Error: {avg_dist:.4f} m")
    
    # 4. 저장
    print("\n💾 Saving Models...")
    joblib.dump(model_x, os.path.join(Config.MODEL_DIR, 'lgbm_model_x.pkl'))
    joblib.dump(model_y, os.path.join(Config.MODEL_DIR, 'lgbm_model_y.pkl'))
    
    # Feature Importance 출력 (디버깅용)
    print("\n🔍 Top 5 Feature Importance (X-Model):")
    importances = pd.DataFrame({
        'feature': X_train.columns, 
        'importance': model_x.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances.head(5))

if __name__ == '__main__':
    run_training()