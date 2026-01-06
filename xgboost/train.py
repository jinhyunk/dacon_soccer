"""
XGBoost 모델 학습 스크립트
"""
import pandas as pd
import numpy as np

from config import (
    TRAIN_PATH,
    SEQUENCE_MODE,
    XGB_PARAMS,
    EARLY_STOPPING_ROUNDS,
    log_config,
    ensure_model_dir,
)
from dataset import build_dataset, split_dataset
from model import (
    XGBCoordinatePredictor,
    evaluate_predictions,
    print_metrics,
)


def main():
    # 설정 출력
    log_config()
    ensure_model_dir()
    
    # 데이터 로드
    print(f"\nLoading data from {TRAIN_PATH}...")
    df = pd.read_csv(TRAIN_PATH)
    print(f"Data shape: {df.shape}")
    
    # 데이터셋 생성
    print("\nBuilding dataset...")
    features_df, targets, game_ids = build_dataset(
        df, mode=SEQUENCE_MODE, return_game_ids=True
    )
    
    # Train/Valid 분리 (game_id 기준)
    print("\nSplitting dataset by game_id...")
    X_train, X_valid, y_train, y_valid, feature_cols = split_dataset(
        features_df, targets, game_ids,
        test_size=0.2,
        split_by_game=True,
    )
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(f"  {feature_cols[:10]}...")
    
    # 모델 생성 및 학습
    print("\n" + "=" * 60)
    print("Starting XGBoost Training")
    print("=" * 60)
    
    model = XGBCoordinatePredictor(params=XGB_PARAMS)
    model.fit(
        X_train, y_train,
        X_valid, y_valid,
        feature_cols=feature_cols,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    
    # 검증 성능 평가
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    
    # Train 평가
    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_predictions(y_train, y_train_pred)
    print_metrics(train_metrics, prefix="[Train] ")
    
    # Valid 평가
    y_valid_pred = model.predict(X_valid)
    valid_metrics = evaluate_predictions(y_valid, y_valid_pred)
    print_metrics(valid_metrics, prefix="[Valid] ")
    
    # 피처 중요도 출력
    model.print_feature_importance(top_k=15)
    
    # SHAP 분석 (validation 데이터 사용)
    model.print_shap_importance(X_valid, top_k=15)
    
    # 모델 저장
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    model.save()
    
    print("\nTraining completed!")
    print(f"Best Validation Mean Distance: {valid_metrics['mean_distance']:.4f} m")


if __name__ == "__main__":
    main()

