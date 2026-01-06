"""
XGBoost 모델 래퍼
Multi-output (X, Y 동시 예측) 또는 Separate (X, Y 별도 예측) 지원
"""
from typing import Tuple, List, Optional
import numpy as np
import json
import os

import xgboost as xgb

from config import (
    XGB_PARAMS, MODEL_PATH_X, MODEL_PATH_Y, MODEL_PATH_MULTI, 
    MODEL_DIR, USE_MULTI_OUTPUT
)


class XGBCoordinatePredictor:
    """
    좌표 예측을 위한 XGBoost 모델 래퍼
    
    USE_MULTI_OUTPUT=True: 하나의 트리에서 X, Y 동시 예측 (multi-output)
    USE_MULTI_OUTPUT=False: X, Y 별도 모델로 예측 (separate)
    """
    
    def __init__(self, params: dict = None, use_multi_output: bool = None):
        """
        Args:
            params: XGBoost 하이퍼파라미터 (None이면 기본값 사용)
            use_multi_output: Multi-output 모드 사용 여부 (None이면 config 값 사용)
        """
        self.params = params or XGB_PARAMS.copy()
        self.use_multi_output = use_multi_output if use_multi_output is not None else USE_MULTI_OUTPUT
        
        # Multi-output 모델 또는 별도 모델
        self.model_multi: Optional[xgb.XGBRegressor] = None  # multi-output용
        self.model_x: Optional[xgb.XGBRegressor] = None      # separate용
        self.model_y: Optional[xgb.XGBRegressor] = None      # separate용
        
        # 피처 컬럼 저장 (추론 시 동일한 순서 보장)
        self.feature_cols: Optional[List[str]] = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        feature_cols: List[str] = None,
        early_stopping_rounds: int = 50,
    ) -> Tuple[dict, dict]:
        """
        모델 학습
        
        Args:
            X_train: (N, n_features) 학습 피처
            y_train: (N, 2) 타깃 좌표 [x, y]
            X_valid: (M, n_features) 검증 피처
            y_valid: (M, 2) 검증 타깃
            feature_cols: 피처 컬럼 이름 리스트
            early_stopping_rounds: early stopping rounds
        
        Returns:
            history: 학습 히스토리
        """
        self.feature_cols = feature_cols
        
        if self.use_multi_output:
            return self._fit_multi_output(X_train, y_train, X_valid, y_valid, early_stopping_rounds)
        else:
            return self._fit_separate(X_train, y_train, X_valid, y_valid, early_stopping_rounds)
    
    def _fit_multi_output(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        early_stopping_rounds: int = 50,
    ) -> Tuple[dict, dict]:
        """Multi-output 모드로 학습 (하나의 트리에서 X, Y 동시 예측)"""
        print("\n" + "="*50)
        print("[Training Multi-Output Model (X, Y jointly)]")
        print("="*50)
        
        # Multi-output 파라미터 설정
        params_multi = self.params.copy()
        # params_multi["multi_strategy"] = "one_output_per_tree"
        params_multi["multi_strategy"] = "multi_output_tree"  # 하나의 트리에서 여러 출력
        
        if X_valid is not None and y_valid is not None and early_stopping_rounds > 0:
            params_multi["early_stopping_rounds"] = early_stopping_rounds
            print(f"Early stopping enabled: {early_stopping_rounds} rounds")
        
        self.model_multi = xgb.XGBRegressor(**params_multi)
        
        # eval_set 구성
        eval_set = [(X_train, y_train)]
        if X_valid is not None and y_valid is not None:
            eval_set.append((X_valid, y_valid))
        
        self.model_multi.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True,
        )
        
        # Early stopping 결과 출력
        print("\n" + "="*50)
        print("[Early Stopping Summary]")
        print("="*50)
        
        if hasattr(self.model_multi, 'best_iteration') and self.model_multi.best_iteration is not None:
            print(f"Best iteration: {self.model_multi.best_iteration}")
            if hasattr(self.model_multi, 'best_score') and self.model_multi.best_score is not None:
                print(f"Best score (RMSE): {self.model_multi.best_score:.6f}")
        else:
            print(f"No early stopping (used all {self.params.get('n_estimators', 'N/A')} trees)")
        
        print("="*50)
        
        return {}, {}
    
    def _fit_separate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: np.ndarray = None,
        y_valid: np.ndarray = None,
        early_stopping_rounds: int = 50,
    ) -> Tuple[dict, dict]:
        """Separate 모드로 학습 (X, Y 별도 모델)"""
        # X 좌표 모델 학습
        print("\n[Training X coordinate model (Separate)]")
        
        params_with_es = self.params.copy()
        if X_valid is not None and y_valid is not None and early_stopping_rounds > 0:
            params_with_es["early_stopping_rounds"] = early_stopping_rounds
            print(f"  Early stopping enabled: {early_stopping_rounds} rounds")
        
        self.model_x = xgb.XGBRegressor(**params_with_es)
        
        eval_set_x = [(X_train, y_train[:, 0])]
        if X_valid is not None and y_valid is not None:
            eval_set_x.append((X_valid, y_valid[:, 0]))
        
        self.model_x.fit(
            X_train, y_train[:, 0],
            eval_set=eval_set_x,
            verbose=True,
        )
        
        # Y 좌표 모델 학습
        print("\n[Training Y coordinate model (Separate)]")
        if X_valid is not None and y_valid is not None and early_stopping_rounds > 0:
            print(f"  Early stopping enabled: {early_stopping_rounds} rounds")
        
        self.model_y = xgb.XGBRegressor(**params_with_es)
        
        eval_set_y = [(X_train, y_train[:, 1])]
        if X_valid is not None and y_valid is not None:
            eval_set_y.append((X_valid, y_valid[:, 1]))
        
        self.model_y.fit(
            X_train, y_train[:, 1],
            eval_set=eval_set_y,
            verbose=True,
        )
        
        # Early stopping 결과 출력
        print("\n" + "="*50)
        print("[Early Stopping Summary]")
        print("="*50)
        
        if hasattr(self.model_x, 'best_iteration'):
            print(f"X model - Best iteration: {self.model_x.best_iteration}")
            print(f"X model - Best score (RMSE): {self.model_x.best_score:.6f}")
        else:
            print(f"X model - No early stopping (used all {self.params.get('n_estimators', 'N/A')} trees)")
        
        if hasattr(self.model_y, 'best_iteration'):
            print(f"Y model - Best iteration: {self.model_y.best_iteration}")
            print(f"Y model - Best score (RMSE): {self.model_y.best_score:.6f}")
        else:
            print(f"Y model - No early stopping (used all {self.params.get('n_estimators', 'N/A')} trees)")
        
        print("="*50)
        
        return {}, {}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        좌표 예측
        
        Args:
            X: (N, n_features) 피처
        
        Returns:
            predictions: (N, 2) 예측 좌표 [x, y]
        """
        if self.use_multi_output:
            if self.model_multi is None:
                raise RuntimeError("모델이 학습되지 않았습니다. fit() 또는 load()를 먼저 호출하세요.")
            return self.model_multi.predict(X)
        else:
            if self.model_x is None or self.model_y is None:
                raise RuntimeError("모델이 학습되지 않았습니다. fit() 또는 load()를 먼저 호출하세요.")
            pred_x = self.model_x.predict(X)
            pred_y = self.model_y.predict(X)
            return np.column_stack([pred_x, pred_y])
    
    def save(self, path_x: str = MODEL_PATH_X, path_y: str = MODEL_PATH_Y, path_multi: str = MODEL_PATH_MULTI) -> None:
        """
        모델 저장
        
        Args:
            path_x: X 좌표 모델 저장 경로 (separate 모드)
            path_y: Y 좌표 모델 저장 경로 (separate 모드)
            path_multi: Multi-output 모델 저장 경로
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        if self.use_multi_output:
            if self.model_multi is None:
                raise RuntimeError("저장할 모델이 없습니다.")
            self.model_multi.save_model(path_multi)
            print(f"Multi-output model saved to: {path_multi}")
        else:
            if self.model_x is None or self.model_y is None:
                raise RuntimeError("저장할 모델이 없습니다.")
            self.model_x.save_model(path_x)
            self.model_y.save_model(path_y)
            print(f"Models saved to:\n  {path_x}\n  {path_y}")
        
        # 피처 컬럼 및 모드 정보 저장
        meta = {
            "feature_cols": self.feature_cols,
            "use_multi_output": self.use_multi_output
        }
        meta_path = os.path.join(MODEL_DIR, "model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    
    def load(self, path_x: str = MODEL_PATH_X, path_y: str = MODEL_PATH_Y, path_multi: str = MODEL_PATH_MULTI) -> None:
        """
        모델 로드
        
        Args:
            path_x: X 좌표 모델 경로 (separate 모드)
            path_y: Y 좌표 모델 경로 (separate 모드)
            path_multi: Multi-output 모델 경로
        """
        # 메타 정보 로드
        meta_path = os.path.join(MODEL_DIR, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
                self.feature_cols = meta.get("feature_cols")
                self.use_multi_output = meta.get("use_multi_output", USE_MULTI_OUTPUT)
        
        if self.use_multi_output:
            self.model_multi = xgb.XGBRegressor()
            self.model_multi.load_model(path_multi)
            print(f"Multi-output model loaded from: {path_multi}")
        else:
            self.model_x = xgb.XGBRegressor()
            self.model_y = xgb.XGBRegressor()
            self.model_x.load_model(path_x)
            self.model_y.load_model(path_y)
            print(f"Models loaded from:\n  {path_x}\n  {path_y}")
    
    def get_feature_importance(self, importance_type: str = "gain") -> dict:
        """
        피처 중요도 반환
        
        Args:
            importance_type: "weight", "gain", "cover" 중 하나
        
        Returns:
            Multi-output: {"combined": importance_dict}
            Separate: {"x": importance_dict, "y": importance_dict}
        """
        if self.use_multi_output:
            if self.model_multi is None:
                raise RuntimeError("모델이 학습되지 않았습니다.")
            importance = self.model_multi.get_booster().get_score(importance_type=importance_type)
            return {"combined": importance}
        else:
            if self.model_x is None or self.model_y is None:
                raise RuntimeError("모델이 학습되지 않았습니다.")
            importance_x = self.model_x.get_booster().get_score(importance_type=importance_type)
            importance_y = self.model_y.get_booster().get_score(importance_type=importance_type)
            return {"x": importance_x, "y": importance_y}
    
    def _map_feature_name(self, feat: str) -> str:
        """피처 이름 매핑 (f0, f1 -> 실제 이름)"""
        if self.feature_cols and feat.startswith("f"):
            try:
                idx = int(feat[1:])
                if idx < len(self.feature_cols):
                    return self.feature_cols[idx]
            except ValueError:
                pass
        return feat
    
    def print_feature_importance(self, top_k: int = 20) -> None:
        """피처 중요도 출력"""
        # imp_type = "weight" if self.use_multi_output else "gain"
        # importance = self.get_feature_importance(importance_type=imp_type)
        importance = self.get_feature_importance(importance_type="gain")
        
        if self.use_multi_output:
            print("\n[Feature Importance - Multi-Output Model (by gain)]")
            sorted_imp = sorted(importance["combined"].items(), key=lambda x: x[1], reverse=True)[:top_k]
            for feat, score in sorted_imp:
                feat_name = self._map_feature_name(feat)
                print(f"  {feat_name}: {score:.4f}")
        else:
            print("\n[Feature Importance - X model (by gain)]")
            sorted_x = sorted(importance["x"].items(), key=lambda x: x[1], reverse=True)[:top_k]
            for feat, score in sorted_x:
                feat_name = self._map_feature_name(feat)
                print(f"  {feat_name}: {score:.4f}")
            
            print("\n[Feature Importance - Y model (by gain)]")
            sorted_y = sorted(importance["y"].items(), key=lambda x: x[1], reverse=True)[:top_k]
            for feat, score in sorted_y:
                feat_name = self._map_feature_name(feat)
                print(f"  {feat_name}: {score:.4f}")
    
    def print_shap_importance(self, X_sample: np.ndarray, top_k: int = 20) -> None:
        """
        SHAP value 기반 피처 중요도 출력 (XGBoost 내장 pred_contribs 사용)
        
        Args:
            X_sample: 샘플 데이터 (SHAP 계산용, 너무 크면 느림)
            top_k: 출력할 상위 피처 개수
        """
        # 모델 체크
        if self.use_multi_output:
            if self.model_multi is None:
                print("\n[SHAP] 모델이 학습되지 않았습니다.")
                return
        else:
            if self.model_x is None or self.model_y is None:
                print("\n[SHAP] 모델이 학습되지 않았습니다.")
                return
        
        # 샘플 크기 제한 (너무 크면 느림)
        max_samples = min(len(X_sample), 1000)
        X_shap = X_sample[:max_samples]
        
        print(f"\n[SHAP Analysis] Using {max_samples} samples...")
        
        def get_shap_values_single(model, X_data):
            """단일 출력 모델에서 SHAP values 추출"""
            dmatrix = xgb.DMatrix(X_data, feature_names=self.feature_cols)
            booster = model.get_booster()
            contribs = booster.predict(dmatrix, pred_contribs=True)
            # 마지막 열은 base value이므로 제외
            return contribs[:, :-1]
        
        def get_shap_values_multi(model, X_data):
            """Multi-output 모델에서 SHAP values 추출"""
            dmatrix = xgb.DMatrix(X_data, feature_names=self.feature_cols)
            booster = model.get_booster()
            # Multi-output의 경우 (n_samples, n_features+1, n_outputs) 형태
            contribs = booster.predict(dmatrix, pred_contribs=True)
            # 마지막 피처(base value) 제외, (n_samples, n_features, n_outputs)
            if contribs.ndim == 3:
                return contribs[:, :-1, :]  # (samples, features, outputs)
            else:
                return contribs[:, :-1]
        
        if self.use_multi_output:
            print("\n[SHAP - Multi-Output Model]")
            try:
                shap_values = get_shap_values_multi(self.model_multi, X_shap)
                
                if shap_values.ndim == 3:
                    # X output (index 0)
                    print("\n  [X output]")
                    mean_abs_x = np.abs(shap_values[:, :, 0]).mean(axis=0)
                    if self.feature_cols:
                        feat_imp_x = list(zip(self.feature_cols, mean_abs_x))
                    else:
                        feat_imp_x = [(f"f{i}", v) for i, v in enumerate(mean_abs_x)]
                    feat_imp_x.sort(key=lambda x: x[1], reverse=True)
                    for feat, score in feat_imp_x[:top_k]:
                        print(f"    {feat}: {score:.6f}")
                    
                    # Y output (index 1)
                    print("\n  [Y output]")
                    mean_abs_y = np.abs(shap_values[:, :, 1]).mean(axis=0)
                    if self.feature_cols:
                        feat_imp_y = list(zip(self.feature_cols, mean_abs_y))
                    else:
                        feat_imp_y = [(f"f{i}", v) for i, v in enumerate(mean_abs_y)]
                    feat_imp_y.sort(key=lambda x: x[1], reverse=True)
                    for feat, score in feat_imp_y[:top_k]:
                        print(f"    {feat}: {score:.6f}")
                    
                    # Combined importance (average across X and Y)
                    print("\n  [Combined (avg of X, Y)]")
                    mean_abs_combined = (mean_abs_x + mean_abs_y) / 2
                    if self.feature_cols:
                        feat_imp_combined = list(zip(self.feature_cols, mean_abs_combined))
                    else:
                        feat_imp_combined = [(f"f{i}", v) for i, v in enumerate(mean_abs_combined)]
                    feat_imp_combined.sort(key=lambda x: x[1], reverse=True)
                    for feat, score in feat_imp_combined[:top_k]:
                        print(f"    {feat}: {score:.6f}")
                else:
                    # 2D인 경우 (fallback)
                    mean_abs = np.abs(shap_values).mean(axis=0)
                    if self.feature_cols:
                        feat_imp = list(zip(self.feature_cols, mean_abs))
                    else:
                        feat_imp = [(f"f{i}", v) for i, v in enumerate(mean_abs)]
                    feat_imp.sort(key=lambda x: x[1], reverse=True)
                    for feat, score in feat_imp[:top_k]:
                        print(f"    {feat}: {score:.6f}")
                        
            except Exception as e:
                print(f"  SHAP 계산 실패: {e}")
                return
        else:
            # Separate 모드 (기존 로직)
            print("\n[SHAP - X model]")
            try:
                shap_values_x = get_shap_values_single(self.model_x, X_shap)
                mean_abs_shap_x = np.abs(shap_values_x).mean(axis=0)
                
                if self.feature_cols:
                    feat_importance_x = list(zip(self.feature_cols, mean_abs_shap_x))
                else:
                    feat_importance_x = [(f"f{i}", v) for i, v in enumerate(mean_abs_shap_x)]
                
                feat_importance_x.sort(key=lambda x: x[1], reverse=True)
                for feat, score in feat_importance_x[:top_k]:
                    print(f"  {feat}: {score:.6f}")
            except Exception as e:
                print(f"  SHAP 계산 실패: {e}")
            
            print("\n[SHAP - Y model]")
            try:
                shap_values_y = get_shap_values_single(self.model_y, X_shap)
                mean_abs_shap_y = np.abs(shap_values_y).mean(axis=0)
                
                if self.feature_cols:
                    feat_importance_y = list(zip(self.feature_cols, mean_abs_shap_y))
                else:
                    feat_importance_y = [(f"f{i}", v) for i, v in enumerate(mean_abs_shap_y)]
                
                feat_importance_y.sort(key=lambda x: x[1], reverse=True)
                for feat, score in feat_importance_y[:top_k]:
                    print(f"  {feat}: {score:.6f}")
            except Exception as e:
                print(f"  SHAP 계산 실패: {e}")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    field_x: float = 105.0,
    field_y: float = 68.0,
) -> dict:
    """
    예측 결과 평가
    
    Args:
        y_true: (N, 2) 정규화된 실제 좌표
        y_pred: (N, 2) 정규화된 예측 좌표
        field_x: 필드 X 크기
        field_y: 필드 Y 크기
    
    Returns:
        평가 지표 딕셔너리
    """
    # 실제 좌표로 변환
    true_x = y_true[:, 0] * field_x
    true_y = y_true[:, 1] * field_y
    pred_x = y_pred[:, 0] * field_x
    pred_y = y_pred[:, 1] * field_y
    
    # 유클리드 거리 계산
    distances = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
    
    metrics = {
        "mean_distance": distances.mean(),
        "std_distance": distances.std(),
        "min_distance": distances.min(),
        "max_distance": distances.max(),
        "median_distance": np.median(distances),
        "p90_distance": np.percentile(distances, 90),
        "mse_x": ((pred_x - true_x)**2).mean(),
        "mse_y": ((pred_y - true_y)**2).mean(),
        "mae_x": np.abs(pred_x - true_x).mean(),
        "mae_y": np.abs(pred_y - true_y).mean(),
    }
    
    return metrics


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """평가 지표 출력"""
    print(f"\n{prefix}[Evaluation Metrics]")
    print(f"  Mean Distance: {metrics['mean_distance']:.4f} m")
    print(f"  Std Distance:  {metrics['std_distance']:.4f} m")
    print(f"  Min Distance:  {metrics['min_distance']:.4f} m")
    print(f"  Max Distance:  {metrics['max_distance']:.4f} m")
    print(f"  Median:        {metrics['median_distance']:.4f} m")
    print(f"  90th Percentile: {metrics['p90_distance']:.4f} m")
    print(f"  MSE (X): {metrics['mse_x']:.4f}")
    print(f"  MSE (Y): {metrics['mse_y']:.4f}")
    print(f"  MAE (X): {metrics['mae_x']:.4f} m")
    print(f"  MAE (Y): {metrics['mae_y']:.4f} m")

