import pandas as pd
import numpy as np
import joblib
import itertools
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

# --- COMPLEX MODELS ---
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

# --- SIMPLE MODELS (New!) ---
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import config

class ModelTrainer:
    def __init__(self):
        self.pipeline = None
        self.label_encoder = None

    def detect_problem_type(self, df, target):
        y = df[target]
        if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
            return "classification"
        if pd.api.types.is_numeric_dtype(y) and y.nunique() < 20: 
            return "classification"
        return "regression"

    def get_cv_score(self, X_sub, y, problem_type):
        """Helper to score features using a fast proxy (LightGBM)."""
        if X_sub.shape[1] == 0: return -np.inf
        
        cat_cols = X_sub.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X_sub.select_dtypes(include=['number']).columns.tolist()
        
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        
        if problem_type == "classification":
            model = LGBMClassifier(n_estimators=30, random_state=42, verbose=-1, n_jobs=1)
            scoring = 'accuracy'
            cv = StratifiedKFold(3, shuffle=True, random_state=42)
        else:
            model = LGBMRegressor(n_estimators=30, random_state=42, verbose=-1, n_jobs=1)
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(3, shuffle=True, random_state=42)
            
        pipe = Pipeline([('pre', pre), ('model', model)])
        try:
            return cross_val_score(pipe, X_sub, y, cv=cv, scoring=scoring).mean()
        except:
            return -np.inf

    def train(self, df: pd.DataFrame, target: str, problem_type: str = "auto", auto_optimize: bool = False):
        
        # 1. SETUP
        detected_type = self.detect_problem_type(df, target)
        X = df.drop(columns=[target])
        y = df[target]
        
        if detected_type == "classification":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            print(f"🕵️ Detected: Classification (Encoded classes: {self.label_encoder.classes_})")
        else:
            self.label_encoder = None
            print(f"🕵️ Detected: Regression")

        all_features = X.columns.tolist()
        final_features = all_features
        opt_status = "Standard Training"

        # 2. FEATURE SELECTION (Hybrid Engine)
        if auto_optimize:
            n_features = len(all_features)
            # Use Brute Force for small data
            if n_features <= 12:
                print(f"🔹 Optimization: Exhaustive Search...")
                opt_status = "Exhaustive Search"
                best_score = -np.inf
                for L in range(1, n_features + 1):
                    for subset in itertools.combinations(all_features, L):
                        cols = list(subset)
                        score = self.get_cv_score(X[cols], y, detected_type)
                        if score > best_score:
                            best_score = score
                            final_features = cols
            # Use Stepwise for large data
            else:
                print(f"🔸 Optimization: Stepwise Selection...")
                opt_status = "Stepwise Selection"
                current_features = all_features
                best_score = self.get_cv_score(X[current_features], y, detected_type)
                improved = True
                while improved and len(current_features) > 1:
                    improved = False
                    best_subset = None
                    best_subset_score = -np.inf
                    for feat in current_features:
                        temp_cols = [f for f in current_features if f != feat]
                        score = self.get_cv_score(X[temp_cols], y, detected_type)
                        if score > best_subset_score:
                            best_subset_score = score
                            best_subset = temp_cols
                    if best_subset_score >= best_score:
                        best_score = best_subset_score
                        current_features = best_subset
                        improved = True
                final_features = current_features

        # 3. EXPANDED MODEL TOURNAMENT (Simple vs Complex)
        print(f"⚔️ Model Tournament on {len(final_features)} features...")
        X_final = X[final_features]
        
        candidates = {}
        
        if detected_type == "classification":
            scoring = 'accuracy'
            cv = StratifiedKFold(3, shuffle=True, random_state=42)
            
            # --- The Contenders ---
            # 1. The Simple Baseline
            candidates["LogisticRegression"] = LogisticRegression(max_iter=1000, random_state=42)
            candidates["DecisionTree"] = DecisionTreeClassifier(max_depth=5, random_state=42)
            
            # 2. The Heavy Hitters
            candidates["RandomForest"] = RandomForestClassifier(n_estimators=50, random_state=42)
            candidates["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
            candidates["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

        else:
            scoring = 'neg_root_mean_squared_error'
            cv = KFold(3, shuffle=True, random_state=42)
            
            # --- The Contenders ---
            # 1. The Simple Baseline (ElasticNet is Linear Reg + Safety Regularization)
            candidates["ElasticNet"] = ElasticNet(random_state=42) 
            candidates["DecisionTree"] = DecisionTreeRegressor(max_depth=5, random_state=42)
            
            # 2. The Heavy Hitters
            candidates["RandomForest"] = RandomForestRegressor(n_estimators=50, random_state=42)
            candidates["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42)
            candidates["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

        # Pipeline Prep
        cat_cols = X_final.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X_final.select_dtypes(include=['number']).columns.tolist()
        pre = ColumnTransformer([
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])

        best_score = -np.inf
        best_name = "LightGBM"
        best_pipe = None
        tournament_results = {}

        for name, model in candidates.items():
            pipe = Pipeline([('pre', pre), ('model', model)])
            try:
                scores = cross_val_score(pipe, X_final, y, cv=cv, scoring=scoring)
                avg = scores.mean()
                tournament_results[name] = avg
                print(f"   👉 {name}: {avg:.4f}")
                
                # Update Winner
                if avg > best_score:
                    best_score = avg
                    best_name = name
                    best_pipe = pipe
            except Exception as e:
                print(f"   ❌ {name} Failed: {e}")

        # 4. FINAL FIT
        self.pipeline = best_pipe
        self.pipeline.fit(X_final, y)
        preds = self.pipeline.predict(X_final)
        
        metrics = {}
        if detected_type == "classification":
            metrics["accuracy"] = round(accuracy_score(y, preds), 3)
            metrics["f1"] = round(f1_score(y, preds, average='weighted'), 3)
        else:
            metrics["rmse"] = round(mean_squared_error(y, preds, squared=False), 3)
            metrics["r2"] = round(r2_score(y, preds), 3)

        metrics["detected_type"] = detected_type
        metrics["kept_features"] = final_features
        metrics["optimization_status"] = f"{opt_status} + {best_name}"
        metrics["tournament_results"] = tournament_results
        
        le_classes = self.label_encoder.classes_.tolist() if self.label_encoder else None

        joblib.dump({
            "pipeline": self.pipeline,
            "features": final_features,
            "target": target,
            "problem_type": detected_type,
            "label_classes": le_classes,
            "cat_cols": cat_cols,
            "num_cols": num_cols,
            "X_train_sample": X_final.head(50)
        }, config.MODELS_DIR / config.MODEL_FILENAME)
        
        return metrics