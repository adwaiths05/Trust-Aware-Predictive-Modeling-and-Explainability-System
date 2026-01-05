import joblib
import shap
import pandas as pd
import numpy as np
import config

class ModelExplainer:
    def __init__(self):
        self.artifacts = self._load_artifacts()
        
    def _load_artifacts(self):
        path = config.MODELS_DIR / config.MODEL_FILENAME
        if not path.exists():
            raise FileNotFoundError("Model not found. Please train a model first.")
        return joblib.load(path)

    def explain_global(self):
        """Returns global feature importance dictionary."""
        pipeline = self.artifacts['pipeline']
        # Access the step named 'model' (was 'classifier' in old version)
        if 'model' in pipeline.named_steps:
            model = pipeline.named_steps['model']
        else:
            model = pipeline.named_steps['classifier'] # Fallback for older saves
        
        preprocessor = pipeline.named_steps['preprocessor']
        
        try:
            num_cols = self.artifacts['num_cols']
            cat_cols = self.artifacts['cat_cols']
            
            # Reconstruct feature names from OneHotEncoder
            if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
                ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
            else:
                ohe_feature_names = cat_cols # Fallback
                
            all_features = num_cols + list(ohe_feature_names)
            
            # Get importance
            importances = model.feature_importances_
            
            # Create dict, sort desc, take top 10
            feat_imp = sorted(zip(all_features, importances), key=lambda x: x[1], reverse=True)[:10]
            return dict(feat_imp)
        except Exception as e:
            print(f"Explain Global Error: {e}")
            return {"error": "Could not extract global feature importance"}

    def explain_local(self, input_data: dict):
        """Calculates SHAP values for a single prediction."""
        pipeline = self.artifacts['pipeline']
        
        # 1. Get Model & Preprocessor
        if 'model' in pipeline.named_steps:
            model = pipeline.named_steps['model']
        else:
            model = pipeline.named_steps['classifier']
            
        preprocessor = pipeline.named_steps['preprocessor']
        X_sample = self.artifacts['X_train_sample']
        problem_type = self.artifacts.get('problem_type', 'classification')
        
        # 2. Preprocess Input (Raw Dict -> Transformed Array)
        input_df = pd.DataFrame([input_data])
        
        # Transform using the exact pipeline logic
        transformed_input = preprocessor.transform(input_df)
        
        # 3. Compute SHAP
        # TreeExplainer is best for LightGBM/CatBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_input)
        
        # 4. Handle Shape Differences (Regression vs Classification)
        if problem_type == 'classification':
            # Binary Classification: shap_values is a list of [Class0_Array, Class1_Array]
            if isinstance(shap_values, list):
                sv = shap_values[1][0] # We want contributions to Class 1 (Positive)
            else:
                # Some newer SHAP versions might return single array for binary
                sv = shap_values[0]
        else:
            # Regression: shap_values is a single array
            if len(shap_values.shape) > 1:
                sv = shap_values[0]
            else:
                sv = shap_values

        # 5. Map SHAP values back to Feature Names
        num_cols = self.artifacts['num_cols']
        cat_cols = self.artifacts['cat_cols']
        
        if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
        else:
            ohe_names = cat_cols

        feature_names = num_cols + list(ohe_names)
        
        # Safety check for length mismatch
        if len(feature_names) != len(sv):
            # If lengths don't match, return raw dict by index (failsafe)
            return {f"Feature_{i}": val for i, val in enumerate(sv)}
            
        return dict(zip(feature_names, sv))