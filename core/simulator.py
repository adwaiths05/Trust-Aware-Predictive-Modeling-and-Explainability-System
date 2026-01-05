import pandas as pd
import dice_ml
import joblib
import numpy as np
import config

class CounterfactualEngine:
    def __init__(self):
        self.artifacts = self._load_artifacts()
        
    def _load_artifacts(self):
        path = config.MODELS_DIR / config.MODEL_FILENAME
        if not path.exists():
            raise FileNotFoundError("Model not found. Please train a model first.")
        return joblib.load(path)

    def generate_counterfactuals(self, input_features: dict):
        """
        Uses DiCE to find the smallest changes needed to flip the prediction.
        """
        # 1. Safety Check: DiCE is mainly for Classification
        problem_type = self.artifacts.get('problem_type', 'classification')
        if problem_type != 'classification':
            return {"message": "Counterfactual simulation is currently only supported for Classification models (e.g., Churn, Fraud)."}

        # 2. Setup Data for DiCE
        # DiCE needs to know the feature ranges. We use the saved training sample.
        target = self.artifacts['target']
        num_cols = self.artifacts['num_cols']
        
        # Load sample data
        train_dataset = self.artifacts.get('X_train_sample')
        if train_dataset is None:
            return {"error": "Training data sample missing. Please re-train the model."}
        
        # Add dummy target column required by DiCE structure
        data_df = train_dataset.copy()
        data_df[target] = 0 
        
        # Initialize DiCE Data
        d = dice_ml.Data(
            dataframe=data_df, 
            continuous_features=num_cols, 
            outcome_name=target
        )
        
        # 3. Setup Model for DiCE
        pipeline = self.artifacts['pipeline']
        
        # Wrap the pipeline so DiCE can use it
        # DiCE expects a generic sklearn wrapper
        m = dice_ml.Model(model=pipeline, backend="sklearn")
        
        # 4. Generate Counterfactuals
        exp = dice_ml.Dice(d, m, method="random")
        
        # Convert input dict to DataFrame
        query_df = pd.DataFrame([input_features])
        
        try:
            # Generate 1 counterfactual, looking for the opposite class
            dice_exp = exp.generate_counterfactuals(
                query_df, 
                total_CFs=1, 
                desired_class="opposite",
                verbose=False
            )
            
            # Extract result as DataFrame
            cf_df = dice_exp.cf_examples_list[0].final_cfs_df
            
            if cf_df is None or cf_df.empty:
                return {"message": "Could not find a valid counterfactual strategy."}
            
            # 5. Clean & Format Result (Crucial for JSON serialization)
            # Remove the target column from the suggestion
            result = cf_df.drop(columns=[target], errors='ignore').iloc[0].to_dict()
            
            # Convert numpy types (int64/float64) to Python standard types (int/float)
            # This prevents the "JSONDecodeError" you saw earlier
            clean_result = {}
            for k, v in result.items():
                if isinstance(v, (np.integer, int)):
                    clean_result[k] = int(v)
                elif isinstance(v, (np.floating, float)):
                    clean_result[k] = round(float(v), 2)
                else:
                    clean_result[k] = v
                    
            return clean_result

        except Exception as e:
            print(f"DiCE Error: {e}")
            return {"error": f"Simulation failed: {str(e)}"}