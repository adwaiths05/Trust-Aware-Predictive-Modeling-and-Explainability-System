from pydantic import BaseModel
from typing import List, Optional, Union, Dict

# --- 1. Model Training Schemas ---
class TrainRequest(BaseModel):
    target_column: str
    drop_columns: Optional[List[str]] = []
    model_type: str = "lightgbm"  # or 'catboost'
    problem_type: str = "classification" # or 'regression'

class ModelMetrics(BaseModel):
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    auc: Optional[float] = None
    rmse: Optional[float] = None # For regression

# --- 2. Prediction Schemas ---
class PredictionRequest(BaseModel):
    # Flexible input: Dict key is feature name, value is user input
    features: Dict[str, Union[int, float, str]] 

class PredictionResponse(BaseModel):
    prediction: Union[int, float, str]
    probability: Optional[float] = None
    shap_values: Optional[Dict[str, float]] = None # Local feature importance

# --- 3. Simulation (Counterfactual) Schemas ---
class SimulationRequest(BaseModel):
    features: Dict[str, Union[int, float, str]]
    target_class: int # e.g., We want to flip result to "1" (Approved)

class SimulationResponse(BaseModel):
    original_prediction: Union[int, str]
    new_prediction: Union[int, str]
    changes_needed: Dict[str, Union[int, float, str]] # e.g., {"Salary": 55000}