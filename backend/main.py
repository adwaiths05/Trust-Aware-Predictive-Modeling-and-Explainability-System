from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import sys
from pathlib import Path

# Add project root to sys path
sys.path.append(str(Path(__file__).parent.parent))

from core.data_processor import DatasetManager
from core.trainer import ModelTrainer
from core.explainer import ModelExplainer
from core.simulator import CounterfactualEngine
from core.llm_insights import generate_narrative
import config

app = FastAPI(title="AI Decision Engine")

# --- Schemas ---
class TrainRequest(BaseModel):
    filename: str
    target_column: str
    drop_columns: List[str] = []
    auto_optimize: bool = False # <--- NEW FIELD

class PredictRequest(BaseModel):
    features: Dict[str, Any]

class ExplainRequest(BaseModel):
    features: Dict[str, Any]
    prediction: Any
    probability: Optional[float] = None

# --- Endpoints ---

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = config.DATA_DIR / file.filename
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        return {"message": f"File '{file.filename}' saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def train_model(req: TrainRequest):
    try:
        dm = DatasetManager()
        dm.load_data(req.filename)
        
        X, y, problem_type = dm.preprocess(req.target_column, req.drop_columns)
        full_df = pd.concat([X, y], axis=1)
        
        trainer = ModelTrainer()
        # Pass the new flag to the trainer
        metrics = trainer.train(
            full_df, 
            req.target_column, 
            problem_type, 
            auto_optimize=req.auto_optimize
        )
        
        metrics["detected_type"] = problem_type
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict")
def predict(req: PredictRequest):
    try:
        explainer = ModelExplainer() # Loads pipeline
        pipeline = explainer.artifacts['pipeline']
        problem_type = explainer.artifacts.get('problem_type', 'classification')
        
        # Prepare input
        input_df = pd.DataFrame([req.features])
        
        # Run Inference
        pred = pipeline.predict(input_df)[0]
        
        result = {}
        
        if problem_type == "classification":
            result["prediction"] = int(pred)
            # Get probability of the positive class (1)
            probs = pipeline.predict_proba(input_df)[0]
            result["probability"] = float(round(probs[1], 4))
        else:
            # Regression: No probability, just the number
            result["prediction"] = float(round(pred, 2))
            result["probability"] = None 

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain(req: ExplainRequest):
    try:
        # 1. Calculate SHAP Values
        explainer = ModelExplainer()
        shap_vals = explainer.explain_local(req.features)
        
        # Get top 5 drivers (sorted by absolute impact)
        top_drivers = dict(sorted(shap_vals.items(), key=lambda item: abs(item[1]), reverse=True)[:5])
        
        # 2. Generate LLM Narrative
        # Adjust context based on problem type (Probability for Class, Value for Reg)
        context = {
            "prediction": req.prediction,
            "shap_drivers": top_drivers
        }
        
        if req.probability is not None:
            context["probability"] = f"{req.probability * 100:.1f}%"
        else:
            context["probability"] = "N/A (Regression Task)"

        narrative = generate_narrative(context)
        
        return {
            "shap_values": top_drivers,
            "narrative": narrative
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
def simulate_counterfactual(req: PredictRequest):
    try:
        cf_engine = CounterfactualEngine()
        # Note: DiCE is primarily for Classification. 
        # For Regression, we might need custom logic or stick to classification for MVP.
        changes = cf_engine.generate_counterfactuals(req.features)
        return {"changes": changes}
    except Exception as e:
        return {"changes": None, "error": str(e)}

# Run command: uvicorn backend.main:app --reload