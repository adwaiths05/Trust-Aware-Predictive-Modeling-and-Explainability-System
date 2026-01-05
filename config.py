import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# 2. Define Paths (Dynamic)
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure dirs exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 3. Load & Validate API Key
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Optional: Crash immediately if key is missing so you don't debug later
if not MISTRAL_API_KEY:
    # It's okay to warn instead of crash if you want to allow running without LLM
    print("⚠️ WARNING: MISTRAL_API_KEY not found in .env. LLM features will fail.")

RANDOM_SEED = 42
MODEL_FILENAME = "lgbm_pipeline.pkl"