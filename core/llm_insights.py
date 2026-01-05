import requests
import json
import config

def generate_narrative(context_data: dict):
    """
    Generates a narrative adaptable to Classification (Risk/Prob) or Regression (Value).
    """
    api_key = config.MISTRAL_API_KEY
    if not api_key or "YOUR_KEY" in api_key:
        return "⚠️ Please set your MISTRAL_API_KEY in .env or config.py"

    url = "https://api.mistral.ai/v1/chat/completions"
    
    # 1. Unpack context
    prediction = context_data.get('prediction')
    probability = context_data.get('probability') # Will be None or "N/A" for Regression
    shap_drivers = context_data.get('shap_drivers', {})
    
    # 2. Dynamic Prompt Engineering
    if "N/A" in str(probability) or probability is None:
        # --- REGRESSION PROMPT (e.g., House Price, Salary) ---
        task_desc = "Predicting a continuous numerical value (Regression)."
        result_desc = f"Predicted Value: {prediction}"
    else:
        # --- CLASSIFICATION PROMPT (e.g., Churn, Fraud) ---
        task_desc = "Classifying a binary outcome (Classification)."
        result_desc = f"Prediction: Class {prediction} (Confidence/Probability: {probability})"

    prompt = f"""
    You are an expert Data Analyst AI. 
    Task: {task_desc}
    
    Model Result:
    {result_desc}
    
    Top Key Drivers (Features that pushed the model to this decision):
    {json.dumps(shap_drivers, indent=2)}
    
    Write a short, professional executive summary (max 3 sentences).
    1. Explain clearly what the prediction is.
    2. Explain WHY (citing the key drivers).
    3. Suggest one logical intervention or next step.
    """
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error from LLM: {response.text}"
    except Exception as e:
        return f"Failed to connect to LLM: {str(e)}"