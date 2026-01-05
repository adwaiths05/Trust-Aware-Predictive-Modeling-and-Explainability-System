import pandas as pd
import numpy as np
import os

def create_dummy_data():
    np.random.seed(42)
    n = 1000
    
    data = pd.DataFrame({
        'CreditScore': np.random.randint(300, 850, n),
        'Age': np.random.randint(18, 70, n),
        'Tenure': np.random.randint(0, 10, n),
        'Balance': np.random.uniform(0, 250000, n).round(2),
        'NumOfProducts': np.random.randint(1, 4, n),
        'HasCrCard': np.random.choice([0, 1], n),
        'IsActiveMember': np.random.choice([0, 1], n),
        'EstimatedSalary': np.random.uniform(10000, 200000, n).round(2),
        'Geography': np.random.choice(['France', 'Germany', 'Spain'], n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Exited': np.random.choice([0, 1], n, p=[0.8, 0.2]) # Target
    })
    
    # Introduce some correlations for SHAP to find
    # Older people with high balance in Germany are more likely to churn (exit)
    mask = (data['Age'] > 50) & (data['Geography'] == 'Germany')
    data.loc[mask, 'Exited'] = np.random.choice([0, 1], mask.sum(), p=[0.3, 0.7])

    if not os.path.exists('data'):
        os.makedirs('data')
    data.to_csv('data/churn_data.csv', index=False)
    print("✅ Generated data/churn_data.csv")

if __name__ == "__main__":
    create_dummy_data()