import pandas as pd
import numpy as np
import os

def create_synthetic_data(output_path='data/raw/creditcard.csv', n_samples=10000):
    """
    Generates industry-level synthetic data to simulate real-world fraud patterns.
    """
    np.random.seed(42)
    
    # 1. Base Features (Simulating PCA transformed features V1 to V28)
    data = np.random.randn(n_samples, 28)
    cols = [f'V{i}' for i in range(1, 29)]
    df = pd.DataFrame(data, columns=cols)
    
    # 2. Amount Feature (Simulating transaction amounts)
    df['Amount'] = np.random.exponential(scale=100, size=n_samples)
    
    # 3. Fraud Logic Simulation
    # Fraud transactions are rare (approx 1.5%) and often involve higher amounts
    df['Class'] = 0
    fraud_indices = np.random.choice(df.index, size=int(n_samples * 0.015), replace=False)
    df.loc[fraud_indices, 'Class'] = 1
    
    # Injecting pattern: Fraudulent transactions have 2.5x higher amounts on average
    df.loc[df['Class'] == 1, 'Amount'] = df.loc[df['Class'] == 1, 'Amount'] * 2.5
    
    # 4. Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Success: {n_samples} samples saved at {output_path}")

if __name__ == "__main__":
    create_synthetic_data()