import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
import struct

# Configuration
FEATURE_COUNT = 448

def train_and_export():
    # Load data (Optimized for Large CSVs)
    df = pd.read_csv('training_data.csv', header=None)
    X = df.iloc[:, :FEATURE_COUNT].values
    y = df.iloc[:, -1].values

    # 1. Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. Robust Regression (Huber is outlier-resistant for HFT)
    model = HuberRegressor(epsilon=1.35, max_iter=1000)
    model.fit(X_scaled, y)

    # 3. MATHEMATICAL FOLDING
    # We fold the scaler into the weights so C++ doesn't have to scale at runtime
    final_weights = model.coef_ / scaler.scale_
    final_bias = model.intercept_ - np.sum((model.coef_ * scaler.mean_) / scaler.scale_)

    # 4. Export as Binary for C++ FastLoader
    with open('weights.bin', 'wb') as f:
        f.write(final_weights.astype(np.float64).tobytes())
    
    with open('bias.bin', 'wb') as f:
        f.write(struct.pack('d', final_bias))

    print(f"Model exported. Weights range: {final_weights.min():.6f} to {final_weights.max():.6f}")

if __name__ == "__main__":
    train_and_export()
