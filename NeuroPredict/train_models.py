import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def make_dirs():
    os.makedirs('models', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)

def train_crop_model():
    print("Training Crop Recommendation Model...")
    # Synthetic Data: N, P, K, temperature, humidity, pH, rainfall
    np.random.seed(42)
    X = np.random.rand(500, 7) * 100
    y = np.random.choice(['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'], 500)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'models/crop_model.pkl')
    
    # Save synthetic dataset
    cols = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']
    df = pd.DataFrame(X, columns=cols)
    df['label'] = y
    df.to_csv('datasets/crop_recommendation_synthetic.csv', index=False)
    print("Crop model saved.")

def train_irrigation_model():
    print("Training Irrigation Model...")
    # Synthetic Data: soil moisture, temperature, humidity
    np.random.seed(42)
    X = np.random.rand(500, 3) * 100
    y = np.random.choice([0, 1], 500) # 0: No, 1: Yes
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'models/irrigation_model.pkl')
    
    # Save synthetic dataset
    cols = ['soil_moisture', 'temperature', 'humidity']
    df = pd.DataFrame(X, columns=cols)
    df['irrigation_needed'] = y
    df.to_csv('datasets/irrigation_synthetic.csv', index=False)
    print("Irrigation model saved.")

def train_yield_model():
    print("Training Yield Model...")
    # Synthetic Data: Year, Rainfall, Pesticides, Avg Temp
    np.random.seed(42)
    X = np.random.rand(500, 4) * [20, 200, 50, 40] # scaling
    X[:, 0] += 2000 # Year between 2000 and 2020
    # Yield is numeric
    y = np.random.rand(500) * 10 + 2 # Tons per hectare
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    joblib.dump(model, 'models/yield_model.pkl')
    
    # Save synthetic dataset
    cols = ['Year', 'Rainfall', 'Pesticides', 'Avg_Temp']
    df = pd.DataFrame(X, columns=cols)
    df['Yield'] = y
    df.to_csv('datasets/yield_synthetic.csv', index=False)
    print("Yield model saved.")


if __name__ == "__main__":
    print("=== NeuroAgro AI Model Training Script ===")
    print("Running with synthetic data for fast setup.")
    make_dirs()
    train_crop_model()
    train_irrigation_model()
    train_yield_model()
    print("All models successfully created!")
