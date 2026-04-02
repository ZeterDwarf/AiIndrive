import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
import joblib
import os

def train_scoring_model():
    # 1. Load Data
    data_path = "subsidies_scoring_data.csv"
    if not os.path.exists(data_path):
        print(f"File {data_path} not found.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records for training.")

    # 2. Select Features for ML
    features = [
        'Productivity_Growth', 
        'Tax_Return_Index', 
        'Tech_Score', 
        'Past_Violations',
        'Regional_Mult'
    ]
    
    X = df[features]
    y = df['Merit_Score']
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Model Definition & Hyperparameter Tuning
    print("Настройка модели XGBoost с помощью GridSearchCV (Кросс-валидация)...")
    
    xgb = XGBRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [4, 5, 7]
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=kf,
        n_jobs=-1,
        verbose=1
    )
    
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print(f"Лучшие параметры выбраны: {grid.best_params_}")
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Training Results:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # 6. Save Model and Scaler
    model_path = "subsidies_scoring_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Check feature importance
    importances = model.feature_importances_
    for name, imp in zip(features, importances):
        print(f"Feature '{name}': {imp:.4f}")

if __name__ == "__main__":
    train_scoring_model()
