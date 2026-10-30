import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import linprog

# =============================================================================
# 1. LOAD & ENCODE LOGISTICS DATASET
# =============================================================================
df = pd.read_csv("logistics_predictive_dataset.csv")

categorical_cols = ['route', 'transport_mode', 'traffic_level', 'weather']
df_encoded = pd.get_dummies(df.drop(columns=['shipment_id']), columns=categorical_cols, drop_first=True)

X = df_encoded.drop(columns=['actual_transit_time_hrs'])
y = df_encoded['actual_transit_time_hrs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================================================================
# 2. MODEL BENCHMARKING
# =============================================================================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
}

print("=== Predictive Model Benchmarks ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name:<25} | MAE: {mae:.3f} hrs | RMSE: {rmse:.3f} hrs | R2: {r2:.4f}")

# =============================================================================
# 3. HYPERPARAMETER TUNING FOR CHAMPION MODEL
# =============================================================================
param_grid = {
    'n_estimators': [100, 150],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 6]
}
grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

tuned_preds = best_model.predict(X_test)
print(f"\nTuned Champion R2: {r2_score(y_test, tuned_preds):.4f}, MAE: {mean_absolute_error(y_test, tuned_preds):.3f} hrs")

# =============================================================================
# 4. LINEAR PROGRAMMING OPTIMIZATION (FREIGHT ALLOCATION)
# =============================================================================
# Objective: Minimize cost ($/ton): Road ($35), Rail ($18), Air ($95)
c = [35.0, 18.0, 95.0]

# Constraints: Capacities (Road <= 600, Rail <= 500, Air <= 200) & SLA transit <= 13.5 hrs
A_ub = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [12.0, 18.0, 4.0]
]
b_ub = [600.0, 500.0, 200.0, 13500.0]

# Equality Constraint: Total Demand = 1000 tons
A_eq = [[1.0, 1.0, 1.0]]
b_eq = [1000.0]

bounds = [(0, None), (0, None), (0, None)]

res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

print("\n=== Linear Programming Modal Allocation ===")
print(f"Optimal Road Allocation: {res.x[0]:.2f} tons")
print(f"Optimal Rail Allocation: {res.x[1]:.2f} tons")
print(f"Optimal Air Allocation:  {res.x[2]:.2f} tons")
print(f"Minimized Freight Cost:  ${res.fun:,.2f}")