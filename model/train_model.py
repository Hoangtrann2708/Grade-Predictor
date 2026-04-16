import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# Repo root (this file lives in model/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'model')

# -----------------------------------------------
# STEP 1: CREATE BETTER, MORE REALISTIC DATA
# -----------------------------------------------
np.random.seed(42)
n = 500

study_hours = np.random.uniform(1, 10, n)
attendance = np.random.uniform(30, 100, n)
assignment_avg = np.random.uniform(40, 100, n)
past_gpa = np.random.uniform(1.5, 4.0, n)
sleep_hours = np.random.uniform(4, 10, n)

final_grade = (
    study_hours * 3.5
    + attendance * 0.3
    + assignment_avg * 0.3
    + past_gpa * 8.0
    + sleep_hours * 1.2
    + np.random.normal(0, 3, n)
)
final_grade = np.clip(final_grade, 0, 100)

df = pd.DataFrame({
    'study_hours': study_hours,
    'attendance': attendance,
    'assignment_avg': assignment_avg,
    'past_gpa': past_gpa,
    'sleep_hours': sleep_hours,
    'final_grade': final_grade,
})

# -----------------------------------------------
# STEP 2: PREPARE DATA
# -----------------------------------------------
X = df.drop('final_grade', axis=1)
y = df['final_grade']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------
# STEP 3: TRY MULTIPLE MODELS, PICK THE BEST
# -----------------------------------------------
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("Testing all models...\n")
best_model = None
best_score = -999
best_name = ""

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name}: MAE={mae:.2f}, R²={r2:.3f}")
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_name = name

print(f"\nBest model: {best_name} with R²={best_score:.3f}")

# -----------------------------------------------
# STEP 4: SAVE THE BEST MODEL + SCALER
# -----------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, 'grade_model.pkl'), 'wb') as f:
    pickle.dump(best_model, f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("Best model and scaler saved successfully!")
