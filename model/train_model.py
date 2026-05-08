"""
Train a regression model that predicts the *final exam* score from the other
syllabus components (midterms, homework, attendance, quizzes, project) — used
by the "Performance forecast (ML)" engine in the Flask app.

Why this shape (read before retraining):
─────────────────────────────────────────
• Course Overall is just  sum(score_i * weight_i / 100)  by definition. That
  part is algebra and stays in app.py.
• What ML *can* help with is filling in the one component the student does
  not have yet — typically the final exam — using the components they already
  have. So the label here is `final_exam`; features are the *other* component
  scores plus a mask saying which of those they actually entered.

Synthetic generation (replace with real rows when available):
• Each simulated student gets a latent "ability" drawn uniformly.
• Each component score = ability + per-component noise (correlated by design).
• `final_exam` = ability + final-specific noise (the label).
• During training we randomly mask a subset of the *input* components so the
  model handles partial inputs (e.g., user only has Midterm 1 + Attendance).

Outputs (written to ./model next to this file):
  grade_model.pkl  – best estimator on scaled features
  scaler.pkl       – StandardScaler fitted on training X
  ml_meta.json     – feature/input layout so app.py can build matching rows
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, 'model')

# Components the user might have a score for (always 0–100). `final_exam` is
# intentionally NOT in this list — it's always the label.
INPUT_COMPONENTS = [
    'homework_avg',
    'midterm_1',
    'midterm_2',
    'project',
    'quiz_avg',
    'attendance_pct',
]
TARGET = 'final_exam'

# Per-component noise (std) around the latent ability. Kept small so the
# synthetic correlation between observed components and the final exam is
# realistic but not deterministic. Tune with real data when available.
COMPONENT_NOISE_SD = {
    'homework_avg':  6.0,
    'midterm_1':     8.0,
    'midterm_2':     8.0,
    'project':       7.0,
    'quiz_avg':      6.0,
    'attendance_pct': 5.0,
    'final_exam':    9.0,
}


def build_synthetic_dataset(n: int = 12000, seed: int = 42, min_active: int = 1):
    """Generate (X, y) where y is final_exam and X is masked input components."""
    rng = np.random.default_rng(seed)

    ability = rng.uniform(45.0, 95.0, size=n)

    raw_scores = np.zeros((n, len(INPUT_COMPONENTS)), dtype=np.float64)
    for j, comp in enumerate(INPUT_COMPONENTS):
        sd = COMPONENT_NOISE_SD[comp]
        raw_scores[:, j] = np.clip(ability + rng.normal(0.0, sd, size=n), 0.0, 100.0)

    finals = np.clip(
        ability + rng.normal(0.0, COMPONENT_NOISE_SD[TARGET], size=n), 0.0, 100.0
    )

    masks = np.zeros_like(raw_scores)
    for i in range(n):
        k = rng.integers(min_active, len(INPUT_COMPONENTS) + 1)
        idx = rng.choice(len(INPUT_COMPONENTS), size=k, replace=False)
        masks[i, idx] = 1.0

    masked_scores = raw_scores * masks
    X = np.hstack([masked_scores, masks])

    columns = (
        [f'score_{c}' for c in INPUT_COMPONENTS]
        + [f'uses_{c}' for c in INPUT_COMPONENTS]
    )
    df_X = pd.DataFrame(X, columns=columns)
    df_y = pd.Series(finals, name=TARGET)
    return df_X, df_y, masks


def main():
    print('Building synthetic latent-ability dataset (final exam as label)…')
    X_df, y_series, masks_arr = build_synthetic_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=14, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=4, random_state=42
        ),
    }

    print('\nTesting models…\n')
    best_model = None
    best_name = ''
    best_r2 = -999.0

    for name, model in models.items():
        model.fit(X_train_s, y_train)
        preds = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        print(f'{name}: MAE={mae:.2f}, R²={r2:.4f}')
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_name = name

    print(f'\nBest model: {best_name} with R²={best_r2:.4f}')

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'grade_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    meta = {
        'target': TARGET,
        'input_components': INPUT_COMPONENTS,
        'feature_columns': list(X_df.columns),
        'description': (
            'Predicts final_exam score from the other syllabus components. '
            'Features are masked component scores followed by binary mask bits '
            '(same order as input_components, length 2 * K).'
        ),
        'best_model': best_name,
        'best_r2_holdout': round(float(best_r2), 6),
        'mean_active_inputs_per_row': round(float(masks_arr.sum(axis=1).mean()), 3),
        'note': (
            'Compute Overall in the app: weighted sum of entered components + '
            'predicted final * remaining_weight / 100.'
        ),
    }
    with open(os.path.join(MODEL_DIR, 'ml_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('\nSaved grade_model.pkl, scaler.pkl, ml_meta.json')


if __name__ == '__main__':
    main()
