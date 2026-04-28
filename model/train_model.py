"""
Train a regression model on SYNTHETIC syllabus-style component grades → final grade.

Important (read before relying on predictions):
─────────────────────────────────────────────────
• If the user already enters every component score AND the syllabus weights are
  known, the course final is *defined* by weighted sum — that is algebra, not ML.

• This script simulates many different syllabi in one dataset: each row has a
  random subset of categories (homework, midterms, attendance, …) with random
  nonnegative weights that sum to 1. The label is exactly that weighted sum of
  0–100 component scores (+ small noise). Any sklearn model is learning to
  approximate that mapping from masked features.

• Replace synthetic generation with REAL historical rows when you have them:
  (student_id optional, component scores…, recorded final). Then retrain.

Outputs (written to ./model next to this file):
  grade_model.pkl  – best estimator on scaled features
  scaler.pkl       – StandardScaler fitted on training X
  ml_meta.json     – feature names + short description for wiring into Flask

Threshold vs percentile grading is NOT trained here — that stays in app.py /
the UI after you have a predicted numeric final (same as today).
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

# Fixed superset of gradebook categories (scores are 0–100 unless noted).
# Extend this list if your app collects more columns — keep training aligned.
COMPONENTS = [
    'homework_avg',
    'midterm_1',
    'midterm_2',
    'final_exam',
    'project',
    'quiz_avg',
    'attendance_pct',  # treat as a 0–100 “score bucket” when the class grades it
]


def random_syllabus_masks(rng: np.random.Generator, n_rows: int, min_active: int = 2):
    """Each row: random subset of components (different syllabi)."""
    masks = np.zeros((n_rows, len(COMPONENTS)), dtype=np.float64)
    for i in range(n_rows):
        k = rng.integers(min_active, len(COMPONENTS) + 1)
        idx = rng.choice(len(COMPONENTS), size=k, replace=False)
        masks[i, idx] = 1.0
    return masks


def random_weights_on_active(mask_row: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Nonnegative weights on active slots, sum to 1."""
    active = mask_row > 0
    if not np.any(active):
        return mask_row
    raw = rng.random(len(COMPONENTS)) * active.astype(float)
    s = raw.sum()
    if s <= 1e-9:
        raw = active.astype(float)
        s = raw.sum()
    return raw / s


def build_synthetic_dataset(n: int = 8000, seed: int = 42):
    rng = np.random.default_rng(seed)

    masks = random_syllabus_masks(rng, n)
    scores = rng.uniform(35, 100, size=(n, len(COMPONENTS)))

    finals = np.zeros(n, dtype=np.float64)
    weights_list = []

    for i in range(n):
        w = random_weights_on_active(masks[i], rng)
        weights_list.append(w)
        pure = np.sum(w * masks[i] * scores[i])
        noise = rng.normal(0, 1.2)
        finals[i] = np.clip(pure + noise, 0, 100)

    weights_list = np.array(weights_list)

    # Features: masked scores + mask bits so the model sees which columns matter.
    masked_scores = masks * scores
    feature_blocks = [masked_scores, masks]
    X = np.hstack(feature_blocks)

    column_names = (
        [f'score_{c}' for c in COMPONENTS]
        + [f'uses_{c}' for c in COMPONENTS]
    )

    df_X = pd.DataFrame(X, columns=column_names)
    df_y = pd.Series(finals, name='final_grade')
    return df_X, df_y, scores, masks, weights_list


def main():
    print('Building synthetic multi-syllabus dataset…')
    X_df, y_series, _, masks_arr, _ = build_synthetic_dataset()

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
            n_estimators=120, max_depth=12, random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=120, max_depth=4, random_state=42
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
        'feature_columns': list(X_df.columns),
        'components': COMPONENTS,
        'description': (
            'Masked component scores + binary syllabus masks. '
            'Training labels are weighted sums with per-row random syllabi.'
        ),
        'best_model': best_name,
        'best_r2_holdout': round(float(best_r2), 6),
        'note': (
            'When real syllabus weights and all scores are known, compute final '
            'with weighted sum in the app; use this ML only if you ingest '
            'matching features from the client or imputation workflows.'
        ),
    }
    with open(os.path.join(MODEL_DIR, 'ml_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('\nSaved grade_model.pkl, scaler.pkl, ml_meta.json')
    print(f'(Mean active components per row: {masks_arr.sum(axis=1).mean():.2f})')


if __name__ == '__main__':
    main()
