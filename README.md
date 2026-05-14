# Course Grade Predictor

Course Grade Predictor is a Flask web app that turns raw assignment scores into a course Overall score and a letter grade. Two complementary engines:

- **Syllabus weighted score** — algebraic `sum(score * weight / 100)`. The default; correct whenever you have every component graded.
- **Performance forecast (ML)** — when the final exam (or any remaining slot) is missing, a scikit-learn model trained in `model/train_model.py` predicts that score from the components you already have, then the app rebuilds the Overall.

Both engines feed the same downstream grading logic: **Custom Score** (your own A/B/C/D thresholds, plus optional A-, B+, …) or **Curve / Percentile** (Overall vs class avg + std dev).

## Demo

- **Vercel (recommended, fast first paint):** [https://grade-predictor-nine.vercel.app](https://grade-predictor-nine.vercel.app)
- **Render (full Flask API + ML, free tier cold-starts in ~30–60s after idle):** [https://grade-predictor-1-zjns.onrender.com](https://grade-predictor-1-zjns.onrender.com)
- **Local:** [http://127.0.0.1:5000](http://127.0.0.1:5000)

The ML engine only runs on the Render deployment (it needs the trained `*.pkl` artifacts). Vercel serves the static UI and falls back to in-browser syllabus math when the API is not reachable.

## Why This Project

Different classes use different grading rules and scales:

- many items are graded on a `0-100` scale, or as raw points like `29/30`,
- some weight components flat, some apply a curve,
- some are still missing one or more components when you want to project your final grade.

This app normalizes each row to a **0–100%** score (plain number as out of 100, fraction like `17/20`, or percent like `85%`), applies the syllabus weights you provide, and either reports your current Overall or predicts the missing slot with ML before reporting it.

## Features

- **Two prediction engines**
  - Syllabus weighted score (deterministic)
  - Performance ML forecast — fills the remaining weight (typically the final exam) with a trained regressor and rebuilds Overall
- **Per-row component picker (ML mode)** — pick `Homework`, `Midterm 1/2`, `Quiz avg`, `Project`, `Attendance`, or `Final (weight slot only)`; ML no longer relies on parsing the item name
- **Two grading modes**
  - Custom Score thresholds (with optional A-, B+, B-, C+, C-, D+, D- variants)
  - Curve / Percentile based on class average, std dev, and median (overall or per-component)
- **Flexible score entry** — `85`, `85%`, or `29/30` all work; fixed 100-point scale
- **Extra credit** — extra-credit rows add to Overall but do not consume base weight
- **Explainable response** — every `/predict` call returns:
  - `prediction_source`: `ml_performance_forecast` | `weighted_requirements`
  - `ml_status`: `ok` | `model_missing` | `no_classified_components` | `no_remaining_weight` | `predict_error`
  - `ml_predicted_final_score`, `ml_overall_percent`, `ml_remaining_weight_percent`, `syllabus_prediction_percent`
- **UX safeguards** — friendly validation errors, request status banner, `/health` endpoint, in-browser fallback when the API is offline

## Tech Stack

- **Backend:** Python, Flask, NumPy, SciPy (curve stats), scikit-learn (ML inference)
- **Training pipeline (offline, `model/train_model.py`):** scikit-learn `RandomForestRegressor` / `GradientBoostingRegressor` on synthetic latent-ability data, exporting `grade_model.pkl`, `scaler.pkl`, `ml_meta.json`
- **Frontend:** HTML, CSS, vanilla JavaScript
- **Production server:** Gunicorn
- **Deployment:** Vercel (static UI / fast preview), Render or Railway (Flask API + ML)

## Local Setup

```bash
pip install -r requirements.txt
python model/train_model.py     # generates model/grade_model.pkl + scaler.pkl + ml_meta.json
python app.py                   # or: py app.py on Windows
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

The ML engine activates only when those three model artifacts are present. Without them, the app keeps working in Syllabus mode and reports `ml_status: model_missing`.

## API

### `GET /health`
Returns `{ ok, status, uptime_seconds }`.

### `POST /predict`
Request body (JSON):

```json
{
  "prediction_engine": "performance_ml",
  "grading_mode": "custom_score",
  "use_letter_grades": true,
  "threshold_a": 90, "threshold_b": 80, "threshold_c": 70, "threshold_d": 60,
  "requirements": [
    { "name": "Midterm 1",  "score": 80, "weight": 20, "component_key": "midterm_1" },
    { "name": "Midterm 2",  "score": 85, "weight": 30, "component_key": "midterm_2" },
    { "name": "Homework",   "score": 90, "weight": 20, "component_key": "homework_avg" }
  ]
}
```

Notable response fields:

- `prediction_percent` — Overall 0–100 used for grading
- `prediction_source` — which engine produced it
- `ml_status` — diagnostic for the ML path
- `grade_letter`, `message`, `needed_for_a`
- `percentile`, `z_score`, `diff_from_avg`, `diff_from_median` (when class stats are provided)

`component_key` is optional but recommended for ML mode. Allowed values: `homework_avg`, `midterm_1`, `midterm_2`, `project`, `quiz_avg`, `attendance_pct`, `final_exam_placeholder`. The placeholder row reserves syllabus weight for the missing final — its score is ignored and that weight is what the model fills.

## Deployment

### Vercel (UI + browser fallback)
Connect the GitHub repo. Production tracks `main`; each branch / PR gets a Preview URL. ML inference is not available here; the page calls `/predict` and falls back to local syllabus math if the API isn't reachable.

### Render (full Python + ML)

- **Build command:** `pip install -r requirements.txt && python model/train_model.py`
- **Start command:** `gunicorn --bind 0.0.0.0:$PORT app:app`

### Railway

- **Build command:** `pip install -r requirements.txt && python model/train_model.py`
- **Start command:** `gunicorn --bind 0.0.0.0:$PORT app:app`

Free tiers may cold-start the first request after a period of inactivity — expected platform behavior.

## Repository Structure

- `app.py` — Flask routes, validation, syllabus + ML pipeline
- `templates/index.html` — UI (engine selector, component picker, results)
- `static/style.css` — styling
- `model/train_model.py` — synthetic data + training; writes `grade_model.pkl`, `scaler.pkl`, `ml_meta.json`
- `model/ml_meta.json` — feature column order and target metadata read by the app
- `requirements.txt` — dependencies
- `render.yaml`, `Procfile`, `runtime.txt` — deploy configuration
