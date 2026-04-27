# Course Grade Predictor

Course Grade Predictor is a Flask web app built around **syllabus-style grading**, not an ML-only guess:

- weighted course components (including extra credit),
- custom score thresholds (A/B/C/D with optional A-, B+, etc.),
- curve / percentile mode.

The `/predict` endpoint scores from **those inputs only** (weights + scores + your chosen rules). Optional improvement rows are UI context; offline scripts in `model/train_model.py` remain for learning the ML training workflow.

## Demo

- **Instant live demo (Vercel — use this first, including new branches / PR previews):**  
  [https://grade-predictor-nine.vercel.app](https://grade-predictor-nine.vercel.app)
- **Local:** [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Full Flask API on Render (optional — free tier may cold-start ~30–60s after idle):**  
  [https://grade-predictor-1-zjns.onrender.com](https://grade-predictor-1-zjns.onrender.com)

**After you push a branch:** open the **Vercel** preview or production URL for a fast UI load. Prefer that over the Render link for demos and recruiter screens.

Recruiter note: Share the **Vercel** link for an instant first paint; mention Render only if you need to show the hosted Python API waking up.

## Why This Project

Different classes use different grading systems:

- some use `0-100`,
- some use `0-10`,
- some use point-based scores like `29/30`.

This app normalizes these inputs, applies the selected grading logic, and provides an actionable result.

## Features

- **Dual grading modes**
  - Custom score thresholds
  - Curve / percentile mode
- **Flexible score entry**
  - Accepts plain values, percentages (`85%`), and fractions (`29/30`)
  - Uses a configurable "Default Out Of" base
- **Weighted requirement math**
  - Syllabus-style calculation
  - Extra credit support
- **Advanced custom scale**
  - Add grade variants like `A-`, `B+`, `B-`
- **Validation and UX safeguards**
  - Clear form validation errors
  - Request status/loading states
  - Health check endpoint (`/health`)

## Tech Stack

- **Backend:** Python, Flask
- **Training pipeline (offline, `model/train_model.py`):** scikit-learn, NumPy, SciPy, pandas — not loaded by `/predict` at runtime
- **Frontend:** HTML, CSS, JavaScript
- **Production server:** Gunicorn
- **Deployment:** Vercel (frontend / instant UI), Render / Railway (Flask API)

## Local Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Train and save model artifacts:

```bash
python model/train_model.py
```

1. Start app:

```bash
python app.py
```

Open `http://127.0.0.1:5000`  
(On Windows, `py app.py` is also supported.)

## API Endpoints

- `GET /` - main UI
- `GET /health` - service health check
- `POST /predict` - prediction and grading response

## Deployment

### Vercel (recommended for demos)

Connect the GitHub repo so **Production** uses your main branch and **Preview** deploys each branch / PR. Use those URLs when you want an instant-opening UI (especially after pushing a new branch).

### Render

- **Build command**
`pip install -r requirements.txt && python model/train_model.py`
- **Start command**
`gunicorn --bind 0.0.0.0:$PORT app:app`

### Railway

- **Build command**
`pip install -r requirements.txt && python model/train_model.py`
- **Start command**
`gunicorn --bind 0.0.0.0:$PORT app:app`

## Reliability Note (Free Tier)

Free-tier services can sleep after inactivity, causing cold-start delay on first request.
This is platform behavior, not an application bug.

## Repository Structure

- `app.py` - Flask routes, validation, grading logic
- `templates/` - UI templates
- `static/` - styling
- `model/train_model.py` - model training and artifact export
- `requirements.txt` - dependencies
- `render.yaml`, `railway.toml`, `Procfile` - deploy configuration
- `REPORT.md` - live test checklist/report

