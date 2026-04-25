# Grade Predictor

Grade Predictor is a Flask + ML web app that helps students estimate their final performance using:
- weighted course components (including extra credit),
- custom score thresholds (A/B/C/D with optional A-, B+, etc.),
- and curve/percentile grading mode.

It is built as a practical student tool, not just a model demo.

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
- **ML:** scikit-learn, NumPy, SciPy, pandas
- **Frontend:** HTML, CSS, JavaScript
- **Production server:** Gunicorn
- **Deployment:** Render / Railway

## Local Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train and save model artifacts:

```bash
python model/train_model.py
```

3. Start app:

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
- `DAY12_REPORT.md` - live test checklist/report
