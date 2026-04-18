## Grade Predictor
A web app that predicts your grade from study habits, attendance, weighted coursework, optional class stats, and custom or curve-based grading.

## Tech stack
- Python + Flask (API + server-rendered UI)
- scikit-learn (regression model)
- HTML / CSS / JavaScript
- Gunicorn (production HTTP server on the host)

## Run locally
1. Create a virtual environment (recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model and write `model/grade_model.pkl` and `model/scaler.pkl` (required before first run; files are gitignored):

```bash
python model/train_model.py
```

3. Start the dev server:

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). On Windows you can use `py` instead of `python` if needed.

## Deploy (Day 11) — Render

1. Push this repo to GitHub (public or private).
2. In [Render](https://render.com), create a **Web Service**, connect the repo, pick the same branch.
3. Use these commands (or deploy from the included `render.yaml` Blueprint):

   - **Build command:** `pip install -r requirements.txt && python model/train_model.py`
   - **Start command:** `gunicorn --bind 0.0.0.0:$PORT app:app`

4. Deploy and open the URL Render assigns (e.g. `https://your-app.onrender.com`).

Free tier services may **spin down after idle**; the first request after sleep can take ~30–60 seconds.

## Deploy — Railway
1. Push the repo to GitHub.
2. In [Railway](https://railway.app), **New Project** → **Deploy from GitHub** → select this repo.
3. Railway can read `railway.toml`: it runs the same build (install + train) and starts Gunicorn on `$PORT`. If the dashboard overrides commands, set:

   - **Build:** `pip install -r requirements.txt && python model/train_model.py`
   - **Start:** `gunicorn --bind 0.0.0.0:$PORT app:app`

## Repo layout
- `app.py` — Flask app and `/predict` validation + ML inference
- `templates/`, `static/` — UI
- `model/train_model.py` — generates pickles used at runtime
- `requirements.txt` — Python dependencies
- `Procfile` — process type for platforms that read it (e.g. Heroku-style)
- `runtime.txt` — Python version hint for hosts that support it
- `DAY12_REPORT.md` — live test checklist/report template for Day 12
