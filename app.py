from flask import Flask, render_template, request, jsonify
import math
import pickle
import time
import numpy as np
from scipy import stats

app = Flask(__name__)
APP_STARTED_AT = int(time.time())


def _num_field(data, key, errors, *, label, lo, hi, default=None, required=False):
    """Parse a float in [lo, hi]. Appends to errors and returns None on failure."""
    raw = data.get(key, default) if key in data else default
    if raw is None or (isinstance(raw, str) and raw.strip() == ''):
        if not required and default is not None:
            return float(default)
        errors.append({'field': key, 'message': f'{label} is required.'})
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        errors.append({'field': key, 'message': f'{label} must be a valid number.'})
        return None
    if math.isnan(v) or math.isinf(v):
        errors.append({'field': key, 'message': f'{label} is not a valid number.'})
        return None
    if v < lo or v > hi:
        errors.append({'field': key, 'message': f'{label} must be between {lo} and {hi}.'})
        return None
    return v


def _validate_predict(data):
    """Returns (parsed dict or None, list of error dicts)."""
    errors = []
    if data is None:
        return None, [{'field': '_body', 'message': 'Request body must be valid JSON.'}]
    if not isinstance(data, dict):
        return None, [{'field': '_body', 'message': 'Request body must be a JSON object.'}]

    study_hours = _num_field(
        data, 'study_hours', errors, label='Study hours', lo=0, hi=24, default=4.0
    )
    attendance = _num_field(
        data, 'attendance', errors, label='Attendance', lo=0, hi=100, default=85.0
    )
    assignment_avg = _num_field(
        data, 'assignment_avg', errors, label='Assignment average', lo=0, hi=100, default=75.0
    )
    past_gpa = _num_field(
        data, 'past_gpa', errors, label='Past GPA', lo=0, hi=4.5, default=2.8
    )
    sleep_hours = _num_field(
        data, 'sleep_hours', errors, label='Sleep hours', lo=0, hi=24, default=7.0
    )

    if any(x is None for x in (study_hours, attendance, assignment_avg, past_gpa, sleep_hours)):
        return None, errors

    grading_mode = data.get('grading_mode', 'custom_score')
    if grading_mode not in ('custom_score', 'curve'):
        errors.append({
            'field': 'grading_mode',
            'message': 'grading_mode must be "custom_score" or "curve".',
        })

    raw_reqs = data.get('requirements')
    parsed_requirements = []
    if raw_reqs is None:
        reqs = []
    elif not isinstance(raw_reqs, list):
        errors.append({'field': 'requirements', 'message': 'requirements must be a JSON array.'})
        reqs = []
    else:
        reqs = raw_reqs
    if isinstance(reqs, list):
        for i, item in enumerate(reqs):
            if not isinstance(item, dict):
                errors.append({
                    'field': f'requirements[{i}]',
                    'message': 'Each requirement must be an object.',
                })
                continue
            sc = item.get('score')
            wt = item.get('weight')
            try:
                scf = float(sc)
                wtf = float(wt)
            except (TypeError, ValueError):
                errors.append({
                    'field': f'requirements[{i}]',
                    'message': 'Each requirement needs numeric score and weight.',
                })
                continue
            if math.isnan(scf) or math.isnan(wtf):
                errors.append({
                    'field': f'requirements[{i}]',
                    'message': 'Score and weight cannot be NaN.',
                })
                continue
            if scf < 0 or scf > 100:
                errors.append({
                    'field': f'requirements[{i}].score',
                    'message': 'Requirement scores must be between 0 and 100.',
                })
            if wtf <= 0 or wtf > 100:
                errors.append({
                    'field': f'requirements[{i}].weight',
                    'message': 'Each weight must be between 0 and 100 (exclusive of 0).',
                })
            parsed_requirements.append({
                'name': str(item.get('name', f'Requirement {i + 1}')),
                'score': scf,
                'weight': wtf,
            })

    curve_pct_a = curve_pct_b = curve_pct_c = None
    curve_class_avg = curve_class_sd = curve_class_median = None
    th_a = th_b = th_c = th_d = None
    opt_class_avg = opt_class_sd = opt_class_median = None

    if grading_mode == 'curve':
        curve_pct_a = _num_field(data, 'pct_a', errors, label='A %', lo=0, hi=100, default=30)
        curve_pct_b = _num_field(data, 'pct_b', errors, label='B %', lo=0, hi=100, default=40)
        curve_pct_c = _num_field(data, 'pct_c', errors, label='C %', lo=0, hi=100, default=20)
        curve_class_avg = _num_field(
            data, 'class_avg', errors, label='Class average', lo=0, hi=100, default=75, required=True
        )
        curve_class_sd = _num_field(data, 'class_sd', errors, label='Class std dev', lo=0, hi=50, default=10)
        curve_class_median = _num_field(
            data, 'class_median', errors, label='Class median', lo=0, hi=100, default=75
        )
        if curve_pct_a is not None and curve_pct_b is not None and curve_pct_c is not None:
            if curve_pct_a + curve_pct_b + curve_pct_c > 100.01:
                errors.append({
                    'field': 'pct_total',
                    'message': 'A% + B% + C% cannot exceed 100%. D/F is the remaining percent.',
                })

    elif grading_mode == 'custom_score':
        th_a = _num_field(
            data, 'threshold_a', errors, label='Threshold A', lo=0, hi=100, default=90
        )
        th_b = _num_field(
            data, 'threshold_b', errors, label='Threshold B', lo=0, hi=100, default=80
        )
        th_c = _num_field(
            data, 'threshold_c', errors, label='Threshold C', lo=0, hi=100, default=70
        )
        th_d = _num_field(
            data, 'threshold_d', errors, label='Threshold D', lo=0, hi=100, default=60
        )
        if all(x is not None for x in (th_a, th_b, th_c, th_d)):
            if not (th_a >= th_b >= th_c >= th_d):
                errors.append({
                    'field': 'thresholds',
                    'message': 'A threshold must be >= B >= C >= D (e.g. 90 / 80 / 70 / 60).',
                })

        if data.get('class_avg') not in (None, ''):
            opt_class_avg = _num_field(
                data, 'class_avg', errors, label='Class average (optional)', lo=0, hi=100, required=True
            )
            if opt_class_avg is not None:
                opt_class_sd = _num_field(
                    data, 'class_sd', errors, label='Class std dev', lo=0, hi=50, default=10
                )
                opt_class_median = _num_field(
                    data,
                    'class_median',
                    errors,
                    label='Class median',
                    lo=0,
                    hi=100,
                    default=opt_class_avg,
                )

    if errors:
        return None, errors

    parsed = {
        'study_hours': study_hours,
        'attendance': attendance,
        'assignment_avg': assignment_avg,
        'past_gpa': past_gpa,
        'sleep_hours': sleep_hours,
        'grading_mode': grading_mode,
        'requirements': parsed_requirements,
    }
    if grading_mode == 'curve':
        parsed.update({
            'pct_a': curve_pct_a,
            'pct_b': curve_pct_b,
            'pct_c': curve_pct_c,
            'class_avg': curve_class_avg,
            'class_sd': curve_class_sd,
            'class_median': curve_class_median,
        })
    elif grading_mode == 'custom_score':
        parsed.update({
            'threshold_a': th_a,
            'threshold_b': th_b,
            'threshold_c': th_c,
            'threshold_d': th_d,
        })
        if opt_class_avg is not None:
            parsed['class_avg'] = opt_class_avg
            parsed['class_sd'] = opt_class_sd
            parsed['class_median'] = opt_class_median

    return parsed, []

with open('model/grade_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'ok': True,
        'status': 'up',
        'uptime_seconds': int(time.time()) - APP_STARTED_AT,
    })


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    parsed, errors = _validate_predict(data)
    if errors:
        return jsonify({'ok': False, 'error': 'Validation failed', 'errors': errors}), 400

    p = parsed
    study_hours = p['study_hours']
    attendance = p['attendance']
    assignment_avg = p['assignment_avg']
    past_gpa = p['past_gpa']
    sleep_hours = p['sleep_hours']
    grading_mode = p['grading_mode']

    # ── ML prediction ──
    features = np.array([[study_hours, attendance, assignment_avg,
                          past_gpa, sleep_hours]])
    features_scaled = scaler.transform(features)
    ml_prediction = model.predict(features_scaled)[0]
    ml_prediction = round(float(np.clip(ml_prediction, 0, 100)), 1)

    # If user provided course components, use their weighted score for grading.
    # This matches real syllabus math (score * weight), which users expect.
    requirements = p.get('requirements', [])
    if requirements:
        total_weight = sum(r['weight'] for r in requirements if r['weight'] > 0)
        if total_weight > 0:
            weighted_sum = sum(r['score'] * r['weight'] for r in requirements)
            prediction = round(float(np.clip(weighted_sum / total_weight, 0, 100)), 1)
            prediction_source = 'weighted_requirements'
        else:
            prediction = ml_prediction
            prediction_source = 'ml_model'
    else:
        prediction = ml_prediction
        prediction_source = 'ml_model'

    if grading_mode == 'curve':
        # ── Curve / Percentile grading ──
        class_avg = p['class_avg']
        class_sd = p['class_sd']
        class_median = p['class_median']

        pct_a = p['pct_a']
        pct_b = p['pct_b']
        pct_c = p['pct_c']

        # Calculate z-score of this student
        if class_sd > 0:
            z_score = (prediction - class_avg) / class_sd
        else:
            z_score = 0.0

        # Percentile of this student (what % of class scored below them)
        percentile = round(stats.norm.cdf(z_score) * 100, 1)

        # Determine grade based on percentile cutoffs
        cutoff_a = 100 - pct_a
        cutoff_b = 100 - pct_a - pct_b
        cutoff_c = 100 - pct_a - pct_b - pct_c

        if percentile >= cutoff_a:
            grade_letter = 'A'
            message = 'Excellent! You are in the top tier of your class!'
        elif percentile >= cutoff_b:
            grade_letter = 'B'
            message = 'Great job! Above the class average!'
        elif percentile >= cutoff_c:
            grade_letter = 'C'
            message = 'Average performance. Push harder!'
        elif percentile >= max(cutoff_c - 10, 0):
            grade_letter = 'D'
            message = 'Below average. Need improvement!'
        else:
            grade_letter = 'F'
            message = 'Failing. Please seek help immediately!'

        score_for_a = round(class_avg + stats.norm.ppf(cutoff_a / 100) * class_sd, 1) if cutoff_a > 0 else class_avg
        score_for_b = round(class_avg + stats.norm.ppf(cutoff_b / 100) * class_sd, 1) if cutoff_b > 0 else 0
        score_for_c = round(class_avg + stats.norm.ppf(cutoff_c / 100) * class_sd, 1) if cutoff_c > 0 else 0

        needed_for_a = max(0, round(score_for_a - prediction, 1))

        diff_from_avg = round(prediction - class_avg, 1)
        diff_from_median = round(prediction - class_median, 1)

        return jsonify({
            'ok': True,
            'prediction': prediction,
            'prediction_source': prediction_source,
            'ml_prediction': ml_prediction,
            'grade_letter': grade_letter,
            'message': message,
            'grading_mode': 'curve',
            'percentile': percentile,
            'z_score': round(z_score, 2),
            'needed_for_a': needed_for_a,
            'diff_from_avg': diff_from_avg,
            'diff_from_median': diff_from_median,
            'class_avg': class_avg,
            'class_sd': class_sd,
            'class_median': class_median,
            'score_for_a': score_for_a,
            'score_for_b': score_for_b,
            'score_for_c': score_for_c,
            'pct_a': pct_a,
            'pct_b': pct_b,
            'pct_c': pct_c,
        })

    # ── Custom score-based grading ──
    threshold_a = p['threshold_a']
    threshold_b = p['threshold_b']
    threshold_c = p['threshold_c']
    threshold_d = p['threshold_d']

    if prediction >= threshold_a:
        grade_letter = 'A'
        message = 'Excellent! Keep it up!'
    elif prediction >= threshold_b:
        grade_letter = 'B'
        message = 'Great job! Almost there!'
    elif prediction >= threshold_c:
        grade_letter = 'C'
        message = 'Not bad! You can do better!'
    elif prediction >= threshold_d:
        grade_letter = 'D'
        message = 'Need to work harder!'
    else:
        grade_letter = 'F'
        message = 'Please study more!'

    needed_for_a = max(0, round(threshold_a - prediction, 1))

    response = {
        'ok': True,
        'prediction': prediction,
        'prediction_source': prediction_source,
        'ml_prediction': ml_prediction,
        'grade_letter': grade_letter,
        'message': message,
        'grading_mode': 'custom_score',
        'needed_for_a': needed_for_a,
        'threshold_a': threshold_a,
        'threshold_b': threshold_b,
        'threshold_c': threshold_c,
        'threshold_d': threshold_d,
    }

    if 'class_avg' in p:
        class_avg = p['class_avg']
        class_median = p['class_median']
        class_sd = p['class_sd']
        response['class_avg'] = class_avg
        response['class_median'] = class_median
        response['class_sd'] = class_sd
        response['diff_from_avg'] = round(prediction - class_avg, 1)
        response['diff_from_median'] = round(prediction - class_median, 1)
        if class_sd > 0:
            response['z_score'] = round((prediction - class_avg) / class_sd, 2)
            response['percentile'] = round(stats.norm.cdf((prediction - class_avg) / class_sd) * 100, 1)

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)