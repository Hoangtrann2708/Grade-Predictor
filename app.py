from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from scipy import stats

app = Flask(__name__)

with open('model/grade_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # ── Core inputs ──
    study_hours    = float(data.get('study_hours', 0))
    attendance     = float(data.get('attendance', 0))
    assignment_avg = float(data.get('assignment_avg', 0))
    past_gpa       = float(data.get('past_gpa', 2.0))
    sleep_hours    = float(data.get('sleep_hours', 7.0))

    # ── ML prediction ──
    features = np.array([[study_hours, attendance, assignment_avg,
                          past_gpa, sleep_hours]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prediction = round(float(np.clip(prediction, 0, 100)), 1)

    # ── Grading mode ──
    grading_mode = data.get('grading_mode', 'custom_score')

    if grading_mode == 'curve':
        # ── Curve / Percentile grading ──
        class_avg    = float(data.get('class_avg', 75))
        class_sd     = float(data.get('class_sd', 10))
        class_median = float(data.get('class_median', 75))

        # Grade distribution percentages (from top)
        pct_a = float(data.get('pct_a', 30))
        pct_b = float(data.get('pct_b', 40))
        pct_c = float(data.get('pct_c', 20))
        # pct_d = remaining

        # Calculate z-score of this student
        if class_sd > 0:
            z_score = (prediction - class_avg) / class_sd
        else:
            z_score = 0.0

        # Percentile of this student (what % of class scored below them)
        percentile = round(stats.norm.cdf(z_score) * 100, 1)

        # Determine grade based on percentile cutoffs
        # Top pct_a% = A, next pct_b% = B, next pct_c% = C, rest = D/F
        cutoff_a = 100 - pct_a             # e.g. top 30% → need 70th percentile+
        cutoff_b = 100 - pct_a - pct_b     # e.g. next 40% → need 30th percentile+
        cutoff_c = 100 - pct_a - pct_b - pct_c  # e.g. next 20% → need 10th percentile+

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

        # Calculate what score is needed for each grade boundary
        score_for_a = round(class_avg + stats.norm.ppf(cutoff_a / 100) * class_sd, 1) if cutoff_a > 0 else class_avg
        score_for_b = round(class_avg + stats.norm.ppf(cutoff_b / 100) * class_sd, 1) if cutoff_b > 0 else 0
        score_for_c = round(class_avg + stats.norm.ppf(cutoff_c / 100) * class_sd, 1) if cutoff_c > 0 else 0

        # How far from A
        needed_for_a = max(0, round(score_for_a - prediction, 1))

        # Comparison with class
        diff_from_avg    = round(prediction - class_avg, 1)
        diff_from_median = round(prediction - class_median, 1)

        return jsonify({
            'prediction': prediction,
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

    else:
        # ── Custom score-based grading ──
        # User sets their own thresholds (defaults to standard 90/80/70/60)
        threshold_a = float(data.get('threshold_a', 90))
        threshold_b = float(data.get('threshold_b', 80))
        threshold_c = float(data.get('threshold_c', 70))
        threshold_d = float(data.get('threshold_d', 60))

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

        # Optional: if class stats provided, include comparison
        response = {
            'prediction': prediction,
            'grade_letter': grade_letter,
            'message': message,
            'grading_mode': 'custom_score',
            'needed_for_a': needed_for_a,
            'threshold_a': threshold_a,
            'threshold_b': threshold_b,
            'threshold_c': threshold_c,
            'threshold_d': threshold_d,
        }

        # If user also provided class stats for reference
        if data.get('class_avg'):
            class_avg    = float(data.get('class_avg', 0))
            class_median = float(data.get('class_median', class_avg))
            class_sd     = float(data.get('class_sd', 10))
            response['class_avg']        = class_avg
            response['class_median']     = class_median
            response['class_sd']         = class_sd
            response['diff_from_avg']    = round(prediction - class_avg, 1)
            response['diff_from_median'] = round(prediction - class_median, 1)
            if class_sd > 0:
                response['z_score']    = round((prediction - class_avg) / class_sd, 2)
                response['percentile'] = round(stats.norm.cdf((prediction - class_avg) / class_sd) * 100, 1)

        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)