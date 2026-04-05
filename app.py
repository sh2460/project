from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    hours = float(request.form['hours'])
    attendance = float(request.form['attendance'])
    assignments = float(request.form['assignments'])
    previous_score = float(request.form['previous_score'])

    features = np.array([[hours, attendance, assignments, previous_score]])
    prediction = model.predict(features)[0]

    # Category logic
    if prediction >= 75:
        category = "Good 🟢"
        suggestion = "Excellent performance! Keep it up."
    elif prediction >= 50:
        category = "Average 🟡"
        suggestion = "Improve consistency and practice more."
    else:
        category = "Poor 🔴"
        suggestion = "Increase study hours and focus on basics."

    # ✅ ONLY ONE RETURN (IMPORTANT)
    return render_template(
        'result.html',
        prediction=round(prediction, 2),
        category=category,
        suggestion=suggestion,
        hours=hours,
        attendance=attendance,
        assignments=assignments,
        previous_score=previous_score
    )


import os

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)