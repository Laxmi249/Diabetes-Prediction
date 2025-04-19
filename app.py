from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import numpy as np
import random

app = Flask(__name__)
app.secret_key = 'your_strong_secret_key'

# Load the trained model and scaler
try:
    model = joblib.load('static/models/diabetes_logistic_model.pkl')
    scaler = joblib.load('static/models/diabetes_scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None

# Health tips
health_tips = [
    "Drink plenty of water every day.",
    "Aim for at least 30 minutes of physical activity daily.",
    "Get enough sleep to recharge your body.",
    "Eat a balanced diet with fruits, vegetables, and lean proteins.",
    "Practice mindful eating to avoid overeating.",
    "Avoid smoking and excessive alcohol consumption.",
    "Take breaks to reduce stress and boost productivity.",
    "Wash your hands regularly to prevent infection.",
    "Limit screen time to protect your eyes and mental health.",
    "Stay positive and practice gratitude daily."
]

def get_random_tip():
    return random.choice(health_tips)

def predict_diabetes(data):
    if model:
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)[0]
        return 'Diabetic' if prediction == 1 else 'Non-Diabetic'
    return 'Model not available'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Retrieve the form data
        data = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
        
        # Predict diabetes
        prediction = predict_diabetes(np.array([data]))
        
        # Flash the prediction result
        flash(f'The prediction result is: {prediction}')
        return redirect(url_for('index'))  # Redirect back to the index page

    daily_tip = get_random_tip()  # For GET requests
    return render_template('index.html', daily_tip=daily_tip)

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')

@app.route('/prevention')
def prevention():
    return render_template('prevention.html')

if __name__ == '__main__':
    app.run(debug=True)
