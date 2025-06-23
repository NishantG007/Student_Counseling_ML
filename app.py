from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'logistic_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return render_template('index.html', prediction_text="Model not loaded.")

    try:
        # Get input from the form in the correct order
        input_features = [
            float(request.form['age']),
            float(request.form['academic_pressure']),
            float(request.form['cgpa']),
            float(request.form['study_satisfaction']),
            float(request.form['sleep_duration']),
            float(request.form['dietary_habits']),
            float(request.form['have_you_ever_had_suicidal_thoughts_']),
            float(request.form['work_study_hours']),
            float(request.form['financial_stress']),
            float(request.form['family_history_of_mental_illness']),
        ]

        input_array = np.array([input_features])
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            result_text = "You need counselling"
        else:
            result_text = "You don't need counselling as of now"

        return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
    
