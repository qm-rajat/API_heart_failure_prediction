from flask import Flask, request, jsonify, session, send_file
import pickle
import numpy as np
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
app.secret_key = "somekaey"

# Load the trained model
with open('model/a.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Mapping dictionaries
sex_map = {"Male": 1, "Female": 0}
fasting_bs_map = {"Yes": 1, "No": 0}
exercise_angina_map = {"Yes": 1, "No": 0}
chest_pain_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
resting_ecg_map = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
st_slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}

# Function to predict heart failure
def display_prediction(input_data):
    input_vector = np.array([ 
        input_data['age'],
        sex_map[input_data['sex']],
        chest_pain_map[input_data['chest_pain_type']],
        input_data['resting_bp'],
        input_data['cholesterol'],
        fasting_bs_map[input_data['fasting_bs']],
        resting_ecg_map[input_data['resting_ecg']],
        input_data['max_hr'],
        exercise_angina_map[input_data['exercise_angina']],
        input_data['oldpeak'],
        st_slope_map[input_data['st_slope']]
    ]).reshape(1, -1)
    
    # Predict heart failure
    prediction = model.predict(input_vector)
    return prediction[0]

@app.route('/')
def home():
    return jsonify({"message": "Heart Failure Prediction API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Input validation
    required_fields = ['age', 'sex', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_bs', 
                       'resting_ecg', 'max_hr', 'exercise_angina', 'oldpeak', 'st_slope']
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return jsonify({"error": "Missing fields", "fields": missing_fields}), 400
    
    # Prepare input data
    input_data = {
        'age': int(data['age']),
        'sex': data['sex'],
        'chest_pain_type': data['chest_pain_type'],
        'resting_bp': int(data['resting_bp']),
        'cholesterol': int(data['cholesterol']),
        'fasting_bs': data['fasting_bs'],
        'resting_ecg': data['resting_ecg'],
        'max_hr': int(data['max_hr']),
        'exercise_angina': data['exercise_angina'],
        'oldpeak': float(data['oldpeak']),
        'st_slope': data['st_slope']
    }

    # Make prediction
    prediction = display_prediction(input_data)
    result = 'high risk' if prediction == 1 else 'low risk'

    # Store input data and result in session for receipt generation
    session['input_data'] = input_data
    session['result'] = result
    
    # Return prediction as JSON response
    return jsonify({"result": result, "input_data": input_data})

@app.route('/download', methods=['GET'])
def download_receipt():
    input_data = session.get('input_data')
    result = session.get('result')

    if input_data is None or result is None:
        return jsonify({"error": "No prediction data available"}), 404

    # Generate the receipt as a text file
    receipt_file = 'receipt.txt'
    with open(receipt_file, 'w') as f:
        f.write(f"Prediction Result: {result}\n")
        f.write("Input Data:\n")
        for key, value in input_data.items():
            f.write(f"{key}: {value}\n")

    # Clear session after download
    session.clear()

    # Send the file as a download
    return send_file(receipt_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
