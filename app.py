from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the model
try:
    model_path = os.path.join(os.path.dirname(__file__), 'accident_severity_model.pkl')
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Mapping severity levels for better output display (adjust according to the actual model's classes)
severity_mapping = {
    1: 'Low Severity',
    2: 'Moderate Severity',
    3: 'High Severity',
    4: 'Severe Accident'
}

# Weather condition mapping used by our dummy model.
# NOTE: If your real scikit-learn model handles strings (e.g. using OneHotEncoder), 
# you should remove this mapping and pass the string directly.
weather_mapping = {
    'Clear': 0, 'Cloudy': 1, 'Rain': 2, 'Snow': 3, 'Storm': 4, 'Fog': 5
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded. Ensure accident_severity_model.pkl is present.'}), 500

    try:
        data = request.json
        
        # Extract features
        temperature = float(data.get('temperature', 0))
        visibility = float(data.get('visibility', 0))
        wind_speed = float(data.get('wind_speed', 0))
        weather_condition_str = data.get('weather_condition', 'Clear')

        # Translate weather condition to numeric based on our mapping.
        weather_condition = weather_mapping.get(weather_condition_str, 0)

        # Create DataFrame
        input_df = pd.DataFrame([{
            'temperature': temperature,
            'visibility': visibility,
            'wind_speed': wind_speed,
            'weather_condition': weather_condition
        }])

        # Make prediction
        prediction = model.predict(input_df)
        
        # Fetch the numeric prediction and map it to a human-readable string
        pred_value = prediction[0]
        result_text = severity_mapping.get(pred_value, f"Severity Level: {pred_value}")

        return jsonify({'severity_prediction': result_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app locally
    # gunicorn will be used for production execution
    app.run(debug=True, host='0.0.0.0', port=5000)
