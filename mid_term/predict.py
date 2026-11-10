
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify


# Find model files in the same folder as predict.py
BASE_DIR = Path(__file__).resolve().parent

# Load DictVectorizer
with open(BASE_DIR / 'dv.bin', 'rb') as f:
    dv = pickle.load(f)

# Load XGBoost model
model = xgb.Booster()
model.load_model(str(BASE_DIR / 'model.json'))

print("Model and DictVectorizer loaded successfully!")


# Create Flask application
app = Flask(__name__)


# Prediction function
def predict(booking):
    booking = booking.copy()

    # Handle missing values
    if pd.isna(booking.get('children')):
        booking['children'] = 0

    if pd.isna(booking.get('country')):
        booking['country'] = 'unknown'

    agent = booking.get('agent')

    if pd.isna(agent):
        agent = -1

    booking['agent'] = str(int(agent))

    # Encode booking features
    X = dv.transform([booking])

    # Convert to XGBoost format
    dmatrix = xgb.DMatrix(X)

    # Predict cancellation probability
    probability = model.predict(dmatrix)[0]

    return float(probability)


# API endpoint
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    booking = request.get_json(silent=True)

    if not isinstance(booking, dict):
        return jsonify({
            'error': 'Please send a JSON object with booking information.'
        }), 400

    try:
        probability = predict(booking)
    except (ValueError, TypeError) as e:
        return jsonify({'error': str(e)}), 400

    result = {
        'cancellation_probability': probability,
        'is_canceled': bool(probability >= 0.5)
    }

    return jsonify(result)


# Run Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9696)
