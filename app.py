from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

app = Flask(__name__)
CORS(app)  # This prevents "Cross-Origin" errors

# Load the files you downloaded from Colab
model = pickle.load(open('final_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# The 11 features your model expects (from your Colab correlation analysis)
FEATURES = ['id', 'sttl', 'ct_state_ttl', 'dload', 'ct_dst_sport_ltm', 
            'dmean', 'rate', 'swin', 'dwin', 'ct_src_dport_ltm', 'ct_dst_src_ltm']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}

    # Validate required features
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({
            'error': 'Missing required fields',
            'missing': missing,
            'status': 400
        }), 400

    # Ensure ordered features (fix sorting/order mismatch issues)
    df = pd.DataFrame([data], columns=FEATURES)

    try:
        # The model is trained on selected 11 features; scaler was saved for full dataset,
        # so we skip scaler transform to avoid mismatch and use the model directly.
        prediction = model.predict(df)
    except Exception as err:
        return jsonify({'error': str(err), 'status': 500}), 500

    result = "🚨 ATTACK" if int(prediction[0]) == 1 else "✅ NORMAL"
    return jsonify({'result': result, 'status': 200})

if __name__ == '__main__':
    app.run(port=5000, debug=True)