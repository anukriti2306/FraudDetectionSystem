from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('models/decision_tree_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def index():
    return "✅ Fraud Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Expecting 30 features: 'Time', 'V1' through 'V28', 'Amount'
        expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_df = pd.DataFrame([data])

        # Validate input
        if not all(col in input_df.columns for col in expected_columns):
            return jsonify({'error': 'Missing one or more required input features.'}), 400

        # Scale 'Time' and 'Amount'
        input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

        # Predict
        prediction = model.predict(input_df)
        result = 'Fraudulent' if prediction[0] == 1 else 'Legitimate'

        return jsonify({'prediction': int(prediction[0]), 'result': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
