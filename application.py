from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('models/xgb_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve input features from form
        features = []

        # Time input
        time = float(request.form['Time'])
        features.append(time)

        # V1 to V28 inputs
        for i in range(1, 29):
            feature_name = f'V{i}'
            value = float(request.form[feature_name])
            features.append(value)

        # Amount input
        amount = float(request.form['Amount'])
        features.append(amount)

        # Convert to numpy array and reshape
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)
        output = int(prediction[0])

        result = 'Fraudulent Transaction' if output == 1 else 'Legitimate Transaction'

    except Exception as e:
        result = f"Error: {e}"

    return render_template('index.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)