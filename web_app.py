
from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # Load the StandardScaler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Load the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [float(request.form['pregnancies']),
                float(request.form['glucose']),
                float(request.form['bp']),
                float(request.form['skin_thickness']),
                float(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['dpf']),
                float(request.form['age'])]

        # Convert to NumPy array and reshape
        input_data = np.array(data).reshape(1, -1)

        # Scale the input data
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1] * 100  # Get probability

        # Set result message
        result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"

        # Return JSON response
        return jsonify({
            'prediction': result,
            'probability': f"{probability:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
