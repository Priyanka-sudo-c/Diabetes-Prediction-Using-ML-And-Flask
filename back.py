from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
try:
    with open("diabetes_model.pkl", "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded."}), 500
        
        data = request.get_json()
        input_features = np.array([[
            float(data["pregnancies"]), float(data["glucose"]),
            float(data["bloodPressure"]), float(data["skinThickness"]),
            float(data["insulin"]), float(data["bmi"]),
            float(data["diabetesPedigree"]), float(data["age"])
        ]])

        # Ensure model is trained before prediction
        if not hasattr(model, "predict"):
            return jsonify({"error": "Model not trained."}), 500

        prediction = model.predict(input_features)[0]
        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
