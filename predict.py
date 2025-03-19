import numpy as np
import pickle

# Load the saved model and scaler
with open("diabetes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


# Function to make predictions
def predict_diabetes(input_data):
    # Convert input data to a numpy array and reshape for model input
    input_array = np.array(input_data).reshape(1, -1)

    # Standardize the input
    input_scaled = scaler.transform(input_array)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Interpret result
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return result


# Example input for prediction [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigree, Age]
sample_input = [2, 120, 70, 23, 50, 30.5, 0.5, 25]  # Modify values as needed
result = predict_diabetes(sample_input)
print(f"Prediction: {result}")
