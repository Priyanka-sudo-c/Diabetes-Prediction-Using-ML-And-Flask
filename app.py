import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk  # Import PIL for image support
import numpy as np
import pickle

# Load the trained model
try:
    model = pickle.load(open("diabetes_model.pkl", "rb"))
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("Loaded model does not support prediction functions.")
except Exception as e:
    messagebox.showerror("Model Error", f"Error loading the model: {str(e)}")
    exit()


# Function to validate inputs
def validate_inputs():
    try:
        pregnancies = int(entry_pregnancies.get())
        glucose = int(entry_glucose.get())
        blood_pressure = int(entry_blood_pressure.get())
        skin_thickness = int(entry_skin_thickness.get())
        insulin = int(entry_insulin.get())
        bmi = float(entry_bmi.get())
        diabetes_pedigree = float(entry_diabetes_pedigree.get())
        age = int(entry_age.get())
        cholesterol = float(entry_cholesterol.get())  # Updated label
        exercise_hours = float(entry_exercise_hours.get())  # Updated label

        if any(value < 0 for value in
               [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age, cholesterol,
                exercise_hours]):
            messagebox.showerror("Input Error", "Values must be non-negative.")
            return None

        return np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age,
                          cholesterol, exercise_hours]])

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")
        return None


# Function to predict diabetes
def predict_diabetes():
    input_data = validate_inputs()
    if input_data is not None:
        try:
            prediction = model.predict(input_data)
            probability = model.predict_proba(input_data)[0][1] * 100

            result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
            messagebox.showinfo("Prediction Result", f"Prediction: {result}\nProbability: {probability:.2f}%")

        except Exception as e:
            messagebox.showerror("Prediction Error", f"Error during prediction: {str(e)}")


# GUI Setup
root = tk.Tk()
root.title("Diabetes Prediction System")
root.geometry("500x650")

# Load background image
bg_image = Image.open("Diabetes.png")  # Replace with your image path
bg_image = bg_image.resize((500, 650), Image.LANCZOS)  # Resize to match window size
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a Canvas widget for the background
canvas = tk.Canvas(root, width=500, height=650)
canvas.pack(fill="both", expand=True)

# Add the image to the canvas
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Title Label
title_label = tk.Label(root, text="🩺 Diabetes Prediction System", font=("Arial", 18, "bold"), bg="#4CAF50", fg="white")
title_label.place(x=50, y=10, width=400)

# Frame for input fields
frame = tk.Frame(root, bg="#DFF2BF")
frame.place(x=50, y=50, width=400, height=450)

# Input Fields
fields = [
    ("Pregnancies", "entry_pregnancies"),
    ("Glucose", "entry_glucose"),
    ("Blood Pressure", "entry_blood_pressure"),
    ("Skin Thickness", "entry_skin_thickness"),
    ("Insulin", "entry_insulin"),
    ("BMI", "entry_bmi"),
    ("Diabetes Pedigree", "entry_diabetes_pedigree"),
    ("Age", "entry_age"),
    ("Cholesterol Level", "entry_cholesterol"),
    ("Exercise Hours per Week", "entry_exercise_hours")
]

entries = {}

for i, (label, var_name) in enumerate(fields):
    tk.Label(frame, text=f"{label}:", font=("Arial", 12), bg="#DFF2BF").grid(row=i, column=0, sticky="w", padx=5, pady=2)
    entry = ttk.Entry(frame, font=("Arial", 12))
    entry.grid(row=i, column=1, padx=5, pady=2)
    entries[var_name] = entry

# Assigning entry variables
entry_pregnancies = entries["entry_pregnancies"]
entry_glucose = entries["entry_glucose"]
entry_blood_pressure = entries["entry_blood_pressure"]
entry_skin_thickness = entries["entry_skin_thickness"]
entry_insulin = entries["entry_insulin"]
entry_bmi = entries["entry_bmi"]
entry_diabetes_pedigree = entries["entry_diabetes_pedigree"]
entry_age = entries["entry_age"]
entry_cholesterol = entries["entry_cholesterol"]
entry_exercise_hours = entries["entry_exercise_hours"]

# Predict Button
predict_button = tk.Button(root, text="🔍 Predict Diabetes", command=predict_diabetes, font=("Arial", 14, "bold"),
                           bg="#388E3C", fg="white", padx=10, pady=5)
predict_button.place(x=150, y=550, width=200)

# Run the GUI Loop
root.mainloop()