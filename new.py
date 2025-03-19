import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("diabetes.csv")  # Make sure 'diabetes.csv' is in your project folder

# Prepare Features and Target
X = df.drop(columns=["Outcome"])  # Exclude target column
y = df["Outcome"]  # Target variable (1 = Diabetic, 0 = Non-Diabetic)

#  Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Train a RandomForest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Save the trained model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model successfully trained and saved as 'diabetes_model.pkl'.")


























import pickle
import os
import numpy as np
from flask import Flask, render_template, request


import pickle
import os
print(os.path.abspath("model.pkl"))


# Load the model from the pickle file
with open("diabetes.pkl", "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully!")
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f, encoding="latin1")

import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

import sys
print(sys.version)

app = Flask(__name__)
import os

file_path = "C:/Users/DELL/PycharmProjects/PythonProject8/model.pkl"  # Update if needed
if os.path.exists(file_path):
    print("File exists")
    print("File size:", os.path.getsize(file_path), "bytes")  # Must be > 0
else:
    print("File does not exist")
import os

file_path = "C:/Users/DELL/PycharmProjects/PythonProject8/model.pkl"  # Ensure this is the correct path
print("File exists:", os.path.exists(file_path))
print("File size:", os.path.getsize(file_path), "bytes")


#  Ensure the model file exists before loading
model_path = "diabetes_model.pkl"
if os.path.exists(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except EOFError:
        raise ValueError("⚠ Model file is corrupt. Retrain and save the model again!")
else:
    raise FileNotFoundError(" Model file 'diabetes_model.pkl' not found!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        features = [float(request.form[key]) for key in request.form]
        input_data = np.array([features])

        # Predict using the model
        prediction = model.predict(input_data)[0]

        # Interpret prediction
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        return render_template('predict.html', result=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

    from sklearn.preprocessing import StandardScaler
    import pickle
    from flask import Flask, render_template, request
    import tkinter as tk

    root = tk.Tk()
    from tkinter import filedialog
    import tkinter as tk

    from tkinter import Tk, filedialog, messagebox
    from pandastable import Table
    import pandas as pd

    import pickle

    # Load the model
    with open("oh_rc.pkl", "rb") as file:
        model = pickle.load(file)

    print("Loaded model type:", type(model))  # Check what is inside the pickle file


    class DiabetesApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Diabetes Data Viewer")
            self.root.geometry("800x600")

            self.load_button = tk.Button(root, text="Load .pkl File", command=self.load_file, font=("Arial", 12))
            self.load_button.pack(pady=10)

            self.frame = tk.Frame(root)
            self.frame.pack(fill=tk.BOTH, expand=True)

        def load_file(self):
            file_path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
            if file_path:
                try:
                    df = pd.read_pickle(file_path)
                    self.show_data(df)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load file:\n{e}")

        def show_data(self, df):
            for widget in self.frame.winfo_children():
                widget.destroy()

            table = Table(self.frame, dataframe=df, showtoolbar=True, showstatusbar=True)
            table.show()


    if __name__ == "__main__":
        root = tk.Tk()
        app = DiabetesApp(root)
        root.mainloop()

    import pandas as pd
    import numpy as np
    import pickle
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Load dataset
    df = pd.read_csv("diabetes.csv")  # Make sure you have the correct dataset

    # Define Features and Target
    X = df.drop(columns=["Outcome"])  # Features
    y = df["Outcome"]  # Target Variable

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save Model
    with open("oh_rc.pkl", "wb") as file:
        pickle.dump(model, file)

    print("Model trained and saved successfully!")
























import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
import numpy as np

# Initialize the GUI
root = tk.Tk()
root.title("Diabetes Prediction")
root.geometry("500x400")

# Global variable to hold the loaded model
loaded_model = None

import pickle

# Function to load the .pkl model
def load_model():
    global loaded_model
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
    if file_path:
        try:
            with open(file_path, "rb") as file:
                loaded_model = pickle.load(file)
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")


# Function to make predictions
def predict_diabetes():
    if loaded_model is None:
        messagebox.showwarning("Warning", "Please load the model first!")
        return

    try:
        # Collect user inputs
        values = [
            float(entry_pregnancies.get()),
            float(entry_glucose.get()),
            float(entry_bp.get()),
            float(entry_skin_thickness.get()),
            float(entry_insulin.get()),
            float(entry_bmi.get()),
            float(entry_dpf.get()),
            float(entry_age.get())
        ]

        # Convert input to numpy array and reshape
        input_data = np.array(values).reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(input_data)

        # Display result
        if prediction[0] == 1:
            result_label.config(text="Diabetes Detected!", fg="red")
        else:
            result_label.config(text="No Diabetes", fg="green")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {str(e)}")


# UI Elements
tk.Button(root, text="Load .pkl File", command=load_model, font=("Arial", 12)).pack(pady=10)

# Labels and Entry Fields
tk.Label(root, text="Pregnancies:").pack()
entry_pregnancies = tk.Entry(root)
entry_pregnancies.pack()

tk.Label(root, text="Glucose Level:").pack()
entry_glucose = tk.Entry(root)
entry_glucose.pack()

tk.Label(root, text="Blood Pressure:").pack()
entry_bp = tk.Entry(root)
entry_bp.pack()

tk.Label(root, text="Skin Thickness:").pack()
entry_skin_thickness = tk.Entry(root)
entry_skin_thickness.pack()

tk.Label(root, text="Insulin Level:").pack()
entry_insulin = tk.Entry(root)
entry_insulin.pack()

tk.Label(root, text="BMI:").pack()
entry_bmi = tk.Entry(root)
entry_bmi.pack()

tk.Label(root, text="Diabetes Pedigree Function:").pack()
entry_dpf = tk.Entry(root)
entry_dpf.pack()

tk.Label(root, text="Age:").pack()
entry_age = tk.Entry(root)
entry_age.pack()

# Prediction Button
tk.Button(root, text="Predict", command=predict_diabetes, font=("Arial", 12), bg="blue", fg="white").pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack()

# Run the GUI
root.mainloop()

import pickle
from sklearn.linear_model import LogisticRegression

# Train and save a simple model
model = LogisticRegression()
X_train = [[1, 2], [3, 4], [5, 6], [7, 8]]
y_train = [0, 1, 0, 1]
model.fit(X_train, y_train)

with open('oh_rc.pkl.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved successfully!")

