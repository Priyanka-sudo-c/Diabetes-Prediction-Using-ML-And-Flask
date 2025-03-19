import pickle
from sklearn.svm import SVC
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_diabetes()
X, y = data.data, (data.target > 140).astype(int)  # Convert target into binary (for example)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = SVC(probability=True)
model.fit(X_train, y_train)

# Save model
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")
