import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("1. Loading the dataset...")

df = pd.read_csv("asl_dataset.csv") # Load CSV file

X = df.drop("label", axis=1) # 'X' is the 42 coordinates (x0, y0... x20, y20)
y = df["label"] # 'y' is the letter (A, B, C, D, E, F)

print(f"Loaded {len(df)} total hand samples.")

print("\n2. Splitting data for testing...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # 80% train, 20% test

print("3. Training the AI Brain (Random Forest)...")
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train.values, y_train) # Train the model on the training data

print("\n4. Giving the AI a Pop Quiz...")
predictions = model.predict(X_test.values)

accuracy = accuracy_score(y_test, predictions) # Grade accuracy on the test set
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\n5. Saving the trained brain...")
joblib.dump(model, "asl_model.pkl")

print("SUCCESS! Model saved as 'asl_model.pkl'")