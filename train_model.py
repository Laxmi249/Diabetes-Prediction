import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Dummy training data
X_train = np.random.rand(100, 8)
y_train = np.random.randint(0, 2, 100)

# Train a simple model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
model_path = "C:/Users/hp/OneDrive/Documents/diabetes_prediction/model.pkl"
with open(model_path, "wb") as file:
    pickle.dump(model, file)

print(f"Model saved at: {model_path}")
