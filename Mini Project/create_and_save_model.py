from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Create a dummy dataset
X = np.random.rand(100, 7)
y = np.random.randint(0, 5, 100)

# Create and fit a model
model = RandomForestClassifier(n_estimators=10)
model.fit(X, y)

# Create and fit a scaler
scaler = StandardScaler()
scaler.fit(X)

# Save the model and scaler
joblib.dump(model, 'crop_prediction_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully")

# Verify that we can load them
loaded_model = joblib.load('crop_prediction_model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

print("Model and scaler loaded successfully")
