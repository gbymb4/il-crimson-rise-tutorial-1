# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:47:56 2025

@author: taske
"""

# Making Predictions
# Uses the trained model to make predictions

import numpy as np
from sklearn.linear_model import LinearRegression

# Data setup and model training (copy from previous files if running independently)
np.random.seed(42)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))
X = temperature.reshape(-1, 1)
y = energy_consumption

model = LinearRegression()
model.fit(X, y)

# Make predictions on our training data
predictions = model.predict(X)

# Let's predict for a new temperature
new_temp = np.array([[27]])  # 27°C (note: must be 2D)
predicted_energy = model.predict(new_temp)
print(f"Predicted energy consumption at 27°C: {predicted_energy[0]:.2f} kWh")

# Predict for multiple temperatures
test_temps = np.array([[20], [25], [30], [35]])
test_predictions = model.predict(test_temps)
print("\nPredictions for multiple temperatures:")
for temp, pred in zip(test_temps.flatten(), test_predictions):
    print(f"  {temp}°C: {pred:.2f} kWh")
    
    