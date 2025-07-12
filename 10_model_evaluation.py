# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:50:35 2025

@author: taske
"""

# Model Evaluation
# Evaluates how good the model is using metrics

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Data setup and model training (copy from previous files if running independently)
np.random.seed(42)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))
X = temperature.reshape(-1, 1)
y = energy_consumption

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# How good is our model?
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print("Model Performance:")
print(f"Mean Squared Error: {mse:.2f} kWh²")
print(f"Root Mean Squared Error: {np.sqrt(mse):.2f} kWh")
print(f"R² Score: {r2:.3f}")
print(f"Model explains {r2*100:.1f}% of the variance in energy consumption")

