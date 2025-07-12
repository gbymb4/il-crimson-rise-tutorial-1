# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:53:46 2025

@author: taske
"""

# Coding Activity 6: Model Evaluation
# Analyzes model performance and identifies best/worst predictions

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Data setup and model training (copy from previous files if running independently)
np.random.seed(42)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))
X = temperature.reshape(-1, 1)
y = energy_consumption

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
r2 = r2_score(y, predictions)

# Let's understand model performance
print("Model Performance Analysis:")
print(f"R² Score: {r2:.3f}")
print()

# Your interpretation:
if r2 > 0.8:
    print("✓ Excellent fit - model explains most variance")
elif r2 > 0.6:
    print("✓ Good fit - model captures main relationship")
elif r2 > 0.4:
    print("⚠ Moderate fit - some predictive power")
else:
    print("⚠ Poor fit - limited predictive power")

# Calculate residuals (errors)
residuals = y - predictions
print("\nError Analysis:")
print(f"Largest prediction error: {max(abs(residuals)):.2f} kWh")
print(f"Average absolute error: {np.mean(abs(residuals)):.2f} kWh")
print(f"Standard deviation of errors: {np.std(residuals):.2f} kWh")

# Find best and worst predictions
best_idx = np.argmin(abs(residuals))
worst_idx = np.argmax(abs(residuals))
print(f"\nBest prediction: {temperature[best_idx]}°C (error: {residuals[best_idx]:.2f} kWh)")
print(f"Worst prediction: {temperature[worst_idx]}°C (error: {residuals[worst_idx]:.2f} kWh)")

