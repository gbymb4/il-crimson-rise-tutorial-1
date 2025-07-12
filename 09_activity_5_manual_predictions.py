# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:49:01 2025

@author: taske
"""

# Coding Activity 5: Manual Predictions
# Verifies understanding by calculating predictions manually

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

slope = model.coef_[0]
intercept = model.intercept_

# Let's verify our understanding by calculating manually
def manual_prediction(temp, slope, intercept):
    """Calculate energy consumption using the linear equation"""
    return slope * temp + intercept

# Test it
test_temp = 27
manual_result = manual_prediction(test_temp, slope, intercept)
sklearn_result = model.predict([[test_temp]])[0]

print(f"Manual calculation for {test_temp}°C: {manual_result:.2f} kWh")
print(f"Sklearn prediction for {test_temp}°C: {sklearn_result:.2f} kWh")
print(f"Difference: {abs(manual_result - sklearn_result):.6f} kWh")
print(f"Match: {abs(manual_result - sklearn_result) < 0.01}")

# Try a few more temperatures
print("\nTesting more temperatures:")
for temp in [22, 33, 38]:
    manual = manual_prediction(temp, slope, intercept)
    sklearn = model.predict([[temp]])[0]
    print(f"{temp}°C - Manual: {manual:.2f}, Sklearn: {sklearn:.2f}")
    
    