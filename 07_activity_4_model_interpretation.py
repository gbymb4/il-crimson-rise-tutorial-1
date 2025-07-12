# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:45:36 2025

@author: taske
"""

# Coding Activity 4: Model Interpretation
# Understands what the model learned

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

# Let's understand what the model learned
slope = model.coef_[0]
intercept = model.intercept_

print(f"Model equation: Energy = {slope:.2f} × Temperature + {intercept:.2f}")
print()

# Model interpretation
print("Model interpretation:")
print(f"1. If temperature increases by 1°C, energy consumption increases by {slope:.2f} kWh")
print(f"2. At 0°C, the model predicts {intercept:.2f} kWh energy consumption")
print(f"3. The baseline energy consumption (intercept) is {intercept:.2f} kWh")

# Let's test our understanding
temp_change = 5  # 5°C increase
energy_change = slope * temp_change
print(f"\nIf temperature increases by {temp_change}°C:")
print(f"Energy consumption increases by {energy_change:.2f} kWh")

