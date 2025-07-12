# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:44:18 2025

@author: taske
"""

# Model Creation and Training
# Creates and trains a linear regression model

import numpy as np
from sklearn.linear_model import LinearRegression

# Data setup (copy from 02_data_setup.py if running independently)
np.random.seed(42)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))

# Prepare data for sklearn
X = temperature.reshape(-1, 1)  # Features must be 2D for sklearn
y = energy_consumption            # Target can be 1D

# Create the model
model = LinearRegression()

# Train the model (this is where the "learning" happens)
model.fit(X, y)

# The model has now learned the best slope and intercept
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Model equation: Energy = {model.coef_[0]:.2f} × Temperature + {model.intercept_:.2f}")

