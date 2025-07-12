# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:51:58 2025

@author: taske
"""

# Results Visualization
# Creates plots showing actual data vs predictions and residuals

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data setup and model training (copy from previous files if running independently)
np.random.seed(42)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))
X = temperature.reshape(-1, 1)
y = energy_consumption

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

# Plot actual data vs predictions
plt.figure(figsize=(12, 6))

# Create subplot for better visualization
plt.subplot(1, 2, 1)
plt.scatter(temperature, energy_consumption, color='blue', alpha=0.7, label='Actual Data', s=50)
plt.plot(temperature, predictions, color='red', linewidth=2, label='Linear Regression')
plt.xlabel('Temperature (°C)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Linear Regression: Energy Consumption Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# Add residual plot
plt.subplot(1, 2, 2)
residuals = y - predictions
plt.scatter(predictions, residuals, color='green', alpha=0.7, s=50)
plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
plt.xlabel('Predicted Energy Consumption (kWh)')
plt.ylabel('Residuals (kWh)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

