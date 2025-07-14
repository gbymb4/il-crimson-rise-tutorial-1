# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 13:19:17 2025

@author: taske
"""

# ML Homework - Linear Regression Practice
# Session 1 Follow-up Assignment
# 
# Instructions: Complete all TODO sections marked with comments
# Run the script section by section and observe the outputs
# Save your completed script as 'homework_completed.py'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducible results
np.random.seed(42)

print("=" * 60)
print("ENVIRONMENTAL ML HOMEWORK - LINEAR REGRESSION PRACTICE")
print("=" * 60)

# ============================================================================
# PART 1: NUMPY OPERATIONS PRACTICE
# ============================================================================

print("\nPART 1: NUMPY OPERATIONS PRACTICE")
print("-" * 40)

# Dataset 1: Temperature and Energy Consumption (from class)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = np.array([95.5, 101.2, 108.8, 115.3, 122.1, 128.9, 135.4, 142.7, 149.2, 156.8])

print("Original Data:")
print(f"Temperature (°C): {temperature}")
print(f"Energy Consumption (kWh): {energy_consumption}")

# TODO 1: Calculate basic statistics using numpy functions
# Hint: Use np.mean(), np.max(), np.min(), np.std()
print("\nTODO 1: Basic Statistics")
avg_temp = # TODO: Calculate average temperature
max_energy = # TODO: Find maximum energy consumption
min_energy = # TODO: Find minimum energy consumption
temp_std = # TODO: Calculate standard deviation of temperature

print(f"Average temperature: {avg_temp:.2f}°C")
print(f"Maximum energy consumption: {max_energy:.2f} kWh")
print(f"Minimum energy consumption: {min_energy:.2f} kWh")
print(f"Temperature standard deviation: {temp_std:.2f}°C")

# TODO 2: Array operations and filtering
# Hint: Use boolean indexing like array[array > threshold]
print("\nTODO 2: Array Operations")
hot_days = # TODO: Find temperatures above 30°C
hot_day_energy = # TODO: Find energy consumption for hot days (temp > 30°C)
energy_range = # TODO: Calculate range of energy consumption (max - min)

print(f"Hot days (>30°C): {hot_days}")
print(f"Energy consumption on hot days: {hot_day_energy}")
print(f"Energy consumption range: {energy_range:.2f} kWh")

# TODO 3: Array reshaping for sklearn
# Hint: Use .reshape(-1, 1) to convert 1D array to 2D column vector
print("\nTODO 3: Array Reshaping")
X_temp = # TODO: Reshape temperature array for sklearn (should be 2D)
y_energy = # TODO: Keep energy_consumption as 1D array (or assign it directly)

print(f"Original temperature shape: {temperature.shape}")
print(f"Reshaped X_temp shape: {X_temp.shape}")
print(f"y_energy shape: {y_energy.shape}")

# TODO 4: Create predictions array using numpy
# Hint: Use np.linspace() to create evenly spaced temperature values
print("\nTODO 4: Create Prediction Range")
temp_range = # TODO: Create array from 10°C to 45°C with 36 points using np.linspace()
temp_range_2d = # TODO: Reshape for sklearn predictions

print(f"Temperature range for predictions: {temp_range[:5]}...{temp_range[-5:]}")
print(f"Temperature range shape: {temp_range_2d.shape}")

# ============================================================================
# PART 2: LINEAR REGRESSION PRACTICE
# ============================================================================

print("\n" + "=" * 60)
print("PART 2: LINEAR REGRESSION PRACTICE")
print("=" * 60)

# TODO 5: Create and train linear regression model
# Hint: Use LinearRegression(), then .fit() method
print("\nTODO 5: Model Training")
model = # TODO: Create LinearRegression instance
# TODO: Train the model using X_temp and y_energy

slope = model.coef_[0]
intercept = model.intercept_

print(f"Model trained successfully!")
print(f"Slope (coefficient): {slope:.3f}")
print(f"Intercept: {intercept:.3f}")
print(f"Model equation: Energy = {slope:.3f} × Temperature + {intercept:.3f}")

# TODO 6: Make predictions
# Hint: Use .predict() method
print("\nTODO 6: Making Predictions")
predictions = # TODO: Make predictions on training data (X_temp)
range_predictions = # TODO: Make predictions on temp_range_2d

# Test individual predictions
test_temperatures = [20, 25, 30, 35]
print(f"Individual predictions:")
for temp in test_temperatures:
    temp_2d = np.array([[temp]])
    pred = # TODO: Make prediction for this temperature
    print(f"  {temp}°C: {pred[0]:.2f} kWh")

# TODO 7: Calculate model performance metrics
# Hint: Use mean_squared_error() and r2_score() from sklearn.metrics
print("\nTODO 7: Model Evaluation")
mse = # TODO: Calculate Mean Squared Error
r2 = # TODO: Calculate R-squared score
rmse = # TODO: Calculate Root Mean Squared Error (hint: use np.sqrt())

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Model explains {r2*100:.1f}% of the variance")

# TODO 8: Calculate residuals and find best/worst predictions
# Hint: Residuals = actual - predicted
print("\nTODO 8: Residual Analysis")
residuals = # TODO: Calculate residuals (y_energy - predictions)
abs_residuals = # TODO: Calculate absolute residuals
best_idx = # TODO: Find index of smallest absolute residual (hint: use np.argmin())
worst_idx = # TODO: Find index of largest absolute residual (hint: use np.argmax())

print(f"Residual Analysis:")
print(f"Best prediction: {temperature[best_idx]}°C (error: {residuals[best_idx]:.2f} kWh)")
print(f"Worst prediction: {temperature[worst_idx]}°C (error: {residuals[worst_idx]:.2f} kWh)")
print(f"Mean absolute error: {np.mean(abs_residuals):.2f} kWh")

# ============================================================================
# PART 3: VISUALIZATION
# ============================================================================

print("\n" + "=" * 60)
print("PART 3: VISUALIZATION")
print("=" * 60)

# TODO 9: Create comprehensive visualization
# Hint: Use plt.figure(), plt.subplot(), plt.scatter(), plt.plot()
print("\nTODO 9: Creating Plots")

# TODO: Create a figure with size (15, 5)
plt.figure(figsize=(15, 5))

# Plot 1: Original data with regression line
# TODO: Create first subplot (1 row, 3 columns, position 1)
plt.subplot(1, 3, 1)
# TODO: Create scatter plot of temperature vs energy_consumption
# TODO: Plot regression line using temp_range and range_predictions
plt.xlabel('Temperature (°C)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Linear Regression: Energy vs Temperature')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Residuals plot
# TODO: Create second subplot (1 row, 3 columns, position 2)
plt.subplot(1, 3, 2)
# TODO: Create scatter plot of predictions vs residuals
# TODO: Add horizontal line at y=0 using plt.axhline()
plt.xlabel('Predicted Energy (kWh)')
plt.ylabel('Residuals (kWh)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# Plot 3: Actual vs Predicted
# TODO: Create third subplot (1 row, 3 columns, position 3)
plt.subplot(1, 3, 3)
# TODO: Create scatter plot of y_energy vs predictions
# TODO: Add perfect prediction line (y=x) using plt.plot()
plt.xlabel('Actual Energy (kWh)')
plt.ylabel('Predicted Energy (kWh)')
plt.title('Actual vs Predicted')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# PART 4: ADDITIONAL DATASET PRACTICE
# ============================================================================

print("\n" + "=" * 60)
print("PART 4: ADDITIONAL DATASET PRACTICE")
print("=" * 60)

# Dataset 2: Solar Panel Output vs Sunlight Hours
sunlight_hours = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
solar_output = np.array([8.5, 12.3, 16.8, 20.1, 24.5, 28.9, 32.4, 36.7, 40.2, 44.8])

print("Solar Panel Dataset:")
print(f"Sunlight Hours: {sunlight_hours}")
print(f"Solar Output (kWh): {solar_output}")

# TODO 10: Complete analysis for solar dataset
# Hint: Follow the same steps as above
print("\nTODO 10: Solar Dataset Analysis")

# TODO: Reshape sunlight_hours for sklearn
X_sunlight = # TODO: Reshape sunlight_hours

# TODO: Create and train model for solar data
solar_model = # TODO: Create LinearRegression instance
# TODO: Train the model

# TODO: Make predictions
solar_predictions = # TODO: Make predictions on X_sunlight

# TODO: Calculate performance metrics
solar_mse = # TODO: Calculate MSE
solar_r2 = # TODO: Calculate R²

print(f"Solar Model Results:")
print(f"Slope: {solar_model.coef_[0]:.3f} kWh per hour")
print(f"Intercept: {solar_model.intercept_:.3f} kWh")
print(f"R² Score: {solar_r2:.3f}")
print(f"MSE: {solar_mse:.2f}")

# TODO 11: Compare models
print("\nTODO 11: Model Comparison")
print("Model Comparison:")
print(f"Energy Model - R²: {r2:.3f}, RMSE: {rmse:.2f}")
print(f"Solar Model - R²: {solar_r2:.3f}, RMSE: {np.sqrt(solar_mse):.2f}")

# TODO: Determine which model performs better
if r2 > solar_r2:
    print("The energy consumption model performs better")
else:
    print("The solar panel model performs better")

# ============================================================================
# PART 5: PRACTICAL APPLICATION
# ============================================================================

print("\n" + "=" * 60)
print("PART 5: PRACTICAL APPLICATION")
print("=" * 60)

# TODO 12: Real-world predictions
print("\nTODO 12: Real-world Predictions")

# Building energy scenarios
scenarios = {
    "Cool Day": 18,
    "Mild Day": 24,
    "Warm Day": 30,
    "Hot Day": 36
}

print("Building Energy Consumption Predictions:")
for scenario, temp in scenarios.items():
    # TODO: Make prediction for this temperature
    temp_2d = np.array([[temp]])
    prediction = # TODO: Use model to predict energy consumption
    print(f"{scenario} ({temp}°C): {prediction[0]:.2f} kWh")

# Solar panel scenarios
solar_scenarios = {
    "Cloudy Day": 3,
    "Partly Cloudy": 6,
    "Sunny Day": 9,
    "Very Sunny": 11
}

print("\nSolar Panel Output Predictions:")
for scenario, hours in solar_scenarios.items():
    # TODO: Make prediction for this sunlight duration
    hours_2d = np.array([[hours]])
    prediction = # TODO: Use solar_model to predict output
    print(f"{scenario} ({hours}h sunlight): {prediction[0]:.2f} kWh")

# TODO 13: Manual calculation verification
print("\nTODO 13: Manual Verification")
print("Manual Calculation Check:")

# TODO: Pick a temperature and calculate energy consumption manually
test_temp = 27
manual_energy = # TODO: Calculate using slope * temp + intercept
model_energy = # TODO: Use model.predict() for same temperature

print(f"Temperature: {test_temp}°C")
print(f"Manual calculation: {manual_energy:.2f} kWh")
print(f"Model prediction: {model_energy[0]:.2f} kWh")
print(f"Match: {abs(manual_energy - model_energy[0]) < 0.01}")

# ============================================================================
# PART 6: REFLECTION QUESTIONS
# ============================================================================

print("\n" + "=" * 60)
print("PART 6: REFLECTION QUESTIONS")
print("=" * 60)

print("""
REFLECTION QUESTIONS (Answer in comments or separate file):

1. Data Analysis:
   - Which dataset had a stronger linear relationship? How can you tell?
   - What do the slopes tell you about energy consumption vs solar output?
   - Are there any data points that seem unusual (outliers)?

2. Model Performance:
   - Which model performed better based on R² and RMSE?
   - What does an R² of 0.95 mean in practical terms?
   - How would you explain the residual plots to a non-technical person?

3. Real-world Applications:
   - What factors might affect building energy consumption besides temperature?
   - How might weather patterns affect the accuracy of these models?
   - What other environmental datasets could benefit from linear regression?

4. Model Limitations:
   - What happens if you try to predict energy consumption at 50°C?
   - Why might linear models fail for extreme values?
   - When would you need more complex models (hint: think about next session)?

5. Technical Understanding:
   - Why do we need to reshape arrays for sklearn?
   - What's the difference between MSE and RMSE?
   - How is linear regression related to neural networks?

TO COMPLETE YOUR HOMEWORK:
1. Fill in all TODO sections
2. Run the script and verify all outputs make sense
3. Answer the reflection questions
4. Save your completed script as 'homework_completed.py'
5. Take screenshots of your plots
6. Be prepared to discuss your results in the next session!
""")

print("\n" + "=" * 60)
print("HOMEWORK COMPLETE - GOOD LUCK!")
print("=" * 60)