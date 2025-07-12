# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:55:18 2025

@author: taske
"""

# Neural Network Connection
# Shows how linear regression is similar to a simple neural network

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

# Linear regression IS a neural network:
print("Neural Network Interpretation:")
print("=" * 50)
print("Input Layer: Temperature (1 neuron)")
print("Output Layer: Energy Consumption (1 neuron)")
print(f"Weight: {model.coef_[0]:.3f}")
print(f"Bias: {model.intercept_:.3f}")
print("Activation Function: ReLU")
print("Loss Function: Mean Squared Error")
print("Optimizer: Normal Equation (closed-form solution)")
print()
print("In neural network terms with ReLU activation function:")
print("output = ReLU(weight × input + bias)")
print(f"energy = {model.coef_[0]:.3f} × temperature + {model.intercept_:.3f}")

# Let's implement this as a simple "neural network" function
def simple_neural_network(input_temp, weight, bias):
    """
    Simple neural network with one input, one output, linear activation
    """
    # Forward pass
    output = max(0, weight * input_temp + bias)
    return output

# Test our neural network
print("\nTesting our neural network implementation:")
for temp in [20, 25, 30, 35]:
    nn_result = simple_neural_network(temp, slope, intercept)
    sklearn_result = model.predict([[temp]])[0]
    print(f"Temperature {temp}°C:")
    print(f"  Neural Network: {nn_result:.2f} kWh")
    print(f"  Sklearn: {sklearn_result:.2f} kWh")
    print(f"  Match: {abs(nn_result - sklearn_result) < 0.001}")
    print()
    
    