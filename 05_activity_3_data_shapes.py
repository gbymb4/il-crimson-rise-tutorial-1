# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:43:13 2025

@author: taske
"""

# Coding Activity 3: Understanding Data Shapes
# Explores why we need to reshape data for sklearn

import numpy as np

# Data setup (copy from 02_data_setup.py if running independently)
np.random.seed(42)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))

# Prepare data for sklearn
X = temperature.reshape(-1, 1)  # Features must be 2D for sklearn
y = energy_consumption            # Target can be 1D

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")

# Why do we need to reshape? Let's explore
print("\nOriginal temperature shape:", temperature.shape)
print("Original temperature:", temperature)
print()
print("Reshaped X shape:", X.shape)
print("Reshaped X:")
print(X)

# Let's understand what .reshape(-1, 1) does:
# -1 means "figure out this dimension automatically"
# 1 means "make the second dimension size 1"

# Try this to see what happens:
print("\nWhat if we reshape to (-1, 2)?")
try:
    bad_reshape = temperature.reshape(-1, 2)
    print("Success:", bad_reshape.shape)
except ValueError as e:
    print("Error:", e)
    print("This fails because we can't fit 10 elements into 2 columns evenly")
    
    