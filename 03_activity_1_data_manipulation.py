# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:40:50 2025

@author: taske
"""

# Coding Activity 1: Basic Data Manipulation
# Run 02_data_setup.py first, or include the data setup here

import numpy as np

# Data setup (copy from 02_data_setup.py if running independently)
np.random.seed(42)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))

# Task 1: Calculate the average temperature
# Method 1: Using sum() and len()
avg_temp = sum(temperature) / len(temperature)
print(f"Average temperature: {avg_temp}°C")

# Method 2: Using NumPy (more common in data science)
avg_temp_np = np.mean(temperature)
print(f"Average temperature (NumPy): {avg_temp_np}°C")

# Task 2: Find the maximum energy consumption
max_energy = max(energy_consumption)  # or np. max(energy_consumption)
print(f"Maximum energy consumption: {max_energy:.2f} kWh")

# Task 3: Create a list of temperatures above 30°C
hot_days = [temp for temp in temperature if temp > 30]  # List comprehension
# Alternative: hot_days = temperature[temperature > 30]  # NumPy boolean indexing
print(f"Hot days (>30°C): {hot_days}")

# Task 4: Calculate the range of energy consumption
energy_range = max(energy_consumption) - min(energy_consumption)
# Alternative: energy_range = np.max(energy_consumption) - np.min(energy_consumption)
print(f"Energy consumption range: {energy_range:.2f} kWh")

