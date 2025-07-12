# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:39:35 2025

@author: taske
"""

# Initial Data Setup
# Creates environmental data for temperature vs energy consumption

import numpy as np

# Set random seed for reproducible results
np.random.seed(42)

# Environmental data: Temperature vs Energy Consumption
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))

print("Temperature (°C):", temperature)
print("Energy Consumption (kWh):", energy_consumption)

