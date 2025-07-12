# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:42:19 2025

@author: taske
"""

# Coding Activity 2: Data Visualization
# Creates scatter plot of temperature vs energy consumption

import numpy as np
import matplotlib.pyplot as plt

# Data setup (copy from 02_data_setup.py if running independently)
np.random.seed(42)
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))

# Create the visualization
plt.figure(figsize=(10, 6))
plt.scatter(temperature, energy_consumption, color='blue', alpha=0.7, s=50)
plt.xlabel('Temperature (°C)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Building Energy Consumption vs Temperature')
plt.grid(True, alpha=0.3)

# Add some styling
plt.style.use('default')  # Ensures consistent appearance across systems
plt.tight_layout()
plt.show()

