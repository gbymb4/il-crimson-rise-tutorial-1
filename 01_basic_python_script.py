# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:37:26 2025

@author: taske
"""

# Coding Activity 0: Basic Python Test
# This tests basic Python skills and confirms environment

print("Hello, Environmental AI!")

# Simple calculation
temperature_celsius = 25
temperature_fahrenheit = temperature_celsius * 9/5 + 32
print(f"Temperature: {temperature_celsius}°C = {temperature_fahrenheit}°F")

# Basic list operation
temperatures = [20, 25, 30, 35]
print(f"Average temperature: {sum(temperatures)/len(temperatures)}°C")

# Test conda environment
import sys
print(f"Python path: {sys.executable}")

# Test script path
import os
print(f"Script path: {os.getcwd()}")

