# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 12:36:24 2025

@author: taske
"""

# Environment Test Script
# Run this first to verify your Python environment is set up correctly

try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    import sklearn
    
    print("✓ All packages imported successfully!")
    print(f"✓ NumPy version: {np.__version__}")
    print(f"✓ Matplotlib version: {plt.matplotlib.__version__}")
    print(f"✓ Scikit-learn version: {sklearn.__version__}")
    
    # Quick test
    x = np.array([1, 2, 3])
    print(f"✓ NumPy working: {x}")
    
    # Test basic plotting
    plt.figure(figsize=(6, 4))
    plt.plot([1, 2, 3], [1, 4, 9])
    plt.title("Test Plot")
    plt.show()
    print("✓ Matplotlib working!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("We need to install missing packages")
    
    