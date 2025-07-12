# Introduction to Machine Learning: Environmental Data Regression - Session 1

## Session Overview
**Duration**: 1 hour  
**Goal**: Assess student's Python proficiency while introducing core ML concepts through environmental data regression  
**Focus**: Linear regression with scikit-learn as foundation for neural networks

## Session Timeline

| Time      | Activity                                    |
| --------- | ------------------------------------------- |
| 0:00 - 0:15 | 1. Python Setup & Environment Check        |
| 0:15 - 0:25 | 2. Introduction & Problem Setup            |
| 0:25 - 0:40 | 3. Data Exploration & Python Assessment    |
| 0:40 - 0:55 | 4. Linear Regression Implementation        |
| 0:55 - 1:00 | 5. Results Analysis & Next Steps           |

---

## 1. Python Setup & Environment Check (15 minutes)

### Pre-Session Setup Check
**Before we start coding, let's make sure your Python environment is ready:**

**Check Python Installation:**
```bash
# Option 1: Check if Anaconda is installed
conda --version

# Option 2: Check Python version
python --version

# If you're on macOS/Linux and the above doesn't work, try:
python3 --version
```

**Expected output**: 
- Anaconda version 4.0+ (preferred)
- Python 3.8 or higher

### Installing Anaconda (if not already installed)
**If you don't have Anaconda installed:**

**Windows:**
- Download from: https://www.anaconda.com/products/distribution
- Run the installer and follow the setup wizard
- Choose "Add Anaconda to PATH" during installation

**macOS:**
- Download from: https://www.anaconda.com/products/distribution
- Run the .pkg installer
- Restart your terminal after installation

**Linux:**
- Download the .sh file from: https://www.anaconda.com/products/distribution
- Run: `bash Anaconda3-[version]-Linux-x86_64.sh`
- Follow the prompts and restart your terminal

### Creating a Conda Environment
**Let's create a dedicated environment for our ML project:**

```bash
# Create a new conda environment
conda create -n ml_env python=3.9

# Activate the environment
# Windows:
conda activate ml_env

# macOS/Linux:
conda activate ml_env
# (or sometimes: source activate ml_env)
```

**Why use environments?** Keeps our ML packages separate from other projects and prevents conflicts.

### Installing Required Packages with Conda
**Install packages using conda (preferred) or pip as backup:**

```bash
# Method 1: Install with conda (recommended)
conda install numpy matplotlib scikit-learn

# Method 2: If conda package isn't available, use pip
pip install numpy matplotlib scikit-learn

# Method 3: Install specific versions if needed
conda install numpy=1.21 matplotlib=3.5 scikit-learn=1.0
```

### **Environment Test Activity** (5 minutes)
**Let's verify everything works by running this test:**

```python
# Test script - run this first
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
```

**Assessment notes**: *This immediately reveals their Python setup experience and comfort with package management*

### Development Environment Setup
**Let's decide how we'll write code today:**

**Option 1: Anaconda Spyder (Recommended for beginners)**
```bash
# Launch Spyder IDE
spyder

# Or if not in PATH:
# Windows: Look for Spyder in Start Menu
# macOS: Look for Spyder in Applications
# Linux: Try 'anaconda-navigator' then launch Spyder
```

**Option 2: Jupyter Notebook (Great for learning)**
```bash
# Launch Jupyter Notebook
jupyter notebook

# Or launch through Anaconda Navigator
anaconda-navigator
```

**Option 3: Python Script (.py file)**
```bash
# Create a file called 'energy_prediction.py'
# Run from command line/terminal:

# Windows (Command Prompt):
python energy_prediction.py

# Windows (PowerShell):
python .\energy_prediction.py

# macOS/Linux:
python energy_prediction.py
# or sometimes:
python3 energy_prediction.py
```

**Option 4: Interactive Python (REPL)**
```bash
# In terminal/command prompt:
python

# Or use IPython for better interactive experience:
ipython
```

**Question for student**: "Which environment are you most comfortable with? Have you used any of these before?"

### **Coding Activity 0: Basic Python Test** (3 minutes)
```python
# Let's test basic Python skills
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
```

**Assessment notes**: *Observe comfort with basic syntax, variables, and print statements*

---

## 2. Introduction & Problem Setup (10 minutes)

### Student Background & Motivation
**Opening questions:**
- "Tell me about your interest in environmental AI applications"
- "What programming experience do you have with Python?"
- "Have you worked with any data analysis before?"
- "Are you familiar with Anaconda or other package managers?"

### Problem Statement
**Real-world scenario**: Building energy optimization for carbon footprint reduction
- **Goal**: Predict building energy consumption based on outside temperature
- **Why this matters**: HVAC systems account for 40% of building energy use
- **AI application**: Smart building management systems

### Mathematical Foundation
**Linear relationships in environmental systems:**
- As temperature increases, energy consumption typically increases (cooling)
- We can model this as: `Energy = slope × Temperature + intercept`
- This is exactly what linear regression finds automatically

---

## 3. Data Exploration & Python Assessment (15 minutes)

### Initial Data Setup
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set random seed for reproducible results
np.random.seed(42)

# Environmental data: Temperature vs Energy Consumption
temperature = np.array([15, 18, 22, 25, 28, 30, 32, 35, 38, 40])
energy_consumption = 50 + 2.5 * temperature + np.random.normal(0, 5, len(temperature))

print("Temperature (°C):", temperature)
print("Energy Consumption (kWh):", energy_consumption)
```

### **Coding Activity 1: Basic Data Manipulation** (5 minutes)
**Instructions**: "Let's start with some basic data analysis. I'll guide you through the first one:"

```python
# Task 1: Calculate the average temperature
# Method 1: Using sum() and len()
avg_temp = sum(temperature) / len(temperature)
print(f"Average temperature: {avg_temp}°C")

# Method 2: Using NumPy (more common in data science)
avg_temp_np = np.mean(temperature)
print(f"Average temperature (NumPy): {avg_temp_np}°C")

# Now you try the rest:
# Task 2: Find the maximum energy consumption
max_energy = # Student implements (hint: use max() or np.max())

# Task 3: Create a list of temperatures above 30°C
hot_days = # Student implements (hint: use list comprehension or np.where())

# Task 4: Calculate the range of energy consumption
energy_range = # Student implements (hint: max - min)

print(f"Maximum energy consumption: {max_energy} kWh")
print(f"Hot days (>30°C): {hot_days}")
print(f"Energy consumption range: {energy_range} kWh")
```

**Assessment notes**: *Observe comfort with basic Python operations vs NumPy functions*

### Data Visualization
```python
# Let's visualize our data
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
```

### **Coding Activity 2: Data Interpretation** (3 minutes)
**Question**: "Looking at this scatter plot, what relationship do you see between temperature and energy consumption?"

**Follow-up**: "If I asked you to draw a line through these points, how would you do it?"

---

## 4. Linear Regression Implementation (15 minutes)

### Step 1: Data Preparation
```python
# Prepare data for sklearn
X = temperature.reshape(-1, 1)  # Features must be 2D for sklearn
y = energy_consumption            # Target can be 1D

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
```

### **Coding Activity 3: Understanding Data Shapes** (3 minutes)
```python
# Why do we need to reshape? Let's explore
print("Original temperature shape:", temperature.shape)
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
```

**Explanation**: "Scikit-learn expects 2D arrays for features (rows = samples, columns = features). Even with one feature, we need a 2D array."

**Assessment notes**: *Check understanding of array operations and sklearn requirements*

### Step 2: Model Creation and Training
```python
# Create the model
model = LinearRegression()

# Train the model (this is where the "learning" happens)
model.fit(X, y)

# The model has now learned the best slope and intercept
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Model equation: Energy = {model.coef_[0]:.2f} × Temperature + {model.intercept_:.2f}")
```

### **Coding Activity 4: Model Interpretation** (4 minutes)
```python
# Let's understand what the model learned
slope = model.coef_[0]
intercept = model.intercept_

print(f"Model equation: Energy = {slope:.2f} × Temperature + {intercept:.2f}")
print()

# Your turn: What does this equation tell us?
print("Model interpretation:")
print(f"1. If temperature increases by 1°C, energy consumption increases by {slope:.2f} kWh")
print(f"2. At 0°C, the model predicts {intercept:.2f} kWh energy consumption")
print(f"3. The baseline energy consumption (intercept) is {intercept:.2f} kWh")

# Let's test our understanding
temp_change = 5  # 5°C increase
energy_change = slope * temp_change
print(f"\nIf temperature increases by {temp_change}°C:")
print(f"Energy consumption increases by {energy_change:.2f} kWh")
```

**Follow-up questions**:
- "Does this slope make sense for building energy consumption?"
- "What might the intercept represent in real-world terms?"
- "Why might energy consumption not be zero at 0°C?"

### Step 3: Making Predictions
```python
# Make predictions on our training data
predictions = model.predict(X)

# Let's predict for a new temperature
new_temp = np.array([[27]])  # 27°C (note: must be 2D)
predicted_energy = model.predict(new_temp)
print(f"Predicted energy consumption at 27°C: {predicted_energy[0]:.2f} kWh")

# Predict for multiple temperatures
test_temps = np.array([[20], [25], [30], [35]])
test_predictions = model.predict(test_temps)
print(f"\nPredictions for multiple temperatures:")
for temp, pred in zip(test_temps.flatten(), test_predictions):
    print(f"  {temp}°C: {pred:.2f} kWh")
```

### **Coding Activity 5: Manual Predictions** (3 minutes)
```python
# Let's verify our understanding by calculating manually
def manual_prediction(temp, slope, intercept):
    """Calculate energy consumption using the linear equation"""
    return slope * temp + intercept

# Test it
test_temp = 27
manual_result = manual_prediction(test_temp, slope, intercept)
sklearn_result = model.predict([[test_temp]])[0]

print(f"Manual calculation for {test_temp}°C: {manual_result:.2f} kWh")
print(f"Sklearn prediction for {test_temp}°C: {sklearn_result:.2f} kWh")
print(f"Difference: {abs(manual_result - sklearn_result):.6f} kWh")
print(f"Match: {abs(manual_result - sklearn_result) < 0.01}")

# Your turn: Try a few more temperatures
for temp in [22, 33, 38]:
    manual = manual_prediction(temp, slope, intercept)
    sklearn = model.predict([[temp]])[0]
    print(f"{temp}°C - Manual: {manual:.2f}, Sklearn: {sklearn:.2f}")
```

### Step 4: Model Evaluation
```python
# How good is our model?
mse = mean_squared_error(y, predictions)
r2 = r2_score(y, predictions)

print(f"Model Performance:")
print(f"Mean Squared Error: {mse:.2f} kWh²")
print(f"Root Mean Squared Error: {np.sqrt(mse):.2f} kWh")
print(f"R² Score: {r2:.3f}")
print(f"Model explains {r2*100:.1f}% of the variance in energy consumption")
```

---

## 5. Results Analysis & Next Steps (5 minutes)

### Visualization with Model
```python
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
```

### **Coding Activity 6: Model Evaluation** (5 minutes)
```python
# Let's understand model performance
print(f"Model Performance Analysis:")
print(f"R² Score: {r2:.3f}")
print()

# Your interpretation:
if r2 > 0.8:
    print("✓ Excellent fit - model explains most variance")
elif r2 > 0.6:
    print("✓ Good fit - model captures main relationship")
elif r2 > 0.4:
    print("⚠ Moderate fit - some predictive power")
else:
    print("⚠ Poor fit - limited predictive power")

# Calculate residuals (errors)
residuals = y - predictions
print(f"\nError Analysis:")
print(f"Largest prediction error: {max(abs(residuals)):.2f} kWh")
print(f"Average absolute error: {np.mean(abs(residuals)):.2f} kWh")
print(f"Standard deviation of errors: {np.std(residuals):.2f} kWh")

# Find best and worst predictions
best_idx = np.argmin(abs(residuals))
worst_idx = np.argmax(abs(residuals))
print(f"\nBest prediction: {temperature[best_idx]}°C (error: {residuals[best_idx]:.2f} kWh)")
print(f"Worst prediction: {temperature[worst_idx]}°C (error: {residuals[worst_idx]:.2f} kWh)")
```

### Bridge to Neural Networks
**Key insight**: "What we just did is actually a simple neural network!"

```python
# Linear regression IS a neural network:
print("Neural Network Interpretation:")
print("=" * 50)
print("Input Layer: Temperature (1 neuron)")
print("Output Layer: Energy Consumption (1 neuron)")
print(f"Weight: {model.coef_[0]:.3f}")
print(f"Bias: {model.intercept_:.3f}")
print("Activation Function: Linear (identity)")
print("Loss Function: Mean Squared Error")
print("Optimizer: Normal Equation (closed-form solution)")
print()
print("In neural network terms:")
print("output = weight × input + bias")
print(f"energy = {model.coef_[0]:.3f} × temperature + {model.intercept_:.3f}")
```

### **Final Coding Activity: Neural Network Connection** (3 minutes)
```python
# Let's implement this as a simple "neural network" function
def simple_neural_network(input_temp, weight, bias):
    """
    Simple neural network with one input, one output, linear activation
    """
    # Forward pass
    output = weight * input_temp + bias
    return output

# Test our neural network
for temp in [20, 25, 30, 35]:
    nn_result = simple_neural_network(temp, slope, intercept)
    sklearn_result = model.predict([[temp]])[0]
    print(f"Temperature {temp}°C:")
    print(f"  Neural Network: {nn_result:.2f} kWh")
    print(f"  Sklearn: {sklearn_result:.2f} kWh")
    print(f"  Match: {abs(nn_result - sklearn_result) < 0.001}")
    print()
```

### Real-world Applications & Next Steps
**Environmental AI applications using this foundation:**
- Smart building management systems
- Solar panel output optimization
- Carbon footprint prediction models
- Energy grid demand forecasting
- HVAC system optimization
- Smart thermostat algorithms

**Preview of next session:**
- Multiple input variables (temperature, humidity, occupancy, time of day)
- Non-linear relationships requiring more complex models
- Introduction to PyTorch for neural networks
- Feature engineering for environmental data

---

## Troubleshooting Common Setup Issues

### Anaconda Installation Issues
**Problem**: "conda: command not found"
**Solutions**:
- **Windows**: Restart Command Prompt/PowerShell after installation
- **macOS**: Add to PATH: `export PATH="/opt/anaconda3/bin:$PATH"`
- **Linux**: Add to bashrc: `echo 'export PATH="/home/username/anaconda3/bin:$PATH"' >> ~/.bashrc`

### Environment Activation Issues
**Problem**: Environment activation doesn't work
**Solutions**:
```bash
# Try these alternatives:
conda activate ml_env
source activate ml_env
conda env list  # Check if environment exists

# On Windows Git Bash:
winpty conda activate ml_env

# Reset conda:
conda init
# Then restart terminal
```

### Package Installation Issues
**Problem**: "PackageNotFoundError"
**Solutions**:
```bash
# Update conda first:
conda update conda

# Try conda-forge channel:
conda install -c conda-forge scikit-learn

# Use pip as fallback:
pip install scikit-learn

# Check what's installed:
conda list
pip list
```

### Jupyter/Spyder Launch Issues
**Problem**: Jupyter or Spyder won't start
**Solutions**:
```bash
# Try launching through navigator:
anaconda-navigator

# Or install/reinstall:
conda install jupyter spyder

# Check if environment is activated:
conda info --envs
```

### Matplotlib Display Issues
**Problem**: Plots not showing
**Solutions**:
```python
# Try different backends:
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg' on some systems
import matplotlib.pyplot as plt

# Force display:
plt.show(block=True)

# For Jupyter, add this magic command:
%matplotlib inline
```

### Python Version Conflicts
**Problem**: Wrong Python version being used
**Solutions**:
```bash
# Check which Python:
which python
which python3

# In conda environment:
conda info
python --version

# Create environment with specific Python:
conda create -n ml_env python=3.9
```

---

## Assessment Rubric for Tutor

### Python Proficiency Indicators

**Strong Python Skills:**
- Quickly adapts to conda commands and environment management
- Completes coding activities with minimal guidance
- Understands array operations and reshaping intuitively
- Suggests code improvements or alternative approaches
- Comfortable with function definitions and imports

**Moderate Python Skills:**
- Follows conda setup with some guidance
- Completes most activities with minor help
- Understands basic operations but needs help with sklearn syntax
- Can modify existing code successfully
- Shows good problem-solving approach

**Developing Python Skills:**
- Needs significant help with conda setup
- Struggles with basic syntax or operations
- Requires step-by-step guidance for coding activities
- Focuses more on understanding concepts than implementation
- May benefit from additional Python fundamentals

### Environment Management Skills

**Experienced with Package Management:**
- Familiar with conda or similar tools
- Understands virtual environments
- Can troubleshoot installation issues
- Knows when to use conda vs pip

**New to Package Management:**
- Needs explanation of environments and package management
- Follows instructions well but needs guidance
- Benefits from understanding why we use these tools
- Good candidate for learning best practices

### Mathematical Understanding Indicators

**Strong Mathematical Foundation:**
- Immediately grasps slope/intercept interpretation
- Connects linear equation to real-world meaning
- Understands evaluation metrics (R², MSE) intuitively
- Makes connections to broader mathematical concepts
- Asks insightful questions about model assumptions

**Solid Mathematical Foundation:**
- Understands basic linear relationships
- Can interpret model outputs with minimal guidance
- Grasps prediction concept well
- Shows interest in mathematical details
- Benefits from visual explanations

**Developing Mathematical Foundation:**
- Needs help interpreting mathematical concepts
- Focuses on implementation over theory
- Requires explanation of statistical concepts
- Benefits strongly from visual explanations
- May need more mathematical context in future sessions

---

## Homework Assignment

### Setup Task (Essential for next session)
1. **Environment Setup**: Ensure your conda environment is working
   ```bash
   conda activate ml_env
   python -c "import numpy, matplotlib, sklearn; print('All packages working!')"
   ```

2. **Save Your Work**: Create a file called `energy_prediction.py` with today's complete code

3. **Test Your Setup**: Run the complete script and save a screenshot of the final plot

### Core Programming Task
```python
# Extend the energy consumption model
# Add these new data points:
new_temps = np.array([16, 19, 24, 29, 36])
new_energy = np.array([92, 98, 108, 125, 142])

# Tasks:
# 1. Combine with existing data using np.concatenate()
# 2. Retrain the model on the combined dataset
# 3. Compare old vs new model performance (R², MSE)
# 4. Make predictions for: 20°C, 26°C, 33°C
# 5. Create a visualization showing both datasets and both models

# Bonus: Try predicting energy consumption for extreme temperatures
# (e.g., 5°C, 45°C) and discuss whether the predictions make sense
```

### Advanced Challenge (Optional)
```python
# Research task: Find real building energy consumption data
# Suggestions:
# - Search for "building energy consumption dataset"
# - Look for government energy data
# - Try datasets from Kaggle or UCI ML repository
# 
# If you find a dataset:
# 1. Load it into Python
# 2. Identify temperature and energy consumption columns
# 3. Apply our linear regression approach
# 4. Compare results with our synthetic data
```

### Reflection Questions
1. How did adding new data affect the model's slope and intercept?
2. Which model performed better? How can you tell?
3. What other environmental factors might affect energy consumption?
4. What limitations do you see with this linear approach?
5. How might weather patterns in different climates affect this model?

### Preparation for Next Session
- **Bring your laptop with the conda environment ready**
- **Think of 2-3 environmental AI applications you'd like to explore**
- **Consider what makes relationships in environmental data non-linear**
- **If you get stuck with setup, we'll troubleshoot together at the start of next session**

### Common Setup Issues to Try First
If you encounter problems:
1. Try restarting your terminal/command prompt
2. Make sure you're in the correct conda environment
3. Try `conda update conda` if packages won't install
4. Document any error messages to discuss in the next session

---

## Key Takeaways

This session establishes that:
1. **Machine learning is about finding patterns in data automatically**
2. **Linear regression is the simplest neural network (one layer, linear activation)**
3. **The "learning" happens when we optimize parameters (weights and biases)**
4. **Environmental applications provide meaningful, interpretable context for AI techniques**
5. **Proper environment management is crucial for reproducible data science**

The student should leave understanding that we've built a solid foundation using industry-standard tools (Anaconda, scikit-learn) that will naturally extend to more complex neural networks and real-world environmental datasets in future sessions.

### Next Session Preview
- **Multi-feature regression** (temperature, humidity, occupancy, time)
- **Non-linear relationships** requiring polynomial features or neural networks
- **Introduction to PyTorch** for deep learning
- **Real-world environmental datasets**
- **Feature engineering** for environmental data

The progression from simple linear regression to neural networks will feel natural and logical, with each concept building on the previous one.