import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

print("="*40)
print("REVISIT: INVESTIGATION A")
print("="*40)

# 1. Load Dataset
df = pd.read_csv("merged_clean_dataset.csv")

# Convert boolean variable 'rat_present_during_landing'converted to integer (0/1)
df['rat_present_during_landing'] = df['rat_present_during_landing'].astype(int)


# 2. Define Variables 
features = [
    'rat_present_during_landing', 
    'rat_arrival_number',         
    'rat_minutes'                
]

target = 'log_latency'

model_data = df[features + [target]].dropna()

X = model_data[features]
y = model_data[target]

# 3. Evaluate The Model’s Performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")


X_train_const = sm.add_constant(X_train)

# Fit the Ordinary Least Squares (OLS) model using the training data.
model = sm.OLS(y_train, X_train_const).fit()


# Evaluate Model's Predictive Performance on the Test Set ---
X_test_const = sm.add_constant(X_test)

# Use the trained model to make predictions on the unseen test data.
y_pred = model.predict(X_test_const)

# Calculate the performance metrics.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Create a baseline prediction
baseline_pred = np.full_like(y_test, y_train.mean())

# Calculate baseline MAE and RMSE
mae_baseline = mean_absolute_error(y_test, baseline_pred)
rmse_baseline = np.sqrt(mean_squared_error(y_test, baseline_pred))

print("\n=== Model Predictive Performance on Test Data ===")
print(f"Mean Absolute Error (MAE):      {mae:.4f}")
print(f"Baseline MAE:                   {mae_baseline:.4f}")
print(f"Mean Squared Error (MSE):       {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Baseline RMSE:                  {rmse_baseline:.4f}")
print(f"R2:                             {r2:.4f}")


# 4. Build and Fit the OLS Model 

# Calculate the constant(intercept) of the regression line
X_const = sm.add_constant(X)
final_model = sm.OLS(y, X_const).fit()

# Summary for Evaluation 
print("\n=== OLS Model Summary (Revisiting InvestigationA) ===")
print(final_model.summary())


# 5. Build, Evaluate, and Analyze the Optimized Model (V2) 
print("="*60)
print("\n   Building and analyzing the optimized model(New Model) ")
print("="*60)


features_new = [
    'rat_minutes'
]
target = 'log_latency'

# Create the new feature sets using the same data split
X_train_new = X_train[features_new]
X_test_new= X_test[features_new]

# New Model Performance Evaluation
X_train_new_const = sm.add_constant(X_train_new)
model_new = sm.OLS(y_train, X_train_new_const).fit()

X_test_new_const = sm.add_constant(X_test_new)
y_pred_new = model_new.predict(X_test_new_const)

# Calculate new models performance metrics
mae_new = mean_absolute_error(y_test, y_pred_new)
rmse_new = np.sqrt(mean_squared_error(y_test, y_pred_new))
r2_new = r2_score(y_test, y_pred_new)

print("\n=== New Model Predictive Performance on Test Data ===")
print(f"MAE(new):      {mae_new:.4f}")
print(f"Baseline MAE:  {mae_baseline:.4f}")
print(f"RMSE(new):     {rmse_new:.4f}")
print(f"Baseline RMSE: {rmse_baseline:.4f}")
print(f"R2(new):       {r2_new:.4f}")

# New Model OLS Summary for Inference
print("\n=== New Model OLS Summary ===")
print(model_new.summary())

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X_test_new['rat_minutes'], y_test, alpha=0.6, label='Actual Data')
plt.plot(X_test_new['rat_minutes'], y_pred_new, color='red', linewidth=2, label='OLS Regression Line')
plt.title('Optimized Model: log_latency vs. rat_minutes')
plt.xlabel('rat_minutes')
plt.ylabel('log_latency')
plt.legend()
plt.grid(True)
plt.show()

fitted_values_new = model_new.fittedvalues
residuals_new = model_new.resid

# Plot 1: Residuals vs. Fitted Plot for New Model
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values_new, residuals_new, alpha=0.6)
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Fitted Values (New Model)')
plt.ylabel('Residuals (New Model)')
plt.title('Residuals vs. Fitted Plot for Optimized Model (New)')
plt.grid(True)
plt.show()

# Plot 2: Q-Q Plot of Residuals for New Model
fig = sm.qqplot(residuals_new, line='45', fit=True)
plt.title('Normal Q-Q Plot of Residuals for Optimized Model (New)')
plt.show()





