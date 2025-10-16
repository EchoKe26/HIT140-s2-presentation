import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import statsmodels.api as sm

# Part 1: Seasonal Model Analysis
print("="*40)
print("   PART 1: SEASONAL MODEL ANALYSIS")
print("="*40)

# 1. Data Loading
try:
    df = pd.read_csv("merged_clean_dataset.csv")
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print("Error: 'merged_clean_dataset.csv' not found. Exiting analysis.")
    exit()


# In[44]:


# 2. Define Feature Set 
# target(Y), feature(X)
target = "log_latency"
features = [
    "rat_present_during_landing", 
    "food_availability",          
    "hours_after_sunset",        
    "rat_minutes",
    "rat_arrival_number",
    "reward"                      
]

required_cols = features + [target] + ['season']

# Convert all required columns to a numeric type
for col in ["rat_present_during_landing", "reward", "season"]:
    if col in df.columns:
        df[col] = df[col].astype(int)

for col in ["food_availability", "hours_after_sunset", "rat_minutes", "rat_arrival_number", "log_latency"]:
     if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=required_cols, inplace=True)

# Split dataset by seasons
winter_df = df[df["season"] == 0].copy()
spring_df = df[df["season"] == 1].copy()

print("\n--- Seasonal DataFrame Statistics ---")
print(f"\nWinter DataFrame shape: {winter_df.shape}")
print(f"\nSpring DataFrame shape: {spring_df.shape}")

print("--- Data Preparation Complete ---")
print(f"Number of complete rows for analysis: {len(df)}")
# Check the type in the split datafram
print("\nData types in the final winter_df:")
print(winter_df[features].info())


# 3. Trains a Linear Regression model and evaluates predictive performance

def train_evaluate_lr(df, features, target, season_name, test_size=0.2, random_state=42):
    X = df[features]
    y = df[target]

    if len(X) < 2 or len(y) < 2:
        print(f"Not enough data for {season_name} model. Skipping.")
        return None, np.nan, np.nan, np.nan
    
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = test_size, random_state=random_state)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    y_pred = lr.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== {season_name} LR Predictive Performance ===")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"R²: {r2:.4f}")

    return lr, mae, mse, r2


# 4. Statsmodels OLS Inference Function 

def ols_inference_block(df, features, target, season_name="All"):
    
    X = sm.add_constant(df[features])
    y = df[target]

    try:
        model = sm.OLS(y, X).fit()
    except ValueError as e:
        print(f"Critical ValueError during OLS fitting for {season_name}. Error: {e}")
        return None

    print(f"\n=== OLS Inference Summary({season_name}) ===")
    xnames = features + ['const']
    print(model.summary(xname=xnames))

    # Residual diagnostics
    resid = model.resid
    fitted = model.fittedvalues
    
    #Visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Residuals vs Fitted 
    axes[0].scatter(fitted, resid, alpha=0.5)
    axes[0].axhline(0, color="red", lw=1)
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residuals vs Fitted")

    # Plot 2: QQ Plot 
    # sm.qqplot has a specific 'ax' parameter to tell it which subplot to draw on
    sm.qqplot(resid, line="45", fit=True, ax=axes[1])
    axes[1].set_title("QQ Plot of Residuals")

    fig.suptitle(f"Diagnostic Plots for {season_name} Model", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()
    
    return model
    


# 5. Execution 

print("\n" + "="*50)
print("   PREDICTIVE MODEL EVALUATION")
print("="*50)

# Train Winter and Spring models 
LR_winter, mae_w, mse_w, r2_w = train_evaluate_lr(winter_df, features, target, "Winter")
LR_spring, mae_s, mse_s, r2_s = train_evaluate_lr(spring_df, features, target, "Spring")

# Model Performance Comparison
comparison = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "R²"],
    "Winter": [mae_w, mse_w, np.sqrt(mse_w) if not np.isnan(mse_w) else np.nan, r2_w],
    "Spring": [mae_s, mse_s, np.sqrt(mse_s) if not np.isnan(mse_s) else np.nan, r2_s]
})

print("\nModel Performance Comparison:")
print(comparison.round(4))

# Visualization 
if not any(comparison['Winter'].isna()) and not any(comparison['Spring'].isna()):
    metrics_to_plot = {
        "R²": [r2_w, r2_s],
        "MAE": [mae_w, mae_s],
        "RMSE": [np.sqrt(mse_w), np.sqrt(mse_s)]
    }

    seasons = ["Winter", "Spring"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    for ax, (name, vals) in zip(axes, metrics_to_plot.items()):
        ax.bar(seasons, vals, color=["skyblue", "lightgreen"])
        ax.set_title(name)
        ax.set_ylabel(name)

    plt.suptitle("Model Performance by Seasons")
    plt.show()

print("\n" + "="*50)
print("   RUNNING OLS INFERENCE AND DIAGNOSTICS")
print("="*50)
plt.show()



# OLS Inference for hypothesis testing
ols_winter = ols_inference_block(winter_df, features, target, season_name="Winter")
ols_spring = ols_inference_block(spring_df, features, target, season_name="Spring")


# Part 2: Optimize the Spring Model
print("\n\n" + "="*60)
print("PART 2: OPTIMIZE THE SPRING MODEL")
print("="*60)

# Define Variables for the new model
features_new = [
    'rat_present_during_landing',
    'rat_arrival_number',
    'hours_after_sunset'
]
target = 'log_latency'

model_data_new = spring_df[features_new + [target]].dropna()
X_new = model_data_new[features_new]
y_new = model_data_new[target]

# Split Data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=42)
print(f"\nSpring data split for the new optimized model:")
print(f"Training samples: {len(X_train_new)}")
print(f"Testing samples:  {len(X_test_new)}")

# Establish the Baseline Performance 
baseline_prediction_new = y_train_new.mean()
baseline_preds_new = np.full_like(y_test_new, baseline_prediction_new)
baseline_mae_new = mean_absolute_error(y_test_new, baseline_preds_new)
baseline_rmse_new = np.sqrt(mean_squared_error(y_test_new, baseline_preds_new))

# Build the new model
X_train_new_const = sm.add_constant(X_train_new)
model_new = sm.OLS(y_train_new, X_train_new_const).fit()

# Evaluate the new model's predictive performance on the test set
X_test_new_const = sm.add_constant(X_test_new)
y_pred_new = model_new.predict(X_test_new_const)
mae_new = mean_absolute_error(y_test_new, y_pred_new)
rmse_new = np.sqrt(mean_squared_error(y_test_new, y_pred_new))
r2_new = r2_score(y_test_new, y_pred_new)

print("\n--- New Optimized Model Predictive Performance on Test Data ---")
print(f"New Model MAE:       {mae_new:.4f}")
print(f"Baseline MAE:        {baseline_mae_new:.4f}")
print(f"New Model RMSE:      {rmse_new:.4f}")
print(f"Baseline RMSE:       {baseline_rmse_new:.4f}")
print(f"New Model R-squared: {r2_new:.4f}")



# Print new optimized model OLS Summary
print("\n--- New Optimized Model OLS Summary ---")
print(model_new.summary())



# Visualization
print("--- Generating visualizations for the new optimized model.---")

# Partial Regression plots
fig = plt.figure(figsize=(12, 6))
sm.graphics.plot_partregress_grid(model_new, fig=fig)
fig.suptitle('Partial Regression Plots for Optimized Spring Model', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Diagnostic Plots 
fitted_values_new = model_new.fittedvalues
residuals_new = model_new.resid

plt.figure(figsize=(8, 6))
plt.scatter(fitted_values_new, residuals_new, alpha=0.6)
plt.axhline(0, color='red', linewidth=2)
plt.xlabel('Fitted Values (New Model)')
plt.ylabel('Residuals (New Model)')
plt.title('Residuals vs. Fitted Plot for Optimized Model')
plt.grid(True)
plt.show()

print("\nOptimization analysis for the spring model is complete.")

