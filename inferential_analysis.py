import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

sns.set_style("whitegrid")

df = pd.read_csv("merged_clean_dataset.csv")

print("Columns:", df.columns)
print("\nHead of dataset:")
print(df.head())
print("\n","="*50, "\n")

# 1. Log Latency vs Rat Presence
# Q: Do bats delay their landing more when rats are present?
# H0: Bat landing delay is not related to rat presence (True vs False).
# H1: Bat landing delay is related to rat presence (True vs False).

latency_true = df[df["rat_present_during_landing"]==True]["log_latency"].dropna()
latency_false = df[df["rat_present_during_landing"]==False]["log_latency"].dropna()

# Normality test using Z-score & Empirical Rule 
def check_normality_empirical(data, group_name):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    
    within_1 = np.mean(np.abs(z_scores) <= 1) * 100
    within_2 = np.mean(np.abs(z_scores) <= 2) * 100
    within_3 = np.mean(np.abs(z_scores) <= 3) * 100

    print(f"Normality Check (Z-score & Empirical Rule) for {group_name}")
    print(f"Mean = {mean:.4f}, Std = {std:.4f}")
    print(f"Within ±1σ: {within_1:.2f}% (Expected ~68%)")
    print(f"Within ±2σ: {within_2:.2f}% (Expected ~95%)")
    print(f"Within ±3σ: {within_3:.2f}% (Expected ~99.7%)")

    # Visualization
    plt.figure(figsize=(8,6))
    sns.histplot(data, bins=20, kde=True, color="salmon", edgecolor="black")
    plt.axvline(mean, color='red', linestyle='-', linewidth=2, label='Mean')
    for i, color in zip([1, 2, 3], ['green', 'orange', 'purple']):
        plt.axvline(mean + i*std, color=color, linestyle='--', linewidth=1, label=f'+{i}σ')
        plt.axvline(mean - i*std, color=color, linestyle='--', linewidth=1)
    plt.title(f"Distribution of {group_name} with ±1σ, ±2σ, ±3σ")
    plt.xlabel("Log Latency")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


check_normality_empirical(latency_true, "Log Latency (Rat Present = True)")
check_normality_empirical(latency_false, "Log Latency (Rat Present = False)")
print("\n")


# Mann-Whitney U test
u_stat, p_latency_presence = stats.mannwhitneyu(latency_true, latency_false, alternative='two-sided')
print(f"Log Latency - Mann-Whitney U: U={u_stat:.2f}, p={p_latency_presence:.4f}")

alpha = 0.05
if p_latency_presence < alpha:
    print(f"p = {p_latency_presence:.4f} < {alpha}, Reject H0. Evidence suggests bat landing delay is related to rat presence.")
else:
    print(f"p = {p_latency_presence:.4f} > {alpha}, Fail to reject H0. No evidence that bat landing delay is related to rat presence.")
print("\n","="*50, "\n")

# Boxplot visualization
plt.figure(figsize=(8,6))
sns.boxplot(x="rat_present_during_landing", y="log_latency", data=df,
            order=[False, True])
plt.title("Log Latency by Rat Presence")
plt.xlabel("Rat Present")
plt.ylabel("Log Latency to Food")
plt.show()
plt.close()


# 2. Risk-taking behaviour vs Rat Presence
#Q: Are bats less likely to take risk when rats are present?
#H0：Risk-taking behavior is independent of rat presence.
#H1：Bats are more likely to avoid risks when rats are present.
risk_ct = pd.crosstab(df["rat_present_during_landing"], df["risk"])
risk_pct = risk_ct.div(risk_ct.sum(axis=1), axis=0) * 100
print("\nRisk vs Rat Presence (%)")
print(risk_pct)
print('\n')

# Chi-square test
chi2_risk, p_risk, dof, _ = stats.chi2_contingency(risk_ct)
print(f"Risk - Chi-square: χ²={chi2_risk:.2f}, p={p_risk:.4f}, df={dof}")

alpha = 0.05
if p_risk < alpha:
    print("Result: Reject H0. Risk-taking behavior is significantly associated with rat presence.")
else:
    print("Result: Fail to reject H0. No significant association between risk-taking behavior and rat presence.")

print("\n","="*50, "\n")

# Stacked bar chart
risk_pct.plot(kind="bar", stacked=True, figsize=(8,6), color=["skyblue","salmon"])
plt.title("Risk-Taking Behavior by Rat Presence (%)")
plt.ylabel("Percentage")
plt.xlabel("Rat Present")
plt.legend(["Risk-Avoidance (0)", "Risk-Taking (1)"], loc='upper right')
plt.show()
plt.close()


# 3. Log Latency vs Rat Pressure
# Q: Do bats delay their landing more under higher rat pressure?
# H0: Bat landing delay is not related to rat pressure (High vs Low).
# H1: Bat landing delay is related to rat pressure (High vs Low).

# Filter High and Low only
latency_df = df[df["rat_pressure_quartile"].isin(["High", "Low"])].copy()
latency_high = latency_df[latency_df["rat_pressure_quartile"]=="High"]["log_latency"].dropna()
latency_low = latency_df[latency_df["rat_pressure_quartile"]=="Low"]["log_latency"].dropna()

#Normality test
def check_normality_empirical(data, group_name):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    
    within_1 = np.mean(np.abs(z_scores) <= 1) * 100
    within_2 = np.mean(np.abs(z_scores) <= 2) * 100
    within_3 = np.mean(np.abs(z_scores) <= 3) * 100

    print(f"Normality Check (Z-score & Empirical Rule) for {group_name}")
    print(f"Mean = {mean:.4f}, Std = {std:.4f}")
    print(f"Within ±1σ: {within_1:.2f}% (Expected ~68%)")
    print(f"Within ±2σ: {within_2:.2f}% (Expected ~95%)")
    print(f"Within ±3σ: {within_3:.2f}% (Expected ~99.7%)")

    #Visualization
    plt.figure(figsize=(8,6))
    sns.histplot(data, bins=20, kde=True, color="skyblue", edgecolor="black")
    plt.axvline(mean, color='red', linestyle='-', linewidth=2, label='Mean')
    for i, color in zip([1, 2, 3], ['green', 'orange', 'purple']):
        plt.axvline(mean + i*std, color=color, linestyle='--', linewidth=1, label=f'+{i}σ')
        plt.axvline(mean - i*std, color=color, linestyle='--', linewidth=1)
    plt.title(f"Distribution of {group_name} with ±1σ, ±2σ, ±3σ")
    plt.xlabel("Log Latency")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


check_normality_empirical(latency_high, "Log Latency (High Rat Pressure)")
check_normality_empirical(latency_low, "Log Latency (Low Rat Pressure)")
print("\n")


# Mann-Whitney U test
u_stat, p_latency_pressure = stats.mannwhitneyu(latency_high, latency_low, alternative='two-sided')
print(f"Log Latency - Mann-Whitney U: U={u_stat:.2f}, p={p_latency_pressure:.4f}")

if p_latency_pressure < alpha:
    print(f"p = {p_latency_pressure:.4f} < {alpha}, Reject H0. Evidence suggests bat landing delay is related to rat pressure.")
else:
    print(f"p = {p_latency_pressure:.4f} > {alpha}, Fail to reject H0. No evidence that bat landing delay is related to rat pressure.")

print("\n","="*50, "\n")

# Boxplot visualization
plt.figure(figsize=(8,6))
sns.boxplot(x="rat_pressure_quartile", y="log_latency", data=latency_df,
            order=["Low","High"])
plt.title("Log Latency by Rat Pressure (High vs Low)")
plt.xlabel("Rat Pressure Level")
plt.ylabel("Log Latency to Food")
plt.show()
plt.close()


# 4. Risk-taking behaviour vs Rat Pressure
# Q: Are bats less likely to take risks under high rat pressure?
# H0: Risk-taking behavior is independent of rat pressure.
# H1: Bats are more likely to avoid risks under high rat pressure.
risk_df = df[df["rat_pressure_quartile"].isin(["High","Low"])].copy()

risk_ct = pd.crosstab(risk_df["rat_pressure_quartile"], risk_df["risk"])
risk_pct = risk_ct.div(risk_ct.sum(axis=1), axis=0) * 100
print("\nRisk vs Rat Pressure (%)")
print(risk_pct)
print('\n')

# Chi-square test
chi2_risk, p_risk, dof, _ = stats.chi2_contingency(risk_ct)
print(f"Risk - Chi-square: χ²={chi2_risk:.2f}, p={p_risk:.4f}, df={dof}")

alpha = 0.05
if p_risk < alpha:
    print("Result: Reject H0. Risk-taking behavior is significantly associated with rat pressure.")
else:
    print("Result: Fail to reject H0. No significant association between risk-taking behavior and rat pressure.")

print("\n","="*50, "\n")

# Stacked bar chart
risk_pct.plot(kind="bar", stacked=True, figsize=(8,6), color=["skyblue","salmon"])
plt.title("Risk-Taking Behavior by Rat Pressure (High vs Low) (%)")
plt.ylabel("Percentage")
plt.xlabel("Rat Pressure")
plt.legend(["Risk-Avoidance (0)", "Risk-Taking (1)"], loc='upper right')
plt.show()
plt.close()

# For Time Limitation, we didn't present No.3&4 in the video.




