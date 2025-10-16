import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

print("="*40)
print("INVESTIGATION B: INFERENTIAL ANALYSIS")
print("="*40)

# Data Loading and Preprocessing 
try:
    # Load the dataset
    df = pd.read_csv('merged_clean_dataset.csv')
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"Error loading data: {e}")
    import sys
    sys.exit(1)

# Create descriptive labels for categorical variables 
df['season_label'] = df['season'].map({0: 'Winter', 1: 'Spring'})
df['rat_present'] = pd.to_numeric(df['rat_present_during_landing'], errors='coerce').astype('Int64')
df['rat_label'] = df['rat_present'].map({0:'RP=0 (Absent)', 1:'RP=1 (Present)'})
df['risk_taking'] = df['risk'].astype(int)


# Create a clean dataframe 
print("\nChecking for Missing Values Before Dropping ")
print(df[['season_label', 'rat_label', 'log_latency', 'risk_taking']].isnull().sum())
df_clean = df.dropna(subset=['season_label', 'rat_label', 'log_latency', 'risk_taking'])
print(f"\nNumber of rows remaining after .dropna(): {len(df_clean)}")

print("--- Data Preprocessing Complete ---")
print(f"Total samples for analysis: {len(df_clean)}")
print("Sample distribution by season:")
print(df_clean['season_label'].value_counts())


# 1. Analysis: Rat Encounter Proportion by Season
print("\n1. Analysis: Rat Encounter Proportion by Season")

# Perform Chi-square test 
contingency_rp_season = pd.crosstab(df_clean['season_label'], df_clean['rat_present_during_landing'])
chi2_stat, p_value, _, _ = chi2_contingency(contingency_rp_season)

# Calculate and print proportions for each season
prop_by_season_rp = df_clean.groupby('season_label')['rat_present_during_landing'].mean()
print(f"Winter Rat Encounter Proportion: {prop_by_season_rp['Winter']:.2%}")
print(f"Spring Rat Encounter Proportion: {prop_by_season_rp['Spring']:.2%}")
print(f"Chi-square Test P-value: {p_value:.4f} {'(Significant)' if p_value < 0.05 else '(Not Significant)'}")

# Visualize the proportions
plt.figure(figsize=(5, 4))
sns.barplot(data=df_clean, x='season_label', y='rat_present_during_landing', errorbar=('ci', 95))
plt.title('Rat Encounter Proportion by Season')
plt.xlabel("Season")
plt.ylabel("Proportion of Rat Encounters")
plt.tight_layout()
plt.show()


# 2. Cross-Analysis: Log Latency vs. Rat Presence by Season
print("\n2. Cross-Analysis: Log Latency(Vigilance) vs. Rat Presence by Season")

# Display descriptive statistics 
desc_stats_latency = df_clean.groupby(['season_label', 'rat_label'])['log_latency'].agg(['mean', 'median', 'std', 'count'])
print("\nDescriptive Statistics for Log Latency:")
print(desc_stats_latency)

# Perform stratified Mann-Whitney U tests 
print("\nStatistical Tests(Mann-Whitney U tests) for Log Latency(Vigilance):")
winter_absent = df_clean[(df_clean['season_label'] == 'Winter') & (df_clean['rat_label'] == 'RP=0 (Absent)')]['log_latency']
winter_present = df_clean[(df_clean['season_label'] == 'Winter') & (df_clean['rat_label'] == 'RP=1 (Present)')]['log_latency']
u_stat_winter, p_value_winter = mannwhitneyu(winter_absent, winter_present, alternative='two-sided')
print(f"Winter: P-value (Latency vs Rat Presence) = {p_value_winter:.4f} {'(Significant)' if p_value_winter < 0.05 else '(Not Significant)'}")

spring_absent = df_clean[(df_clean['season_label'] == 'Spring') & (df_clean['rat_label'] == 'RP=0 (Absent)')]['log_latency']
spring_present = df_clean[(df_clean['season_label'] == 'Spring') & (df_clean['rat_label'] == 'RP=1 (Present)')]['log_latency']
u_stat_spring, p_value_spring = mannwhitneyu(spring_absent, spring_present, alternative='two-sided')
print(f"Spring: P-value (Latency vs Rat Presence) = {p_value_spring:.4f} {'(Significant)' if p_value_spring < 0.05 else '(Not Significant)'}")

# Visualization (Group Boxplot)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='season_label', y='log_latency', hue='rat_label', palette='pastel')
plt.title('Log Latency(Vigilance) vs. Rat Presence by Season')
plt.ylabel('Log Latency(Vigilance)')
plt.xlabel('Season')
plt.legend(title='Rat Presence')
plt.tight_layout()
plt.show()


# 3. Cross-Analysis: Risk-Taking vs. Rat Presence by Season
print("\n3. Cross-Analysis: Risk-Taking vs. Rat Presence by Season")

# Display proportions of risk-taking behavior for each of the four groups
prop_risk = df_clean.groupby(['season_label', 'rat_label'])['risk_taking'].mean().unstack()
print("\nProportion of Risk-Taking Behavior:")
print(prop_risk)

# Perform stratified Chi-square test to check for significance within each season
print("\nStatistical Tests(Chi-square test) for Risk-Taking:")
winter_df = df_clean[df_clean['season_label'] == 'Winter']
contingency_winter = pd.crosstab(winter_df['risk_taking'], winter_df['rat_present_during_landing'])
chi2, p_value_winter_risk, _, _ = chi2_contingency(contingency_winter)
print(f"Winter: P-value (Risk-Taking vs Rat Presence) = {p_value_winter_risk:.4f} {'(Significant)' if p_value_winter_risk < 0.05 else '(Not Significant)'}")

spring_df = df_clean[df_clean['season_label'] == 'Spring']
contingency_spring = pd.crosstab(spring_df['risk_taking'], spring_df['rat_present_during_landing'])
chi2, p_value_spring_risk, _, _ = chi2_contingency(contingency_spring)
print(f"Spring: P-value (Risk-Taking vs Rat Presence) = {p_value_spring_risk:.4f} {'(Significant)' if p_value_spring_risk < 0.05 else '(Not Significant)'}")


# Visualization (Group Barplot)
plt.figure(figsize=(8, 6))
sns.barplot(data=df_clean, x='season_label', y='risk_taking', hue='rat_label', palette='pastel', errorbar=('ci', 95))
plt.title('Risk-Taking Proportion vs. Rat Presence by Season')
plt.ylabel('Proportion of Risk-Taking')
plt.xlabel('Season')
plt.legend(title='Rat Presence')
plt.tight_layout()
plt.show()

print("\n--- Inferential Analysis Complete ---")




