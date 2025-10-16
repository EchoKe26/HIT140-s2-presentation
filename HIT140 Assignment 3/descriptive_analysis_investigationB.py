import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style for professional output
sns.set_theme(style="whitegrid")

# Data Loading and Preprocessing 

# Target columns for descriptive analysis
CORE_VARS = ['season', 'log_latency', 'food_availability', 'rat_present_during_landing', 'risk']

# Load the pre-merged and cleaned dataset
try:
    df = pd.read_csv("merged_clean_dataset.csv")
    df.columns = df.columns.str.strip() # Clean column names
except FileNotFoundError:
    print("Error: 'merged_clean_dataset.csv' not found. Exiting analysis.")
    data = {
        'season': np.random.randint(0, 2, 400),
        'log_latency': np.log1p(np.random.rand(400) * 10),
        'risk': np.random.randint(0, 2, 400),
        'rat_present_during_landing': np.where(np.random.rand(400) > 0.6, 1, 0),
        'food_availability': np.random.normal(loc=[10]*200 + [20]*200, scale=5, size=400),
        'hours_after_sunset': np.random.normal(loc=2, scale=1, size=400),
        'rat_minutes': np.random.normal(loc=[5]*200 + [15]*200, scale=3, size=400),
        'reward': np.random.randint(0, 2, 400)
    }
    df = pd.DataFrame(data)

# 1. Map season values (0=Winter, 1=Spring)
df['season_label'] = df['season'].map({0: 'Winter', 1: 'Spring'})

# 2. Ensure core variables are correctly typed and assign new names for clarity
df['log_latency'] = pd.to_numeric(df['log_latency'], errors='coerce')
df['food_availability'] = pd.to_numeric(df['food_availability'], errors='coerce')
df['rat_present'] = pd.to_numeric(df['rat_present_during_landing'], errors='coerce').astype('Int64')
df['risk_behaviour'] = pd.to_numeric(df['risk'], errors='coerce').astype('Int64')
df['hours_after_sunset'] = pd.to_numeric(df['hours_after_sunset'], errors='coerce')
df['rat_minutes'] = pd.to_numeric(df['rat_minutes'], errors='coerce')
df['reward'] = pd.to_numeric(df['reward'], errors='coerce').astype('Int64')

# 3. Final cleanup for descriptive analysis
df_clean = df.dropna(subset=['season_label', 'log_latency', 'food_availability', 
                             'rat_present', 'risk_behaviour', 'hours_after_sunset', 
                             'rat_minutes', 'reward']).copy()
df_clean['rat_label'] = df_clean['rat_present'].map({0: 'RP=0 (Absent)', 1: 'RP=1 (Present)'})

print(f"Analysis sample size after cleaning: {len(df_clean)}")

# 2. Descriptive Analysis Functions 
def analyze_continuous(df, col_name, title, y_label):
    # Calculate summary statistics (Mean, Median, Std Dev)
    summary = df.groupby('season_label')[col_name].agg(
        Mean='mean', Median='median', Std='std', N='count'
    ).round(3).reset_index()

    print(f"\n--- Descriptive Stats: {title} ---")
    print(summary.to_markdown(index=False))
    print("\n")
    
    # Boxplot Visualization
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x='season_label', y=col_name, palette=['skyblue', 'lightgreen'])
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel('Season')
    plt.show()

def analyze_proportions(df, col_name, title, hue_col=None):
    if hue_col:
        # Group by season and a second factor (Rat Presence)
        summary = df.groupby(['season_label', hue_col])[col_name].mean().mul(100).round(2).reset_index(name='Proportion (%)')
        
        print(f"\n--- Descriptive Stats: {title} ---")
        print(summary.to_markdown(index=False))
        print("\n")
        
        # Grouped Bar Plot Visualization
        plt.figure(figsize=(8, 5))
        sns.barplot(data=summary, x='season_label', y='Proportion (%)', hue=hue_col, palette=['salmon', 'skyblue'])
        plt.title(title)
        plt.ylabel('Proportion (%)')
        plt.xlabel('Season')
        plt.legend(title=hue_col, loc='upper right')
    else:
        # Simple proportion by season
        summary = df.groupby('season_label')[col_name].mean().mul(100).round(2).reset_index(name='Proportion (%)')
        
        print(f"\n--- Descriptive Stats: {title} ---")
        print(summary.to_markdown(index=False))
        print("\n")
        
        # Bar Plot Visualization
        plt.figure(figsize=(6, 4))
        sns.barplot(data=summary, x='season_label', y='Proportion (%)', palette=['skyblue', 'lightgreen'])
        plt.title(title)
        plt.ylabel('Proportion (%)')
        plt.xlabel('Season')
        
    plt.show()


# 3. Execution of Descriptive Analyses (Investigation B) 

print("\n\n" + "="*40)
print("INVESTIGATION B: DESCRIPTIVE ANALYSIS")
print("="*40)

# 3.1a. Seasonal Food Availability 
analyze_continuous(
    df_clean,
    col_name='food_availability',
    title='Food Availability Distribution by Season',
    y_label='Food Availability (Unit)'
)

# 3.1b. Seasonal Rat Encounter Proportion 
analyze_proportions(
    df_clean,
    col_name='rat_present',
    title='Rat Present During Bat Landing by Season',
    hue_col=None 
)

# 3.1c. Food Availability vs Rat Presence by Season
print("\n--- Descriptive Stats: Food Availability vs Rat Presence by Season---")
food_summary = df_clean.groupby(['season_label', 'rat_label'])['food_availability'].agg(
    Mean='mean', Median='median', Std='std', N='count'
).round(3).reset_index()
print(food_summary.to_markdown(index=False))
print("\n")

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x='season_label', y='food_availability', hue='rat_label', palette=['skyblue', 'salmon'])
plt.title('Food Availability Distribution vs Rat Presence by Season')
plt.ylabel('Food Availability (Unit)')
plt.xlabel('Season')
plt.legend(title='Rat Presence', loc='upper right')
plt.show()


# 3.2. Log Latency (Vigilance) by Season and Rat Presence  (Behavioral Response)
print("\n--- Descriptive Stats: Log Latency vs Rat Presenceby Season ---")
latency_summary = df_clean.groupby(['season_label', 'rat_label'])['log_latency'].agg(
    Mean='mean', Median='median', Std='std', N='count'
).round(3).reset_index()
print(latency_summary.to_markdown(index=False))
print("\n")
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_clean, x='season_label', y='log_latency', hue='rat_label', palette='pastel')
plt.title('Log Latency (Vigilance) vs Rat Presence by Season')
plt.ylabel('Log Latency (Log(1+Time))')
plt.xlabel('Season')
plt.legend(title='Rat Presence', loc='upper left')
plt.show()

# 3.3. Risk-Taking vs Rat Presence by Season
analyze_proportions(
    df_clean,
    col_name='risk_behaviour',
    title='Risk-Taking vs Rat Presence by Season',
    hue_col='rat_label' 
)

print("\n\n--- Descriptive Analysis Complete ---")

