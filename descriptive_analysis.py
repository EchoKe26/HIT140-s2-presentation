import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

df = pd.read_csv("merged_clean_dataset.csv", parse_dates=["window_start"])
print("Columns:", df.columns)
print(df.head())
print("="*50)


# 1. Basic summary 
print("\nSummary statistics:")
print(df.describe(include="all"))
print("="*50)
print("\n")

print("Key Variables Distribution Analysis\n")

# 2. Distribution of key numerical variables
key_numeric_cols = ['log_latency', 'rat_minutes']
key_numeric_cols = [col for col in key_numeric_cols if col in df.columns]

if key_numeric_cols:
    n_cols = len(key_numeric_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))  
    if n_cols == 1:
        axes = [axes]  
        
    for i, col in enumerate(key_numeric_cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

        stats_text = f'Mean: {df[col].mean():.2f}\nMedian: {df[col].median():.2f}\nStd: {df[col].std():.2f}'
        axes[i].text(0.65, 0.75, stats_text, transform=axes[i].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.show()


#3. rat presence analysis
rat_counts = df["rat_present_during_landing"].value_counts(dropna=False)
rat_percent = df["rat_present_during_landing"].value_counts(normalize=True) * 100

rat_summary = pd.DataFrame({
    "count": rat_counts,
    "percentage": rat_percent.round(2)
})
print("\nRat presence count and precentage:\n")
total_count = rat_summary["count"].sum()
rat_summary.loc["Total"] = [total_count, 100.00]
print(rat_summary)
print("="*50)
print("\n")

      
# Pie chart for rat presence
plot_data = rat_summary.drop(index="Total")
plt.figure(figsize=(6,6))
plt.pie(
    plot_data["count"],
    labels=plot_data.index,
    autopct='%1.1f%%',   
    colors=["salmon", "skyblue"], 
    startangle=90,      
    wedgeprops={'edgecolor': 'black'} 
)
plt.title("Bat Landings by Rat Presence")
plt.tight_layout()
plt.show()

quartile_counts = df["rat_pressure_quartile"].value_counts().sort_index()
quartile_percent = df["rat_pressure_quartile"].value_counts(normalize=True).sort_index() * 100
quartile_minmax = df.groupby("rat_pressure_quartile")["rat_minutes"].agg(["min","max"]).round(2)


quartile_summary = pd.DataFrame({
    "min": quartile_minmax["min"],
    "max": quartile_minmax["max"],
    "std": df.groupby("rat_pressure_quartile")["rat_minutes"].std().round(2),
    "count": quartile_counts,
    "percentage(%)": quartile_percent.round(2)
})

quartile_order = ["Low", "Medium", "High"]
quartile_summary = quartile_summary.reindex(quartile_order + ["Total"])
quartile_summary.loc["Total"] = ["-", "-", "-", quartile_counts.sum(), 100.0]

print("\nRat pressure quartile summary:")
print(quartile_summary)
print("\n","="*50, "\n")


# Bar Chart
colors = ["lightgreen", "gold", "salmon"]

plt.figure(figsize=(10,5))
bars = plt.bar(quartile_summary.index[:-1], quartile_summary["percentage(%)"][:-1], color=colors)

for bar, perc in zip(bars, quartile_summary["percentage(%)"][:-1]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{perc:.1f}%", ha='center', va='bottom')

range_labels = [f"{q}: {row['min']:.2f}–{row['max']:.2f}" 
                 for q, row in quartile_summary.iloc[:-1].iterrows()]
plt.legend(bars, range_labels, title="Range (min–max)", loc='upper right')

plt.title("Bat Landings by Rat Pressure Quartile (%)")
plt.xlabel("Rat Pressure Quartile")
plt.ylabel("Percentage (%)")
plt.tight_layout()
plt.show()


# 5. Risk and reward vs rat presence 

# Mean risk by rat presence
risk_stats = df.groupby("rat_present_during_landing")["risk"].agg(["mean","std","count"])
risk_stats.loc["Total"] = ["-", "-" , risk_stats["count"].sum()]
print("\nRisk stats by rat presence:")
print(risk_stats)
print("\n")

# Mean reward by rat presence
reward_stats = df.groupby("rat_present_during_landing")["reward"].agg(["mean","std","count"])
reward_stats.loc["Total"] = ["-", "-", reward_stats["count"].sum()]
print("\nReward stats by rat presence:")
print(reward_stats)
print("\n","="*50, "\n")

#Visualization
summary = df.groupby("rat_present_during_landing")[["risk","reward"]].mean()

summary.plot(kind="bar", figsize=(8,5))
plt.title("Mean Risk and Reward by Rat Presence")
plt.ylabel("Mean value (0-1)")
plt.xlabel("Rat present during landing")
plt.ylim(0,1)
plt.xticks(rotation=0)  
plt.legend(title="Measure")  
plt.tight_layout()
plt.show()
plt.close()


# 6. Behavior categories vs rat presence 
if "habit_clean" in df.columns:
    # Crosstab counts
    behavior_counts = pd.crosstab(df["rat_present_during_landing"], df["habit_clean"])
    behavior_counts.loc["Total"] = behavior_counts.sum()
    print("\nBehavior counts by rat presence:")
    print(behavior_counts)
    
print("\n","="*50, "\n")


# Percentage table
behavior_precent = behavior_counts.div(behavior_counts.sum(axis=1), axis=0) * 100
print("\nBehavior percentages by rat presence:")
print(behavior_precent.round(2))

# Grouped bar plot
behavior_counts_tf = behavior_counts.loc[behavior_counts.index.isin([True, False])]
behavior_percent_tf = behavior_counts_tf.div(behavior_counts_tf.sum(axis=1), axis=0) * 100

ax = behavior_percent_tf.T.plot(kind="bar", figsize=(10, 8))
plt.title("Behavior percentages by rat presence")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=45)
plt.legend(title="Rat Present")
    
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", label_type="edge", padding=2)

plt.tight_layout()
plt.show()
plt.close()


# 7. Latency vs Rat Presence
if "log_latency" in df.columns:
    plt.figure()
    sns.boxplot(x="rat_present_during_landing", y="log_latency", data=df)
    plt.title("Log latency to food (rat present vs not)")
    plt.show()
    plt.close()
   

# 8. Latency vs Rat pressure
# Filter only High and Low rat pressure groups
latency_df = df[df["rat_pressure_quartile"].isin(["High", "Low"])].copy()

plt.figure(figsize=(8,6))
sns.boxplot(x="rat_pressure_quartile", y="log_latency", data=latency_df)
plt.title("Log Latency by Rat Pressure Level (High vs Low)")
plt.xlabel("Rat Pressure Level")
plt.ylabel("Log Latency to Food")
plt.show()
plt.close()





