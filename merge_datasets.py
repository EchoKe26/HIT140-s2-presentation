import pandas as pd
import numpy as np

df1 = pd.read_csv("dataset1_cleaned.csv", parse_dates=["start_time", "sunset_time", "rat_period_start", "rat_period_end"])
df2 = pd.read_csv("dataset2_cleaned.csv", parse_dates=["time"])

print("Dataset1 shape:", df1.shape)
print("Dataset2 shape:", df2.shape)
print("Dataset1 total rows:", len(df1))
print("Dataset2 total rows:", len(df2))
print("\n")
print("="*50)
print("\n")


# 1. Create "window_start" column in df1
df1["hours_after_sunset"] = pd.to_numeric(df1["hours_after_sunset"], errors="coerce")
df1["window_start"] = df1["sunset_time"] + pd.to_timedelta(
    np.floor(df1["hours_after_sunset"] * 2) / 2, unit="h"
)
print("1. Added 'window_start' to dataset1")
print("Total rows with window_start:", len(df1[df1['window_start'].notna()]))
print("Missing hours_after_sunset count:", df1["hours_after_sunset"].isna().sum())
print(df1[["hours_after_sunset", "window_start"]].head())
print("\n")
print("="*50)
print("\n")


# 2. Align dataset2 column names
df2 = df2.rename(columns={"time": "window_start"})
print("2. Renamed 'time' to 'window_start' in dataset2")
print(df2.head())
print("Total rows with window_start:", len(df2[df2['window_start'].notna()]))
print("="*50)
print("\n")


# 3. Merge datasets on window_start + month
original_count = len(df1)
merged = df1.merge(df2, on=["window_start", "month"], how="inner", suffixes=("", "_win"))
print("3. Merge datasets on window_start + month")
print(f"Merged dataset shape: {merged.shape}")
print(f"Total rows after merge: {len(merged)}")
print(f"Rows lost during merge: {original_count - len(merged)}")
print("Missing values per key column:")
print(f"  - window_start missing: {df1['window_start'].isna().sum()} in df1, {df2['window_start'].isna().sum()} in df2")
print(f"  - month missing: {df1['month'].isna().sum()} in df1, {df2['month'].isna().sum()} in df2")

print("\n")
print("="*50)
print("\n")


# 4. Add log_latency 
if "bat_landing_to_food" in merged.columns:
    merged["log_latency"] = np.log1p(merged["bat_landing_to_food"])
    print("4. Added 'log_latency' column")
    print(merged[["bat_landing_to_food", "log_latency"]].head())
    print(f"Total rows with log_latency: {len(merged[merged['log_latency'].notna()])}")
    print(f"Missing bat_landing_to_food count: {merged['bat_landing_to_food'].isna().sum()}")
else:
    print("'bat_landing_to_food' not found, skipping log_latency")
    print("Available columns:", [col for col in merged.columns 
                                 if 'bat' in col.lower() or 'food' in col.lower()])

print("\n")
print("="*50)
print("\n")

# 5. Rat pressure binning 
q1 = merged["rat_minutes"].quantile(0.25)
q3 = merged["rat_minutes"].quantile(0.75)

# Create rat pressure bins
def bin_rat_pressure_quartile(x):
    if pd.isna(x):
        return np.nan
    elif x <= q1:
        return "Low"
    elif x >= q3:
        return "High"
    else:
        return "Medium"
        
merged["rat_pressure_quartile"] = merged["rat_minutes"].apply(bin_rat_pressure_quartile)

print("5. Rat pressure binning")
print(merged["rat_pressure_quartile"].value_counts())

total_non_missing = merged["rat_pressure_quartile"].notna().sum()
total_rows = len(merged)

print(f"Total rows after merged: {len(merged["rat_pressure_quartile"])}")
print(f"Non-missing values: {total_non_missing}")
print(f"Missing values: {total_rows - total_non_missing}")

print("\n")
print("="*50)
print("\n")


# 6. Rat presence during bat landing 
if {"seconds_after_rat_arrival", "rat_minutes"}.issubset(merged.columns):
    s = pd.to_numeric(merged["seconds_after_rat_arrival"], errors="coerce")
    m = pd.to_numeric(merged["rat_minutes"], errors="coerce")
    merged["rat_present_during_landing"] = (s >= 0) & (s <= m * 60)
    print("6. Added 'rat_present_during_landing' column")
    print(merged["rat_present_during_landing"].value_counts(dropna=False))
else:
    merged["rat_present_during_landing"] = np.nan

print(f"Total rows: {len(merged["rat_present_during_landing"])}")

print("\n")
print("="*50)
print("\n")


# 7. Clean habit column (keep only bat behavior actions)
def clean_habit(x):
    if pd.isna(x):
        return "others"
    x = str(x).lower()
    if "fast" in x:
        return "fast"
    elif "fight" in x or "attack" in x:
        return "fight"
    elif "pick" in x or "eat" in x:
        return "pick"
    else:
        return "others"
        
merged["habit_clean"] = merged["habit_clean"].apply(clean_habit)
print("7. Cleaned 'habit_clean' to keep only bat actions")
print(merged["habit_clean"].value_counts())
print(f"Total cleaned habit_clean values: {len(merged[merged['habit_clean'].notna()])}")
print(f"Missing after cleaning: {merged['habit_clean'].isna().sum()}")

print("\n")
print("="*50)
print("\n")


# 8. Save merged dataset 
merged.to_csv("merged_clean_dataset.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
print("Saved merged dataset to 'merged_clean_dataset.csv'")

