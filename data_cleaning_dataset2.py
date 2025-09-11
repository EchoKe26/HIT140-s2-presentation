import pandas as pd 

df = pd.read_csv("dataset2.csv")

print(df.head())
print("="*50)
print(df.info())


# 1. Convert 'time' column to datetime
if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], dayfirst=True, errors="coerce")
print(df["time"].head())
print(f"Total count for 'time' value: {len(df["time"])}")
print("="*50)
print("\n")


# 2. Convert numeric columns
num_cols = [
    "month",
    "hours_after_sunset",
    "bat_landing_number",
    "food_availability",
    "rat_minutes",
    "rat_arrival_number"
]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
print(df[num_cols].head())
print("="*50)
print("\n")

      
# 3. Check basic info and missing values
print(df.info())
print(df.isnull().sum())
print("="*50)
print("\n")

# 4. Save cleaned dataset
df.to_csv("dataset2_cleaned.csv", index=False)
print("Cleaned dataset saved as dataset2_cleaned.csv")

