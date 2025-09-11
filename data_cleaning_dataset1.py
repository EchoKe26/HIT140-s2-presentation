import pandas as pd 

df = pd.read_csv("dataset1.csv")
print("Dataset1 overview")
print(df.head())
print("="*50)
print(df.info())
print("="*50)

# 1. Convert datetime columns 
for col in ["start_time", "sunset_time", "rat_period_start", "rat_period_end"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

print("1. Convert datetime columns:\n")
print(f"Datetime columns have been converted.")
print(f"[dataset1] Rows after converting datetime columns : {len(df)}")
print("\n")
print("="*50)
print("\n")

# 2. Convert numeric columns 
num_cols = ["bat_landing_to_food", "seconds_after_rat_arrival", "risk", "reward", "hours_after_sunset"]
df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

print("2. Convert numeric columns:\n")
print(f"Numeric columns have been converted.")
print(f"[dataset1] Rows after converting numeric columns : {len(df)}")
print("\n")
print("="*50)
print("\n")

# 3. Drop rows with critical missing or invalid values 
drop_mask = df["start_time"].isna() | df["risk"].isna() | (df["bat_landing_to_food"] < 0)
df = df.loc[~drop_mask].copy()
print("3. Drop rows with critical missing or invalid values:\n")
print(f"[dataset1] Rows after dropping missing/invalid: {len(df)}")
print("\n")
print("="*50)
print("\n")
      
# 4. Fill minor missing values 
print("4. Fill minor missing values:\n")
if df["hours_after_sunset"].isna().any():
    median_val = df["hours_after_sunset"].median()
    df["hours_after_sunset"] = df["hours_after_sunset"].fillna(median_val)
    print(f"[dataset1] Filled missing hours_after_sunset with median = {median_val:.3f}")
else:
    print("No minor missing value.")

print("\n")   
print("="*50)
print("\n")

# 5. Check for dulpicated rows
print("5. Check for dulpicated rows:\n")
duplicates = df.duplicated()
print(f"Number of duplicate rows: {duplicates.sum()}")
df = df.drop_duplicates()
print(f"Number of rows after dropping duplicates: {len(df)}")
print("\n")
print("="*50)
print("\n")

# 6. Check numeric outliers 
print("6. Check numeric outliers:\n")
for col in df.columns:
    if col.endswith("_outlier"):
        df.drop(columns=[col], inplace=True)

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    
    # create outlier mask
    outlier_mask = (df[col] < lower) | (df[col] > upper)
    
    # only create column if there are outliers
    if outlier_mask.sum() > 0:
        outlier_col = col + "_outlier"
        df[outlier_col] = outlier_mask
        print(f"{col} outliers: {outlier_mask.sum()}")

print("\n")
print("="*50)
print("\n")


# 7. Time logic check
print("7. Time logic check:\n")
if "rat_period_start" in df.columns and "rat_period_end" in df.columns:
    invalid_time = df[df["rat_period_start"] > df["rat_period_end"]]
    print(f"Rows with rat_period_start > rat_period_end: {len(invalid_time)}")


print("\n")
print("="*50)
print("\n")

# 8. Check all the categories in "habit" row
print("8. Check all the categories in 'habit' row:\n")
pd.set_option('display.max_rows', None)
print(df['habit'].value_counts())
print("="*50)
print("\n")

# 9. Clean "habit" column by merging similar categories
def clean_habit(x):
    if pd.isna(x):
        return 'others'
    x = x.lower()
    if 'fast' in x:
        return 'fast'
    elif 'fight' in x or 'attack' in x:
        return 'fight'
    elif 'pick' in x or 'eat' in x:
        return 'pick'
    elif 'rat' in x and 'bat' in x:
        return 'rat_bat_others'
    elif 'rat' in x:
        return 'rat'
    elif 'bat' in x:
        return 'bat'
    else:
        return 'others'
        
df['habit_clean'] = df['habit'].apply(clean_habit)
print("9. Clean 'habit' column:\n")
print(df['habit_clean'].value_counts())
print("\n","habit_clean total rows:", len(df['habit_clean']))

print("\n")
print("="*50)
print("\n")

# 10. Save cleaned dataset
df.to_csv("dataset1_cleaned.csv", index=False)
print("Cleaned dataset saved as dataset1_cleaned.csv")

