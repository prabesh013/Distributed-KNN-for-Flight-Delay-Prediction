import pandas as pd


FILE_PATH = (
    "data/raw/2024/january/"
    "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_1.csv"
)


df = pd.read_csv(FILE_PATH)


# Basic dataset information
print("\n====================")
print("DATASET SIZE")
print("====================")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# Column names
print("\n====================")
print("COLUMN NAMES")
print("====================")

for i, column in enumerate(df.columns):
    print(f"{i}: {column}")


# Relevant columns for our project
important_columns = [
    "FlightDate",
    "Month",
    "DayOfWeek",
    "CRSDepTime",
    "DepDelay",
    "DepDelayMinutes",
    "TaxiOut",
    "CRSElapsedTime",
    "Distance",
    "ArrDelay",
    "ArrDel15",
    "Cancelled",
    "Diverted"
]


print("\n====================")
print("SAMPLE DATA")
print("====================")

print(df[important_columns].head(10).to_string())


print("\n====================")
print("DATA TYPES")
print("====================")

print(df[important_columns].dtypes)


print("\n====================")
print("MISSING VALUES")
print("====================")

print(df[important_columns].isnull().sum())


print("\n====================")
print("ARRIVAL DELAY TARGET")
print("====================")

print(df["ArrDel15"].value_counts(dropna=False))

print("\nPercentages:")

print(
    df["ArrDel15"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
)


print("\n====================")
print("CANCELLED")
print("====================")

print(df["Cancelled"].value_counts(dropna=False))


print("\n====================")
print("DIVERTED")
print("====================")

print(df["Diverted"].value_counts(dropna=False))


print("\n====================")
print("NUMERICAL SUMMARY")
print("====================")

features = [
    "DepDelayMinutes",
    "TaxiOut",
    "CRSElapsedTime",
    "Distance",
    "ArrDelay"
]

print(df[features].describe())