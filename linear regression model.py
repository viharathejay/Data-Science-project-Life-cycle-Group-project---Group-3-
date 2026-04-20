import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Load files
# ----------------------------
weather = pd.read_excel(r"C:\Users\lavan\OneDrive\Documents\weather clean.xlsx")
crime = pd.read_csv(r"C:\Users\lavan\OneDrive\Documents\final_cleaned_crime_dataset.csv")

# ----------------------------
# Clean weather dataset
# ----------------------------
weather = weather.rename(columns={
    'Date': 'date',
    'Stations ': 'station'
})

weather['date'] = pd.to_datetime(weather['date'])
weather['month'] = weather['date'].dt.to_period('M')
weather['station'] = weather['station'].str.strip().str.lower()

# ----------------------------
# Clean crime dataset
# ----------------------------
crime['date'] = pd.to_datetime(crime['date'])
crime['month'] = crime['date'].dt.to_period('M')
crime['station'] = crime['station'].str.strip().str.lower()

# ----------------------------
# Standardise station names
# ----------------------------
def standardise_station(name):
    if pd.isna(name):
        return name

    name = str(name).strip().lower()

    if 'kings cross' in name:
        return 'kings cross'
    elif 'leeds' in name:
        return 'leeds'
    elif 'newcastle' in name:
        return 'newcastle'
    elif 'edinburgh' in name:
        return 'edinburgh'
    else:
        return name

weather['station'] = weather['station'].apply(standardise_station)
crime['station'] = crime['station'].apply(standardise_station)

# ----------------------------
# Aggregate datasets
# ----------------------------
weather_monthly = weather.groupby(['station', 'month']).agg({
    'temperature_avg': 'mean',
    'humidity_avg': 'mean',
    'precipitation': 'sum'
}).reset_index()

crime_monthly = crime.groupby(['station', 'month']).agg({
    'disorder_flag': 'sum'
}).reset_index()

crime_monthly = crime_monthly.rename(columns={'disorder_flag': 'crime_count'})

# ----------------------------
# Merge datasets
# ----------------------------
df = pd.merge(weather_monthly, crime_monthly, on=['station', 'month'], how='inner')

print("Merged dataset preview:")
print(df.head())

print("\nStations in merged dataset:")
print(sorted(df['station'].unique()))

# ----------------------------
# Handle missing values
# ----------------------------
df['temperature_avg'] = df['temperature_avg'].fillna(df['temperature_avg'].mean())
df['humidity_avg'] = df['humidity_avg'].fillna(df['humidity_avg'].mean())
df['precipitation'] = df['precipitation'].fillna(0)

# ----------------------------
# Linear regression model
# ----------------------------
X = df[['temperature_avg', 'precipitation', 'humidity_avg']]
y = df['crime_count']

model = LinearRegression()
model.fit(X, y)

results = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nRegression coefficients:")
print(results)

print("\nIntercept:")
print(model.intercept_)

print("\nR^2 score:")
print(model.score(X, y))

# ----------------------------
# King's Cross only graph
# ----------------------------
kings_cross_df = df[df['station'] == 'kings cross']

print("\nKing's Cross data:")
print(kings_cross_df[['month', 'precipitation', 'crime_count']])

plt.figure(figsize=(8, 6))
sns.regplot(
    x=kings_cross_df['precipitation'],
    y=kings_cross_df['crime_count']
)

plt.title("King’s Cross: Precipitation vs Crime Count")
plt.xlabel("Precipitation (mm of rainfall)")
plt.ylabel("Crime Count (number of incidents)")
plt.tight_layout()
plt.show()