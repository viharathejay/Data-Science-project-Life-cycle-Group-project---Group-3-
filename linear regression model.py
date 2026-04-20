import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------
# Load the datasets
# ----------------------------

# Load the cleaned weather data from Excel
weather = pd.read_excel(r"C:\\Users\\lavan\\OneDrive\\Documents\\weather clean.xlsx")

# Load the cleaned crime data from CSV
crime = pd.read_csv(r"C:\\Users\\lavan\\OneDrive\\Documents\\final_cleaned_crime_dataset.csv")


# ----------------------------
# Clean weather dataset
# ----------------------------

# Rename columns so they are easier to work with
weather = weather.rename(columns={
    'Date': 'date',
    'Stations ': 'station'
})

# Convert date column to datetime format
weather['date'] = pd.to_datetime(weather['date'])

# Create a month column for monthly analysis
weather['month'] = weather['date'].dt.to_period('M')

# Clean station names by removing spaces and using lowercase
weather['station'] = weather['station'].str.strip().str.lower()


# ----------------------------
# Clean crime dataset
# ----------------------------

# Convert crime date column to datetime format
crime['date'] = pd.to_datetime(crime['date'])

# Create a month column to match the weather dataset
crime['month'] = crime['date'].dt.to_period('M')

# Clean station names in the same way
crime['station'] = crime['station'].str.strip().str.lower()


# ----------------------------
# Standardise station names
# ----------------------------

# Function to make station names consistent across both datasets
def standardise_station(name):
    # Keep missing values unchanged
    if pd.isna(name):
        return name

    # Remove spaces and convert text to lowercase
    name = str(name).strip().lower()

    # Match different versions of the same station name
    if 'kings cross' in name:
        return 'kings cross'
    elif 'leeds' in name:
        return 'leeds'
    elif 'newcastle' in name:
        return 'newcastle'
    elif 'edinburgh' in name:
        return 'edinburgh'
    else:
        # Return the cleaned name if no match is found
        return name


# Apply station name cleaning to both datasets
weather['station'] = weather['station'].apply(standardise_station)
crime['station'] = crime['station'].apply(standardise_station)


# ----------------------------
# Aggregate datasets
# ----------------------------

# Group weather data by station and month
# Mean is used for temperature and humidity
# Sum is used for total monthly precipitation
weather_monthly = weather.groupby(['station', 'month']).agg({
    'temperature_avg': 'mean',
    'humidity_avg': 'mean',
    'precipitation': 'sum'
}).reset_index()

# Group crime data by station and month
# Sum disorder_flag to get monthly crime count
crime_monthly = crime.groupby(['station', 'month']).agg({
    'disorder_flag': 'sum'
}).reset_index()

# Rename the crime column to make it clearer
crime_monthly = crime_monthly.rename(columns={'disorder_flag': 'crime_count'})


# ----------------------------
# Merge datasets
# ----------------------------

# Merge weather and crime data using station and month
# Only matching rows in both datasets are kept
df = pd.merge(weather_monthly, crime_monthly, on=['station', 'month'], how='inner')

# Preview the merged dataset
print("Merged dataset preview:")
print(df.head())

# Show which stations are included after merging
print("\nStations in merged dataset:")
print(sorted(df['station'].unique()))


# ----------------------------
# Handle missing values
# ----------------------------

# Fill missing temperature values with the overall mean
df['temperature_avg'] = df['temperature_avg'].fillna(df['temperature_avg'].mean())

# Fill missing humidity values with the overall mean
df['humidity_avg'] = df['humidity_avg'].fillna(df['humidity_avg'].mean())

# Fill missing precipitation values with 0
df['precipitation'] = df['precipitation'].fillna(0)


# ----------------------------
# Linear regression model
# ----------------------------

# Select weather variables as input features
X = df[['temperature_avg', 'precipitation', 'humidity_avg']]

# Select crime count as the target variable
y = df['crime_count']

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Store coefficients in a table for easier interpretation
results = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

# Print regression coefficients
print("\nRegression coefficients:")
print(results)

# Print model intercept
print("\nIntercept:")
print(model.intercept_)

# Print R^2 score to show model fit
print("\nR^2 score:")
print(model.score(X, y))


# ----------------------------
# King's Cross only graph
# ----------------------------

# Filter the merged data for King's Cross only
kings_cross_df = df[df['station'] == 'kings cross']

# Print King's Cross monthly values
print("\nKing's Cross data:")
print(kings_cross_df[['month', 'precipitation', 'crime_count']])

# Plot precipitation against crime count for King's Cross
plt.figure(figsize=(8, 6))
sns.regplot(
    x=kings_cross_df['precipitation'],
    y=kings_cross_df['crime_count']
)

# Add chart title and axis labels
plt.title("King’s Cross: Precipitation vs Crime Count")
plt.xlabel("Precipitation (mm of rainfall)")
plt.ylabel("Crime Count (number of incidents)")

# Adjust spacing and show the graph
plt.tight_layout()
plt.show()
