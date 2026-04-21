import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ----------------------------
# Load files
# ----------------------------
crime = pd.read_csv(r"C:\Users\lavan\OneDrive\Documents\final_cleaned_crime_dataset.csv")
football = pd.read_csv(r"C:\Users\lavan\OneDrive\Documents\football clean file.csv")

# ----------------------------
# Clean crime dataset
# ----------------------------
crime['date'] = pd.to_datetime(crime['date'])
crime['station'] = crime['station'].str.strip().str.lower()

def standardise_crime_station(name):
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

crime['station_std'] = crime['station'].apply(standardise_crime_station)

target_stations = ['kings cross', 'leeds', 'newcastle', 'edinburgh']
crime = crime[crime['station_std'].isin(target_stations)].copy()

print("Crime dataset preview:")
print(crime.head())
print("\nCrime shape:", crime.shape)

# ----------------------------
# Clean football dataset
# ----------------------------
football['nearest_lner_station'] = football['nearest_lner_station'].str.strip().str.lower()

def standardise_football_station(name):
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

football['station_std'] = football['nearest_lner_station'].apply(standardise_football_station)
football = football[football['station_std'].isin(target_stations)].copy()

print("\nFootball dataset preview:")
print(football.head())
print("\nFootball shape:", football.shape)

# ----------------------------
# Create football pressure variables by station
# ----------------------------
football_station = football.groupby('station_std').agg(
    football_match_count=('stadium_name', 'count'),
    avg_estimated_attendance=('estimated_attendance', 'mean'),
    avg_stadium_capacity=('stadium_capacity', 'mean')
).reset_index()

print("\nFootball station summary:")
print(football_station)

# ----------------------------
# Merge football exposure into crime data
# ----------------------------
df = crime.merge(football_station, on='station_std', how='left')

df['football_match_count'] = df['football_match_count'].fillna(0)
df['avg_estimated_attendance'] = df['avg_estimated_attendance'].fillna(0)
df['avg_stadium_capacity'] = df['avg_stadium_capacity'].fillna(0)

df['football_exposure'] = (df['football_match_count'] > 0).astype(int)

print("\nMerged dataset preview:")
print(df[['station_std', 'disorder_flag', 'football_match_count', 'avg_estimated_attendance', 'avg_stadium_capacity', 'football_exposure']].head())

# ----------------------------
# Create dummy variables for station
# ----------------------------
station_dummies = pd.get_dummies(df['station_std'], prefix='station', drop_first=True)
station_dummies = station_dummies.astype(int)

# ----------------------------
# Build feature matrix
# ----------------------------
X = pd.concat([
    df[['football_match_count', 'avg_estimated_attendance', 'avg_stadium_capacity', 'football_exposure']],
    station_dummies
], axis=1)

y = df['disorder_flag']

print("\nFeature columns:")
print(X.columns.tolist())

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# Scale numeric variables
# ----------------------------
numeric_cols = ['football_match_count', 'avg_estimated_attendance', 'avg_stadium_capacity']

scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# ----------------------------
# Logistic Regression model
# ----------------------------
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# ----------------------------
# Predictions and evaluation
# ----------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------
# Coefficients
# ----------------------------
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nLogistic Regression Coefficients:")
print(coef_df)

# ----------------------------
# Station-level summary for report
# ----------------------------
station_summary = df.groupby('station_std').agg(
    total_incidents=('disorder_flag', 'count'),
    disorder_incidents=('disorder_flag', 'sum'),
    disorder_rate=('disorder_flag', 'mean'),
    football_match_count=('football_match_count', 'first'),
    avg_estimated_attendance=('avg_estimated_attendance', 'first')
).reset_index()

print("\nStation summary:")
print(station_summary)