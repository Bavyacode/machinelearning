import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv('HousePricePrediction.csv')

# Drop rows with missing target value
df = df.dropna(subset=['SalePrice'])

# Separate features and target variable
X = df.drop(['SalePrice', 'Id'], axis=1)
y = df['SalePrice']

# Handle missing values

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Fill missing values for numeric columns with median
X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

# Fill missing values for categorical columns with mode
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

# Encode categorical features
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
