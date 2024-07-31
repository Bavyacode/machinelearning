import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv('CREDITSCORE.csv')

# Drop irrelevant columns or IDs
data = data.drop(['ID', 'Customer_ID', 'SSN', 'Name'], axis=1)

# Handle missing values
data = data.dropna()

# Convert categorical features to numeric
label_encoders = {}
categorical_columns = ['Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_Behaviour', 'Payment_of_Min_Amount']

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le
    #print(f"{col} classes:", le.classes_) """ # Print classes for debugging

# Define features and target variable
X = data.drop('Credit_Score', axis=1)
y = data['Credit_Score']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print data types and columns for debugging
#print("X_train columns:", X_train.columns)
#print("X_train data types:", X_train.dtypes)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Example of predicting a new data point
new_data = pd.DataFrame({
    'Month': [1],
    'Age': [30],
    'Occupation': ['Engineer'],
    'Annual_Income': [50000],
    'Monthly_Inhand_Salary': [4000],
    'Num_Bank_Accounts': [2],
    'Num_Credit_Card': [1],
    'Interest_Rate': [0.05],
    'Num_of_Loan': [1],
    'Type_of_Loan': ['Auto Loan'],
    'Delay_from_due_date': [0],
    'Num_of_Delayed_Payment': [0],
    'Changed_Credit_Limit': [0],
    'Num_Credit_Inquiries': [1],
    'Credit_Mix': ['Good'],
    'Outstanding_Debt': [1000],
    'Credit_Utilization_Ratio': [0.2],
    'Credit_History_Age': [5],
    'Payment_of_Min_Amount': ['Yes'],
    'Total_EMI_per_month': [200],
    'Amount_invested_monthly': [500],
    'Payment_Behaviour': ['High_spent_Medium_value_payments'],
    'Monthly_Balance': [2000]
})

# Ensure the new_data DataFrame contains all necessary columns
missing_cols = set(X.columns) - set(new_data.columns)
if missing_cols:
    print(f"Warning: Missing columns in new data: {missing_cols}")
    for col in missing_cols:
        new_data[col] = 0  # or some default value

# Convert categorical features in new data
for col in categorical_columns:
    if col in new_data.columns:
        le = label_encoders[col]
        try:
            new_data[col] = le.transform(new_data[col])
        except ValueError as e:
            print(f"Error transforming {col}: {e}")
            # Handle unseen labels
            new_data[col] = le.transform([le.classes_[0]])  # Map unseen labels to the first class
    else:
        print(f"Warning: {col} not found in new data")

# Ensure new_data columns match the trained model's feature set
new_data = new_data[X.columns]

# Print columns for debugging
#print("new_data columns:", new_data.columns)

# Scale the new data
new_data = scaler.transform(new_data)
new_prediction = model.predict(new_data)
print("Predicted Credit Score:", new_prediction)
