# Loan Approval Prediction

# Import Required Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# Load the Dataset

data = pd.read_csv(r"C:\Users\fatai\Desktop\ML AND AI Learning\loan_approval_dataset.csv") #Update with your path
data.columns = data.columns.str.strip().str.lower()

# Display first few rows
print(data.head())

# Check info
print("\n Dataset Info:")
print(data.info())

# Check for Missing Values
print("\nMissing Values:\n", data.isnull().sum())



#  Handle Missing Values

# Fill numeric columns with median and categorical with mode
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0])
    else:
        data[col].fillna(data[col].median())

print("\n Missing values handled successfully!")


#  Encode Categorical Columns

label_encoders = {}
for col in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

print("\n Categorical features encoded!")


#  Split Features and Target

X = data.drop('loan_status', axis=1)  # assuming 'Loan_Status' is target column
y = data['loan_status']


#  Split into Train & Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n Dataset split: Train =", len(X_train), ", Test =", len(X_test))


#  Feature Scaling

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#  Train Classification Model

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("\n Model training completed!")


#  Model Evaluation

y_pred = model.predict(X_test)

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n Classification Report:")
print(classification_report(y_test, y_pred, digits=3))

print("\n Model evaluation complete!")
