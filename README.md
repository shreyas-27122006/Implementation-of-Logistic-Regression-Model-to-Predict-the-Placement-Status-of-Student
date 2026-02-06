# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and prepare the dataset containing student features (such as marks, CGPA, skills, etc.) and the target variable (Placement Status).
2. Split the dataset into training data and testing data.
3. Train the Logistic Regression model using the training dataset.
4. Test the model and predict the placement status using the testing dataset and evaluate the result.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SHREYAS M
RegisterNumber: 250013237
*/
# Step 1: Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
data = pd.read_csv("C:/Users/acer/Downloads/Placement_Data (2).csv")

# Display first few rows
print(data.head())

# Step 3: Data Preprocessing

# Drop serial number column (not useful for prediction)
data.drop('sl_no', axis=1, inplace=True)

# Fill missing salary values with 0
data['salary'].fillna(0, inplace=True)

# Convert categorical columns into numerical values
le = LabelEncoder()
categorical_columns = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Step 4: Split features and target
X = data.drop('status', axis=1)
y = data['status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:
![WhatsApp Image 2026-02-06 at 11 24 32 AM](https://github.com/user-attachments/assets/93dd15cf-3371-46ab-bf0e-c14848923865)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
