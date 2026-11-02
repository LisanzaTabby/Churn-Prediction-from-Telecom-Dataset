import pandas as pd
# import matplotlib.pyplot as plt

# for our ML model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Task 1: Loading the Dataset and checking the data types of each column, and for missing values
df = pd.read_csv('Customer Churn.csv')

print(df.info())

# Task 2: Data Pre-processing
# a) Checking and handling missing values in the dataset
df_missing_values = df.isnull().sum()

print("Missing Values in Each Column:", df_missing_values)

# b) Checking & Handling duplicate values in the dataset
df_duplicate_values = df.duplicated().sum()

print("Duplicate Values in the Dataset:", df_duplicate_values)

# 300 duplicates were found, hence removal

df = df.drop_duplicates()

print("Dataset with no duplicates:", df.shape)

print ("Data types of each column:", df.dtypes)


# Task 3: Feature Selection for the x and y variables
# Independent variables
X = df.drop(columns=[
    'Churn',
    'Age Group',
    'Status'
])

# Dependent variable
y = df['Churn']

print("Independent variables:", X.columns)
print("\nTarget variable:", y.name)

# Task 4: Splitting the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify = y
)

print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)

# Numerical features
numerical_features = [
    'Call  Failure',
    'Complains',
    'Subscription  Length',
    'Charge  Amount',
    'Seconds of Use',
    'Frequency of use',
    'Frequency of SMS',
    'Distinct Called Numbers',
    'Age',
    'Customer Value'
]

# Categorical features
categorical_features = [
    'Tariff Plan'
]

# Preprocessing for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine both preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Task 5: Model Training 

model = Pipeline(steps=[
    ('preprocessor', preprocessor),

    ('classifier', LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    ))
])
model.fit(X_train, y_train)

print("Model training completed successfully!")


# Task 6: Model Evaluation and Accuracy assessment

y_prediction = model.predict(X_test)

X_test ['Predicted Churn'] = y_prediction

X_test ['Actual Churn'] = y_test.values

predicted_results = X_test[['Predicted Churn', 'Actual Churn']]

# Exporting the prediction results to a CSV file

X_test.to_csv('Customer_Churn_predictions.csv', index=False)


# print(prediction_results.head(10))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_prediction)

precision = precision_score(
    y_test, y_prediction, zero_division=0
)

recall = recall_score(
    y_test, y_prediction, zero_division=0
)

f1 = f1_score(
    y_test, y_prediction, zero_division=0
)

# Display results
print("MODEL EVALUATION RESULTS")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("Precision:", round(precision * 100, 2), "%")

print("Recall:", round(recall * 100, 2), "%")

print("F1 Score:", round(f1 * 100, 2), "%")