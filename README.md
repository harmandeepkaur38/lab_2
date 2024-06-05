harmandeepakaur_model_v2
# Import necessary libraries
import pandas as pd
import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Logisitc regression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target
# Set column names
columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data.columns = columns

# Drop the ID column
data = data.drop(columns=['ID'])

# Encode the Diagnosis column
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# Split data into features and target variable
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Spliting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=40)
# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#  Initialize Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier and logistic regression 
rf_classifier.fit(X_train, y_train)
model = LogisticRegression()
model.fit(X_train, y_train)

#  Make predictions on the test set
predictions = rf_classifier.predict(X_test)

#  Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Random Forest Classifier Accuracy:", accuracy)# Predict the test set results
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print('-'* 90)
Accuracy: 0.9824561403508771
Classification Report:
               precision    recall  f1-score   support

           0       0.98      0.99      0.99       186
           1       0.98      0.97      0.97        99

    accuracy                           0.98       285
   macro avg       0.98      0.98      0.98       285
weighted avg       0.98      0.98      0.98       285

# Print classification report and confusion matrix for more detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
Random Forest Classifier Accuracy: 0.9649122807017544

Classification Report:
              precision    recall  f1-score   support

           0       0.98      0.93      0.95        43
           1       0.96      0.99      0.97        71

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114


Confusion Matrix:
[[40  3]
 [ 1 70]]
harmandeepkaur_model_v2
