import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("/content/Telco_Customer_Churn_Dataset  (1).csv")


numeric_data = data.select_dtypes(include=['number'])
data[numeric_data.columns] = numeric_data.fillna(numeric_data.median())

label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

print("Data preparation completed. Preview:")
print(data.head())
from sklearn.model_selection import train_test_split

X = data.drop("Churn", axis=1)  
y = data["Churn"]             

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data split completed.")
print("Training set:",X_train.shape, "Testing set:",X_test.shape)
import numpy as np

correlation = data.corr()["Churn"].abs().sort_values(ascending=False)
selected_features = correlation[correlation > 0.1].index.drop("Churn")

X_train = X_train[selected_features]
X_test = X_test[selected_features]

print("Selected features:", list(selected_features))
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=42)

print("Model selection completed. Random Forest chosen.")

model.fit(X_train, y_train)

print("Model training completed.")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

print("Model Evaluation Results:")
print("Accuracy: " + str(round(accuracy, 2)))
print("Precision: " + str(round(precision, 2)))
print("Recall: " + str(round(recall, 2)))
print("F1 Score: " + str(round(f1, 2)))
print("ROC-AUC: " + str(round(roc_auc, 2)))
