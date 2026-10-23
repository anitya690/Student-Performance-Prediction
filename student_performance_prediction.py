# Student Performance Prediction
# Dataset: Student Performance Factors (Kaggle)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Read dataset
df = pd.read_csv('StudentPerformanceFactors.csv')

print(df.head())
print(df.info())
print(df.describe())
print('\nMissing values:')
print(df.isnull().sum())
print('\nDuplicate rows:', df.duplicated().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].median())

# OBJECTIVE 1: Study the distribution of exam scores
print('\nOBJECTIVE 1: Exam score distribution')
print('Mean:', df['Exam_Score'].mean())
print('Median:', df['Exam_Score'].median())
print('Minimum:', df['Exam_Score'].min())
print('Maximum:', df['Exam_Score'].max())
print('Standard deviation:', df['Exam_Score'].std())

plt.hist(df['Exam_Score'], bins=20, edgecolor='black')
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Number of Students')
plt.show()

plt.boxplot(df['Exam_Score'])
plt.title('Boxplot of Exam Scores')
plt.ylabel('Exam Score')
plt.show()

# OBJECTIVE 2: Study habits and exam performance
print('\nOBJECTIVE 2: Study habits and exam performance')
study_columns = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Physical_Activity']
print(df[study_columns + ['Exam_Score']].corr(numeric_only=True))

for column in study_columns:
    plt.scatter(df[column], df['Exam_Score'])
    plt.title(column + ' vs Exam Score')
    plt.xlabel(column)
    plt.ylabel('Exam Score')
    plt.show()

# OBJECTIVE 3: Previous scores and current exam scores
print('\nOBJECTIVE 3: Previous scores and current exam scores')
print('Correlation:', df['Previous_Scores'].corr(df['Exam_Score']))
plt.scatter(df['Previous_Scores'], df['Exam_Score'])
plt.title('Previous Scores vs Current Exam Score')
plt.xlabel('Previous Scores')
plt.ylabel('Exam Score')
plt.show()

# OBJECTIVE 4: Educational support factors and exam performance
print('\nOBJECTIVE 4: Educational support factors and exam performance')
support_columns = ['Parental_Involvement', 'Access_to_Resources', 'Teacher_Quality', 'Tutoring_Sessions']
for column in support_columns:
    print('\nAverage score by', column)
    print(df.groupby(column)['Exam_Score'].mean().sort_values(ascending=False))
    df.groupby(column)['Exam_Score'].mean().plot(kind='bar')
    plt.title('Average Exam Score by ' + column)
    plt.ylabel('Average Exam Score')
    plt.show()

# OBJECTIVE 5: Learning environment and exam performance
print('\nOBJECTIVE 5: Learning environment and exam performance')
environment_columns = ['Internet_Access', 'School_Type', 'Family_Income', 'Peer_Influence', 'Motivation_Level']
for column in environment_columns:
    print('\nAverage score by', column)
    print(df.groupby(column)['Exam_Score'].mean().sort_values(ascending=False))
    df.groupby(column)['Exam_Score'].mean().plot(kind='bar')
    plt.title('Average Exam Score by ' + column)
    plt.ylabel('Average Exam Score')
    plt.show()

# Machine learning: create performance groups
# These are project-defined groups, not official grades.
df['Performance_Level'] = pd.cut(
    df['Exam_Score'],
    bins=[-np.inf, 64, 69, np.inf],
    labels=['Low', 'Medium', 'High']
)
print('\nPerformance level counts:')
print(df['Performance_Level'].value_counts())

# Separate input and output columns
X = df.drop(['Exam_Score', 'Performance_Level'], axis=1)
y = df['Performance_Level']

# Convert text columns into numbers using Label Encoding
le = LabelEncoder()

for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = le.fit_transform(X[column].astype(str))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Scale the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# MODEL 1: Logistic Regression
print("\nMODEL 1: Logistic Regression")
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
y_pred1 = model1.predict(X_test)
accuracy1 = accuracy_score(y_test, y_pred1)
print("Accuracy:", round(accuracy1, 3))
print("Classification Report:")
print(classification_report(y_test, y_pred1, zero_division=0))


# MODEL 2: K-Nearest Neighbors (KNN)
print("\nMODEL 2: K-Nearest Neighbors")
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
accuracy2 = accuracy_score(y_test, y_pred2)
print("Accuracy:", round(accuracy2, 3))
print("Classification Report:")
print(classification_report(y_test, y_pred2, zero_division=0))


# MODEL 3: Gaussian Naive Bayes
print("\nMODEL 3: Gaussian Naive Bayes")
model3 = GaussianNB()
model3.fit(X_train, y_train)
y_pred3 = model3.predict(X_test)
accuracy3 = accuracy_score(y_test, y_pred3)
print("Accuracy:", round(accuracy3, 3))
print("Classification Report:")
print(classification_report(y_test, y_pred3, zero_division=0))


# MODEL 4: Random Forest
print("\nMODEL 4: Random Forest")
model4 = RandomForestClassifier(n_estimators=100, random_state=42)
model4.fit(X_train, y_train)
y_pred4 = model4.predict(X_test)
accuracy4 = accuracy_score(y_test, y_pred4)
print("Accuracy:", round(accuracy4, 3))
print("Classification Report:")
print(classification_report(y_test, y_pred4, zero_division=0))


# MODEL COMPARISON
print("\\nMODEL COMPARISON")
print("Logistic Regression Accuracy:", round(accuracy1, 3))
print("KNN Accuracy:", round(accuracy2, 3))
print("Gaussian Naive Bayes Accuracy:", round(accuracy3, 3))
print("Random Forest Accuracy:", round(accuracy4, 3))

# Select the best model using accuracy
best_accuracy = max(accuracy1, accuracy2, accuracy3, accuracy4)

if best_accuracy == accuracy1:
    best_name = "Logistic Regression"
    best_predictions = y_pred1
elif best_accuracy == accuracy2:
    best_name = "K-Nearest Neighbors"
    best_predictions = y_pred2
elif best_accuracy == accuracy3:
    best_name = "Gaussian Naive Bayes"
    best_predictions = y_pred3
else:
    best_name = "Random Forest"
    best_predictions = y_pred4

print("\\nBest model:", best_name)
print("\\nClassification Report:")
print(classification_report(y_test, best_predictions, zero_division=0))

print("Confusion Matrix:")
print(confusion_matrix(y_test, best_predictions))

print("\\nModel comparison is complete.")
