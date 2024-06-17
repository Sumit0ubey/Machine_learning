import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the Titanic dataset
TitanicData = pd.read_csv('titanic_train.csv')
TitanicData.rename(columns={'Sex': 'Gender'}, inplace=True)

# Summary of the dataset
print("\nBasic information on Titanic dataset: ")
TitanicData.info()
print("\nColumns Name: \n", TitanicData.columns, "\n")
print("Some data from that dataset: \n", TitanicData.head(), "\n")

# Short analysis with graphs
print("Some common stats on the dataset: \n", TitanicData.describe(), "\n")

permission = input("Do you want to see graphs (yes/no): ")
if permission.lower() in ['yes', 'y']:
    # Correlation heatmap
    sns.heatmap(TitanicData.corr(numeric_only=True), annot=True, cmap='viridis')
    plt.show()

    # Pairplot
    sns.pairplot(TitanicData)
    plt.show()

    # Null values heatmap
    sns.heatmap(TitanicData.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Null values in different columns')
    plt.show()

# Survived Column Analysis
print("-" * 10, " Survived Column analysis ", "-" * 10)
print("Value Count of: ")
print(TitanicData['Survived'].value_counts())
print("Mean of the Survived Column: ", TitanicData['Survived'].mean())
print()

if permission.lower() in ['yes', 'y']:
    plt.pie(TitanicData['Survived'].value_counts(), labels=['Not Survived', 'Survived'], colors=['red', 'green'],
            autopct='%1.1f%%', startangle=140)
    plt.title('Survived')
    plt.show()

# Pclass Column Analysis
print("-" * 10, " Pclass Column analysis ", "-" * 10)
print("Value Count of: ")
print(TitanicData['Pclass'].value_counts())
print()

if permission.lower() in ['yes', 'y']:
    pclass_counts = TitanicData['Pclass'].value_counts().sort_index()
    plt.bar(pclass_counts.index, pclass_counts.values, color=['blue', 'orange', 'green'])
    plt.title('Number of Passengers by Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.xticks(ticks=[1, 2, 3], labels=['1st Class', '2nd Class', '3rd Class'])
    plt.show()

    sns.countplot(data=TitanicData, x='Pclass', hue='Survived', palette=['red', 'green'])
    plt.title('Survival Rate by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.legend(labels=['Not Survived', 'Survived'])
    plt.show()

# Gender Column Analysis
print("-" * 10, " Gender Column analysis ", "-" * 10)
print("Value Count of: ")
print(TitanicData['Gender'].value_counts())
print()

if permission.lower() in ['yes', 'y']:
    gender_counts = TitanicData['Gender'].value_counts().sort_index()
    plt.bar(gender_counts.index, gender_counts.values, color=['blue', 'green'])
    plt.title('Number of Passengers by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
    plt.show()

    sns.countplot(data=TitanicData, x='Gender', hue='Survived', palette=['red', 'green'])
    plt.title('Survival Rate by Gender')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Female', 'Male'])
    plt.legend(labels=['Not Survived', 'Survived'])
    plt.show()

# Age Column Analysis
print("-" * 10, " Age Column analysis ", "-" * 10)
print(TitanicData['Age'])
print('Sum of Null Value in Age Column: ', TitanicData['Age'].isnull().sum())
print("Mean of Age: ", TitanicData['Age'].mean())

if permission.lower() in ['yes', 'y']:
    sns.boxplot(y='Age', data=TitanicData, orient='v')
    plt.title('Box plot on Age')
    plt.show()

# Filling missing values based on class
print("Filling missing values based on Pclass median values.")
TitanicData['Age'] = TitanicData.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))

if permission.lower() in ['yes', 'y']:
    sns.boxplot(x='Pclass', y='Age', data=TitanicData, palette='viridis')
    plt.title('Age Distribution by Passenger Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Age')
    plt.show()
print()

# Embarked Column Analysis
print("-" * 10, " Embarked Column analysis ", "-" * 10)
print("Value Count of: ")
print(TitanicData['Embarked'].value_counts())
print("Number of null values in Embarked: ", TitanicData['Embarked'].isnull().sum())
print()

if permission.lower() in ['yes', 'y']:
    embarked_counts = TitanicData['Embarked'].value_counts()
    sns.barplot(x=embarked_counts.index, y=embarked_counts.values, palette='viridis')
    plt.title('Number of unique Values in Embarked')
    plt.show()

# Fill missing values with the most common value
TitanicData['Embarked'] = TitanicData['Embarked'].fillna('S')

# Data Preprocessing
# Convert categorical variables to dummy variables
TitanicData = pd.get_dummies(TitanicData, columns=['Gender', 'Embarked'], drop_first=True)

# Drop less relevant columns
TitanicData.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], inplace=True, axis=1)
print("Cleaned dataset of Titanic: ")
print(TitanicData.info())
print()

# Splitting the dataset
X = TitanicData.drop('Survived', axis=1)
y = TitanicData['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
logistic_regression = LogisticRegression(max_iter=1000)
logistic_regression.fit(X_train, y_train)
log_y_predict = logistic_regression.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, log_y_predict))

# Decision Tree Classifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)
print("Decision Tree:\n", classification_report(y_test, y_predict))

# Grid Search for Decision Tree
search_options = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None] + list(range(5, 10)),
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': [None, 'sqrt', 'log2'],  # Removed 'auto'
    'max_leaf_nodes': [None] + list(range(5, 10)),
    'min_impurity_decrease': [0.0, 0.1],
}

grid_search_dtc = GridSearchCV(estimator=dtc, param_grid=search_options, cv=5)
grid_search_dtc.fit(X_train, y_train)
best_dtc = grid_search_dtc.best_estimator_
y_pred_dtc = best_dtc.predict(X_test)
print("Grid Search Decision Tree:\n", classification_report(y_test, y_pred_dtc))

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rfc))

# Grid Search for Random Forest
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 2, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Grid Search Random Forest:\n", classification_report(y_test, y_pred_rf))

# K-Nearest Neighbors
knn = KNeighborsClassifier()

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)
print("K-Nearest Neighbors:\n", classification_report(y_test, y_pred_knn))

# Grid Search for K-Nearest Neighbors
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
grid_search_knn = GridSearchCV(estimator=knn, param_grid=param_grid_knn, cv=5)
grid_search_knn.fit(X_train_scaled, y_train)
best_knn = grid_search_knn.best_estimator_
y_pred_best_knn = best_knn.predict(X_test_scaled)
print("Grid Search K-Nearest Neighbors:\n", classification_report(y_test, y_pred_best_knn))

# Model Comparison
models = ['Logistic Regression', 'Decision Tree', 'Grid Search Decision Tree', 'Random Forest',
          'Grid Search Random Forest', 'K-Nearest Neighbors', 'Grid Search K-Nearest Neighbors']
accuracy_scores = [
    logistic_regression.score(X_test, y_test),
    dtc.score(X_test, y_test),
    best_dtc.score(X_test, y_test),
    rfc.score(X_test, y_test),
    best_rf.score(X_test, y_test),
    knn.score(X_test_scaled, y_test),
    best_knn.score(X_test_scaled, y_test)
]

print("\nComparison of Models:")
for model, score in zip(models, accuracy_scores):
    print(f"{model}: {score:.2f}")
print()

# Plot comparison of models
permission = input("Do you want to see graphs on comparison of models (yes/no): ")
if permission.lower() in ['yes', 'y']:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=models, y=accuracy_scores, palette='viridis', hue=models)
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.show()
