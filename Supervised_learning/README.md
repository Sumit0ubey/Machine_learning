                                                Titanic Survival Analysis and Prediction
                                                
![Stöwer_Titanic](https://github.com/Sumit0ubey/Machine_learning/assets/149804568/7b28866d-5575-4ff2-8b89-6fed253af662)

This project analyzes the Titanic dataset and builds machine learning models to predict the survival of passengers based on various features.

Table of Contents
Dataset
Project Structure
Installation
Usage
Results
Contributing
License
Dataset
The dataset used in this project is the Titanic - Machine Learning from Disaster dataset from Kaggle. The dataset contains information about passengers on the Titanic and whether they survived or not. Key features include:

Survived: Survival (0 = No, 1 = Yes)
Pclass: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
Name: Passenger name
Sex: Gender
Age: Age in years
SibSp: Number of siblings/spouses aboard the Titanic
Parch: Number of parents/children aboard the Titanic
Ticket: Ticket number
Fare: Passenger fare
Cabin: Cabin number
Embarked: Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
The dataset is included in this repository as titanic_train.csv.

Project Structure

├── titanic_train.csv     # Dataset file
├── titanic_analysis.py   # Main analysis and modeling script
├── README.md             # This readme file
└── requirements.txt      # Dependencies list


The script performs the following tasks:

Data Preprocessing:

Renames the Sex column to Gender.
Handles missing values in the Age and Embarked columns.
Converts categorical variables to numerical using dummy encoding.
Drops irrelevant columns like PassengerId, Name, Ticket, etc.
Exploratory Data Analysis:

Generates summary statistics.
Provides visualizations for understanding the distribution and relationship of features.
Displays correlation heatmaps, pairplots, and analysis plots for Survived, Pclass, Gender, Age, and Embarked.
Modeling:

Trains and evaluates several machine learning models:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
K-Nearest Neighbors Classifier
Uses GridSearchCV to optimize hyperparameters for DecisionTreeClassifier, RandomForestClassifier, and KNeighborsClassifier.
Results:

Prints classification reports for each model.
Compares model performance based on accuracy.
Outputs
Classification reports for each model.
Accuracy comparison of the models.
Visualization plots if enabled by user input.
Graphical Output
If prompted for graphical output, you can type yes to visualize:

Correlation heatmaps
Pairplots
Count plots for Survived, Pclass, Gender
Box plots for Age
