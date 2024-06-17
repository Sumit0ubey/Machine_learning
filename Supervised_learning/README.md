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


This project analyzes the Titanic dataset to predict passenger survival using machine learning models. It includes exploratory data analysis (EDA), data preprocessing, model training, and evaluation. The dataset is sourced from Kaggle and contains information about Titanic passengers such as age, gender, class, and survival status.

Features
Dataset: titanic_train.csv includes columns like Survived, Pclass, Gender, Age, SibSp, Parch, Fare, Cabin, and Embarked.
Models: Implemented models include Logistic Regression, Decision Tree Classifier, Random Forest Classifier, and K-Nearest Neighbors.
Analysis: EDA involves visualizations like correlation heatmaps, pairplots, and histograms to understand feature distributions and relationships.
Model Evaluation: Utilizes GridSearchCV for hyperparameter tuning and evaluates models using classification_report for accuracy comparison.
Instal
