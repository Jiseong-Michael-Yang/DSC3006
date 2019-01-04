# Introduction to Machine Learning (DSC3006)
Assignments and project in Introduction to Machine Learning (Fall 2018)

## 1. Assignments

### 1. Assignment 1: Kaggle InClass Competition

  * Goal  
  The goal of assignment 1 is to practice data preprocessing and classification through a Kaggle InClass Competition. You are expected to understand how Kaggle works and how you can improve your classification model’s performance.
  
  * Task  
  You are provided with a classification dataset and your task is to build a series of models with the goal of improving the performance. You can use any data preprocessing technique and classification method.

  * Data description
    * A detailed description is available at https://archive.ics.uci.edu/ml/datasets/student+performance.
    * The dataset used in the assignment 1 is a slightly modified version with 30 features and 1 categorical target variable. The goal is to use the 30 features and classify each student into one of the FIVE categories.
      * X_train.csv: 264 samples, 30 features (Id should not be counted as a feature)
      * y_train.csv: 264 samples, 1 target (from 1 to 5, each number represents a category)
      * X_test.csv: 131 samples, 30 features (the dataset you test your model)
      
> Please refer to "~/Assignment1/Assignment1.pdf" for the datailed report.   

### 2. Assignment 2: Ensemble Learning

  * Goal  
    The goal of assignment 2 is to practice ensemble learning.
  
  * Task
    * Load the breast cancer wisconsin dataset from sklearn.
      ```
      from sklearn.datasets import load_breast_cancer
      data = load_breast_cancer()
      ```
    
    * Perform 5-fold cross validation, bagging and boosting.
    * Visualize the result.
    * Compare each methods.
    
 > Please refer to "~/Assignmnet2/Assignment2.pdf" for the datailed report. 

## 2. Project: Lego Set Price Prediction (Multiple Linear Regressions & Neural Network)

  ### 1. Introduction
  
  ### 2. Data Description
  
  ### 3. Exploratory Data Analysis (EDA)
  
  ### 4. Preprocessing
  
  ### 5. Model Fitting and Evaluation
  
  ### 6. Conclusion
