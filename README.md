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

  #### Kidults
  
  * Kidults refers to grown-ups who still are consumers of toys.
  * Kidult culture is rising as well as their power in the market.
  * A marketing survey designed by myself to measure the kidults' satisfaction level on Lego's marketing efforts(product, price, place and promotion) had been conducted. 
  
  #### Survey Result
 
  ![image](https://user-images.githubusercontent.com/46237445/50714185-1512e780-10bb-11e9-84db-278e1b8f0f39.png)

  * Respondents' level of satisfaction on the product was quite fair given other areas and the average.
  * However, they expressed a strong dissatisfaction on price-related marketing efforts of Lego with the score of 2.3 out of 5, which is notably lower than the average score of 3.17.
  * This implies the gap between the *percerived value* and *the price*, probably caused by some *outliers* or unreasonably priced product set.

  #### The Goal of the Project
  Decrease the gap between the *percerived value* and *the price* by creating a model that is not or less affected by *outliers*.
    
  ### 2. Data Description
  > Please refer to '~/Final Project/Lego Set Price Prediction_Final_Team2.pdf' for the detailed project description.
  * Dataset
    * ["Are Lego Sets Too Pricey?", Jonathan Bouchet, Kaggle](https://www.kaggle.com/jonathanbouchet/are-lego-sets-too-pricey/data)
    * 12261 rows with 14 columns
    * The dataset does not include the entire product lines. 
  
  * Features
    * Numerical Features
      * list_price **[target]**
      * num_reviews
      * piece_count
      * play_star_rating
      * star_rating
      * var_star_rating
      * prod_id
    * Categorical Features
      * ages
      * theme_name
      * review_difficulty
      * country
      * prod_desc
      * prod_log_desc
      * set_name

  ### 3. Exploratory Data Analysis (EDA)
  
  * Histogram, Barplot and Swarmplot
    * 'ages' and 'list_price' are positively correlated.
    * 'review_difficulty'and 'list_price' are positively correlated.
    * There are considerable number of outliers.
  * Correlation
    * 'piece_count' highly correlated with the 'list_price' positively.
    * Rating-related features are correlated with one another.

  ### 4. Data Preprocessing

  * Remove Unnecessary Features
    * Product Descriptions (prod_desc & prod_log_desc)
    * Product ID (prod_id)
    * Set Names (set_name)
  * Categorical Value Encoding
    * Cateogorical values were one-hot encoded.  
    ![image](https://user-images.githubusercontent.com/46237445/50728223-cc663780-1169-11e9-8702-c0018bbf277a.png)
  * Grouping of Values
    * Country Names (country)
    * Theme Names (theme_name)
    * Age Range (ages)
  * Outlier Removal
    * Outliers detected in the EDA stage has been removed with 'IsolationForest' from 'sklearn'  
    ![image](https://user-images.githubusercontent.com/46237445/50728236-02a3b700-116a-11e9-839e-0555945d1694.png)
  * Standardization
    * Numerical features has been normalized with 'StandardScaler' from 'sklearn'.
  * Multi-collinearity Problem
    * VIF has greatly been stabilized due to the standardization.  
      ![image](https://user-images.githubusercontent.com/46237445/50728231-e30c8e80-1169-11e9-8581-7c146702c656.png)
  * Data Split (Hold-Out)
    * The ratio of the train, validation, and test sets was 60%: 20%: 20%.
  
  ### 5. Model Fitting and Evaluation
  * Models
    * Model with 94 features (full model)
    * Model with 22 features (reduced model 1)
      * Grouping of the features.
    * Model with 3 features by PCA (reduced model 2)
      * Scree Plot
      ![image](https://user-images.githubusercontent.com/46237445/50733642-6d89d800-11d4-11e9-85e8-d387b9960034.png)

      
  * Ordinary Least Squares (OLS)
    * In Multiple Linear Regression (MLR) by OLS, R Squared, Adj. R Squared, F-statistics, and p-value were fair for all three models, though ones with more features displayed higher explanatory power. 
    * Mean squared errors tend to be greater than that of neural network model. 
  * Neural Network (Tensorflow)
    * 2 hidden layers, 50 inputs between hidden layers, drop-out rate of 0.7, and 10,000 iterations.
    * Better performance compared to OLS in general.
  * Least Absolute Shrinkage and Selection Operator (LASSO)
    * Best performance where alpha equals zero.  
    ![image](https://user-images.githubusercontent.com/46237445/50729287-a4cb9b00-117a-11e9-8f5c-ef0bbd1f804a.png)

    * No need for regularization.
  
* Model Summary
  * Mean Squared Error  
    Number of Features | OLS | NN | 
      :---:| :---: | :---: |
      94 | 0.0821 | 0.0331 |
      22 | 0.1595 | 0.0492 |
      3 | 0.1022 | 0.0521 |

  * Efficiency-Accuracy Tradeoff
    * Full model performs better.
    * Full model recommended given sufficient computing power and time.

  ### 6. Conclusion

  #### Significance
  * Setting a basis of Lego Set Price
  * Beneficial to both the customers and the manufacturers
  * Applicable to all consumer goods
  
  #### Limitaions
  * Data incompleteness of not covering all product lines, countries, and etc.
  * Better modeling will be possible with better dataset. 
  
