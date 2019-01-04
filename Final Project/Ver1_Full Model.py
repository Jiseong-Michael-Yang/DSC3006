
# coding: utf-8

# # 1. Data Loading

# In[1]:


# Import required modules
import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import tensorflow as tf


# In[2]:


# Set the working directory.
wd = r"C:\Users\Jiseong Yang\git_projects\DSC3006\Final Project"
wd = wd.replace("'\'", "/")
os.chdir(wd)
os.getcwd()


# In[3]:


# Read in the dataset
lego = pd.read_csv("lego_sets.csv")
lego = pd.DataFrame(lego, columns = lego.columns)

# Set display options
pd.set_option("display.max_columns", 14)
pd.set_option("display.width", 500)

# Data overview
print("data shape:", lego.shape)
lego.describe()


# In[4]:


# Check the missing values
True in lego.isnull()

# Replace NaN with 0
lego.dropna(axis=0, how="any", inplace = True)

# Drop unnecessary features
feat_dis = ["prod_desc", "prod_id", "prod_long_desc", "set_name"]
lego.drop(feat_dis, axis = 1, inplace = True)

# Create a list of features depending on their types.
feat_num = ["num_reviews", "piece_count", "play_star_rating", "star_rating", 
            "val_star_rating", "list_price"]
feat = lego.columns
feat_cat = list(set(feat) - set(feat_num))
feat_cat.sort()


# In[5]:


print(feat_num)
print(feat_cat)


# # 2. Exploratory Data Analysis (EDA)

# ## 2.1 Histograms

# In[37]:


# For a single categorical feature
for i in feat_cat[:3]:    
    plt.pyplot.figure(figsize=(20,3))
    sns.countplot(lego[i])
    plt.pyplot.title("Frequency Table for " + i)
    plt.pyplot.show()


# ## 2.2 Barplots

# In[38]:


# A categorical feature vs. target (price)
for i in feat_cat[:3]:
    plt.pyplot.figure(figsize=(20,3))
    plt.pyplot.bar(lego[i],lego["list_price"])
    plt.pyplot.xlabel(i)
    plt.pyplot.ylabel("Price")
    plt.pyplot.title("Price vs. " + i[0].upper()+i[1:], fontsize = 16)
    plt.pyplot.show()


# ## 2.3 Boxplots

# In[39]:


# A categorical feature vs. target (price)
for i in feat_cat[:3]:
    plt.pyplot.figure(figsize=(20,3))
    sns.boxplot(lego[i],lego["list_price"])
    plt.pyplot.suptitle("Price vs. " + i[0].upper()+i[1:], fontsize = 16)
    plt.pyplot.show()


# ## 2.4 Swarmplots

# ### Caution: plotting swarmplots is extremely time-consuming.

# In[40]:


# A categorical feature vs. target (price)
#for i in feat_cat[:3]:
#    plt.pyplot.figure(figsize=(20,3))
#    sns.swarmplot(lego[i],lego["list_price"])
#    plt.pyplot.suptitle("Price vs. " + i[0].upper()+i[1:], fontsize = 16)
#    plt.pyplot.show()


# ## 2.5 Correlation

# In[6]:


# Pairplot
lego[feat_num]
plt.pyplot.figure(figsize=(5,5))
pairplot_num = sns.pairplot(data = lego[feat_num])
pairplot_num.fig.subplots_adjust(top=0.9)
pairplot_num.fig.suptitle("Correlations Pairplots(Numerical Values)", 
                          fontsize = 16)

# Heatmap
plt.pyplot.figure()
ax = plt.pyplot.axes()
cm_num = np.corrcoef(lego[feat_num].values.T)
hm_num = sns.heatmap(cm_num, cbar=True, annot=True, xticklabels=feat_num, 
                 yticklabels=feat_num, ax = ax)
ax.set_title("Correlations Heatmap (Numerical Values)", fontsize =16)


# # 3. Preprocessing

# In[7]:


# One-hot encoding. 
lego_dummies = pd.get_dummies(lego[feat_cat], drop_first=True)

# Scaling
scaler = StandardScaler()
lego[feat_num] = lego[feat_num].apply(pd.to_numeric)
lego_num_scaled = pd.DataFrame(scaler.fit_transform(lego[feat_num]), 
                               columns=feat_num)

# Finalize the dataset
lego_processed = pd.concat([lego_dummies.reset_index(drop=True), 
                            lego_num_scaled.reset_index(drop=True)], axis = 1)

# Define the input and output features
feat_ind = lego_processed.columns[lego_processed.columns != "list_price"]
feat_dep = lego_processed.columns[lego_processed.columns == "list_price"]


# In[8]:


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(lego_processed[feat_ind], lego_processed[feat_dep], test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .25, shuffle=True)
X_train, X_val, X_test, y_train, y_val, y_test = np.float32(X_train.values), np.float32(X_val.values),                                                 np.float32(X_test.values), np.float32(y_train.values),                                                 np.float32(y_val.values), np.float32(y_test.values)


# # 4. Regression Analysis

# ## 4.1 Multi-collinearity Problem

# ### 4.1.1 VIF(Variance Inflation Factor) before and after scaling

# In[9]:


# Define a function to report VIF
def vif_report(data):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    vif["features"] = feat_num[:-1]
    return vif
    print("VIF Report")
    return vif

# Data before and after scaling
data_input = lego[feat_num[:-1]]
data_input_scaled = lego_processed[feat_num[:-1]]

# Print out the report
for data in [data_input, data_input_scaled]:
    print(vif_report(data))
    print()


# ### 4.1.2 Correlation after Scaling

# In[10]:


# Pairplot
lego[feat_num]
plt.pyplot.figure(figsize=(5,5))
pairplot_num = sns.pairplot(data = lego_processed[feat_num])
pairplot_num.fig.subplots_adjust(top=0.9)
pairplot_num.fig.suptitle("Correlations Pairplots(Numerical Values)", 
                          fontsize = 16)

# Heatmap
plt.pyplot.figure()
ax = plt.pyplot.axes()
cm_num = np.corrcoef(lego_processed[feat_num].values.T)
hm_num = sns.heatmap(cm_num, cbar=True, annot=True, xticklabels=feat_num, 
                 yticklabels=feat_num, ax = ax)
ax.set_title("Correlations Heatmap (Numerical Values)", fontsize =16)


# ## 4.2 OLS(Ordinary Least Squares) Report

# In[11]:


# Fit the OLS model
reg = sm.OLS(lego_processed[feat_dep], lego_processed[feat_ind], data = lego_processed)
def reg_report(mod):
    return mod.fit().summary()
    
# View OLS report
reg_report(reg)


# # 5. Model Fitting and Testing

# ## 5.1 Plain Linear Regression

# In[12]:


# Fit the model
mlr = LinearRegression()
mlr.fit(X_train, y_train)

# Prediction
y_pred_val = mlr.predict(X_val)
y_pred_test = mlr.predict(X_test)

# Define a function to return MSE
def get_mse(y_pred, y_actual):
    return mean_squared_error(y_actual, y_pred)
    
# Define a functdion to create a residual plot with a MSE report
def residual_plot(y_pred, y):  
    plt.pyplot.figure()
    plt.pyplot.scatter(y_pred, y_pred - y, c='red', marker='o', label='Residual Plot', alpha = .3)
    plt.pyplot.xlabel("Predicted Values")
    plt.pyplot.ylabel("Residuals")
    plt.pyplot.axhline(y=0, xmin = -10, xmax = 10, color='black')
    plt.pyplot.title("Residual Plot")
    plt.pyplot.show()
    print("MSE: " , get_mse(y_pred, y))

# Plot the residuals and calculate mean square error
residual_plot(y_pred_val, y_val)
residual_plot(y_pred_test, y_test)


# ## 5.2 Least Absolute Shrinkage and Selection Operator (LASSO)

# In[13]:


# Define a function to draw a line
def draw_line(marker):
    for i in range(50):
            print(marker, end="")
    print()
    
# Define a function to perform LASSO regression
def lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test, alpha, plot):
    # Fit the model
    lasso = Lasso(alpha = alpha)
    lasso.fit(X_train, y_train)
    
    # Make predictions for validation and test sets
    y_pred_lasso_val = lasso.predict(X_val)
    y_pred_lasso_test = lasso.predict(X_test)
    
    # Return MSE and residual plots for validation and test sets
    y_pred_lasso_val = np.reshape(y_pred_lasso_val, (len(y_val), 1)) 
    y_pred_lasso_test = np.reshape(y_pred_lasso_test, (len(y_test), 1))
    
    # Return the output
    if plot == "T":
        draw_line("=")
        print("Residual Plots of the Validation set & Test set\n", "- alpha: ", alpha)
        return residual_plot(y_pred_lasso_val, y_val), residual_plot(y_pred_lasso_test, y_test),             get_mse(y_pred_lasso_val, y_val), get_mse(y_pred_lasso_test, y_test)
    elif plot == "F":
        return get_mse(y_pred_lasso_val, y_val), get_mse(y_pred_lasso_test, y_test)


# In[14]:


# Plot the MSE by alpha value to find the optimal alpha
mse_val_history = []
mse_test_history = []
alpha_list = [x * 0.1 for x in range(10)]    
for alpha in alpha_list:
    mse_val, mse_test = lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test, alpha, "F")[:2]    
    mse_val_history.append(mse_val)
    mse_test_history.append(mse_test)

# Plot for the validtaion set
plt.pyplot.figure()
plt.pyplot.plot(alpha_list, mse_val_history)
plt.pyplot.xlabel("alpha value")
plt.pyplot.ylabel("MSE")
plt.pyplot.title("MSE by Alpha Value with the Vadlidation Set")
plt.pyplot.show()

# Plot for the test set
plt.pyplot.figure()
plt.pyplot.plot(alpha_list, mse_test_history)
plt.pyplot.xlabel("alpha value")
plt.pyplot.ylabel("MSE")
plt.pyplot.title("MSE by Alpha Value with the Test Set")
plt.pyplot.show()

# Perform a LASSO regression with optimal alpha
lasso_regression(X_train, X_val, X_test, y_train, y_val, y_test, 0, "T")


# ## 5.3 Neural Network with Tensorflow

# In[15]:


# Assign placeholders
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 94])
Y = tf.placeholder(tf.float32, [None, 1])

# Set parameters for layers
num_in = 94
num_hidden = 50
num_out = 1
keep_prob = 0.7
        
# Layer 1
W1 = tf.get_variable("weight1", shape=[num_in,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("bias1", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)
L1 =tf.nn.dropout(_L1, keep_prob)

# Layer 2
W2 = tf.get_variable("weight2", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("bias2", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L2 = tf.nn.leaky_relu(tf.matmul(L1, W2) + b2)
L2 =tf.nn.dropout(_L2, keep_prob)
 
# Layer 3
W3 = tf.get_variable("weight3", shape=[num_hidden,num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("bias3", shape=[num_hidden], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
_L3 = tf.nn.leaky_relu(tf.matmul(L2, W3) + b3)
L3 = tf.nn.dropout(_L3, keep_prob)

# Layer 4
W4 = tf.get_variable("weight4", shape=[num_hidden,num_out], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("bias4", shape=[num_out], dtype = tf.float32,
                     initializer=tf.contrib.layers.xavier_initializer())
hypothesis = tf.add(tf.matmul(L3, W4), b4)


# In[16]:


# Hyperparameters
learning_rate = 0.003
keep_prob = tf.placeholder_with_default(0.7, shape=())
epochs = 10000

# Cost function, optimizer and cost history list for visualization
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)
cost_history = np.empty(shape=[1], dtype=float)

# Initializer
init = tf.global_variables_initializer()


# In[17]:


# Close the existing session
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()

# Launch the Graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(init)
    # Starts the optimization
    print()
    print("Starts learning...")
    print()
    for step in range(epochs+1):
        sess.run(train,feed_dict={X: X_train, Y: y_train})
        cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: X_train, Y: y_train}))
        if step % 100 == 0:
                print("step ",step, "cost ", sess.run(cost, feed_dict={
                      X: X_train, Y: y_train}))
    
    # Plot the cost by ilteration
    plt.pyplot.figure()
    plt.pyplot.plot(range(len(cost_history)),cost_history)
    plt.pyplot.axis([0,epochs,0,np.max(cost_history)])
    plt.pyplot.title("Cost by Iterations")
    plt.pyplot.show() 

    # Plot the model accuracy with validation set
    fig, ax = plt.pyplot.subplots()
    ax.scatter(y_val, y_pred_val)
    ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'k--', lw=3)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Actual vs. Predicted with Validation Sets")
    plt.pyplot.show()
    
    # Report the MSE with validation set
    y_pred_val = sess.run(hypothesis, feed_dict={X: X_val})
    mse = tf.reduce_mean(tf.square(y_pred_val - y_val))
    print("MSE: %.4f" % sess.run(mse))

    # Plot the model accuracy with test set
    fig, ax = plt.pyplot.subplots()
    ax.scatter(y_test, y_pred_test)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'b--', lw=2)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Actual vs. Predicted with Test Sets")
    plt.pyplot.show()
    
    # Report the MSE with test set
    y_pred_test = sess.run(hypothesis, feed_dict={X: X_test})
    mse = tf.reduce_mean(tf.square(y_pred_test - y_test))
    print("MSE: %.4f" % sess.run(mse)) 

