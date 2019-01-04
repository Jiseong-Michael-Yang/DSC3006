
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
from sklearn import decomposition
from sklearn.linear_model import Lasso
from sklearn.ensemble import IsolationForest
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


# Define lists to simplify country features by grouping their labels.
europe = ["AT", "BE", "CH", "CZ", "DE", "DN", "ES", "FI", "FR", "GB", "IE", "IT", "LU", "NL", "DN", "NO", "PT", "PL"]
oceania = ["AU", "NZ"]
north_america = ["CA", "US"]
country_list = [europe, oceania, north_america]
country_str = ["europe", "oceania", "north_america"]

# Replace the labels
for i in range(len(country_list)):
        lego.replace(country_list[i], country_str[i], inplace = True)

# Check the labels
np.unique(lego.country)


# In[6]:


# Define lists to simplify the theme features by grouping their labels.
licensed = ['Indoraptor Rampage at Lockwood Estate', 'Stygimoloch Breakout','Pteranodon Chase', 'Dilophosaurus Outpost Attack',
            'Carnotaurus Gyrosphere Escape', 'Jurassic Park Velociraptor Chase', 'Jurassic World™','Star Wars™', 'Architecture', 
            'BrickHeadz', 'Disney™', 'Ideas','Ghostbusters™','Minecraft™','DC Comics™ Super Heroes','DC Super Hero Girls', 
            'DIMENSIONS™']
non_licensed = ['Power Functions', 'BOOST','City', 'Classic', 'Creator 3-in-1', 'Creator Expert', 'Elves', 'Friends', 'MINDSTORMS®', 
                'Minifigures', 'NINJAGO®', 'THE LEGO® NINJAGO® MOVIE™','NEXO KNIGHTS™', 'Technic', 'LEGO® Creator 3-in-1']
hybrid = ['THE LEGO® BATMAN MOVIE', 'DUPLO®', 'Juniors','Speed Champions', 'Marvel Super Heroes',]
theme_list = [licensed,  non_licensed, hybrid]
theme_str = ['liscensed', 'non_licensed', 'hybrid']

# Replace the labels
for i in range(len(theme_list)):
    lego.replace(theme_list[i], theme_str[i], inplace = True)

# Drop the discontinued theme    
index_disc = lego[lego.theme_name == "Angry Birds™"].index
lego.drop(index_disc, axis = 0, inplace = True)
np.unique(lego.theme_name)


# In[7]:


# Define lists to simplify the age features by grouping their labels.
np.unique(lego.ages)
under_two = ['1½-3', '1½-5']
over_two_four = ['2-5', '4+', '4-7', '4-99']
over_five = ['5+', '5-12', '5-8']
over_six = ['6+', '6-12', '6-14']
over_seven = ['7+', '7-12', '7-14']
over_eight= ['8+', '8-12', '8-14']
over_nine = ['9+', '9-12', '9-14', '9-16']
over_ten = ['10+', '10-14', '10-16']
over_eleven_twelve = ['11-16', '12+', '12-16']
over_fourteen_sixteen = ['14+', '16+']
age_list = [under_two, over_two_four, over_five, over_six, over_seven, over_eight, over_nine, over_ten, over_eleven_twelve, over_fourteen_sixteen]
age_str = ["under_two", "over_two_four", "over_five", "over_six", "over_seven", "over_eight", "over_nine", "over_ten", "over_eleven_twelve", "over_fourteen_sixteen"]

# Replace the labels
for i in range(len(age_list)):
    lego.replace(age_list[i], age_str[i], inplace = True)
np.unique(lego.ages)    


# # 2. Exploratory Data Analysis (EDA)

# ## 2.1 Histograms

# In[8]:


# For a single categorical feature
for i in feat_cat:    
    plt.pyplot.figure(figsize=(20,3))
    sns.countplot(lego[i])
    plt.pyplot.title("Frequency Table for " + i)
    plt.pyplot.show()


# ## 2.2 Barplots

# In[9]:


# A categorical feature vs. target (price)
for i in feat_cat:
    plt.pyplot.figure(figsize=(20,3))
    sns.barplot(lego[i],lego["list_price"], ci=None)
    plt.pyplot.xlabel(i)
    plt.pyplot.ylabel("Price")
    plt.pyplot.title("Price vs. " + i[0].upper()+i[1:], fontsize = 16)
    plt.pyplot.show()


# ## 2.3 Boxplots

# In[10]:


# A categorical feature vs. target (price)
for i in feat_cat:
    plt.pyplot.figure(figsize=(20,3))
    sns.boxplot(lego[i],lego["list_price"])
    plt.pyplot.suptitle("Price vs. " + i[0].upper()+i[1:], fontsize = 16)
    plt.pyplot.show()


# ## 2.4 Swarmplots

# ### Caution: plotting swarmplots is extremely time-consuming.

# In[ ]:


# A categorical feature vs. target (price)
#for i in feat_cat:
#    plt.pyplot.figure(figsize=(20,3))
#    sns.swarmplot(lego[i],lego["list_price"])
#    plt.pyplot.suptitle("Price vs. " + i[0].upper()+i[1:], fontsize = 16)
#    plt.pyplot.show()


# ## 2.5 Correlation

# In[11]:


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

# ## 3.1 Categorical Value Encoding, Outlier Removal, and Standardization

# In[12]:


# One-hot encoding. 
lego_dummies = pd.get_dummies(lego[feat_cat], drop_first=True)

# Combine the dataset
lego_processed = pd.concat([lego_dummies.reset_index(drop=True), 
                            lego[feat_num].reset_index(drop=True)], axis = 1)


# In[13]:


# Remove the outliers
clf = IsolationForest(max_samples = "auto", random_state = 1)
clf.fit(lego_processed)
y_pred_outliers = clf.predict(lego_processed)
outliers = pd.DataFrame(y_pred_outliers)
outleirs = outliers.rename(columns={0: 'out'})
lego_processed = pd.concat([lego_processed, outliers], axis = 1)
index_disc = lego_processed[lego_processed.iloc[:,-1]==-1].index
lego_processed.drop(index_disc, axis = 0, inplace = True)
lego_processed.drop(lego_processed.columns[-1], axis = 1, inplace = True)


# In[14]:


# Scaling
scaler = StandardScaler()
lego[feat_num] = lego[feat_num].apply(pd.to_numeric)
lego_num_scaled = pd.DataFrame(scaler.fit_transform(lego[feat_num]), 
                               columns=feat_num)


# In[15]:


# Create the dataset
lego_processed = pd.concat([lego_dummies.reset_index(drop=True),
                            lego_num_scaled.reset_index(drop=True)], axis = 1)


# ## 3.2 Multi-collinearity Problem

# ### 3.2.1 VIF(Variance Inflation Factor) before and after scaling

# In[16]:


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


# ### 3.2.2 Principal Component Analysis (PCA)

# In[17]:


# Find the optimal number of the principal components with scree plot
pca = decomposition.PCA(n_components = 5)
feat_num_ratings_pca = pca.fit_transform(lego_processed[:-1])
pc = ["PC1", "PC2", "PC3", "PC4", "PC5"]
eigen_df = pd.DataFrame({"PC" : pc,
                      "var": pca.explained_variance_ratio_})
sns.factorplot(x = "PC", y = "var", data=eigen_df, color = "r")
plt.pyplot.suptitle("Scree Plot")


# In[18]:


# Revise the dataset with the new principal component 
pca = decomposition.PCA(n_components = 3)
feat_pca = pca.fit_transform(lego_processed[:-1])
lego_processed_pca = feat_pca
lego_processed_pca = pd.DataFrame(lego_processed_pca)
lego_processed_pca.columns = ["PC1", "PC2", "PC3"]
lego_processed_pca.head()

# Define the input and output features
feat_ind = lego_processed_pca.columns
feat_dep = "list_price"

# FInalize and save the dataset
lego_processed = pd.concat([lego_processed_pca[feat_ind], lego_processed[feat_dep]], axis = 1)
lego_processed.to_csv("lego_processed_pca.csv")


# In[19]:


# Remove the misssing values
lego_processed.dropna(how = "any", axis = 0, inplace = True)


# In[20]:


# Check the final dataset
lego_processed.head()


# ### 4.1.4 Splititng the Dataset

# In[21]:


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(lego_processed[feat_ind], lego_processed[feat_dep], test_size=0.2, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .25, shuffle=True)
X_train, X_val, X_test, y_train, y_val, y_test = np.float32(X_train.values), np.float32(X_val.values),                                                 np.float32(X_test.values), np.float32(y_train.values),                                                 np.float32(y_val.values), np.float32(y_test.values)


# In[22]:


y_train, y_val, y_test = np.reshape(y_train, (y_train.shape[0], 1)),                         np.reshape(y_val, (y_val.shape[0], 1)),                         np.reshape(y_test, (y_test.shape[0], 1))


# # 5. Model Fitting and Testing

# ## 5.1 OLS(Ordinary Least Squares) Report

# In[23]:


# Fit the OLS model
reg = sm.OLS(lego_processed[feat_dep], lego_processed[feat_ind], data = lego_processed)
def reg_report(mod):
    return mod.fit().summary()
    
# View OLS report
reg_report(reg)


# ## 5.2 Plain Linear Regression

# In[24]:


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


# ## 5.3 Least Absolute Shrinkage and Selection Operator (LASSO)

# In[25]:


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


# In[26]:


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


# ## 5.4 Neural Network with Tensorflow

# In[27]:


# Assign placeholders
tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, 3])
Y = tf.placeholder(tf.float32, [None, 1])

# Set parameters for layers
num_in = 3
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


# In[28]:


# Hyperparameters
learning_rate = 0.003
keep_prob = tf.placeholder_with_default(0.7, shape=())
epochs = 10000

# Cost function, optimizer and cost history list for visualization
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)
cost_history = []
cost_history = np.empty(shape=[1], dtype=float)

# Initializer
init = tf.global_variables_initializer()


# In[29]:


# Close the existing session
if 'session' in locals() and session is not None:
    print('Close interactive session')
    session.close()


# In[30]:


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

