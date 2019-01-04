
# coding: utf-8

# # 1. Cross Validation

# ## 1.1 Data Preprepartion

# In[2]:


# Import the required modules and dataset from sklearn. 
import time
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
import pandas as pd
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


# In[3]:


# Split the input and targe variables.
data.keys()
data_cols = data["feature_names"]
del data["DESCR"]
data_values = { k:data[k] for k in ["data", "target"] }
data_names = { j:data[j] for j in ["target_names", "feature_names"]}
x = data_values["data"]
y = data_values["target"]


# ## 1.2 Fitting the Model

# In[4]:


# Get the validation score for 5-fold cross validation and their mean and standard deviation,
# using the Decision Tree.
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
data_dt = dt.fit(x, y)
scores_dt = cross_val_score(data_dt, x, y, scoring = "accuracy", cv = 5)
avg_acc_dt = scores_dt.mean()
std_dt = scores_dt.std()
print("Decision Tree Fitting Complete!")

# Get the validation score for 5-fold cross validation and their mean and standard deviation,
# using the Support Vector Machine.
from sklearn.svm import SVC
svc = SVC(kernel='linear')
data_svm = svc.fit(x, y)
scores_svm = cross_val_score(data_svm, x, y, scoring = "accuracy", cv = 5)
avg_acc_svm = scores_svm.mean()
std_svm = scores_svm.std()
print("Support Vector Machine Fitting Complete!")

# Get the validation score for 5-fold cross validation and their mean and standard deviation,
# using the Logistic Regression.
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
data_lr = lr.fit(x, y)
scores_lr = cross_val_score(data_lr, x, y, scoring = "accuracy", cv = 5)
avg_acc_lr = scores_lr.mean()
std_lr = scores_lr.std()
print("Logistic Regression Fitting Complete!")
for i in range(10):
    time.sleep(0.3)
    print(".", end="")
print()
time.sleep(1)
print("Fitting Complete!")


# ## 1.3 Accuracy and Standard Deviation Report

# In[5]:


# Report the result
# Create a Pandas dataframe for the cross validation report.
index = ["ACC", "STD"]
report = {
        "DT": [avg_acc_dt, std_dt],
        "SVM": [avg_acc_svm, std_svm],
        "LR": [avg_acc_lr, std_lr]
        }
report_table = pd.DataFrame(data=report, index=index)
report_table

# Define a class to report the result of cross validation.
class cross_val_report:
    def __init__(self, report_table):
        self.report_table = report_table
        
    def report(self):
        print("◎ Cross Validation Report")
        return report_table

# Retrieve the report.
cross_val_report = cross_val_report(report_table)
cross_val_report.report()


# # 2. Bagging

# ## 2.1 Fitting the Model

# In[ ]:


# Import required modules.
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Assign some classifiers and lists in advance. 
dt = DecisionTreeClassifier()
svc = SVC(kernel='linear')
lr = LogisticRegression()
classifiers = [dt, svc, lr]
n_estimators = [10, 20, 50, 100, 200, 500]

# Create empty dictionaries for later creation of dataframes for bagging results of each classifier.
report_bagging_dt = {
        10: [], 
        20: [],
        50: [],
        100: [],
        200: [],
        500: []
        }
report_bagging_svm = {
        10: [], 
        20: [],
        50: [],
        100: [],
        200: [],
        500: []
       }
report_bagging_lr = {
        10: [], 
        20: [],
        50: [],
        100: [],
        200: [],
        500: []
       }

# Perform the bagging on the dataset.
max_samples = 0.1
print("Launching Bagging...")
for i in range(3):
    for j in range(6):
        if i == 0:
            bag_data_dt = BaggingClassifier(classifiers[i], n_estimators=n_estimators[j], max_samples=max_samples, bootstrap=True)
            bag_data_dt = bag_data_dt.fit(x, y)
            scores_dt = cross_val_score(bag_data_dt, x, y, scoring = "accuracy", cv = 5)
            avg_acc_dt = scores_dt.mean()
            std_dt = scores_dt.std()
            report_bagging_dt[n_estimators[j]] = [avg_acc_dt, std_dt]
            print("Calculation on %d Estimator for %s Complete!" % (n_estimators[j], "Decision Tree"))
            if j == 5:
                print("Calculation on Decision Tree Complete!")
        elif i == 1:
            bag_data_svm = BaggingClassifier(classifiers[i], n_estimators=n_estimators[j], max_samples=max_samples, bootstrap=True)
            bag_data_svm = bag_data_svm.fit(x, y)
            scores_svm = cross_val_score(bag_data_svm, x, y, scoring = "accuracy", cv = 5)
            avg_acc_svm = scores_svm.mean()
            std_svm = scores_svm.std()
            report_bagging_svm[n_estimators[j]] = [avg_acc_svm, std_svm]
            print("Calculation on %d Estimator for %s Complete!" % (n_estimators[j], "SVM"))
            if j == 5:
                print("Calculation on Support Vector Machine Complete!")
        else:
            bag_data_lr = BaggingClassifier(classifiers[i], n_estimators=n_estimators[j], max_samples=max_samples, bootstrap=True)
            bag_data_lr = bag_data_lr.fit(x, y)
            scores_lr = cross_val_score(bag_data_lr, x, y, scoring = "accuracy", cv = 5)
            avg_acc_lr = scores_lr.mean()
            std_lr = scores_lr.std()
            report_bagging_lr[n_estimators[j]] = [avg_acc_lr, std_lr]
            print("Calculation on %d Estimator for %s Complete!" % (n_estimators[j], "Logistic Regression"))
            if j == 5:
                print("Calculation on Logistic Regression Complete!")
print()
print("Bagging Complete!")


# ## 2.2 Accuracy and Standard Deviation Report

# In[ ]:


# Bagging Accucary and Standard Deviation Report
# Convert the bagging result dictionaries into Pandas dataframes.
report_bagging_table_dt = pd.DataFrame(data=report_bagging_dt, index = index)
report_bagging_table_svm = pd.DataFrame(data=report_bagging_svm, index = index)
report_bagging_table_lr = pd.DataFrame(data=report_bagging_lr, index = index)

# Define a class that prints out the bagging results.
class report_bagging:
    def __init__(self, report_bagging_table_dt, report_bagging_table_svm, report_bagging_table_lr):
        self.report_bagging_table_dt = report_bagging_table_dt
        self.report_bagging_table_svm = report_bagging_table_svm
        self.report_bagging_table_lr = report_bagging_table_lr
        
    def dt(self):
        print("◎ Bagging Report for the Decision Tree")
        return report_bagging_table_dt

    def svm(self):
        print("◎ Bagging Report for the Support Vector Machine")
        return report_bagging_table_svm
    
    def lr(self):
        print("◎ Bagging Report for the Logistic Regression")
        return report_bagging_table_lr

report_bagging = report_bagging(report_bagging_table_dt, report_bagging_table_svm, report_bagging_table_lr)
print(report_bagging.dt())
print()
print(report_bagging.svm()) 
print()
print(report_bagging.lr())


# ## 2.3 Visualization

# In[ ]:


# Bagging Visualization
import matplotlib.pyplot as plt
n_estimators = [10, 20, 50, 100, 200, 500]
classifier = ["DT", "SVM", "LR"]


# In[ ]:


# Define axes for the Decision Tree.
x_dt = list(report_bagging_dt.keys())
y_dt = [k for number in n_estimators for k in report_bagging_dt[number]][0:2*len(n_estimators):2] 
std_dt = [k for number in n_estimators for k in report_bagging_dt[number]][1:2*len(n_estimators):2] 

# Plot the bagging result of the Decision Tree.
plt.figure()
plt.errorbar(x_dt, y_dt, std_dt, color = 'r', marker = "^", capsize = 5)

# Plot the accuracy line of the Decision Tree before bagging.
plt.plot(range(0,510), [report_table[classifier[0]][0]]*len(range(0,510)))

# Add the title, axis label, and adjust the axis.
plt.title("The Accuracy Curve - Decision Tree")
plt.xlabel("The Number of Estimators")
plt.ylabel("The Average Accuracy")
plt.axis([0, 510, 0.9, 1])
plt.show()

# Recall the bagging and the cross validation report.
print(report_bagging.dt())
print()
print("Accuarcy: ", cross_val_report.report().iloc[0,0])


# In[ ]:


# Define axes for the Support Vector Machine.
x_svm = list(report_bagging_svm.keys())
y_svm = [k for number in n_estimators for k in report_bagging_svm[number]][0:2*len(n_estimators):2] 
std_svm = [k for number in n_estimators for k in report_bagging_svm[number]][1:2*len(n_estimators):2] 

# Plot the bagging result of the Support Vector Machine.
plt.figure()
plt.errorbar(x_svm, y_svm, std_svm, color = 'g', marker = "^", capsize = 5)

# Plot the accuracy line of the Support Vector Machine before bagging.
plt.plot(range(0,510), [report_table[classifier[1]][0]]*len(range(0,510)))

# Add the title, axis label, and adjust the axis.
plt.title("The Accuracy Curve - Support Vector Machine")
plt.xlabel("The Number of Estimators")
plt.ylabel("The Average Accuracy")
plt.axis([0, 510, 0.9, 1])
plt.show()

# Recall the bagging and the cross validation report.
print(report_bagging.svm())
print()
print("Accuarcy: ", cross_val_report.report().iloc[0,1])


# In[ ]:


# Define axes for the Logistic Regression.
x_lr = list(report_bagging_lr.keys())
y_lr = [k for number in n_estimators for k in report_bagging_lr[number]][0:2*len(n_estimators):2] 
std_lr = [k for number in n_estimators for k in report_bagging_lr[number]][1:2*len(n_estimators):2] 

# Plot the bagging result of the Logistic Regression.
plt.figure()
plt.errorbar(x_lr, y_lr, std_lr, color = 'b', marker = "^", capsize = 5)

# Plot the accuracy line of the Logistic Regression before bagging.
plt.plot(range(0,510), [report_table[classifier[2]][0]]*len(range(0,510)))

# Add the title, axis label, and adjust the axis.
plt.title("The Accuracy Curve - Logistic Regression")
plt.xlabel("The Number of Estimators")
plt.ylabel("The Average Accuracy")
plt.axis([0, 510, 0.9, 1])
plt.show()

# Recall the bagging and the cross validation report.
print(report_bagging.lr())
print()
print("Accuarcy: ", cross_val_report.report().iloc[0,2])


# # 3. Boosting

# ## 3.1 Fitting the Model

# In[ ]:


# Import required modules.
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Assign some classifiers and lists in advance. 
dt = DecisionTreeClassifier()
svc = SVC(kernel='linear')
lr = LogisticRegression()
classifiers = [dt, svc, lr]
n_estimators = [10, 20, 50, 100, 200, 500]

# Create empty dictionaries for later creation of dataframes for bagging results of each classifier.
report_boosting_dt = {
        10: [], 
        20: [],
        50: [],
        100: [],
        200: [],
        500: []
        }
report_boosting_svm = {
        10: [], 
        20: [],
        50: [],
        100: [],
        200: [],
        500: []
       }
report_boosting_lr = {
        10: [], 
        20: [],
        50: [],
        100: [],
        200: [],
        500: []
       }

# Perform the bagging on the dataset.
for i in range(3):
    for j in range(6):
        if i == 0:
            boost_data_dt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators[j], algorithm='SAMME')
            boost_data_dt = boost_data_dt.fit(x, y)
            scores_dt = cross_val_score(boost_data_dt, x, y, scoring = "accuracy", cv = 5)
            avg_acc_dt = scores_dt.mean()
            std_dt = scores_dt.std()
            report_boosting_dt[n_estimators[j]] = [avg_acc_dt, std_dt]
            print("Calculation on Estimator %d for %s Complete!" % (n_estimators[j], "Decision Tree"))
        elif i == 1:
            boost_data_svm = AdaBoostClassifier(classifiers[i], n_estimators=n_estimators[j], algorithm='SAMME')
            boost_data_svm = boost_data_svm.fit(x, y)
            scores_svm = cross_val_score(boost_data_svm, x, y, scoring = "accuracy", cv = 5)
            avg_acc_svm = scores_svm.mean()
            std_svm = scores_svm.std()
            report_boosting_svm[n_estimators[j]] = [avg_acc_svm, std_svm]
            print("Calculation on Estimator %d for %s Complete!" % (n_estimators[j], "SVM"))
        else:
            boost_data_lr = AdaBoostClassifier(classifiers[i], n_estimators=n_estimators[j], algorithm='SAMME')
            boost_data_lr = boost_data_lr.fit(x, y)
            scores_lr = cross_val_score(boost_data_lr, x, y, scoring = "accuracy", cv = 5)
            avg_acc_lr = scores_lr.mean()
            std_lr = scores_lr.std()
            report_boosting_lr[n_estimators[j]] = [avg_acc_lr, std_lr]
            print("Calculation on Estimator %d for %s Complete!" % (n_estimators[j], "Logistic Regression"))
print()
print("Boosting Complete!")


# ## 3.2 Accuracy and Standard Deviation Report

# In[ ]:


# Bagging Accucary and Standard Deviation Report
# Convert the bagging result dictionaries into Pandas dataframes.
report_boosting_table_dt = pd.DataFrame(data=report_boosting_dt, index = index)
report_boosting_table_svm = pd.DataFrame(data=report_boosting_svm, index = index)
report_boosting_table_lr = pd.DataFrame(data=report_boosting_lr, index = index)

# Define a class that prints out the bagging results.
class report_boosting:
    def __init__(self, report_boosting_table_dt, report_boosting_table_svm, report_boosting_table_lr):
        self.report_boosting_table_dt = report_boosting_table_dt
        self.report_boosting_table_svm = report_boosting_table_svm
        self.report_boosting_table_lr = report_boosting_table_lr
        
    def dt(self):
        print("◎ Boosting Report for the Decision Tree")
        return report_boosting_table_dt

    def svm(self):
        print("◎ Boosting Report for the Support Vector Machine")
        return report_boosting_table_svm
    
    def lr(self):
        print("◎ Boosting Report for the Logistic Regression")
        return report_boosting_table_lr

report_boosting = report_boosting(report_boosting_table_dt, report_boosting_table_svm, report_boosting_table_lr)
print(report_boosting.dt())
print()
print(report_boosting.svm()) 
print()
print(report_boosting.lr())


# ## 3.3 Visualization

# In[ ]:


# Boosting Visualization
import matplotlib.pyplot as plt
n_estimators = [10, 20, 50, 100, 200, 500]
classifier = ["DT", "SVM", "LR"]


# In[ ]:


# Define axes for the Decision Tree.
x_dt = list(report_boosting_dt.keys())
y_dt = [k for number in n_estimators for k in report_boosting_dt[number]][0:2*len(n_estimators):2] 
std_dt = [k for number in n_estimators for k in report_boosting_dt[number]][1:2*len(n_estimators):2] 

# Plot the bagging result of the Decision Tree.
plt.figure()
plt.errorbar(x_dt, y_dt, std_dt, color = 'r', marker = "^", capsize = 5)

# Plot the accuracy line of the Decision Tree before boosting.
plt.plot(range(0,510), [report_table[classifier[0]][0]]*len(range(0,510)))

# Add the title, axis label, and adjust the axis.
plt.title("The Accuracy Curve - Decision Tree")
plt.xlabel("The Number of Estimators")
plt.ylabel("The Average Accuracy")
plt.axis([0, 510, 0.9, 1])
plt.show()

# Recall the bagging and the cross validation report.
print(report_boosting.dt())
print()
print("Accuarcy: ", cross_val_report.report().iloc[0,0])


# In[ ]:


# Define axes for the Support Vector Machine.
x_svm = list(report_boosting_svm.keys())
y_svm = [k for number in n_estimators for k in report_boosting_svm[number]][0:2*len(n_estimators):2] 
std_svm = [k for number in n_estimators for k in report_boosting_svm[number]][1:2*len(n_estimators):2] 

# Plot the bagging result of the Support Vector Machine.
plt.figure()
plt.errorbar(x_svm, y_svm, std_svm, color = 'g', marker = "^", capsize = 5)

# Plot the accuracy line of the Support Vector Machine before boosting.
plt.plot(range(0,510), [report_table[classifier[1]][0]]*len(range(0,510)))

# Add the title, axis label, and adjust the axis.
plt.title("The Accuracy Curve - Support Vector Machine")
plt.xlabel("The Number of Estimators")
plt.ylabel("The Average Accuracy")
plt.axis([0, 510, 0.9, 1])
plt.show()

# Recall the boosting and the cross validation report.
print(report_boosting.svm())
print()
print("Accuarcy: ", cross_val_report.report().iloc[0,1])


# In[ ]:


# Define axes for the Logistic Regression.
x_lr = list(report_boosting_lr.keys())
y_lr = [k for number in n_estimators for k in report_boosting_lr[number]][0:2*len(n_estimators):2] 
std_lr = [k for number in n_estimators for k in report_boosting_lr[number]][1:2*len(n_estimators):2] 

# Plot the boosting result of the Logistic Regression.
plt.figure()
plt.errorbar(x_lr, y_lr, std_lr, color = 'b', marker = "^", capsize = 5)

# Plot the accuracy line of the Logistic Regression before boosting.
plt.plot(range(0,510), [report_table[classifier[2]][0]]*len(range(0,510)))

# Add the title, axis label, and adjust the axis.
plt.title("The Accuracy Curve - Logistic Regression")
plt.xlabel("The Number of Estimators")
plt.ylabel("The Average Accuracy")
plt.axis([0, 510, 0.9, 1])
plt.show()

# Recall the boosting and the cross validation report.
print(report_boosting.lr())
print()
print("Accuarcy: ", cross_val_report.report().iloc[0,2])

#%% Getting Tables
report_bagging.dt()
