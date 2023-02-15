#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###MSI 581 Final Project Code, Random Forest and Logistic Regression


# In[92]:


###Import pandas and read csv
import numpy as np # linear algebra
import pandas as pd # data danipulation and processing
import seaborn as sns # charts
import matplotlib.ticker as mtick 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

MSIchurn = pd.read_csv('MSI581Churn.csv')
MSIchurn.head()


# In[93]:


# Converting Total Charges to a numerical data type
MSIchurn.TotalCharges = pd.to_numeric(MSIchurn.TotalCharges, errors='coerce')
#Removing missing values and the customer ID column
MSIchurn.dropna(inplace = True)
df2 = MSIchurn.iloc[:,1:]
#Churn to numerical/binary instead of categorical
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#convert all categorical variables to dummy variables for pre-processing
df_dummies = pd.get_dummies(df2)
df_dummies.head()


# In[95]:


###  Tenure, the big kahuna 
ax = sns.distplot(MSIchurn['tenure'], hist=True, kde=False, 
             bins=int(180/5), color = 'black', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
ax.set_ylabel('Customers')
ax.set_xlabel('Tenure (months)')
ax.set_title('Customers by tenure')
### most customers are either new or long time customers, there is not a lot of middle ground


# In[97]:


### Compare and contrast tenure for the three contract types 
fig, (ax1,ax2,ax3) = plt.subplots(nrows=1, ncols=3, sharey = True, figsize = (20,6))

ax = sns.distplot(MSIchurn[MSIchurn['Contract']=='Month-to-month']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'red',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax1)
ax.set_ylabel('Total Customers')
ax.set_xlabel('Tenure')
ax.set_title('Month to Month ')

ax = sns.distplot(MSIchurn[MSIchurn['Contract']=='One year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'blue',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax2)
ax.set_xlabel('Tenure',size = 12)
ax.set_title('1 yr Contract',size = 12)

ax = sns.distplot(MSIchurn[MSIchurn['Contract']=='Two year']['tenure'],
                   hist=True, kde=False,
                   bins=int(180/5), color = 'green',
                   hist_kws={'edgecolor':'black'},
                   kde_kws={'linewidth': 4},
                 ax=ax3)

ax.set_xlabel('Tenure')
ax.set_title('2 yr Contract')
### The contract length seems to be some what correlated, you are more likely to stay longer with long term contracts 


# In[98]:


### It appears that the main thing that influences churn are the contract and tenure
### Next step is Logistic Regression
# As there is a lot of categorical we had to geneate dummy values 
y = df_dummies['Churn'].values
X = df_dummies.drop(columns = ['Churn'])

# Scaling all the variables to a range of 0 to 1 in order to perform logistic regression to improve accuracy

features = X.columns.values
scaler = MinMaxScaler(feature_range = (0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = features


# In[115]:


# Training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# logistic regression model from SKLearn

LRModel = LogisticRegression()
result = LRModel.fit(X_train, y_train)
prediction_test = LRModel.predict(X_test)

##evaluating accuracy, precition, recall,  f1-score

print ('Accuracy:',metrics.accuracy_score(y_test, prediction_test))
print ('Precision:', metrics.precision_score(y_test, prediction_test))
print ('Recall:', metrics.recall_score(y_test, prediction_test))
print ('f1:', metrics.f1_score(y_test, prediction_test))


# In[116]:


### AUC for Logistic Regression
metrics.plot_roc_curve(model, X_test, y_test) 


# In[100]:


# Variable Importance Plot, there are no p-values as SK learn does not support them 
###highest predictors of churn  
importance = pd.Series(model.coef_[0],
                 index=X.columns.values)
print (importance.sort_values(ascending = False)[:10].plot(kind='bar'))


# In[101]:


print(importance.sort_values(ascending = False)[-10:].plot(kind='bar'))
###highest predictors of staying


# In[110]:


###Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 50)
rf.fit(X_train, y_train)

# Make predictions
prediction_test = rf.predict(X_test)
print ('Accuracy:',metrics.accuracy_score(y_test, prediction_test))
print ('Precision:', metrics.precision_score(y_test, prediction_test))
print ('Recall:', metrics.recall_score(y_test, prediction_test))
print ('f1:', metrics.f1_score(y_test, prediction_test))


# In[111]:


### Look at feature importance chart from Random Forst
importances = rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')


# In[112]:


## AUC for RF
metrics.plot_roc_curve(rf, X_test, y_test) 


# In[ ]:




