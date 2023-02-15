# MIS581-Portfolio-Project
Portfolio Project for CSU Global 
Jordan LaCerte
# PRedicting Telecom Churn using logistic regression and random forest models
This repository contains the Python code that was used to build a logistic regression and random forest model based on a data set for the final portfolio project for  MIS 581: Business Intelligence and Data Analytics, Colorado State University Global
### Data Set 
The data set that being used is from Kaggle and is based on a data set from IBM. This data set can be found here https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download 

This data set contains 21 columns and 7,043 rows. The columns include things like the service the customer has, what benefits they have, demographic information about the customer, and more in the 21 columns. The target variable is churn which is customers that have left in the past month. 
### Code 

The code for this project was done in Python using the Jupyter notebook environemnt. The code has comments as well to explain the though process but the basic outline is first importing all the required packages, then data exploration and visualization, then the machine learning models, and finally evaluation of the models. 

Based on Jupyter's sectioning, section 1 is importing all required packages, section 2 is data cleaning, section 3 is data visualization and exploration, section 4 is splitting the data into training and testing sets to prepare for machine learining, and the remaining sections are all realated to logistic regression and random forest models and their evaluation.

### Visualizations

![image](https://user-images.githubusercontent.com/63986681/219169845-793a2a4d-d6c0-42bd-b817-da58b33e00d5.png)

This first image is showing the distribution of customers and the total tenure they have with the company. This was done to get a better understanding of what the distribution of customes is to see if the company is mainly getting new customers, doing well on retention, and other questions related to tenure. 

![image](https://user-images.githubusercontent.com/63986681/219174151-3f341cd1-32d0-401e-838c-0c21c4c5f1f8.png)


This second visualization shows the tenure distribution further by breaking it down by contract type. This is to get a further understanding of what are the important variables and it appears that contract type may be significant as there appears to be many more people churning in the month to month pool as opposed to the 2 year ones. 

### Model Evaluation

![image](https://user-images.githubusercontent.com/63986681/219174923-034cbc81-242d-4167-b2ea-14d87f36bb98.png)
 
This is an image of the AUC for the logistic regression model. With an AUC of 0.83 it shows that this model is acceptable at predicting customer churn based on the data set that it had been trained on. 

![image](https://user-images.githubusercontent.com/63986681/219175503-1c650524-6979-4ec7-912e-a70598f6faf6.png)
![image](https://user-images.githubusercontent.com/63986681/219175513-32eade06-ffe4-42ef-b20d-1e9da39608f0.png)

This pair of images show the variable importance plots that Logistic regressions provide. It appears that the most significant variables are the customer's contract type, tenure, and if they have the option of fiber avialable to them. 

![image](https://user-images.githubusercontent.com/63986681/219175885-61b42972-cfe2-47aa-b128-d1240d09f006.png)

This is an image of the AUC for the randomforest model. With an AUC of 0.84 it shows that this model is generally acceptable at predicting customer churn based on the data set that it had been trained on. 

![image](https://user-images.githubusercontent.com/63986681/219175896-23b5030b-35c8-4d2f-b11b-6165628909b0.png)

This next image shows the feature importance chart for random forest, again we can see that the most important variables in predicting a customer's churn are the contract type and tenure. This also shows that the extra bells and whistles on a plan may not be as important as the price and time with the company. 

### References
Kaggle. (2018, February 23). Telco customer churn. Kaggle. Retrieved, from https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download 
