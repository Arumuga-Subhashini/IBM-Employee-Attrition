# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 11:56:37 2022

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
Data=pd.read_csv('E:/Illumine i/Employee Attrition.csv')

Data.describe()
Data.info()
Data.describe(include='object')
Data.isnull().sum()
'''There is no Nan values'''

Data.Attrition.replace({"Yes":1,"No":0}, inplace=True)
Data.corr()['Attrition'] 

Data                                  
Data.var()
Data.var()==0 

Data.drop(columns=['EmployeeCount','Over18','StandardHours'], inplace=True)
duplicate=Data.duplicated()
sum(duplicate)

Data['Attrition'].value_counts().plot(kind='pie',explode=[0.1,0.1],autopct='%1.1f%%',shadow=True,colors=['g','r'])
print(Data['Attrition'].value_counts())
'''16% employees have left the company'''

Num_Cols = Data.select_dtypes([np.number]).columns
Num_Cols

for col in Num_Cols:
    plt.figure(figsize=(18,6))
    sns.set()
    sns.displot(x=col,data=Data,stat='probability',hue='Attrition')
    plt.show()


Cat_Cols = Data.select_dtypes(['object']).columns
Cat_Cols

for col in Cat_Cols:
    plt.figure(figsize=(18,6))
    sns.set()
    sns.histplot(x=col,data=Data,stat='probability',hue="Attrition")
    plt.show()
    
    
'''Effect on Distance by education & Attrition'''
sns.scatterplot(x='JobRole',y='DistanceFromHome',hue='Attrition',data=Data)

'''Maximum Attrition happens when the distance is above 10kms and in Sales,LabTech & Research Positions'''

sns.scatterplot(x='MonthlyIncome',y='Education',hue='Attrition',data=Data)
'''Most Attrition Happens when the Monthly Income is less than 15000 and education doesn't have an effect on attrition'''

'''Also PerformanceRating is 3 or above for every employee and influences Attrition marginally. This maybe correlated with the years on current role to understand further'''

'''Outlier Analysis'''   
for col in Num_Cols:
    plt.figure(figsize=(12,8))
    sns.boxplot(data=Data, y=col,color='Blue')
    plt.show()

'''About 7 columns have extreme values'''
Data.shape
'''Monthly Income'''                                 
(Data['MonthlyIncome'] <= 18000).value_counts()
Data = Data[(Data['MonthlyIncome'] <= 180000)]

'''years at company'''
(Data['YearsAtCompany']<= 35).value_counts()
Data=Data[(Data['YearsAtCompany']<= 35)]

'''Numberofcompaniesworked'''
(Data['NumCompaniesWorked'] <= 8).value_counts()
Data = Data[(Data['NumCompaniesWorked'] <= 8)]

'''Totalworkingyears'''
(Data['TotalWorkingYears'] <= 35).value_counts()
Data=Data[(Data['TotalWorkingYears'] <= 35)]

'''YearsInCurrentRole'''
(Data['YearsInCurrentRole'] <= 17).value_counts()
Data=Data[(Data['YearsInCurrentRole'] <= 17)]

'''YearsSinceLastPromotion'''
(Data['YearsSinceLastPromotion'] <= 14).value_counts()
Data=Data[(Data['YearsSinceLastPromotion'] <= 14)]

'''YearswithCurrentManager'''
(Data['YearsWithCurrManager'] <= 15).value_counts()
Data=Data[(Data['YearsWithCurrManager'] <= 15)]

Data.shape

'''85 rows were removed to handle outliers'''

encoded_cat_col = pd.get_dummies(Data[Cat_Cols])
final_model = pd.concat([Data[Num_Cols],encoded_cat_col], axis = 1)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

x = final_model.drop(columns="Attrition")
y = final_model["Attrition"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)
train_Pred = model.predict(x_train)
metrics.confusion_matrix(y_train,train_Pred)
Accuracy_percent_train = (metrics.accuracy_score(y_train,train_Pred))*100
Accuracy_percent_train
test_Pred = model.predict(x_test)
metrics.confusion_matrix(y_test,test_Pred)
Accuracy_percent_test = (metrics.accuracy_score(y_test,test_Pred))*100
Accuracy_percent_test
print(classification_report(y_test, test_Pred))




