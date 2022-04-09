import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv('datafile.tsv', sep='\t')

# print(data.head())
#data.head()
#data.tail()

#data.shape
#print(data.describe())
# print(data.isnull().sum())

#visualize
#print(sns.relplot(x='Diagnosis Age',y='Overall Survival Status',data=data))
#sns.relplot(y='Diagnosis Age',
 #           x='Overall Survival Status'
  #          ,hue="Sex",data=data)
#plt.show()


train=data.drop(['Overall Survival Status','American Joint Committee on Cancer Metastasis Stage Code','American Joint Committee on Cancer Publication Version Type','Subtype','Sample Type','Patient ID','Number of Samples Per Patient'],axis=1)

test=data['Overall Survival Status']


train= train.apply(pd.to_numeric, errors='coerce')
test = test.apply(pd.to_numeric, errors='coerce')
train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

X_train,X_test,y_train,y_test=train_test_split(train,test,test_size=0.3,random_state=2)

regr = LinearRegression()

regr.fit(X_train,y_train)
pred = regr.predict(X_test)
print(pred)
print("accurace : ")
print(regr.score(X_test,y_test))
#git config --global --edit::

