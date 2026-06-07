#Logistical regression-classification algorythm using the concept of regression
#Classification : finite set of output (ex. detecting male or female)
#Regression : infinite ste of output(ex. weather prediction)
#Recommendation: based on previous choices, it will predict future choices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("titanic2.csv")
print(data.isnull().sum())
print("hi")
print(data.info())

#data preprocessing - replacing empty values
data["Age"]=data["Age"].fillna(data["Age"].median(skipna = True))
data["Embarked"]=data["Embarked"].fillna(data["Embarked"].value_counts().idxmax())
print(data.isnull().sum())

#removing the unwanted columns
data=data.drop("Cabin",axis=1)
data=data.drop("PassengerId",axis=1)
data=data.drop("Name",axis=1)
data=data.drop("Ticket",axis=1)
data["TravelAlone"] = np.where((data["SibSp"]+data["Parch"])>0,0,1)
print(data.head())

#converting strings into integers
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data["Sex"] = encoder.fit_transform(data["Sex"])
data["Embarked"] = encoder.fit_transform(data["Embarked"])
print(data.head())

#data analysis
x = data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","TravelAlone"]]
y = data["Survived"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
y_predict = log_reg.predict(x_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_predict)
acc = round(acc*100,2)
print("Accuracy =",acc,"%")