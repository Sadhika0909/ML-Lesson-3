import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("student_placement.csv")
print(data.isnull().sum())
print(data.info())

data["study_hours"]=data["study_hours"].fillna(data["study_hours"].median(skipna = True))
data["attendance"]=data["attendance"].fillna(data["attendance"].median(skipna = True))
data["sleep_hours"]=data["sleep_hours"].fillna(data["sleep_hours"].median(skipna = True))
data["internet_usage"]=data["internet_usage"].fillna(data["internet_usage"].median(skipna = True))
print(data.isnull().sum())

data=data.drop("previous_score",axis=1)
data=data.drop("assignments_completed",axis=1)
print(data.head())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
data["placement_status"] = encoder.fit_transform(data["placement_status"])
print(data.head())
#study_hours,attendance,sleep_hours,internet_usage,assignments_completed,previous_score,exam_score,placement_status

#data analysis
x = data[["study_hours","attendance","sleep_hours","internet_usage","exam_score"]]
y = data["placement_status"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
y_predict = log_reg.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
acc = accuracy_score(y_test,y_predict)
acc = round(acc*100,2)
print("Accuracy =",acc,"%")

matrix = confusion_matrix(y_test,y_predict)
sns.heatmap(matrix, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()