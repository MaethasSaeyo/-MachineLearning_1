import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

dataset = pd.read_csv("Weather.csv")

#train & test set
x = dataset["MinTemp"].values.reshape(-1,1)
y = dataset["MaxTemp"].values.reshape(-1,1)

# 80% - 20%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#trainning
model = LinearRegression()
model.fit(x_train,y_train)
#test
y_pred =  model.predict(x_test) 

#compare true data & predict data
df = pd.DataFrame({'Actually':y_test.flatten(),'Predicted':y_pred.flatten()})

#ใกล้0=ดี
print("MAE",metrics.mean_absolute_error(y_test,y_pred))#การคำนวน Error ว่า y_pred ต่างจาก y_test อยู่เท่าไร ด้วยการนำมาลบกันตรง ๆ แล้วหาค่าเฉลี่ย โดยไม่สนใจเครื่องหมาย
print("MSE",metrics.mean_squared_error(y_test,y_pred))#การคำนวน Error ว่า y_pred ต่างจาก y_test อยู่เท่าไร ด้วยการนำมาลบกัน แล้วยกกำลังสอง
print("RMSE",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))#MSE มาหา Squared Root
print("Score = ",metrics.r2_score(y_test,y_pred))#%ความแม่นยำ