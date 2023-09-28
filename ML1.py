import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split

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

#plt.scatter(x_test,y_test)
#plt.plot(x_test,y_pred,color="red",linewidth=2)
#plt.show()


#compare true data & predict data
df = pd.DataFrame({'Actually':y_test.flatten(),'Predicted':y_pred.flatten()})

df1 = df.head(20)
df1.plot(kind="bar",figsize=(16,10))
plt.show()