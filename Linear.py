import numpy as np
import pandas as pd
dataset=pd.read_csv('/Users/abc/Documents/Book1.csv')

x=dataset.iloc[:, :-1].values
y=dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size= 1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
print(reg.fit(x_train, y_train))
y_pred = reg.predict(x_test)
print(x_test)
print(y_pred)



9

