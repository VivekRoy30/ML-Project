import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv("/Users/abc/Downloads/Student_Performance.csv")
print(dataset)

# Separate features and target
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Identify the columns with categorical data
categorical_features = [index for index, col in enumerate(dataset.columns[:-1]) if dataset[col].dtype == 'object']

# Apply OneHotEncoder to all categorical features
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
x = ct.fit_transform(x)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(len(y_train))
print(len(y_test))

# Model
reg = LinearRegression()
reg.fit(x_train, y_train)

# Predict
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#polynomial regression
#support vector regression
#decision tree regression
#random forest regression
   


