import numpy as np
import pandas as pd
#importing dataset
dataset=pd.read_csv('/Users/abc/Downloads/diabetes2.csv')
print(dataset)
x=dataset.iloc[:, :-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
#split dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=0)
print(len(x_train))
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
print(x_train)
#Model
from sklearn.svm import SVC
classifier= SVC(kernel= 'linear', random_state= 0)
print(classifier.fit(x_train,y_train))
print(classifier.predict(sc.transform([[33,168]])))
y_pred=classifier.predict(x_test)
print(x_test)
print(y_pred)
print(y_test)
#Confusion Matrix and accuracy
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
#LR
#SVM
#K-Nearest Neighbours(KNN)
#Kernel SVM
#Naive Bayes
#Decision Tree Classification
#Random Forest classification


