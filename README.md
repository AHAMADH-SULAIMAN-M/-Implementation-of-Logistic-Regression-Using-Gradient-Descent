<img width="1361" height="546" alt="image" src="https://github.com/user-attachments/assets/18b24e01-0514-4b38-8af6-6e491a53e2b2" /># Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2. Load the dataset.

3.Define X and Y array.

4.Define a function for costFunction,cost and gradient.

5.Define a function to plot the decision boundary.****

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:IRFAN KHAN.N
RegisterNumber:  212224230097

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset= dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta, X, y ):
    h = sigmoid(X.dot(theta)) 
    return -np.sum(y *np.log(h)+ (1- y) *np.log(1-h))
def gradient_descent(theta, x, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot (h-y) /m
        theta-=alpha * gradient
    return theta
theta= gradient_descent (theta,X,y,alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where( h >= 0.5,1 , 0)
    return y_pred

y_pred= predict(theta,X)
accuracy = np.mean(y_pred.flatten()==y) 
print("Accuracy:", accuracy)
print(Y)
print(y_pred)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)  
*/
```

## Output:

# Dataset

<img width="1361" height="546" alt="image" src="https://github.com/user-attachments/assets/6ff8ce9b-9385-4b03-8990-e2207b56226b" />

# Labelling Data

<img width="365" height="341" alt="image" src="https://github.com/user-attachments/assets/4ff36007-22f5-4a47-ab23-6f837ccbdc29" />

# Labelling Columns

<img width="1243" height="567" alt="image" src="https://github.com/user-attachments/assets/55824d1a-efad-42e9-8cef-d3d07cad3a86" />

# Dependent Variable

<img width="884" height="280" alt="image" src="https://github.com/user-attachments/assets/376d882c-d763-40fb-a3a9-e755b252a0d7" />

# Accuracy

<img width="375" height="78" alt="image" src="https://github.com/user-attachments/assets/ba1c742b-b246-4d6b-a940-d28ed59fda70" />

# Y
 
<img width="766" height="153" alt="image" src="https://github.com/user-attachments/assets/4c2879b3-1cb7-4b0c-910f-747e257f6ead" />

# Predicted Data

<img width="92" height="49" alt="image" src="https://github.com/user-attachments/assets/4179e8e6-ab72-4867-9e1f-ee19c743be54" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

