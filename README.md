# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1. Start the program.
STEP 2. Import the required library and read the dataframe.
STEP 3. Write a function computeCost to generate the cost function.
STEP 4. Perform iterations og gradient steps with learning rate.
STEP 5. Plot the Cost function using Gradient Descent and generate the required graph.
STEP 6.End the Program

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: V Rishon Anand
RegisterNumber:24900460  
*/
```
```.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv('/content/50_startups.csv',header=None)
X=(data.iloc[1:, :-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)

theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1, new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![Screenshot 2025-01-03 084837](https://github.com/user-attachments/assets/45354753-070d-4d90-9762-37bc0bf3cb64)

![Screenshot 2025-01-03 084902](https://github.com/user-attachments/assets/75b75ce5-1d29-41ab-b3af-4d8cd808cc9a)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
