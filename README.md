# EX 03 -Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:PRAVEEN.K 
RegisterNumber:212223040152

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
 
*/
```

## Output:
#### TOP 5 ELEMENTS:
![Screenshot 2025-04-07 033050](https://github.com/user-attachments/assets/850e20aa-81f7-4b2a-97ac-29fe52d246be)
![Screenshot 2025-04-07 033059](https://github.com/user-attachments/assets/cdef6f70-7e3a-40b1-bd2b-d386186490a7)


#### DATA DUPLICATE:
![Screenshot 2025-04-07 033105](https://github.com/user-attachments/assets/a63c42d2-b20f-4334-a5c2-da616d600763)


#### PRINT DATA:
![Screenshot 2025-04-07 033118](https://github.com/user-attachments/assets/7f01d14f-ecbb-4bdd-89d5-2458f69b8d97)


#### DATA_STATUS:
![Screenshot 2025-04-07 033127](https://github.com/user-attachments/assets/87ef7cc9-e9dd-42f8-9d29-e46b28164a32)

#### Y_PREDICTION ARRAY:


![Screenshot 2025-04-07 033133](https://github.com/user-attachments/assets/7a774f03-19e0-46b8-9041-a359633d949c)


#### CONFUSION ARRAY:

![Screenshot 2025-04-07 033141](https://github.com/user-attachments/assets/c4ca3abf-6ff6-4468-9f1a-412785d72081)

#### ACCURACY VALUE:

![Screenshot 2025-04-07 033145](https://github.com/user-attachments/assets/8f40966d-bd21-4d16-b915-bbedecfee104)


#### CLASSFICATION REPORT:


![Screenshot 2025-04-07 033151](https://github.com/user-attachments/assets/1d7bea6a-40fa-46fb-8c43-374f606d04a2)

#### PREDICTION:


![Screenshot 2025-04-07 033159](https://github.com/user-attachments/assets/51f33c88-7c5d-4669-becc-7e116d729cff)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
