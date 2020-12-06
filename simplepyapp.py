#importing required libraries
import joblib
import numpy as np

#loading saved model
model = joblib.load('irismodel.model')

#taking inputs
sl = float(input("Enter sepal length:")) #sepal length
sw = float(input("Enter sepal width:")) #sepal width
pl = float(input("Enter petal length:")) #petal length 
pw = float(input("Enter petal width:")) #petal width

#reshaping input to feed into the model
x = np.array([sl, sw, pl, pw]).reshape(1,-1)

#predicting class
p = model.predict(x)
print("Predicted class: ", p)