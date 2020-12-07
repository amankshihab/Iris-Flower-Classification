#importing required libraries
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import joblib
from sklearn.metrics._classification import classification_report, confusion_matrix

#naming columns
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width','class']
#loading dataset
dt = pd.read_csv('iris.csv', names = columns)

#setting model
model = KNeighborsClassifier(n_neighbors = 3)

#independent variables to x
x = np.array(dt.iloc[:, :4])
#dependent variables to y
y = np.array(dt.iloc[:, 4])

#splitting into training and validation sets
xt, xv, yt, yv = train_test_split(x, y, train_size = 0.7, random_state = 1)

#fitting the model to the dataset
model.fit(xt, yt)

#storing predictions to pred
pred = model.predict(xv)
#getting accuracy scores
acc_score = accuracy_score(yv ,pred)
print("Acuracy Score:", acc_score)
print(classification_report(yv, pred))

joblib.dump(model, 'irismodel.model')