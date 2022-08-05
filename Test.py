import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from NNFromScratch import NeuralNetwork

X,y = make_blobs(5000,centers= 2)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)


nn = NeuralNetwork()
nn.Add(3,activation_fun='tanh')
nn.Add(4,activation_fun='relu')
nn.Add(1,activation_fun='sigmoid')



nn.Fit(X_train,y_train , learning_rate = 0.001 , iterations = 10000)
nn.History()



predictions = nn.predict(X_test)

print(y_test[:10])
print(predictions[0,:10])

    