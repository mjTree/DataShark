import NeuralNetwork
import numpy as np

nn = NeuralNetwork.NeuralNetwork([2,2,1],"tanh")

x= np.array([[0,0],[0,1],[1,0],[1,1]])
y= np.array([0,1,1,0])

nn.fit(x, y,0.2,10000)

for i in [[0,0],[0,1],[1,0],[1,1]]:
    print(i,nn.predict(i))
