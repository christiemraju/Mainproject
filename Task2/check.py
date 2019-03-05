import numpy as np
from perceptron import Perceptron
from random import randint

training_inputs = []
labels = []
for i in range(10000):
    a=randint(0,999)
    training_inputs.append(a)
    if a<100:
        labels.append(1)
    else:
        labels.append(0)

#for i t
#labels = np.array([1,0])

perceptron = Perceptron(1)
perceptron.train(training_inputs, labels)
print(perceptron.weights)

while(True):
    inp = input()
    if( perceptron.predict(inp)==1):
        print "Less than 100"
    else:
        print "more than 100"
    if inp == -1:
        break

