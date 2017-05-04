import numpy as np

class Gradient_Descent:
    def __init__(self, weight, input, target, learnrate=0.001):
        self.weight = weight
        self.input = input
        self.target = target
        self.learnrate = learnrate

    def grad_descent(self, iterations):
        for i in range(iterations):
            prediction = self.input * self.weight
            error = (self.target - prediction) ** 2
            delta =  self.target - prediction
            derivative = delta * self.input
            self.weight -= derivative * self.learnrate
            print('I am the new weight %s' % (self.weight))
        print('Final Gradient Descent weight: %s' % (self.weight))


descent = Gradient_Descent(0.4, 3, 1)

descent.grad_descent(1)
