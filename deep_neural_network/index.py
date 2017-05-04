import numpy as np

street_lights= np.array(  [[1,0,1],
                         [0,1,1],
                         [0,0,1],
                         [1,1,1],
                         [0,1,1],
                         [1,0,1]]
                         )
walk_or_stop = np.array([[0,1,0,1,1,0]]).T

class NeuralNetwork:
    def __init__(self, inputs, result, alpha):
        self.inputs = inputs
        self.results = result
        self.alpha = alpha
        self.hidden_size = 4
        self.weights_0_1 = 2 * np.random.random((3, self.hidden_size)) -1
        self.weights_1_2 = 2 * np.random.random((self.hidden_size, 1)) -1

    def set_negative_to_zero(self, val):
        return (val > 0) * val

    def set_negative_to_zero_deriv(self,val):
        return (val > 0)

    def train(self):
     for iteration in range(50):
         layer_2_error = 0
         for i in range(len(self.inputs)):
             layer_0 = self.inputs[i:i+1]
             layer_1= self.set_negative_to_zero(np.dot(layer_0, self.weights_0_1))
             layer_2 = np.dot(layer_1, self.weights_1_2)
             layer_2_error += (self.results[i:i+1] - layer_2) ** 2
             delta = layer_2 - self.results[i:i+1]

             delta_hidden = np.dot(delta, self.weights_1_2.T) * self.set_negative_to_zero_deriv(layer_1)

             self.weights_1_2 -= np.dot(layer_1.T, delta) * self.alpha
             self.weights_0_1 -= np.dot(layer_0.T, delta_hidden) * self.alpha
            #  print("I am the first weight update %s" % (self.weights_0_1))
            #  print("I am the first second update %s" % (self.weights_1_2))
         print("I am the error %s" % (layer_2_error))


net = NeuralNetwork(street_lights, walk_or_stop, .2)
net.train()
