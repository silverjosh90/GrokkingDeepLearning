import numpy as np

    street_lights= np.array(  [[1,0,1],
                             [0,1,1],
                             [0,0,1],
                             [1,1,1],
                             [0,1,1],
                             [1,0,1]]
                             )
    walk_or_stop = np.array([
                             0,
                             1,
                             0,
                             1,
                             1,
                             0
                             ])

class NeuralNetwork:
    def __init__(self, data , weights, results, learnrate):
        self.data = data
        self.weights = weights
        self.results = results
        self.learnrate = learnrate
        self.total_error = 0

    def train(self):
        for iteration in range(40):
            self.total_error = 0
            for row_index in range(len(self.results)):
                input =  self.data[row_index]
                goal_prediction = self.results[row_index]

                prediction = np.dot(input, self.weights)

                error = (goal_prediction - prediction) ** 2
                self.total_error += error
                print("Total Error: %s" % (self.total_error))


                delta = prediction - goal_prediction
                # print('prediction %s' % (prediction))
                # print('Intended goal_prediction %s' % (goal_prediction))

                self.weights -= (delta * input) * self.learnrate
                print(' this ish looks like %s' % (delta * input))
                # print('updated weights %s' % (self.weights))



neural_network = NeuralNetwork(street_lights, np.array([0.5,0.48,-0.7]), walk_or_stop, 0.1)

neural_network.train();
