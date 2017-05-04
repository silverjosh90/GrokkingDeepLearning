import numpy as np

class Learning_Error:
    def __init__(self, weight, learnrate=0.001):
        self.hot_and_cold_weight = weight
        self.learnrate = learnrate
        self.gradient_descent_weight = weight

    def see_errors(self, input, target, iterations):
        for i in range(iterations):
            print(i)
            self.hot_and_cold_error(input,target)
            self.gradient_descent(input,target)

    def __str__(self):
        return "I am the weight %s.\nI am the learning rate %s. " % (self.hot_and_cold_weight, self.learnrate)

    def hot_and_cold_error(self, input, target):
        original_prediction = np.dot(input,self.hot_and_cold_weight)
        self.original_error = (target - original_prediction) ** 2

        weight = (self.hot_and_cold_weight[0] + self.learnrate)
        adding_prediction = np.dot(input, weight)
        self.adding_error = (target - adding_prediction) ** 2


        weight = (self.hot_and_cold_weight[0] - self.learnrate)
        subtraction_prediction = np.dot(input, weight)
        self.subtraction_error = (target - subtraction_prediction) ** 2


        if(self.original_error > self.adding_error or self.original_error > self.subtraction_error ):
            min_error = min(self.original_error, self.adding_error, self.subtraction_error)
            if(min_error == self.adding_error):
                self.hot_and_cold_weight[0] += self.learnrate
            elif(min_error == self.subtraction_error):
                self.hot_and_cold_weight[0] -= self.learnrate
        else:
            print('NOT ANY LOWER')
        print("I am the hot and cold weight %s" % np.around(self.hot_and_cold_weight[0], decimals=9))

    def gradient_descent(self, input, target):
        pred = np.dot(input,self.gradient_descent_weight[0])
        error = (target - pred) ** 2
        print('I am the error %s' % np.around(error, decimals=9))
        delta = target - pred
        weight_delta = np.around(delta * input, decimals=9)
        self.gradient_descent_weight[0] -= weight_delta * self.learnrate
        print("I am gradient_descent weight: %s" % np.around(self.gradient_descent_weight[0], decimals=9))






        # prediction = input * self.gradient_descent_weight[0]
        # error = (target - prediction) ** 2
        #
        # print('prediction %s' % (prediction))
        # # How much the prediction mixed by
        # delta = target - prediction
        # print("I AM THE DELTA %s" % (delta))
        #
        # weight_delta = delta * input
        # print("I AM THE WEIGHT DELTA %s" % (weight_delta))
        # self.gradient_descent_weight[0] -= weight_delta * self.learnrate
        # print('I am the gradient descent weight %s' % (self.gradient_descent_weight))




net = Learning_Error([.4])
net.see_errors(8.5, 1, 200)
print(net)
