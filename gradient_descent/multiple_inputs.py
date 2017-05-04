import numpy as np

class Neural_Network:
    def __init__(self, input, weight, target, learnrate=0.01):
        self.input = input
        self.weight = weight
        self.target = target
        self.learnrate = learnrate


    def run_all_inputs_gradient_descent(self):
        for i, val in enumerate(self.target):
            inputs = [0] * len(self.input)
            for y, inp in enumerate(self.input):
                inputs[y] = self.input[y][i]
            self.grad_descent(inputs, val)

    def grad_descent(self, input, target):
        prediction = 0
        for i, value in enumerate(input):
            prediction += input[i] * self.weight[i]

        print(" I AM THE PREDICTION %s" % (prediction))
        error = (target - prediction) ** 2
        print("I am the error %s" % error)
        delta = target - prediction

        print("I am the delta %s" % delta)

        deriv = self.mult_vector_by_number(delta, input)

        for i in range(len(deriv)):
            self.weight[i] -= self.learnrate * deriv[i]

        print('I am the new weight', self.weight)

    def mult_vector_by_number(self, number, vector):
        output = [0] * len(vector)
        for i, val in enumerate(vector):
            output[i] = val * number

        return output

    def grad_descent_single_input_multiple_outputs(self,input, weight, target):
        prediction = self.mult_vector_by_number(input, weight)
        error = [0] * len(prediction)
        delta = [0] * len(prediction)

        for i in range(len(error)):
            error[i] = (target[i] - prediction[i]) ** 2
            delta[i] = target[i] - prediction[i]
            weight[i] -= (delta[i] * input) * self.learnrate

        print ('I am the delta: %s. I am the error: %s. I am the new weights: %s' % (delta, error, weight))

    def grad_descent_multiple_inputs_multiple_outputs(self, input, weight, target):
        prediction = [0] * len(target)
        for i in range(len(weight)):
            print('Prediction step number %s: \n I am the input %s. \n I am the weight %s \n \n' % (i,input,weight[i]))
            prediction[i] = self.multiply_vectors(input, weight[i])

        error = [0] * len(prediction)
        delta = [0] * len(prediction)
        weight_delta = [0] * len(prediction)

        for i in range(len(prediction)):
            error[i] = (target[i] - prediction[i]) ** 2
            delta[i] = target[i] - prediction[i]
            for v in range(len(input)):
                weight_delta[i] += delta[i] * input[v]
            weight_delta[i] *= self.learnrate

        for outer_index in range(len(weight)):
            for inner_index in range(len(weight[outer_index])):
                weight[outer_index][inner_index] -= weight_delta[outer_index]

        print("I am the updated weights homeslice %s" % (weight))


    def multiply_vectors(self,input, weight):
        assert(len(input) == len(weight))
        output = 0
        for i in range(len(input)):
            output += input[i] * weight[i]
        return output

# Baseball example

toes = [8.5, 9.5, 9.9, 9.0]
wlrec = [0.65, 0.8, 0.8, 0.9]
nfans = [1.2, 1.3, 0.5, 1.0]


input = [toes , wlrec , nfans]
weight = [0.1, 0.2, -.1]

win_or_lose_binary = [1,1,0,1]


mult_input = [8.5, .65, 1.2]
mult_weight = [[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 1.3, 0.1]]
mult_target = [0.1, 1, 0.1]


neural_net = Neural_Network(input, weight, win_or_lose_binary)
# neural_net.run_all_inputs_gradient_descent()
# neural_net.grad_descent_single_input_multiple_outputs(0.65, [0.3, 0.2, 0.9], [0.1, 1, 0.1])
neural_net.grad_descent_multiple_inputs_multiple_outputs(mult_input, mult_weight, mult_target)

# def neural_network(input, weight):
#     prediction = input * weight
#     return prediction

# def w_sum(a,b):
#     assert(len(a) == len(b))
#     output = 0
#     for i in range(a):
#       output += (a[i] * b[i])
#    return output

# def ele_mul(number,vector):
#    output = [0,0,0]
#    assert(len(output) == len(vector))
#    for i in xrange(len(vector)):
#        output[i] = number * vector[i]
#    return output

# def vect_mat_mul(vect,matrix):
    # assert(len(a) == len(b))
    # output = vector_of_zeros(len(vect))
    # for i in range(len(vect)): output[i]=w_sum(vect,matrix[i])
    # return output
