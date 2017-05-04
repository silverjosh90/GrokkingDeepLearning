import numpy as np

class Neural_network:
    def __init__(self, input, input_to_hidden_weights, hidden_to_output_weights):
        self.inputs = input
        self.input_to_hidden_weights = input_to_hidden_weights
        self.hidden_to_output_weights = hidden_to_output_weights
        self.final_output = 0
    def get_predictions(self):
        # Initializing empty arrays equivalent to the amount of output nodes
        hidden_output = [0] * len(self.inputs)
        output = [0] * len(self.inputs)

        # For each set of inputs (single prediction) create a prediction.
        # to reiterate, each list of inputs represent a seperate prediction
        for i,input in enumerate(self.inputs):
            hidden_output[i] = self.predict_from_scratch(input, self.input_to_hidden_weights)

        self.hidden_output = hidden_output

        # Once  output from input_to_hidden layer is found perform the exact same prediction with the nhidden_to_output_weights using the output from input_to_hidden as the new input
        for i, out in enumerate(hidden_output):
            output[i] = self.predict_from_scratch(out, self.hidden_to_output_weights)
            self.output = output


    def predict_from_scratch(self, inputs, weights):
        assert(len(weights) == len(inputs))

        # this should be more dynamic and based on how many outputs are wanted
        prediction = [0,0,0]

        # for every expected output node
        for weight_vect, vector in enumerate(prediction):
            for weight_index, weight in enumerate(weights[weight_vect]):
                multiplied = weight * inputs[weight_index]
                prediction[weight_vect] += multiplied

        return prediction

# A bit unclear. These are three seperate inputs, and therefore yield three different predictions.
inputs = [[9.5,.65,1.7], [10,.35,1.4], [7.5,.95,1.6]]

# Each index in the outer array of weights refers to an output. Therefore the weights below yeild three output nodes
input_to_hidden_weights = [[.4,.3,-.3],[0.2, 0.2, 0.1],[0.0, 1.3, 0.1]]

 
hidden_to_output_weights = [[.1,.1,-.3],[0.1, 0.2, 0.0],[0.0, 1.3, 0.1]]


net = Neural_network(inputs, input_to_hidden_weights, hidden_to_output_weights)

# print(net.predict_from_scratch(inputs,weights))
net.get_predictions()

print(net.hidden_output)
print(net.output)
