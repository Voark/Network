# import cupy as cp
import numpy as np
import activation_functions
import gradient_descent_algorithms
import error_functions


class Network:
    def __init__(self):
        self.hidden_layer_sizes = []
        self.weights = []
        self.biases = []
        # need to add functions
        self.activation_function_options = activation_functions.list_functions()
        self.error_function_options = error_functions.list_functions()
        self.gradient_descent_algorithm_options = gradient_descent_algorithms.list_functions()
        self.activation_functions_text = []
        self.activation_functions = []
        self.error_function_text = None
        self.error_function = None
        self.gradient_descent_algorithm_text = None
        self.gradient_descent_algorithm = None
        self.compiled = 0
        self.history = []

    def set_input(self, input_size):
        if type(input_size) == int:
            self.input_size = input_size
        else:
            print("Ints only")

    def set_output(self, output_size):
        if type(output_size) == int:
            self.output_size = output_size
        else:
            print("Ints only")

    def add_hidden_layers(self, number, size, activation):
        if (
            type(number) == int
            and type(size) == int
            and activation in self.activation_function_options.keys()
        ):
            for i in range(number):
                self.hidden_layer_sizes.append(size)
                self.activation_functions_text.append(activation)
                self.activation_functions.append(self.activation_function_options[activation])
        else:
            print("Number/size are not ints or activation function doesn't exist")
    
    def set_output_activation_function(self, activation):
        if activation in self.activation_function_options.keys():
            self.activation_functions_text.append(activation)
            self.activation_functions.append(self.activation_function_options[activation])
        else:
            print("Activation function doesn't exist")

    def set_error_function(self, error_function):
        if error_function in self.error_function_options.keys():
            self.error_function_text = error_function
            self.error_function = self.error_function_options[error_function]
        else:
            print("No error error function found")

    def set_gradient_descent_algorithm(self, gradient_descent_algorithm):
        if gradient_descent_algorithm in self.gradient_descent_algorithm_options.keys():
            self.gradient_descent_algorithm_text = gradient_descent_algorithm
            self.gradient_descent_algorithm = self.gradient_descent_algorithm_options[gradient_descent_algorithm]
        else:
            print("No gradient descent algorithm found")

    def reset_hidden_layers(self):
        self.weights = []
        self.biases = []
        self.compiled = 0

    def list_hidden_layers(self):
        return self.hidden_layer_sizes
    
    def set_alpha(self, alpha):
        self.alpha = alpha

    def compile_network(self):
        if not self.compiled:
            # create input layer
            try:
                self.weights.append(
                    np.random.normal(size = (self.hidden_layer_sizes[0], self.input_size))
                )
                self.biases.append(np.zeros((self.hidden_layer_sizes[0], 1)))
            except:
                print("Layer sizes and/or input size not defined")
                self.reset_hidden_layers()
                return
            # create hidden layer
            for i in range(1, len(self.hidden_layer_sizes)):
                self.weights.append(
                    np.random.normal(size = (self.hidden_layer_sizes[i], self.hidden_layer_sizes[i - 1])))
                self.biases.append(np.zeros((self.hidden_layer_sizes[i], 1)))
            # create output layer
            try:
                self.weights.append(
                    np.random.normal(size = (self.output_size, self.hidden_layer_sizes[-1]))
                )
                self.biases.append(np.zeros((self.output_size, 1)))
            except:
                print("Outputs not defined")
                self.reset_hidden_layers()
                return
            self.compiled = 1
        else:
            print("Already compiled")

    def forward_prop(self, x):
        z = []
        activations = [x]
        for weights, biases, activation_function, input in zip(self.weights, self.biases, self.activation_functions, activations):
            temp_z = np.add(np.matmul(weights, input), biases)
            z.append(temp_z)
            activations.append(activation_function(temp_z))
        return activations, z

    def back_prop(self, activations, z, y_hat):
        weight_gradient = []
        bias_gradient = []
        delta = np.multiply(
            self.error_function(activations[-1], y_hat, derivative = True), self.activation_functions[-1](z[-1], derivative = True)
        )
        weight_gradient.append(np.matmul(delta, np.transpose(activations[-2])))
        bias_gradient.append(delta)
        for a, z_layer, weights, activation_function in zip(
            reversed(activations[:-2]),
            reversed(z[:-1]),
            reversed(self.weights[1:]),
            reversed(self.activation_functions[:-1]),
        ):
            delta = np.multiply(activation_function(z_layer, derivative = True), np.matmul(np.transpose(weights), delta))
            bias_gradient.append(delta)
            weight_gradient.append(np.matmul(delta, np.transpose(a)))
        return weight_gradient, bias_gradient
    
    def update_weights_and_biases(self, weight_gradient, bias_gradient):
        for i in range(len(self.weights)):
            self.weights[i] += self.alpha * weight_gradient[-(i+1)]
            self.biases[i] += self.alpha * bias_gradient[-(i+1)]

    def train(self, train_x, train_y, test_x, test_y, epochs):
        self.gradient_descent_algorithm(self, train_x, train_y, test_x, test_y, epochs)
    
    def test_accuracy(self, test_x, test_y):
        correct = 0
        for x, y in zip(test_x, test_y):
            x, y = self.reshape_data(x, y)
            a, z = self.forward_prop(x)
            if a[-1].argmax() == y.argmax():
                correct += 1
        return correct / len(test_x)            
    
    def reshape_data(self, x, y):
        return x.reshape(self.input_size, 1), y.reshape(self.output_size, 1)
    
    def save_model(self, file_name):
        #save non weights/biases params
        params = [self.hidden_layer_sizes, self.activation_functions_text, self.error_function_text, self.gradient_descent_algorithm_text, self.alpha, self.compiled]
        with open(file_name + '.txt', 'w') as f:
            for param in params:
                if type(param) == list:
                    param = map(str, param)
                    f.write(','.join(param))
                    f.write('\n')
                else:
                    f.write(str(param) + '\n')
        #save weights and biases
        np.savez(file_name + "_weights", *self.weights)
        np.savez(file_name + "_biases", *self.biases)

    def load_model(self, file_name):
        #write all params to their respective variables
        try:
            with open(file_name + '.txt', 'r') as f:
                lines = f.readlines()
                param_tuple = []
                for i in range(len(lines)):
                    param_tuple.append(lines[i].strip().split(','))
                (self.hidden_layer_sizes, self.activation_functions_text, self.error_function_text, self.gradient_descent_algorithm_text, self.alpha, self.compiled) = param_tuple
        except:
            print('File doesn\'t exist')
            return
        #un-list 1 item lists
        self.error_function_text, self.gradient_descent_algorithm_text = self.error_function_text[0], self.gradient_descent_algorithm_text[0]
        #fix type
        self.hidden_layer_sizes = list(map(int, self.hidden_layer_sizes))
        self.activation_functions = [self.activation_function_options[func] for func in self.activation_functions_text]
        self.error_function = self.error_function_options[self.error_function_text]
        self.gradient_descent_algorithm = self.gradient_descent_algorithm_options[self.gradient_descent_algorithm_text]
        self.alpha = float(self.alpha[0])
        self.compiled = int(self.compiled[0])
        #load weights and biases
        weights = np.load(file_name + '_weights.npz')
        biases = np.load(file_name + '_biases.npz')
        for weight_layer, bias_layer in zip(weights.values(), biases.values()):
            self.weights.append(weight_layer)
            self.biases.append(bias_layer)
        weights.close()
        biases.close()
