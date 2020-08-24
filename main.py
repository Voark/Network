from network import Network
import numpy as cp
import mnist
x_train, t_train, x_test, t_test = mnist.load()
y_train = cp.zeros((10, 60000))
y_test = cp.zeros((10, 10000))
for i in range(60000):
    y_train[t_train[i], i] = 1
for i in range(10000):
    y_test[t_test[i], i] = 1
x_train = x_train.T / 255
x_test = x_test.T / 255
test_network = Network()
test_network.set_input(784)
test_network.set_output(10)
test_network.add_hidden_layers(1, 30, 'sigmoid')
test_network.set_output_activation_function('sigmoid')
test_network.set_error_function('mean squared error')
test_network.set_gradient_descent_algorithm('stochastic')
test_network.set_alpha(.5)
test_network.compile_network()
#test_network.save_model('test_model')

#test_network = Network()
#test_network.load_model('test')
#test_x = [np.transpose(np.array([[1, 1, 2, 2, 5]]))]
#test_y = [np.transpose(np.array([[0, .5]]))]

test_network.train(x_train, y_train, x_test, y_test, 10)
print('done')