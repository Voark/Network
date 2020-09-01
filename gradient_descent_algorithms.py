import numpy as np
#TODO: condense batch, SGD, and mini-batch gradient descent to SGD with variable batch size
#TODO: Add other optimizers (Adam)
def list_functions():
    functions = {'batch': batch_gradient_descent, 'stochastic': stochastic_gradient_descent, 'mini batch': mini_batch_gradient_descent}
    return functions

def batch_gradient_descent(network, train_x, train_y, test_x, test_y, epochs):
    for epoch in range(epochs):
        total_gradient = [[np.zeros(w.shape) for w in reversed(network.weights)], [np.zeros(b.shape) for b in reversed(network.biases)]]
        for x, y in zip(train_x, train_y):
            x, y = network.reshape_data(x, y)
            a, z = network.forward_prop(x)
            weight_gradient, bias_gradient = network.back_prop(a, z, y)
            for i in range(len(network.weights)):
                total_gradient[0][i] += weight_gradient[i]  
                total_gradient[1][i] += bias_gradient[i]
        for i in range(len(total_gradient)):
            for j in range(len(total_gradient[i])):
                total_gradient[i][j] /= train_x.shape[0]
        network.update_weights_and_biases(*total_gradient)
        acc = network.test_accuracy(test_x, test_y) 
        print(f'Epoch: {epoch + 1}. Accuracy: {acc * 100}%')
        network.history.append(acc)

def stochastic_gradient_descent(network, train_x, train_y, test_x, test_y, epochs):
    indices = np.arange(0, train_x.shape[0], 1)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        train_x, train_y = train_x[indices], train_y[indices]
        for x, y in zip(train_x, train_y):
            x, y = network.reshape_data(x, y)
            a, z = network.forward_prop(x)
            weight_gradient, bias_gradient = network.back_prop(a, z, y)
            network.update_weights_and_biases(weight_gradient, bias_gradient)
        acc = network.test_accuracy(test_x, test_y) 
        print(f'Epoch: {epoch + 1}. Accuracy: {acc * 100}%')
        network.history.append(acc)
    
def mini_batch_gradient_descent(network, train_x, train_y, test_x, test_y, epochs, mini_batch_size = 30):
    num_batches = train_x.shape[0] / mini_batch_size 
    indices = np.arange(0, train_x.shape[0], 1)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        train_x, train_y = train_x[indices], train_y[indices]
        split_x, split_y = np.split(train_x, num_batches), np.split(train_y, num_batches)
        for batch_x, batch_y in zip(split_x, split_y):
            total_weight_gradient = [np.zeros(w.shape) for w in reversed(network.weights)]
            total_bias_gradient = [np.zeros(b.shape) for b in reversed(network.biases)]
            for x, y in zip(batch_x, batch_y):
                x, y = network.reshape_data(x, y)
                a, z = network.forward_prop(x)
                weight_gradient, bias_gradient = network.back_prop(a, z, y)
                total_weight_gradient += weight_gradient
                total_bias_gradient += bias_gradient
            for i in range(len(total_bias_gradient)):
                total_weight_gradient[i] /= batch_x.shape[0]
                total_bias_gradient[i] /= batch_x.shape[0]
            network.update_weights_and_biases(total_weight_gradient, total_bias_gradient)
        acc = network.test_accuracy(test_x, test_y) 
        print(f'Epoch: {epoch + 1}. Accuracy: {acc * 100}%')
        network.history.append(acc)
    