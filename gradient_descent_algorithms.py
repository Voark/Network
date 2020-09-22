import cupy as np
#TODO: condense batch, SGD, and mini-batch gradient descent to SGD with variable batch size
#TODO: Add other optimizers (Adam)
def list_functions():
    functions = {'stochastic': stochastic_gradient_descent}
    return functions
    
def stochastic(network, train_x, train_y, test_x, test_y, epochs, mini_batch_size = 30):
    num_batches = train_x.shape[0] / mini_batch_size 
    indices = np.arange(0, train_x.shape[0], 1)
    for epoch in range(epochs):
        np.random.shuffle(indices)
        train_x, train_y = train_x[indices], train_y[indices]
        split_x, split_y = np.split(train_x, int(num_batches)), np.split(train_y, int(num_batches))
        comb_x, comb_y = np.array([np.column_stack(i) for i in split_x]), np.array([np.column_stack(i) for i in split_y])
        for batch_x, batch_y in zip(comb_x, comb_y):
            a, z = network.forward_prop(batch_x)
            weight_gradient, bias_gradient = network.back_prop(a, z, batch_y)
            network.update_weights_and_biases(weight_gradient, bias_gradient)
        acc = network.test_accuracy(test_x, test_y) 
        print(f'Epoch: {epoch + 1}. Accuracy: {acc * 100}%')
        network.history.append(acc)
    