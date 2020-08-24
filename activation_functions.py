import math
def list_functions():
    functions = {'relu': relu, 'sigmoid': sigmoid}
    return functions

def relu(x, derivative = False):
    if not derivative:
        return x * (x > 0)
    else:
        return 1 * (x > 0)

def sigmoid(x, derivative = False):
    if not derivative:
        return 1 / (1 + math.e ** -x)
    else:
        return sigmoid(x) * (1 - sigmoid(x))

