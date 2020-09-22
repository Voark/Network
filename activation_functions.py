import math
import cupy as np

def list_functions():
    functions = {'relu': relu, 'sigmoid': sigmoid}
    return functions

"""
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
"""

sigmoid_kernel = np.ElementwiseKernel(
    'float64 x',
    'float64 y',
    'y = 1.0 / (1 + exp(-x))'
)

sigmoid_derivative_kernel = np.ElementwiseKernel(
    'float64 x',
    'float64 y',
    'y = exp(-x) / pow((1.0 + exp(-x)), 2.0)'
)

def relu(x, derivative = False):
    pass

def sigmoid(x, derivative = False):
    if not derivative:
        return sigmoid_kernel(x)
    else:
        return sigmoid_derivative_kernel(x)
    