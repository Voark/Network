def list_functions():
    functions = {'mean squared error': mean_squared_error}
    return functions

def mean_squared_error(y_out, y_expected, derivative = False):
    if derivative:
        return y_expected - y_out 
    else:
        return .5 * (y_expected - y_out) ** 2
