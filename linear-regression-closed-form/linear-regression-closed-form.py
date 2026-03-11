import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    x_t=np.transpose(X)
    return np.linalg.inv(x_t@X)@(x_t@y)
        