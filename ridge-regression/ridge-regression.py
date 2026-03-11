def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X=np.array(X)
    y=np.array(y)
    x_t=np.transpose(X)
    I=np.identity(X.shape[1])
    return np.linalg.inv(x_t@X + lam*I)@(x_t@y)