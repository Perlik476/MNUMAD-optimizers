import numpy as np

def gradient_descent(f: callable, df: callable, x0: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
    """
    Gradient descent algorithm for unconstrained optimization.
    :param f: function to be minimized
    :param df: gradient of f
    :param x0: initial point
    :param alpha: learning rate
    :param max_iter: maximum number of iterations
    :return: minimum point
    """

    assert alpha > 0, "alpha must be positive"
    assert max_iter > 0, "max_iter must be positive"

    x = x0
    for _ in range(max_iter):
        x = x - alpha * df(x)
    return x


def gauss_newton(f: callable, df: callable, x0: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
    """
    Gauss-Newton algorithm for unconstrained optimization.
    :param f: function to be minimized
    :param df: gradient of f
    :param x0: initial point
    :param alpha: learning rate
    :param max_iter: maximum number of iterations
    :return: minimum point
    """
    x = x0
    for i in range(max_iter):
        print(i)
        x = x - alpha * np.linalg.inv(df(x).T @ df(x)) @ df(x).T @ f(x)
    return x