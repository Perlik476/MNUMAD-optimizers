import numpy as np

def gradient_descent(F: callable, DF: callable, p0: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
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

    p = p0
    for _ in range(max_iter):
        p = p - alpha * DF(p)
    return p


def gauss_newton(F: callable, DF: callable, p0: np.ndarray, max_iter: int) -> np.ndarray:
    """
    Gauss-Newton algorithm for unconstrained optimization.
    :param f: function to be minimized
    :param df: gradient of f
    :param x0: initial point
    :param alpha: learning rate
    :param max_iter: maximum number of iterations
    :return: minimum point
    """
    p = p0
    for i in range(max_iter):
        print(f"iter {i}: p = {p}, ||F(p)|| = {np.linalg.norm(F(p))}")
        try:
            p = p - np.linalg.solve(DF(p).T @ DF(p),  DF(p).T @ F(p))
        except np.linalg.LinAlgError:
            return p
    return p