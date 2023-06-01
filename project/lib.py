from typing import Callable

import numpy as np


def gauss_newton(
    F: Callable, DF: Callable, p0: np.ndarray, max_iter: int, silent: bool = True
) -> tuple[np.ndarray, np.float64]:
    """
    Gauss-Newton algorithm for unconstrained optimization.
    :param f: function to be minimized
    :param df: gradient of f
    :param x0: initial point
    :param max_iter: maximum number of iterations
    :return: minimum point
    """
    assert max_iter > 0, "max_iter must be positive"

    p = p0
    err = np.linalg.norm(F(p))

    for i in range(max_iter):
        err = np.linalg.norm(F(p))
        if not silent:
            print(f"iter {i}: p = {p}, ||F(p)|| = {err}")
        try:
            p = p - np.linalg.solve(DF(p).T @ DF(p), DF(p).T @ F(p))
        except np.linalg.LinAlgError:
            return p, err

    return p, err


def levenberg_marquardt(
    lambda_fun: Callable,
    F: Callable,
    DF: Callable,
    p0: np.ndarray,
    max_iter: int,
    silent: bool = True,
) -> tuple[np.ndarray, np.float64]:
    """
    Levenberg-Marquardt algorithm for unconstrained optimization.
    :param f: function to be minimized
    :param df: gradient of f
    :param x0: initial point
    :param max_iter: maximum number of iterations
    :return: minimum point
    """
    assert max_iter > 0, "max_iter must be positive"

    p = p0
    err = np.linalg.norm(F(p))

    for i in range(max_iter):
        err = np.linalg.norm(F(p))
        if not silent:
            print(f"iter {i}: p = {p}, ||F(p)|| = {err}")
        try:
            p = p - np.linalg.solve(
                DF(p).T @ DF(p) + lambda_fun(F, DF, p, i) * np.eye(p.size),
                DF(p).T @ F(p),
            )
        except np.linalg.LinAlgError:
            return p, err

    return p, err
