from typing import Callable

import numpy as np


class DifferentiableFunction:
    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        DF: Callable[[np.ndarray], np.ndarray],
        N: int,
        M: int
    ):
        """
        :param F: function from R^N to R^M
        :param DF: differential of F
        :param N: dimension of domain
        :param M: dimension of codomain
        """

        self.F = F
        self.DF = DF

        zero = np.zeros(N)
        assert F(zero).size == M, "F must be a function from R^N to R^M"
        assert DF(zero).shape == (M, N), "DF must be a function from R^N to R^(MxN)"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.F(x)

    def differential(self, x: np.ndarray) -> np.ndarray:
        return self.DF(x)


def gradient_descent(
    R: DifferentiableFunction,
    p0: np.ndarray,
    alpha: float,
    max_iter: int,
    silent: bool = True,
) -> tuple[np.ndarray, np.float64]:
    """
    Gradient descent algorithm for nonlinear least squares.
    :param f: function to be minimized
    :param df: gradient of f
    :param p0: initial point
    :param alpha: step size
    :param max_iter: maximum number of iterations
    :return: minimum point
    """

    assert max_iter > 0, "max_iter must be positive"
    assert alpha > 0, "alpha must be positive"

    DR = R.differential
    p = p0
    err = np.linalg.norm(R(p))

    for i in range(max_iter):
        err = np.linalg.norm(R(p))
        if not silent:
            print(f"iter {i}: p = {p}, ||F(p)|| = {err}")
        p = p - alpha * DR(p).T @ R(p)
    
    return p, err


def gauss_newton(
    R: DifferentiableFunction,
    p0: np.ndarray,
    max_iter: int,
    silent: bool = True,
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

    DR = R.differential
    p = p0
    err = np.linalg.norm(R(p))

    for i in range(max_iter):
        err = np.linalg.norm(R(p))
        if not silent:
            print(f"iter {i}: p = {p}, ||F(p)|| = {err}")
        try:
            p = p - np.linalg.solve(DR(p).T @ DR(p), DR(p).T @ R(p))
        except np.linalg.LinAlgError:
            return p, err

    return p, err


class LevenbergMarquardt:
    class LambdaParam:
        def __call__(
            self,
            R: DifferentiableFunction,
            step: Callable[[np.ndarray, float], np.ndarray],
            p: np.ndarray,
            i: int,
        ) -> float:
            raise NotImplementedError
        
    class LambdaParamConstant(LambdaParam):
        def __init__(
            self,
            lambda0: float
        ):
            assert lambda0 >= 0, "lambda0 must be non-negative"
            self.value = lambda0

        def __call__(
            self,
            R: DifferentiableFunction,
            step: Callable[[np.ndarray, float], np.ndarray],
            p: np.ndarray,
            i: int,
        ) -> float:
            return self.value
        
    class LambdaParamDefaultOptimizer(LambdaParam):
        def __init__(
            self,
            lambda0: float,
            lambda_change: float,
            eps: float = 1e-16,
        ):
            assert lambda_change > 1, "lambda_change must be greater than 1"
            assert lambda0 >= 0, "lambda0 must be non-negative"
            assert eps > 0, "eps must be positive"

            self.value = lambda0
            self.value_change = lambda_change
            self.eps = eps

        def __call__(
            self,
            R: DifferentiableFunction,
            step: Callable[[np.ndarray, float], np.ndarray],
            p: np.ndarray,
            i: int,
        ) -> float:
            current_err = np.linalg.norm(R(p))
            next_err = np.linalg.norm(R(step(p, self.value)))

            if self.value < self.eps:
                return self.value

            if current_err > next_err:
                self.value = self.value / self.value_change
            else:
                self.value = self.value * self.value_change
                return self.__call__(R, step, p, i)

            return self.value

    def __init__(
        self,
        R: DifferentiableFunction,
        lambda_param_fun: Callable[
            [
                DifferentiableFunction,
                Callable[[np.ndarray, float], np.ndarray],
                np.ndarray,
                int,
            ],
            float,
        ],
    ):
        self.R = R
        self.DR = R.differential
        self.lambda_param_fun = lambda_param_fun

    def step(self, p: np.ndarray, lambda_param: float) -> np.ndarray:
        return p - np.linalg.solve(
            self.DR(p).T @ self.DR(p) + lambda_param * np.eye(p.size),
            self.DR(p).T @ self.R(p),
        )

    def optimize(
        self,
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
        err = np.linalg.norm(self.R(p))

        for i in range(max_iter):
            err = np.linalg.norm(self.R(p))

            if not silent:
                print(f"iter {i}: p = {p}, ||F(p)|| = {err}")

            try:
                p = self.step(p, self.lambda_param_fun(self.R, self.step, p, i))
            except np.linalg.LinAlgError:
                return p, err

        return p, err
