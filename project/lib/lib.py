from typing import Callable

import numpy as np


class DifferentiableFunction:
    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        DF: Callable[[np.ndarray], np.ndarray],
    ):
        self.F = F
        self.DF = DF

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.F(x)

    def differential(self, x: np.ndarray) -> np.ndarray:
        return self.DF(x)


def gauss_newton(
    F: DifferentiableFunction,
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

    DF = F.differential
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


class LevenbergMarquardt:
    class LambdaParam:
        def __call__(
            self,
            F: DifferentiableFunction,
            next_point: Callable[[np.ndarray, float], np.ndarray],
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
            F: DifferentiableFunction,
            next_point: Callable[[np.ndarray, float], np.ndarray],
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
            F: DifferentiableFunction,
            next_point: Callable[[np.ndarray, float], np.ndarray],
            p: np.ndarray,
            i: int,
        ) -> float:
            current_err = np.linalg.norm(F(p))
            next_err = np.linalg.norm(F(next_point(p, self.value)))

            if self.value < self.eps:
                return self.value

            if current_err > next_err:
                self.value = self.value / self.value_change
            else:
                self.value = self.value * self.value_change
                return self.__call__(F, next_point, p, i)

            return self.value

    def __init__(
        self,
        F: DifferentiableFunction,
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
        self.F = F
        self.DF = F.differential
        self.lambda_param_fun = lambda_param_fun

    def step(self, p: np.ndarray, lambda_param: float) -> np.ndarray:
        return p - np.linalg.solve(
            self.DF(p).T @ self.DF(p) + lambda_param * np.eye(p.size),
            self.DF(p).T @ self.F(p),
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
        err = np.linalg.norm(self.F(p))

        for i in range(max_iter):
            err = np.linalg.norm(self.F(p))

            if not silent:
                print(f"iter {i}: p = {p}, ||F(p)|| = {err}")

            try:
                p = self.step(p, self.lambda_param_fun(self.F, self.step, p, i))
            except np.linalg.LinAlgError:
                return p, err

        return p, err
