from typing import Callable, Optional

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


class LevenbergMarquardt:
    class LambdaParam:
        def __init__(
            self,
            lambda0: float,
            lambda_change: float,
            eps: float = 1e-16,
            lambda_fun: Optional[Callable] = None,
        ):
            assert lambda_change > 1, "lambda_change must be greater than 1"
            assert lambda0 > 0, "lambda0 must be positive"
            assert eps > 0, "eps must be positive"

            self.value = lambda0
            self.value_change = lambda_change
            self.eps = eps

            if lambda_fun is None:
                self.lambda_fun = self.lambda_fun_default
            else:
                self.lambda_fun = lambda_fun

        def __call__(
            self, F: Callable, next_point: Callable, p: np.ndarray, i: int
        ) -> float:
            return self.lambda_fun(F, next_point, p, i)

        def lambda_fun_default(
            self, F: Callable, next_point: Callable, p: np.ndarray, i: int
        ) -> float:
            current_err = np.linalg.norm(F(p))
            next_err = np.linalg.norm(F(next_point(p, self.value)))

            # print(f"lambda_fun: {self.value}, {current_err}, {next_err}")

            if self.value < self.eps:
                return self.value

            if current_err > next_err:
                self.value = self.value / self.value_change
            else:
                self.value = self.value * self.value_change
                return self.lambda_fun_default(F, next_point, p, i)

            return self.value

    def __init__(
        self,
        F: Callable,
        DF: Callable,
        lambda_param: Optional[LambdaParam] = None,
    ):
        self.F = F
        self.DF = DF
        if lambda_param is None:
            self.lambda_param = self.LambdaParam(0.1, 2)
        else:
            self.lambda_param = lambda_param

    def next_point(self, p: np.ndarray, lambda_param: float) -> np.ndarray:
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
                p = self.next_point(
                    p, self.lambda_param(self.F, self.next_point, p, i)
                )
            except np.linalg.LinAlgError:
                return p, err

        return p, err
