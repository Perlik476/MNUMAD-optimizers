from typing import Callable

import numpy as np
from numpy.typing import NDArray

from functions import Function


def gradient_descent(
    R: Function,
    p0: NDArray[np.float64],
    alpha: float,
    max_iter: int,
    silent: bool = True,
) -> tuple[NDArray[np.float64], np.float64]:
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

    for i in range(max_iter):
        if not silent:
            err = np.linalg.norm(R(p))
            print(f"iter {i}: p = {p}, ||R(p)|| = {err}")
        p = p - alpha * DR(p).T @ R(p)
    
    err = np.linalg.norm(R(p))
    return p, err


def gauss_newton(
    R: Function,
    p0: NDArray[np.float64],
    max_iter: int,
    alpha: float = 1,
    step_type: str = "least_squares",
    silent: bool = True,
) -> tuple[NDArray[np.float64], np.float64]:
    """
    Gauss-Newton algorithm for unconstrained optimization.
    :param f: function to be minimized
    :param df: gradient of f
    :param x0: initial point
    :param max_iter: maximum number of iterations
    :return: minimum point
    """
    assert max_iter > 0, "max_iter must be positive"
    assert step_type == "cholesky" or step_type == "least_squares", "step_type must be 'cholesky' or 'least_squares'"

    DR = R.differential
    p = p0

    for i in range(max_iter):
        if not silent:
            err = np.linalg.norm(R(p))
            print(f"iter {i}: p = {p}, ||R(p)|| = {err}")
        try:
            if step_type == "cholesky":
                try:
                    L = np.linalg.cholesky(DR(p).T @ DR(p))
                    y = np.linalg.solve(L, DR(p).T @ R(p))
                    d = np.linalg.solve(L.T, y)
                except np.linalg.LinAlgError:
                    d = np.linalg.solve(DR(p).T @ DR(p), DR(p).T @ R(p))
            elif step_type == "least_squares":
                d = np.linalg.lstsq(DR(p), R(p), rcond=None)[0]
        except np.linalg.LinAlgError:
            err = np.linalg.norm(R(p))
            print(f"Gauss-Newton ({step_type=}) failed in iteration nr {i}. Returning current point.")
            return p, err
        p = p - alpha * d
        
        # A = np.vstack((self.DR(p), np.sqrt(lambda_param) * np.eye(p.size)))
        # b = np.vstack((self.R(p).reshape(-1, 1), np.zeros((p.size, 1))))

        # d = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)

    err = np.linalg.norm(R(p))
    return p, err


def cgnr_normal_equations(A: NDArray[np.float64], b: NDArray[np.float64], max_iter: int, eps: float = 1e-6) -> NDArray[np.float64]:
    """
    Conjugate gradient method for solving normal equations.
    :param A: matrix of the least squares problem
    :param b: right-hand side of the least squares problem
    :param max_iter: maximum number of iterations
    :param eps: tolerance
    :return: solution of the least squares problem
    """
    assert max_iter > 0, "max_iter must be positive"
    assert eps > 0, "eps must be positive"

    assert A.shape[0] == b.shape[0], "A and b must have the same number of rows"

    # B = A.T @ A
    c = A.T @ b

    x = np.zeros_like(c)
    s = c  # s = c - B @ x
    p = s
    beta = 0.
    rho_old = (s.T @ s)[0, 0]
    rho_new = rho_old

    for _ in range(max_iter):
        if np.sqrt(rho_new) < eps:
            break
        p = s + beta * p
        Bp = A.T @ (A @ p)
        alpha = rho_old / (p.T @ Bp)
        x = x + alpha * p
        s = s - alpha * Bp
        rho_new = (s.T @ s)[0, 0]
        beta = rho_new / rho_old
        rho_old = rho_new

    return x.reshape(-1)

class LevenbergMarquardt:
    R: Function
    DR: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    lambda_param_fun: Callable[[Function, Callable[[NDArray[np.float64], float], NDArray[np.float64]], NDArray[np.float64], int], float]
    
    class LambdaParam:
        def __call__(
            self,
            R: Function,
            step: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
            p: NDArray[np.float64],
            i: int,
        ) -> float:
            raise NotImplementedError
        
    class LambdaParamConstant(LambdaParam):
        def __init__(
            self,
            lambda0: float
        ):
            assert lambda0 >= 0, "lambda0 must be non-negative"
            self.init_value = lambda0
            self.value = lambda0

        def __call__(
            self,
            R: Function,
            step: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
            p: NDArray[np.float64],
            i: int,
        ) -> float:
            return self.value
        
    class LambdaParamDefaultModifier(LambdaParam):
        """
        Implementation of the algorithm described in "Choice of damping parameter" section in 
        https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
        """
        def __init__(
            self,
            lambda0: float,
            lambda_change: float,
            eps: float = 1e-16,
            max_iter: int = 100,
        ):
            assert lambda_change > 1, "lambda_change must be greater than 1"
            assert lambda0 >= 0, "lambda0 must be non-negative"
            assert eps > 0, "eps must be positive"

            self.init_value = lambda0
            self.value = lambda0
            self.value_change = lambda_change
            self.eps = eps
            self.max_iter = max_iter

        def __call__(
            self,
            R: Function,
            step: Callable[[NDArray[np.float64], float], NDArray[np.float64]],
            p: NDArray[np.float64],
            i: int,
        ) -> float:
            if i == 0:
                self.value = self.init_value

            current_err = np.linalg.norm(R(p))
            for _ in range(self.max_iter):
                next_err1 = np.linalg.norm(R(step(p, self.value)))
                next_err2 = np.linalg.norm(R(step(p, self.value / self.value_change)))
                if next_err1 > current_err and next_err2 > current_err:
                    self.value *= self.value_change
                elif next_err2 < current_err:
                    self.value /= self.value_change
                    return self.value
                else:
                    return self.value
            return self.value
    def __init__(
        self,
        R: Function,
        lambda_param_fun: Callable[
            [
                Function,
                Callable[[NDArray[np.float64], float], NDArray[np.float64]],
                NDArray[np.float64],
                int,
            ],
            float,
        ],
    ):
        self.R = R
        self.DR = R.differential
        self.lambda_param_fun = lambda_param_fun

    def step_cholesky(self, p:NDArray[np.float64], lambda_param: float) -> NDArray[np.float64]:
        """
        Solve operation is equivalent to the following:
        (DR(p).T @ DR(p) + lambda_param * np.eye(p.size))**(-1) @ DR(p)^T @ R(p)
        """
        try:
            L = np.linalg.cholesky(self.DR(p).T @ self.DR(p) + lambda_param * np.eye(p.size))
            y = np.linalg.solve(L, self.DR(p).T @ self.R(p))
            d = np.linalg.solve(L.T, y)
        except np.linalg.LinAlgError:
            d = np.linalg.solve(
                self.DR(p).T @ self.DR(p) + lambda_param * np.eye(p.size),
                self.DR(p).T @ self.R(p),
            )
        return p - d
    
    def step_least_squares(self, p:NDArray[np.float64], lambda_param: float) -> NDArray[np.float64]:
        """
        Solve operation is equivalent to ridge regression:
        ||DR(p) @ d + R(p)||^2 + sqrt(lambda_param) * ||d||^2 -> min_d!
        which is equivalent to the reformulated least squares problem
        """
        A = np.vstack((self.DR(p), np.sqrt(lambda_param) * np.eye(p.size)))
        b = np.vstack((self.R(p).reshape(-1, 1), np.zeros((p.size, 1))))

        d = np.linalg.lstsq(A, b, rcond=None)[0].reshape(-1)
        return p - d
    
    def step_cgnr(self, p:NDArray[np.float64], lambda_param: float) -> NDArray[np.float64]:
        """
        Solve operation is equivalent to ridge regression:
        ||DR(p) @ d + R(p)||^2 + sqrt(lambda_param) * ||d||^2 -> min_d!
        which is equivalent to the reformulated least squares problem,
        which can be solved iteratively by CGNR method
        """
        A = np.vstack((self.DR(p), np.sqrt(lambda_param) * np.eye(p.size)))
        b = np.vstack((self.R(p).reshape(-1, 1), np.zeros((p.size, 1))))

        d = cgnr_normal_equations(A, b, max_iter=self.step_max_iter, eps=self.step_tol)
        return p - d
    
    def step_svd(self, p:NDArray[np.float64], lambda_param: float) -> NDArray[np.float64]:
        """
        Solve operation is equivalent to ridge regression:
        ||DR(p) @ d + R(p)||^2 + sqrt(lambda_param) * ||d||^2 -> min_d!
        which is equivalent to the reformulated least squares problem,
        which can be solved by SVD decomposition
        """
        A = self.DR(p)
        b = self.R(p)
        U, sigma, VT = np.linalg.svd(A, full_matrices=False)
        V = VT.T
        # y = np.linalg.solve(
        #     np.diag(sigma**2) + lambda_param * np.eye(p.size),
        #     np.diag(sigma) @ U.T @ b
        # )
        y = (sigma / (sigma**2 + lambda_param)) * (U.T @ b)
        x = V @ y

        d = x
        return p - d

    def step(self, p: NDArray[np.float64], lambda_param: float) -> NDArray[np.float64]:
        if self.step_type == "cholesky":
            return self.step_cholesky(p, lambda_param)
        elif self.step_type == "least_squares":
            return self.step_least_squares(p, lambda_param)
        elif self.step_type == "cgnr":
            return self.step_cgnr(p, lambda_param)
        elif self.step_type == "svd":
            return self.step_svd(p, lambda_param)
        else:
            raise NotImplementedError

    def optimize(
        self,
        p0: NDArray[np.float64],
        max_iter: int,
        silent: bool = True,
        step_type: str = "cgnr",
        step_max_iter: int = 100,
        step_tol: float = 1e-6,
    ) -> tuple[NDArray[np.float64], np.float64]:
        """
        Optimize the function R(p) using Levenberg-Marquardt method
        :param p0: initial guess
        :param max_iter: maximum number of iterations
        :param silent: if False, print logs at each iteration
        :param step_type: type of step to use, one of the following: 'cholesky', 'least_squares', 'svd', 'cgnr'
        :param step_max_iter: maximum number of iterations for step method, only used if step_type is 'cgnr'
        :param step_tol: tolerance for step method, only used if step_type is 'cgnr'
        """
        assert max_iter > 0, "max_iter must be positive"
        assert step_type in ["cholesky", "least_squares", "cgnr", "svd"], "step_type must be one of the following: 'cholesky', 'least_squares', 'cgnr', 'svd'"
        self.step_type = step_type
        self.step_max_iter = step_max_iter
        self.step_tol = step_tol

        p = p0

        for i in range(max_iter):
            if not silent:
                err = np.linalg.norm(self.R(p))
                print(f"iter {i}: p = {p}, ||R(p)|| = {err}")
            try:
                p = self.step(p=p, lambda_param=self.lambda_param_fun(self.R, self.step, p, i))
            except np.linalg.LinAlgError:
                err = np.linalg.norm(self.R(p))
                print(f"Levenberg-Marquardt ({step_type=}) failed in iteration nr {i}. Returning current point.")
                return p, err

        err = np.linalg.norm(self.R(p))
        return p, err
