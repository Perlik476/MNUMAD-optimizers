from functools import reduce
from typing import Callable, Optional, Union
import numpy as np
from numpy.typing import NDArray

class Function:
    def __init__(
        self,
        F: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        DF: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        D2F: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        M: int,
        N: int,
    ):
        """
        :param F: function from R^M to R^N
        :param DF: differential of F
        :param M: dimension of domain
        :param N: dimension of codomain
        """
        assert N > 0, "N must be positive"
        assert M > 0, "M must be positive"
        assert type(N) == int, "N must be an integer"
        assert type(M) == int, "M must be an integer"
        
        arg = np.random.randn(M)
        assert F(arg).size == N, f"F must be a function from R^{M} to R^{N}, but F({arg}) is in R^{F(arg).size}"
        if DF(arg).shape != (N, M) and DF(arg).reshape(-1).size == N * M:
            DF_old = DF
            DF = lambda x: DF_old(x).reshape(N, M)
        assert DF(arg).shape == (N, M), f"DF must be a function from R^{M} to R^{N}xR^{M}, but DF({arg}) is in R^{DF(arg).shape}"
        if D2F(arg).shape != (N, M, M) and D2F(arg).reshape(-1).size == N * M * M:
            D2F_old = D2F
            D2F = lambda x: D2F_old(x).reshape(N, M, M)
        assert D2F(arg).shape == (N, M, M), f"D2F must be a function from R^{M} to R^{N}xR^{M}xR^{M}, but D2F({arg}) is in R^{D2F(arg).shape}"

        self.F = F
        self.DF = DF
        self.D2F = D2F
        self.M = M
        self.N = N

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        assert x.size == self.M, f"x must be in R^{self.M}"
        return self.F(x)

    def differential(self, n: int = 1) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        """
        :param n: order of differential
        :return: differential of order n
        """
        assert n >= 0, "n must be non-negative"
        if n == 0:
            return self.F
        elif n == 1:
            return self.DF
        elif n == 2:
            return self.D2F
        else:
            raise NotImplementedError
        
    
    def __add__(self, other):
        return add(self, other)
    
    def __sub__(self, other):
        return add(self, scale(other, -1))
    
    def __mul__(self, other):
        if type(other) == float or type(other) == int:
            return scale(self, other)
        else:
            return multiply(self, other)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if type(other) == float or type(other) == int:
            return scale(self, 1/other)
        else:
            return divide(self, other)

    
def _compose(f: Function, g: Function) -> Function:
    """
    Compose two differentiable functions.
    :param f: function from R^M to R^N
    :param g: function from R^N to R^K
    :return: function from R^M to R^K
    """
    assert f.N == g.M, "f and g must be composable"

    def F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return g(f(x))

    def DF(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return g.differential()(f(x)) @ f.differential()(x)

    def D2F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return g.differential(2)(f(x)) @ f.differential()(x) @ f.differential()(x).T + f.differential(2)(x).T @ g.differential()(f(x)).T

    return Function(F, DF, D2F, f.M, g.N)

def compose(*functions: Function) -> Function:
    """
    Compose multiple differentiable functions.
    :param functions: functions from R^M1 to R^M2, R^M2 to R^M3, ..., R^M(K-1) to R^MK
    :return: function from R^M1 to R^MK
    """
    assert len(functions) > 1, "must compose at least two functions"
    for i in range(len(functions) - 1):
        assert functions[i].N == functions[i + 1].M, "functions must be composable"

    f = functions[0]
    for g in functions[1:]:
        f = _compose(f, g)
    return f

def add(f: Function, g: Function) -> Function:
    """
    Add two differentiable functions.
    :param f: function from R^M to R^N
    :param g: function from R^M to R^N
    :return: function from R^M to R^N
    """
    assert f.M == g.M, "f and g must have the same domain"
    assert f.N == g.N, "f and g must have the same codomain"

    def F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f(x) + g(x)

    def DF(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f.differential()(x) + g.differential()(x)
    
    def D2F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f.differential(2)(x) + g.differential(2)(x)
    
    return Function(F, DF, D2F, f.M, f.N)
    

def scale(f: Function, c: float) -> Function:
    """
    Scale a differentiable function.
    :param f: function from R^M to R^N
    :param c: scalar
    :return: function from R^M to R^N
    """

    def F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return c * f(x)

    def DF(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return c * f.differential()(x)
    
    def D2F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return c * f.differential(2)(x)

    return Function(F, DF, D2F, f.M, f.N)

def multiply(f: Function, g: Function) -> Function:
    """
    Multiply two differentiable functions.
    :param f: function from R^M to R
    :param g: function from R^M to R
    :return: function from R^M to R
    """
    assert f.M == g.M, "f and g must have the same domain"
    assert f.N == g.N == 1, "f and g must have codomain R"

    def F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f(x) * g(x)

    def DF(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f(x) * g.differential()(x) + g(x) * f.differential()(x)
    
    def D2F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f(x) * g.differential(2)(x) + 2 * g.differential()(x) * f.differential()(x) + g(x) * f.differential(2)(x)
    
    return Function(F, DF, D2F, f.M, f.N)

def divide(f: Function, g: Function) -> Function:
    """
    Divide two differentiable functions.
    :param f: function from R^M to R
    :param g: function from R^M to R
    :return: function from R^M to R
    """
    assert f.M == g.M, "f and g must have the same domain"
    assert f.N == g.N == 1, "f and g must have codomain R"

    def F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return f(x) / g(x)

    def DF(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return (f.differential()(x) * g(x) - f(x) * g.differential()(x)) / (g(x) ** 2)
    
    def D2F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        # TODO: check this
        # return (f.differential(2)(x) * g(x) - 2 * f.differential()(x) * g.differential()(x) + f(x) * g.differential(2)(x)) / (g(x) ** 2) - 2 * (f.differential()(x) * g(x) - f(x) * g.differential()(x)) * g.differential()(x) / (g(x) ** 3)
        return (f.differential(2)(x) * g(x)**2 - g(x) * (2 * f.differential()(x) * g.differential()(x) + f(x) + g.differential(2)(x)) + 2 * f(x) * g.differential()(x)**2) / g(x) ** 3

    return Function(F, DF, D2F, f.M, f.N)

def _stack(f: Function, g: Function) -> Function:
    """
    Stack two differentiable functions, i.e. f(x) = [f_1(x), f_2(x)]^T.
    :param f: function from R^M to R^N
    :param g: function from R^M to R^K
    :return: function from R^M to R^(N+K)
    """
    assert f.M == g.M, "f and g must have the same domain"

    def F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.hstack((f(x), g(x)))

    def DF(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.vstack((f.differential()(x), g.differential()(x)))
    
    def D2F(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.vstack((f.differential(2)(x), g.differential(2)(x)))
        
    
    return Function(F, DF, D2F, f.M, f.N + g.N)

def stack(*functions: Function) -> Function:
    """
    Stack multiple differentiable functions.
    :param functions: functions from R^M to R^N1, R^M to R^N2, ..., R^M to R^NK
    :return: function from R^M to R^(N1+N2+...+NK)
    """
    assert len(functions) > 1, "must stack at least two functions"
    for i in range(len(functions) - 1):
        assert functions[i].M == functions[i + 1].M, "functions must have the same domain"

    f = functions[0]
    for g in functions[1:]:
        f = _stack(f, g)
    return f


sin = Function(F=np.sin, DF=lambda x: np.cos(x), D2F=lambda x: -np.sin(x), M=1, N=1)
cos = Function(F=np.cos, DF=lambda x: -np.sin(x), D2F=lambda x: -np.cos(x), M=1, N=1)
exp = Function(F=np.exp, DF=np.exp, D2F=np.exp, M=1, N=1)
log = Function(F=lambda x: np.log(np.abs(x)), DF=lambda x: 1 / x, D2F=lambda x: -1 / x**2, M=1, N=1)
square = Function(F=lambda x: x**2, DF=lambda x: 2 * x, D2F=lambda x: 2 * np.eye(1), M=1, N=1)
abs = Function(F=np.abs, DF=lambda x: np.sign(x), D2F=lambda x: np.zeros_like(x), M=1, N=1)
sqrt = Function(F=lambda x: np.sqrt(np.abs(x)), DF=lambda x: 1 / (2 * np.sqrt(np.abs(x))) * np.sign(x), D2F=lambda x: - 1 / (4 * np.abs(x)**(3 / 2)), M=1, N=1)
norm_sqr = lambda M: Function(F=lambda x: np.array(np.linalg.norm(x)**2), DF=lambda x: 2 * x, D2F=lambda x: 2 * np.eye(len(x)), M=M, N=1)
sum_ = lambda M: Function(F=np.sum, DF=lambda x: np.ones_like(x), D2F=lambda _: np.zeros((M, M)), M=M, N=1)
proj = lambda M: lambda k: Function(F=lambda x: x[k], DF=lambda x: np.eye(M)[k], D2F=lambda x: np.zeros((len(x), len(x))), M=M, N=1)
constM = lambda M: lambda c: Function(F=lambda _: np.array(c), DF=lambda _: np.zeros(M), D2F=lambda _: np.zeros((M, M)), M=M, N=1)
const1 = lambda c: Function(F=lambda _: np.array(c), DF=lambda _: np.zeros(1), D2F=lambda _: np.zeros((1, 1)), M=1, N=1)
mul_const = lambda c: Function(F=lambda x: c * x, DF=lambda x: c * np.eye(1), D2F=lambda x: np.zeros(1), M=1, N=1)