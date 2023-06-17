from typing import Callable
import numpy as np

class Function:
    def __init__(
        self,
        F: Callable[[np.ndarray], np.ndarray],
        DF: Callable[[np.ndarray], np.ndarray],
        M: int,
        N: int
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
        assert F(arg).size == N, "F must be a function from R^M to R^N"

        if N == 1:
            DF_old = DF
            DF = lambda x: DF_old(x).reshape(1, M)
        elif M == 1:
            DF_old = DF
            DF = lambda x: DF_old(x).reshape(N, 1)
        assert DF(arg).shape == (N, M), "DF must be a function from R^M to R^(NxM)"

        self.F = F
        self.DF = DF
        self.M = M
        self.N = N        

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.size == self.M, f"x must be in R^{self.M}"
        return self.F(x)

    def differential(self, x: np.ndarray) -> np.ndarray:
        assert x.size == self.M, f"x must be in R^{self.M}"
        return self.DF(x)
    
    def __add__(self, other):
        return add(self, other)
    
    def __mul__(self, other):
        if type(other) == float or type(other) == int:
            return scale(self, other)
        else:
            return multiply(self, other)
        
    def __rmul__(self, other):
        return self.__mul__(other)
    
def _compose(f: Function, g: Function) -> Function:
    """
    Compose two differentiable functions.
    :param f: function from R^M to R^N
    :param g: function from R^N to R^K
    :return: function from R^M to R^K
    """
    assert f.N == g.M, "f and g must be composable"

    def F(x: np.ndarray) -> np.ndarray:
        return g(f(x))

    def DF(x: np.ndarray) -> np.ndarray:
        return g.differential(f(x)) @ f.differential(x)

    return Function(F, DF, f.M, g.N)

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

    def F(x: np.ndarray) -> np.ndarray:
        return f(x) + g(x)

    def DF(x: np.ndarray) -> np.ndarray:
        return f.differential(x) + g.differential(x)

    return Function(F, DF, f.M, f.N)

def scale(f: Function, c: float) -> Function:
    """
    Scale a differentiable function.
    :param f: function from R^M to R^N
    :param c: scalar
    :return: function from R^M to R^N
    """

    def F(x: np.ndarray) -> np.ndarray:
        return c * f(x)

    def DF(x: np.ndarray) -> np.ndarray:
        return c * f.differential(x)

    return Function(F, DF, f.M, f.N)

def multiply(f: Function, g: Function) -> Function:
    """
    Multiply two differentiable functions.
    :param f: function from R^M to R
    :param g: function from R^M to R
    :return: function from R^M to R
    """
    assert f.M == g.M, "f and g must have the same domain"
    assert f.N == g.N == 1, "f and g must have codomain R"

    def F(x: np.ndarray) -> np.ndarray:
        return f(x) * g(x)

    def DF(x: np.ndarray) -> np.ndarray:
        return f.differential(x) * g(x) + f(x) * g.differential(x)

    return Function(F, DF, f.M, f.N)

def _stack(f: Function, g: Function) -> Function:
    """
    Stack two differentiable functions.
    :param f: function from R^M to R^N
    :param g: function from R^M to R^K
    :return: function from R^M to R^(N+K)
    """
    assert f.M == g.M, "f and g must have the same domain"

    def F(x: np.ndarray) -> np.ndarray:
        return np.hstack((f(x), g(x)))

    def DF(x: np.ndarray) -> np.ndarray:
        return np.vstack((f.differential(x), g.differential(x)))

    return Function(F, DF, f.M, f.N + g.N)

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