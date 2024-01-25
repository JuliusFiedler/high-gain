import numpy as np
from abc import abstractmethod
import sympy as sp
import symbtools as st

class System:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def rhs(self, t, x):
        raise NotImplementedError

    @abstractmethod
    def get_output(self, x):
        raise NotImplementedError

    @abstractmethod
    def get_approx_region(self):
        raise NotImplementedError

    def get_lie_derivs(self, max_order=1):
        # TODO replace with AD
        l = []
        for i in range(max_order):
            l.append(st.lie_deriv(self.h_symb, self.f_symb, self.x, order=i))
        q_symb = sp.Matrix(l)
        return q_symb

    def get_q_func(self):
        self.q_symb = sp.Matrix(self.get_lie_derivs(self.N))
        return sp.lambdify(self.x, self.q_symb)

    def get_alpha_of_x_func(self):
        alpha_x_symb = st.lie_deriv(self.h_symb, self.f_symb, self.x, order=self.N)
        return sp.lambdify(self.x, alpha_x_symb)


class UndampedHarmonicOscillator(System):
    def __init__(self) -> None:
        self.name = "UndampedHarmonicOscillator"
        self.x1, self.x2 = self.x = sp.symbols("x1, x2")
        self.f_symb = sp.Matrix([self.x2, -self.x1])
        self.h_symb = self.x1 + self.x1**2
        self.n = 2
        self.N = 4
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        super().__init__()

    def rhs(self, t, x):
        return [x[1], -x[0]]

    def get_output(self, x):
        return x[0] + x[0] ** 2

    def get_approx_region(self):
        return [[-1, 1], [-1, 1]]

class DuffingOscillator(System):
    def __init__(self) -> None:
        self.x1, self.x2 = self.x = sp.symbols("x1, x2")
        self.f_symb = sp.Matrix([self.x2, self.x1 - self.x1**3])
        self.h_symb = self.x2
        self.n = 2
        self.N = 6
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "DuffingOscillator"
        super().__init__()

    def rhs(self, t, x):
        return [x[1], x[0] - x[0] ** 3]

    def get_output(self, x):
        return x[1]

    def get_approx_region(self):
        return [[-1.5, 1.5], [-1.5, 1.5]]