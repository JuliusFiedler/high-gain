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

    def get_beta_of_x_func(self):
        if not hasattr(self, "g_symb"):
            print(f"Warning, system {self.name} is autonomous!")
            return lambda x: np.zeros(self.N)
        beta = sp.Matrix([0 for i in range(self.N)])
        for i in range(self.N):
            beta[i] = st.lie_deriv(st.lie_deriv(self.h_symb, self.f_symb, self.x, order=i), self.g_symb, self.x)
        return sp.lambdify(self.x, beta)


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

class VanderPol(System):
    def __init__(self) -> None:
        self.x1, self.x2 = self.x = sp.symbols("x1, x2")
        self.f_symb = sp.Matrix([self.x2, -self.x1 - -self.x2 * (self.x1**2 - 1)])
        self.h_symb = self.x2
        self.n = 2
        self.N = 4
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "VanderPol"
        super().__init__()

    def rhs(self, t, x):
        return [x[1], -x[0] - x[1] * (x[0] ** 2 - 1)]

    def get_output(self, x):
        return x[1]

    def get_approx_region(self):
        return [[-2.5, 2.5], [-2.5, 2.5]]


class Lorenz(System):
    def __init__(self) -> None:
        self.x1, self.x2, self.x3 = self.x = sp.symbols("x1, x2, x3")
        self.sigma = 10
        self.rho = 28
        self.beta = 8 / 3
        self.f_symb = sp.Matrix(
            [
                self.sigma * (self.x2 - self.x1),
                self.rho * self.x1 - self.x2 - self.x1 * self.x3,
                self.x1 * self.x2 - self.beta * self.x3,
            ]
        )
        self.h_symb = self.x1
        self.n = 3
        self.N = 4
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "Lorenz"
        super().__init__()

    def rhs(self, t, x):
        return [self.sigma * (x[1] - x[0]), self.rho * x[0] - x[1] - x[0] * x[2], x[0] * x[1] - self.beta * x[2]]

    def get_output(self, x):
        return x[0]

    def get_approx_region(self):
        return [[-20, 20], [-20, 20], [0, 50]]

class Roessler(System):
    def __init__(self) -> None:
        self.x1, self.x2, self.x3 = self.x = sp.symbols("x1, x2, x3")
        self.a = 0.2
        self.b = 0.2
        self.c = 5.7
        self.f_symb = sp.Matrix(
            [
                - self.x2 - self.x3,
                self.x1 + self.a*self.x2,
                self.b - self.c * self.x3 + self.x1 * self.x3
            ]
        )
        self.h_symb = self.x1
        self.n = 3
        self.N = 5
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "Roessler"
        super().__init__()

    def rhs(self, t, x):
        return [-x[1] -x[2], x[0] + self.a*x[1], self.b + x[2]*(x[0] - self.c)]

    def get_output(self, x):
        return x[0]

    def get_approx_region(self):
        return [[-10, 12], [-8, 11], [0, 25]]

class InvPendulum(System):
    def __init__(self) -> None:
        self.x1, self.x2 = self.x = sp.symbols("x1, x2")
        self.u = sp.symbols("u")
        self.kappa1 = 9.81/1
        self.kappa2 = 1
        # \dot{x} = f(x) + g(x)*u
        self.f_symb = sp.Matrix([self.x2, self.kappa1*sp.sin(self.x1)])
        self.g_symb = sp.Matrix([0, self.kappa2])
        self.h_symb = self.x1
        self.n = 2
        self.N = 2
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "InvPendulum"
        super().__init__()

    def rhs(self, t, x, u):
        return [x[1], self.kappa1*np.sin(x[0]) + self.kappa2*u]

    def get_output(self, x):
        return x[0]

    def get_input(self, x):
        return 1

    def get_approx_region(self):
        return [[-2.5, 2.5], [-2.5, 2.5]]
