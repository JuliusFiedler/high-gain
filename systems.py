import numpy as np
from abc import abstractmethod
import sympy as sp
import symbtools as st
import symbtools.modeltools as mt
import pickle
from numpy import sin, cos
from scipy.integrate import solve_ivp

class System:
    def __init__(self) -> None:
        self.trig_state = False
        self.alpha_limit = None
        self.log = False
        self.separate = False
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
        return x[2]

    def get_input(self, x):
        return 1

    def get_approx_region(self):
        return [[-2.5, 2.5], [-2.5, 2.5]]

class InvPendulum2(System):
    def __init__(self) -> None:
        self.x1, self.x2, self.x3 = self.x = sp.symbols("x1, x2, x3")
        m = 1
        g = 9.81
        l = 1
        J = 0.01
        self.kappa = m*g*l/(J+m*l**2)
        self.f_symb = sp.Matrix([-self.x2*self.x3, self.x1*self.x3, -self.kappa*self.x1])
        self.h_symb = self.x3
        self.n = len(self.x)
        self.N = 2
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "InvPendulum2"
        super().__init__()

    def rhs(self, t, x):
        return [-x[1]*x[2], x[0]*x[2], -self.kappa*x[0]]

    def get_output(self, x):
        return x[2]

    def get_approx_region(self):
        return [[-2.5, 2.5], [-2.5, 2.5]]

    def get_x_data(self):
        np.random.seed(2)
        phi0 = np.random.random(size=10)*np.pi*2
        w0 = (np.random.random(size=10) - 0.5)*2
        x0_list = np.vstack((np.sin(phi0), np.cos(phi0), w0)).T
        x0_list = np.concatenate((x0_list, np.array([[1,0,0.1],[1,0,-0.2]])), axis=0)
        x_data = None
        tend = 30
        tt = np.linspace(0, tend, 6000)
        for x0 in x0_list:
            s = solve_ivp(self.rhs, (0, tend), x0, t_eval=tt)
            if x_data is None:
                x_data = s.y
            else:
                x_data = np.concatenate((x_data, s.y), axis=1)
        return x_data

class DoublePendulum(System):
    def __init__(self) -> None:
        self.x1, self.x2, self.x3, self.x4 = self.x = sp.symbols("x1, x2, x3, x4")
        # \dot{x} = f(x) + g(x)*u
        self.h_symb = self.x1
        self.n = 4
        self.N = 4
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "DoublePendulum"
        # if 1:
        #     self.get_rhs()
        # else:
        #     with open(f"models/{self.name}/double_pendulum_rhs.pcl", "rb") as f:
        #         self.rhs = pickle.load(f)
        super().__init__()
        self.trig_state = True
        self.alpha_limit = 1000

    def get_output(self, x):
        return x[0]

    def get_approx_region(self):
        return [[-np.pi, np.pi], [-np.pi, np.pi], [-30, 30], [-100,100]]

    def rhs(self, t, state):
        x1, x2, x3, x4 = state
        return np.array([x3, x4, -4.0*(-0.125*x3**2*sin(x2) - 2.4525*sin(x1 + x2))*(0.5*cos(x2) + 0.25)*(0.25*sin(x2)**2 + 0.0625)**-1 + (0.25*x3*x4*sin(x2) + 0.125*x4**2*sin(x2) - 7.3575*sin(x1) - 2.4525*sin(x1 + x2))*(0.25*sin(x2)**2 + 0.0625)**-1, 16.0*(-0.125*x3**2*sin(x2) - 2.4525*sin(x1 + x2))*(0.25*cos(x2) + 0.375)*(0.25*sin(x2)**2 + 0.0625)**-1 - 4.0*(0.5*cos(x2) + 0.25)*(0.25*x3*x4*sin(x2) + 0.125*x4**2*sin(x2) - 7.3575*sin(x1) - 2.4525*sin(x1 + x2))*(0.25*sin(x2)**2 + 0.0625)**-1])

    def get_x_data(self):
        # x0_list = [[-1,0,0,0],
        #            [2,0,0,0],
        #            [-3,0,0,0],
        #            [0,1,0,0],
        #            [0,-2,0,0],
        #            [0,3,0,0]]
        x0_list = [[-0.1,0,0,0],
                   [0.05,0,0,0.1],
                   [0.13,0,0.01,0],
                   [0.08, 0.1, 0.1, 0.01],
                   ]
        x_data = None
        tend = 30
        tt = np.linspace(0, tend, 6000)
        for x0 in x0_list:
            s = solve_ivp(self.rhs, (0, tend), x0, t_eval=tt)
            if x_data is None:
                x_data = s.y
            else:
                x_data = np.concatenate((x_data, s.y), axis=1)
        return x_data


class DoublePendulum2(System):
    def __init__(self) -> None:
        self.x1, self.x2, self.x3, self.x4, self.x5, self.x6 = self.x = sp.symbols("x1, x2, x3, x4, x5, x6")
        self.pp = sp.var("p1, p2, p3, p4, p5, p6")
        # \dot{x} = f(x) + g(x)*u
        self.h_symb = self.x6
        self.n = 6
        self.N = 9
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "DoublePendulum2"
        # if 1:
        #     self.get_rhs()
        # else:
        with open(f"models/{self.name}/pdot.pcl", "rb") as f:
            pdot = pickle.load(f)
        # self.pdot_func = sp.lambdify(self.pp, pdot)
        with open(f"models/{self.name}/p.pcl", "rb") as f:
            self.p_symb = pickle.load(f)
        self.p_x = sp.lambdify(self.x[:4], self.p_symb)#

        self.f_symb = pdot.subs(zip(self.pp, self.x))
        super().__init__()
        # self.alpha_limit = 10
        # self.log = True
        self.separate = True

    def get_output(self, x):
        return x[5]

    def get_approx_region(self):
        return [[-np.pi, np.pi], [-np.pi, np.pi], [-30, 30], [-100,100]]

    def rhs(self, t, x):
        p1, p2, p3, p4, p5, p6 = x
        return np.array([
            -p2*p5,
            p1*p5,
            -2.0*p2*p5-16.0*(p5+p6)*(p1*(p1*p4-p2*p3)+p2*(p1*p3+p2*p4-0.125)),
            2.0*p1*p5+16.0*(p5+p6)*(p1*(p1*p3+p2*p4-0.125)-p2*(p1*p4-p2*p3)),
            (-156.96*p1*(p1*p3+p2*p4-0.125)-29.43*p1+156.96*p2*(p1*p4-p2*p3)+4.0*p5*p6*(p1*p4-p2*p3)+2.0*p6**2*(p1*p4-p2*p3)+4.0*(8.0*p1*p3+8.0*p2*p4-0.75)*(156.96*p1*(p1*p3+p2*p4-0.125)-156.96*p2*(p1*p4-p2*p3)+2.0*p5**2*(p1*p4-p2*p3)))*(64.0*(p1*p4-p2*p3)**2+0.0625)**(-1),
            (-16.0*(4.0*p1*p3+4.0*p2*p4-0.125)*(156.96*p1*(p1*p3+p2*p4-0.125)-156.96*p2*(p1*p4-p2*p3)+2.0*p5**2*(p1*p4-p2*p3))-4.0*(8.0*p1*p3+8.0*p2*p4-0.75)*(-156.96*p1*(p1*p3+p2*p4-0.125)-29.43*p1+156.96*p2*(p1*p4-p2*p3)+4.0*p5*p6*(p1*p4-p2*p3)+2.0*p6**2*(p1*p4-p2*p3)))*(64.0*(p1*p4-p2*p3)**2+0.0625)**(-1)])

    def get_x_data(self):
        np.random.seed(2)
        # phi0 = np.random.random(size=10)*np.pi*2
        # phi1 = np.random.random(size=10)*np.pi*2
        # w0 = (np.random.random(size=10) - 0.5)*2
        # w1 = (np.random.random(size=10) - 0.5)*2
        phi0 = (np.random.random(size=10)-0.5)*2
        phi1 = np.zeros(10)
        w0 = np.zeros(10)
        w1 = np.zeros(10)
        x0_list = self.p_x(phi0, phi1, w0, w1).T[:,0,:]
        # x0_list = np.concatenate((x0_list, np.array([[1,0,0.1],[1,0,-0.2]])), axis=0)
        x_data = None
        tend = 30
        tt = np.linspace(0, tend, 6000)
        for x0 in x0_list:
            s = solve_ivp(self.rhs, (0, tend), x0, t_eval=tt)
            if x_data is None:
                x_data = s.y
            else:
                x_data = np.concatenate((x_data, s.y), axis=1)
        return x_data