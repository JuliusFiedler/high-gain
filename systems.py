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

    def simulate(self, x0_list, tt):
        x_data = None
        for x0 in x0_list:
            s = solve_ivp(self.rhs, (0, tt[-1]), x0, t_eval=tt, atol=1e-7, rtol=1e-7)
            if x_data is None:
                x_data = s.y
            else:
                x_data = np.concatenate((x_data, s.y), axis=1)
        return x_data

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
        self.N = 4
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
        return self.simulate(x0_list, tt)

class DoublePendulum(System):
    def __init__(self) -> None:
        self.x1, self.x2, self.x3, self.x4 = self.x = sp.symbols("x1, x2, x3, x4")
        # \dot{x} = f(x) + g(x)*u
        self.h_symb = ["x1dot, x2dot"]
        self.n = 4
        self.N = [4,4]
        # self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "DoublePendulum"
        # if 1:
        #     self.get_rhs()
        # else:
        #     with open(f"models/{self.name}/double_pendulum_rhs.pcl", "rb") as f:
        #         self.rhs = pickle.load(f)
        super().__init__()
        # self.trig_state = True
        # self.alpha_limit = 1000

    def get_output(self, x):
        return x[2:]

    def get_approx_region(self):
        return [[-np.pi, np.pi], [-np.pi, np.pi], [-30, 30], [-100,100]]

    def rhs(self, t, state):
        x1, x2, x3, x4 = state
        return np.array([x3, x4, -4.0*(-0.125*x3**2*sin(x2) - 2.4525*sin(x1 + x2))*(0.5*cos(x2) + 0.25)*(0.25*sin(x2)**2 + 0.0625)**-1 + (0.25*x3*x4*sin(x2) + 0.125*x4**2*sin(x2) - 7.3575*sin(x1) - 2.4525*sin(x1 + x2))*(0.25*sin(x2)**2 + 0.0625)**-1, 16.0*(-0.125*x3**2*sin(x2) - 2.4525*sin(x1 + x2))*(0.25*cos(x2) + 0.375)*(0.25*sin(x2)**2 + 0.0625)**-1 - 4.0*(0.5*cos(x2) + 0.25)*(0.25*x3*x4*sin(x2) + 0.125*x4**2*sin(x2) - 7.3575*sin(x1) - 2.4525*sin(x1 + x2))*(0.25*sin(x2)**2 + 0.0625)**-1])

    def get_x_data(self):
        np.random.seed(2)
        n = 70
        x = (np.random.random(size=(2,n))-0.5)*2
        w = np.zeros((2,n))
        x0_list = np.array([*x, *w]).T
        # x0_list = np.concatenate((x0_list, np.array([[1,0,0.1],[1,0,-0.2]])), axis=0)
        tend = 50
        tt = np.linspace(0, tend, 6000)
        return self.simulate(x0_list, tt)


class DoublePendulum2(System):
    def __init__(self) -> None:
        self.x1, self.x2, self.x3, self.x4, self.x5, self.x6 = self.x = sp.symbols("x1, x2, x3, x4, x5, x6")
        self.pp = sp.var("p1, p2, p3, p4, p5, p6")
        # \dot{x} = f(x) + g(x)*u
        self.h_symb = ["phi1fot, phi2dot"]
        self.n = 6
        self.N = [4,4]
        # self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "DoublePendulum2"
        # if 1:
        #     self.get_rhs()
        # else:
        # with open(f"models/{self.name}/pdot.pcl", "rb") as f:
        #     pdot = pickle.load(f)
        # self.pdot_func = sp.lambdify(self.pp, pdot)
        with open(f"models/{self.name}/p.pcl", "rb") as f:
            self.p_symb = pickle.load(f)
        self.p_x = sp.lambdify(self.x[:4], self.p_symb)#

        # self.f_symb = pdot.subs(zip(self.pp, self.x))
        super().__init__()
        # self.alpha_limit = 10
        # self.log = True
        self.separate = True

    def get_output(self, x):
        return x[4:]

    # def get_approx_region(self):
    #     return [[-np.pi, np.pi], [-np.pi, np.pi], [-30, 30], [-100,100]]

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
        n=30
        # phi0 = np.random.random(size=10)*np.pi*2
        # phi1 = np.random.random(size=10)*np.pi*2
        # w0 = (np.random.random(size=10) - 0.5)*2
        # w1 = (np.random.random(size=10) - 0.5)*2
        phi0 = (np.random.random(size=n)-0.5)*4
        phi1 = np.zeros(n)
        w0 = np.zeros(n)
        w1 = np.zeros(n)
        x0_list = self.p_x(phi0, phi1, w0, w1).T[:,0,:]
        # x0_list = np.concatenate((x0_list, np.array([[1,0,0.1],[1,0,-0.2]])), axis=0)
        tend = 30
        tt = np.linspace(0, tend, 6000)
        return self.simulate(x0_list, tt)

class MagneticPendulum(System):
    """
    Source:
    Motter, Gruiz, et. al.: 'Doubly Transient Chaos: The Generic Form of Chaos in Autonomous Dissipative Systems'
    https://arxiv.org/pdf/1310.4209.pdf
    """
    def __init__(self):
        self.n_charges = n_charges = 3
        self.charges = np.ones(n_charges, dtype=float) #* -1
        if n_charges == 1:
            self.magnet_positions = np.array([[0,0]])
        else:
            self.magnet_positions = np.array([[np.cos(phi), np.sin(phi)] for phi in [i*2*np.pi/n_charges for i in np.arange(n_charges)]])
        self.h = 0.3 #0.3
        self.w0 = 0.5
        self.b = 0
        self.angle = 30
        self.measurement_point = [-2, 0.5]

        self.x1, self.x2, self.x3, self.x4 = self.x = sp.symbols("x1, x2, x3, x4")
        self.h_symb = ["x1dot", "x2dot"]
        # self.h_symb = self.x3
        self.n = 4
        self.N = [5, 5]
        # self.N = [5]
        # self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "MagneticPendulum"
        super().__init__()

    def get_output(self, x):
        # return np.cos(self.angle/180*np.pi)*x[0] + np.sin(self.angle/180*np.pi)*x[1]
        # return x[0]
        return x[2:]
        # return [np.sqrt((x[0]-self.measurement_point[0])**2 + (x[1]-self.measurement_point[1])**2)]

    def get_approx_region(self):
        return [[-np.pi, np.pi], [-np.pi, np.pi], [-30, 30], [-100,100]]

    def rhs(self, t, x):
        s = None
        for pos, p in zip(self.magnet_positions, self.charges):
            v = p * (pos- x[0:2]) * ((pos[0]-x[0])**2 + (pos[1]-x[1])**2 + self.h**2)**(-3/2)
            if s is None:
                s = v
            else:
                s += v

        dx1 = x[2]
        dx2 = x[3]
        dx3 = s[0] - self.b*x[2] - self.w0**2*x[0]
        dx4 = s[1] - self.b*x[3] - self.w0**2*x[1]

        return np.array([dx1, dx2, dx3, dx4])

    def get_x_data(self):
        np.random.seed(2)
        n = 70
        x = (np.random.random(size=(2,n))-0.5)*4
        w = np.zeros((2,n))
        x0_list = np.array([*x, *w]).T
        # x0_list = np.concatenate((x0_list, np.array([[1,0,0.1],[1,0,-0.2]])), axis=0)
        tend = 50
        tt = np.linspace(0, tend, 6000)
        return self.simulate(x0_list, tt)

class ThreeBody(System):
    """
    Source:
    https://en.wikipedia.org/wiki/Three-body_problem
    """
    def __init__(self):
        # self.h_symb = self.x1
        self.n = 18
        self.N = 18
        self.z = [sp.var(f"z_{i}") for i in range(self.N)]
        self.name = "ThreeBody"

        self.G = 6.674e-3
        self.m1 = 1
        self.m2 = 1
        self.m3 = 1
        super().__init__()

    def get_output(self, x):
        return x[0:3]

    def rhs(self, t, x):
        G = self.G
        m1 = self.m1
        m2 = self.m2
        m3 = self.m3

        rr1 = x[0:3]
        rr2 = x[3:6]
        rr3 = x[6:9]

        drr1 = x[9:12]
        drr2 = x[12:15]
        drr3 = x[15:18]

        # denominators, as factors for adolc -> minus in exponent
        den12 = ((rr1[0] - rr2[0])**2 + (rr1[1] - rr2[1])**2 + (rr1[2] - rr2[2])**2) ** (-3/2)
        den13 = ((rr1[0] - rr3[0])**2 + (rr1[1] - rr3[1])**2 + (rr1[2] - rr3[2])**2) ** (-3/2)
        den23 = ((rr2[0] - rr3[0])**2 + (rr2[1] - rr3[1])**2 + (rr2[2] - rr3[2])**2) ** (-3/2)

        ddrr1 = [-G*m2 * (rr1[i]-rr2[i]) * den12 - G*m3 * (rr1[i]- rr3[i]) * den13 for i in range(3)]
        ddrr2 = [-G*m2 * (rr2[i]-rr3[i]) * den23 - G*m3 * (rr2[i]- rr1[i]) * den12 for i in range(3)]
        ddrr3 = [-G*m2 * (rr3[i]-rr1[i]) * den13 - G*m3 * (rr3[i]- rr2[i]) * den23 for i in range(3)]
        return np.array([*drr1, *drr2, *drr3, *ddrr1, *ddrr2, *ddrr3])

    def get_x_data(self):
        np.random.seed(2)
        n = 30
        x = (np.random.random(size=(9,n))-0.5)/5
        w = np.zeros((9,n))
        x0_list = np.array([*x, *w]).T
        # x0_list = np.concatenate((x0_list, np.array([[1,0,0.1],[1,0,-0.2]])), axis=0)
        tend = 30
        tt = np.linspace(0, tend, 6000)
        return self.simulate(x0_list, tt)
