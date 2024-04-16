from scipy.integrate import OdeSolver, DenseOutput
from scipy.linalg import expm
from warnings import warn
import numpy as np

def norm(x):
    return np.sqrt(np.linalg.norm(x)/x.size)

def superid(dim):
    return np.block([[np.zeros((dim-1, 1)), np.eye(dim-1)], [np.zeros((1, dim-1)), np.zeros((1, 1))]])

def exparg(A, order):
    """
    return a matrix with special structure:
    [ A 0 ]
    [ 0 J ]
    where J hase ones on the superdiagonal
    """
    dim = np.shape(A)[0]
    return np.block([[A, np.zeros((dim, order))], [np.zeros((order, dim)), superid(order)]])

def expweight(A, W, h):
    """
    compute a weighted sum of exponential functions
    this is the last column of the matrix exponential
    """

    o = np.shape(W)[1]

    # scale
    Wh = h * W

    n = np.linalg.norm(Wh, 1)
    e = np.ceil(np.log2(n))
    scale = 2**(-e)
    invscale = 2**e

    dim = np.shape(W)[0]
    X = np.empty((dim+o, dim+o))
    X[:dim,:dim] = h*A
    X[:dim,dim:] = scale * Wh
    X[dim:,:dim] = np.zeros((o, dim))
    X[dim:,dim:] = superid(o)

    Y = expm(X)

    return invscale * Y[:dim,-1]

    if h == 0:
        return np.zeros(np.shape(W)[0], 1)

    o = np.shape(W)[1]
    Wh = np.dot(W, np.diag([h**(i+1-o) for i in range(o)]))
    #Wh = W
    # FIXME: check the theory

    # scale
    n = np.linalg.norm(Wh, 1) 
    if n > 0:
        e = np.ceil(np.log2(n))
    else:
        e = 0
    scale = 2**(-e)
    invscale = 2**(e)

    dim = np.shape(W)[0]
    A[:dim,dim:dim+o] = scale * Wh

    X = expm(A[:dim+o,:dim+o]*h)
    #print("expw")
    #print(A, W, X, invscale)
    #print(invscale * X[:dim,-1])
    return invscale * X[:dim,-1]

class ExpRB(OdeSolver):
    """
    base class for exponential Rosenbrock integrators
    """

    # Butcher coefficients
    a: list = NotImplemented
    b: np.ndarray = NotImplemented
    be: np.ndarray = NotImplemented
    c:list = NotImplemented
    order: int = NotImplemented
    err_order: int = NotImplemented

    def __init__(self,
                 fun,
                 t0,
                 y0,
                 t_bound,
                 jac=None,
                 fun_t=None,
                 vectorized=False,
                 max_step=np.inf,
                 rtol=1.e-3,
                 atol=1.e-6,
                 **kwargs):
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)

        self.stages = 1 + len(self.c)
        assert np.shape(self.b)[0] == self.stages
        assert np.shape(self.be)[0] == self.stages
        assert len(self.a) == self.stages-1
        for i, a in enumerate(self.a):
            assert np.shape(a)[0] == i+1

        self.exp_order = max([np.shape(self.b)[1], np.shape(self.be)[1]] + [np.shape(a)[1] for a in self.a])

        if kwargs:
            warn("the following arguments have no effect for this integrator: %s"%", ".join("%s"%k for k in kwargs.keys))

        self.atol = atol
        self.rtol = rtol

        self.err_exp = -1. / (self.err_order + 1)

        self.jac = self._jac(jac)
        self.max_step = max_step

        # TODO: make adjustable
        # ratio bound for successive step sizes
        self.min_factor = .2
        self.max_factor = 5.
        # an additional factor to keep step size a bit smaller in order to avoid too large steps
        self.inc = .8

        # next step size
        self.h = 1.

        self.D = np.empty((self.n, self.stages), dtype=self.y.dtype)
        self.W = np.empty((self.n, self.exp_order), dtype=self.y.dtype)
        self.W0 = np.empty((self.n, 2), dtype=self.y.dtype)
        self.A = np.empty((self.n+self.exp_order, self.n+self.exp_order), dtype=self.y.dtype)
        self.A[self.n:,:self.n] = np.zeros((self.exp_order, self.n))
        self.A[self.n:,self.n:] = superid(self.exp_order)

    def _jac(self, jac):
        # TODO: support constant (makes no sense) and sparse Jacobian
        if jac is None:
            raise ValueError("a Jacobian has to be specified") 
        if not callable(jac):
            raise ValueError("the Jacobian must be a function")
        def jac_wrap(t, y):
            self.njev += 1
            return np.asarray(jac(t, y), dtype=float)
        J = jac(self.t, self.y)
        if np.shape(J) != (self.n, self.n):
            raise ValueError("the Jacobian must have shape '%s' but is '%s'"%((self.n, self.n), np.shape(J)))
        return jac_wrap

    def _step_impl(self):
        t = self.t
        y = self.y
        f = self.fun(t, y)

        # derivative wrt. time
        #ft = self.fun_t(t, y)

        J = self.jac(t, y)
        #Ny = f - np.dot(J, y) # nonlinear part

        # augment with additional integrator chain
        #self.A = exparg(J, self.exp_order)
        #self.A[:self.n,:self.n] = J

        #self.W0[:,0] = f
        #self.W0[:,1] = ft

        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)

        while True:
            if self.h < min_step:
                return False, self.TOO_SMALL_STEP
            h = self.h * self.direction
            tend = t + h
            if self.direction * (tend - self.t_bound) > 0:
                tend = self.t_bound
                h = tend - t
                self.h = np.abs(h)

            self.D[:,0] = f
            for i in range(self.stages-1):
                dt = self.c[i] * h
                W = np.dot(self.D[:,:i+1], self.a[i])
                #X = expweight(self.A, W, dt)
                X = expweight(J, W, dt)
                Y = y + X
                self.D[:,i+1] = self.fun(t+dt, Y) - f - np.dot(J, X)#(np.dot(J, Y) - Ny)



            self.W = np.dot(self.D, self.b)
            #dy = expweight(self.A, self.W, h) # increment
            dy = expweight(J, self.W, h) # increment
            #ey = dy - expweight(self.A, np.dot(self.D, self.be), h) # error estimate
            dye = expweight(J, np.dot(self.D, self.be), h)
            ey = dy - dye
            yy = y + dy

            scale = self.atol + self.rtol * np.maximum(np.abs(y), np.abs(yy))
            err = norm(ey / scale)
            if err < 1:
                if err == 0:
                    factor = self.max_factor
                else:
                    factor = min(self.max_factor, self.inc * err**self.err_exp)
                self.h *= factor

                self.t = tend
                self.y = yy
                #print(dy, h)
                #exit(0)
                return True, None
            else:
                self.h *= max(self.min_factor, self.inc * err**self.err_exp)


    def _dense_output_impl(self):
        return ExpRBDense(self.t_old, self.t, self.y_old, self.A, self.W)

class ExpEuler(ExpRB):
    a = []
    b=np.array([[1]])
    be=np.array([[0]])
    c=[]
    order = 2
    err_order = 0

class ExpRB32(ExpRB):
    a = [np.array([[1]])]
    b=np.array([[0, 0, 1], [2, 0, 0]])
    be=np.array([[1], [0]])
    c=[1]
    order = 3
    err_order = 2

class ExpG3(ExpRB):
    a = [np.array([[1]])]
    #b = np.array([[0, 0, 1], [2, 0, 0]])
    b = np.array([[0, 0, 1], [128./9., 0, 0]])
    be = np.array([[1], [0]])
    c = [3./8.]
    order = 4
    err_order = 2

class ExpRB43(ExpRB):
    c = [.5, 1]
    b = np.array([[0, 0, 0, 1], [-48, 16, 0, 0], [12, -2, 0, 0]])
    be = np.array([[0, 0, 1], [16, 0, 0], [-2, 0, 0]])
    a = [np.array([[1]]), np.array([[1], [1]])]
    #a = [np.array([[2]]), np.array([[1], [1]])]
    order = 4
    err_order = 3


class ExpRBDense(DenseOutput):
    """
    interpolator for the solution using the exponential integrator
    """
    def __init__(self, t_old, t, y, A, W):
        # TODO
        super().__init__(t_old, t)
        self.y = y
        self.A = A
        self.W = W

    def _call_impl(self, t):
        return self.y + expweight(self.A, self.W, t - self.t_old)



if __name__ == "__main__":
    # test case: Van-der-Pol
    # dot y_1 = y_2
    # dot y_2 = mu (1-y_1^2)y_2 - y_1 + A sin(omega t)

    # FIXME: implement nonautonomous case

    from scipy.integrate import solve_ivp
    from matplotlib import pyplot as plt

    def fun(t, y):
        A = 0
        omega = 0
        mu = 1
        return [y[1], mu*(1-y[0]**2)*y[1] - y[0] + A*np.sin(omega*t)]
    def jac(t, y):
        A = 0
        omega = 0
        mu = 1
        return np.array([[0, 1], [-2*mu*y[0]*y[1] - 1, -mu*y[0]**2]])

    y0 = [1, 0]

    sol_e = solve_ivp(fun, (0, 20), y0, method=ExpRB32, jac=jac, rtol=1.e-8, atol=1.e-8)
    sol_g = solve_ivp(fun, (0, 20), y0, method=ExpG3, jac=jac, rtol=1.e-8, atol=1.e-8)
    #sol_e = solve_ivp(fun, (0, 20), y0, method=ExpRB43, jac=jac, rtol=1.e-8, atol=1.e-8)
    #sol = solve_ivp(fun, (0, 20), y0, method=ExpEuler, jac=jac)
    sol_r = solve_ivp(fun, (0, 20), y0, rtol=1.e-6, atol=1.e-8)
    #print(sol_g.t)
    #print(sol_g.y)
    print(len(sol_e.t), len(sol_g.t))

    plt.figure()
    plt.plot(sol_r.y[0,:], sol_r.y[1,:])
    plt.plot(sol_e.y[0,:], sol_e.y[1,:])
    plt.plot(sol_g.y[0,:], sol_g.y[1,:])
    plt.show()
