import sympy as sp
import numpy as np

def get_coefs(poles:list):
    return np.flip(np.polynomial.polynomial.polyfromroots(poles)[:-1])
