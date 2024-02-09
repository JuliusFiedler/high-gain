import sympy as sp
import numpy as np

def get_coefs(poles:list):
    return np.flip(np.polynomial.polynomial.polyfromroots(poles)[:-1])

def get_percentage_over_abs(data, limit):
    total = len(data)
    assert total > 1
    count = len(np.where(np.abs(data) > limit)[0])
    return count, total, f"{count/total*100}%"