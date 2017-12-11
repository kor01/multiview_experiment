import sympy as sp
import numpy as np


def linear_coefficient(
    eq, variables, is_homogeneous=True):

  coeff = sp.collect(
    sp.expand(eq), variables, evaluate=False)

  ret = []
  for v in variables:
    if v in coeff:
      ret.append(coeff[v])
    else:
      ret.append(0)

  if not is_homogeneous:
    if 1 in coeff:
      ret.append(coeff[1])
    else:
      ret.append(0)

  ret = sp.Matrix(ret)
  ret = ret.transpose().tolist()[0]
  return ret


def stack_coefficients(variables, generator):
  v0 = generator(variables[0])
  ret = np.zeros((len(variables), v0.size))
  ret[0, :] = v0
  for i, v in enumerate(variables[1:]):
    ret[i + 1, :] = generator(v)
  return ret
