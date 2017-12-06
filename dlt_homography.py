import numpy as np
import sympy as sp
from sympy.utilities.autowrap import autowrap

def dlt_generate_equation(numerical=True):

  """
  DLT direct linear method for solving homography
  generate two equations from one pair of point correspondence in homography
  :return: compile sympy function
  """

  # symbolic point pairs
  x0v = sp.symbols('x0_1:4')
  x0 = sp.Matrix(x0v)
  x1v = sp.symbols('x1_1:4')
  x1 = sp.Matrix(x1v)

  # symbolic homography
  vh = sp.symbols('h1:10')
  h = sp.Matrix(vh).reshape(3, 3)

  # generate 3 equations, first two are linear independent
  equation = x0.cross(h * x1)

  def coeff_as_vector(eff):
    ret = []
    for hi in vh:
      if hi in eff:
        ret.append(eff[hi])
      else:
        ret.append(0)
    return sp.Matrix(ret)

  coeff0 = sp.collect(sp.expand(equation[0]), vh, evaluate=False)
  coeff1 = sp.collect(sp.expand(equation[1]), vh, evaluate=False)

  coeff0 = coeff_as_vector(coeff0)
  coeff1 = coeff_as_vector(coeff1)

  coeff0 = coeff0.transpose().tolist()[0]
  coeff1 = coeff1.transpose().tolist()[0]

  coeff = sp.Matrix((coeff0, coeff1))

  if numerical:
    return autowrap(coeff, args=x0v + x1v)
  else:
    return coeff, x0v + x1v


parameter_generator = dlt_generate_equation(numerical=True)


def generate_dlt_parameter(pairs):
  ret = []
  for pair in pairs:
    ret.append(parameter_generator(*pair[0], *pair[1]))
  return np.vstack(ret)

# solve dlt by svd
def solve_dlt(pairs):
  parameter = generate_dlt_parameter(pairs)
  s, v, d = np.linalg.svd(parameter)
  # every vector corresponding to zero elements in
  solution = d[:, -1].reshape(3, 3)
  return solution
