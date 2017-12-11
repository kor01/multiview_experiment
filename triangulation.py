import sympy as sp
import numpy as np
from sympy.utilities.autowrap import autowrap
from equation import linear_coefficient
from equation import stack_coefficients


def triangulation_coefficient():

  p2dv = sp.symbols('x1_1:3')
  p3dv = sp.symbols('x0_1:4')
  p2d = sp.Matrix(p2dv + (1,))
  p3d = sp.Matrix(p3dv + (1,))

  pv = sp.symbols('p1:12')
  p = sp.Matrix(pv).reshape(3, 4)

  equation = p2d.cross(p * p3d)

  coeff0 = linear_coefficient(
    equation[0], p3dv, is_homogeneous=False)

  coeff1 = linear_coefficient(
    equation[1], p3dv, is_homogeneous=False)

  coeff0 = coeff0.transpose().tolist()[0]
  coeff1 = coeff1.transpose().tolist()[0]
  coeff = sp.Matrix((coeff0, coeff1))

  fn = autowrap(coeff, args=p2dv)

  def generator(x2d, x3d):
    return fn(x2d[0], x2d[1], x3d[0], x3d[1], x3d[2])

  return generator


coefficient_gen = triangulation_coefficient()


def solve_triangulation(pairs):
  """
  solve a small scale triangulation with linear LS
  :param pairs:
  :return:
  """
  parameters = stack_coefficients(pairs, coefficient_gen)
  j, b = parameters[:, :-1], -parameters[:, -1]
  jt = j.transpose()

  optimal = np.linalg.pinv(jt @ j) @ jt @ b

  return optimal
