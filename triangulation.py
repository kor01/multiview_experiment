import sympy as sp
import numpy as np
from sympy.utilities.autowrap import autowrap
from equation import linear_coefficient


def extract_coefficient(equation, variables):
  coeff0 = linear_coefficient(
    equation[0], variables, is_homogeneous=False)
  coeff1 = linear_coefficient(
    equation[1], variables, is_homogeneous=False)
  coeff = sp.Matrix((coeff0, coeff1))
  return coeff


def triangular_source_coefficient():
  p2dv = sp.symbols('x1_1:3')
  p3dv = sp.symbols('x0_1:4')
  p2d = sp.Matrix(p2dv + (1,))
  p3d = sp.Matrix(p3dv)
  equation = p2d.cross(p3d)
  coeff = extract_coefficient(equation, p3dv)
  fn = autowrap(coeff, args=p2dv)

  def generator(pt):
    return fn(pt[0], pt[1])

  return generator


def target_coefficient():
  p2dv = sp.symbols('x0_1:3')
  p3dv = sp.symbols('x1_1:4')
  p2d = sp.Matrix(p2dv + (1,))
  p3d = sp.Matrix(p3dv + (1,))
  pv = sp.symbols('p1:13')
  p = sp.Matrix(pv).reshape(3, 4)
  equation = p2d.cross(p * p3d)
  coeff = extract_coefficient(equation, p3dv)
  fn = autowrap(coeff, args=p2dv + pv)

  def generator(pt, proj):
    return fn(pt[0], pt[1], *list(proj.ravel()))
  return generator


target_gen = target_coefficient()
source_gen = triangular_source_coefficient()


def solve_triangulation(source, target, projection):
  """
  solve a small scale triangulation with linear LS
  """
  source_coeff = source_gen(source)
  target_coeff = target_gen(target, projection)
  coeffs = np.vstack((source_coeff, target_coeff))
  j, b = coeffs[:, :-1], -coeffs[:, -1]
  jt = j.transpose()
  optimal = np.linalg.pinv(jt @ j) @ jt @ b
  return optimal

