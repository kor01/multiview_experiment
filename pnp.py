import sympy as sp
import numpy as np
from equation import linear_coefficient
from equation import stack_coefficients
from least_square import solve_linear_homogeneous
from sympy.utilities.autowrap import autowrap


def extract_coefficient(equation, variables):
  coeff0 = linear_coefficient(
    equation[0], variables, is_homogeneous=True)
  coeff1 = linear_coefficient(
    equation[1], variables, is_homogeneous=True)
  coeff = sp.Matrix((coeff0, coeff1))
  return coeff


def linear_pnp_coefficient():

  p2dv = sp.symbols('x1_1:3')
  p3dv = sp.symbols('x0_1:4')

  p2d = sp.Matrix(p2dv + (1,))
  p3d = sp.Matrix(p3dv + (1,))

  pv = sp.symbols('p1:13')
  p = sp.Matrix(pv).reshape(3, 4)

  equation = p2d.cross(p * p3d)
  coeff =  extract_coefficient(equation, pv)
  fn = autowrap(coeff, args=p2dv + p3dv)

  def generator(pair):
    left, right = pair
    return fn(*list(left), *list(right))

  return generator


pnp_gen = linear_pnp_coefficient()

def estimate_pnp(pairs):
  coeff = stack_coefficients(pairs, pnp_gen)
  ret = solve_linear_homogeneous(coeff).reshape(3, 4)
  return ret


def normalize_vec(vec):
  ret = vec / np.linalg.norm(vec)
  return ret


def approximate_so3(mat):
  """
  find the closest so3 transform under Frobenius norm
  :param mat: a 3x3 matrix
  :return:a 3x3 rotation matrix
  """
  xaxis = normalize_vec(mat[0, :])
  yaxis = normalize_vec(mat[1, :])
  yaxis = np.cross(np.cross(xaxis, yaxis), yaxis)
  yaxis = normalize_vec(yaxis)
  zaxis = np.cross(xaxis, yaxis)

  rot = np.vstack((xaxis, yaxis, zaxis)).transpose()
  #TODO: represent rot as quaternion and perform nonlinear LS
  return rot


def solve_pnp(pairs):
  ret = estimate_pnp(pairs)
  ret[:, :3] = approximate_so3(ret[:, :3])
  return ret
