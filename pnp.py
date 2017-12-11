import sympy as sp
import numpy as np
from equation import linear_coefficient
from least_square import solve_linear_homogeneous
from sympy.utilities.autowrap import autowrap


def homogeneous_pnp_equation():

  p2dv = sp.symbols('x1_1:4')
  p3dv = sp.symbols('x0_1:5')

  p2d = sp.Matrix(p2dv)
  p3d = sp.Matrix(p3dv)

  pv = sp.symbols('p1:12')
  p = sp.Matrix(pv).reshape(3, 4)

  equation = p2d.cross(p * p3d)

  coeff0 = linear_coefficient(
    equation[0], pv)
  coeff1 = linear_coefficient(
    equation[1], pv)

  coeff0 = coeff0.transpose().tolist()[0]
  coeff1 = coeff1.transpose().tolist()[0]

  coeff = sp.Matrix((coeff0, coeff1))

  return autowrap(coeff, args=p2dv + p3dv)


coefficient_gen = homogeneous_pnp_equation()


def generate_pnp_parameter(pairs):
  ret = []
  for pair in pairs:
    ret.append(coefficient_gen(
      pair[0][0], pair[0][1], 1, pair[1][0],
      pair[1][1], pair[1][2], 1))
  return np.vstack(ret)


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
  coeff = generate_pnp_parameter(pairs)
  solution = solve_linear_homogeneous(coeff).reshap(3, 4)
  solution[:, :3] = approximate_so3(solution[:, :3])
  return solution
