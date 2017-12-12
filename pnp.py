import cv2
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
    return fn(left[0], left[1], *list(right))

  return generator


pnp_gen = linear_pnp_coefficient()

def svd_cleanup(proj):
  r, t = proj[:, :3], proj[:, -1]
  u, d, v = np.linalg.svd(r)
  sign = np.linalg.det(u) * np.linalg.det(v)
  r = u @ v * sign
  t = sign * t / d[0]
  print(r @ r.transpose(), t)
  return np.concatenate((r, t[:, None]), axis=-1)


def estimate_pnp(pairs):
  coeff = np.array([pnp_gen(v) for v in pairs])
  coeff = coeff.reshape(-1, coeff.shape[-1])
  ret = solve_linear_homogeneous(coeff).reshape(3, 4)
  ret = svd_cleanup(ret)
  return ret


def gradient_descend(grad, loss):
  pass


def optimize_proj(initial, pairs):
  """
  further optimize projection given initial value and pairs
  :param initial:
  :param pairs:
  :return:
  """
  pass



def normalize_vec(vec):
  ret = vec / np.linalg.norm(vec)
  return ret

def solve_pnp(pairs):
  ret = estimate_pnp(pairs)
  r, t = ret[:, :3], ret[:, -1]
  axis, angle = extract_axis_angle(r)
  return ret

cv2.solvePnP()