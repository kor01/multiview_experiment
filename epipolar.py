import numpy as np
import sympy as sp
from sympy.utilities.autowrap import autowrap
from equation import linear_coefficient
from least_square import solve_linear_homogeneous



def fundamental_equation():
  """
  create fundamental equation coefficients
  :return: fortan function
  """
  x0v = sp.symbols('x0_1:4')
  x0 = sp.Matrix(x0v)
  x1v = sp.symbols('x1_1:4')
  x1 = sp.Matrix(x1v)
  fv = sp.symbols('f1:10')
  f = sp.Matrix(fv).reshape(3, 3)
  equation = sp.transpose(x0) * f * x1
  coeffs = linear_coefficient(equation, fv)
  coeffs = sp.Matrix(coeffs)
  ret = autowrap(coeffs, args=x0v + x1v)

  return ret

coeff_gen = fundamental_equation()


def generate_fundamental_parameter(pairs):

  coeffs = []
  for pair in pairs:
    coeff = coeff_gen(
      pair[0][0], pair[0][1], 1,
      pair[1][0], pair[1][1], 1)
    coeffs.append(coeff)
  coeffs = np.array(coeffs)
  return coeffs


def svd_cleanup(essential):
  u, d, v = np.linalg.svd(essential)
  sigma = (d[0] + d[1]) * 0.5
  d = np.diag([sigma, sigma, 0])
  return u @ d @ v


def estimate_essential(
    pairs, inv_intrinsics):

  pairs = pairs.reshape(-1, 2).transpose()
  pairs = np.matmul(inv_intrinsics, pairs)
  pairs = pairs.reshape(-1, 2, 2)
  coeff = generate_fundamental_parameter(pairs)

  # solve equation
  essential = solve_linear_homogeneous(coeff)
  essential = svd_cleanup(essential.reshape(3, 3))
  return essential


def test_front(rot, t, pairs):
  for _, pt in pairs:
    image = rot @ pt + t
    if image[2] < 0:
      return False
  return True


def estimate_euclidean(essential, pairs):
  """
  estimate euclidean transform from essential matrix
  :return: R and t
  """
  # solve translation
  t = solve_linear_homogeneous(essential.transpose())

  u, d, v = np.linalg.svd(essential)
  w = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])

  rot0 = u @ w @ v
  rot1 = u @ w.transpose() @ v

  candidates = [(rot0, t), (rot0, -t), (rot1, t), (rot1, -1)]

  for c in candidates:
    if test_front(c[0], c[1], pairs):
      return c
  raise AttributeError('non in front')



