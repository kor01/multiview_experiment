import sympy as sp
import numpy as np
from sympy.utilities.autowrap import autowrap


def normalize(vec):
  return vec / np.linalg.norm(vec)


def axis_angle_derivative():
  axis = sp.symbols('omega0:3')
  angle = sp.symbols('theta')
  tv = sp.symbols('t0:3')
  t = sp.Matrix(tv)

  ct, st = sp.cos(angle), sp.sin(angle)
  x, y, z = axis
  axis_hat = sp.Matrix([[0, -z, y], [z, 0, -x], [x, -y, 0]])
  r = sp.eye(3) + st * axis_hat + (1 - ct) * axis_hat * axis_hat

  p2dv = sp.symbols('x1_1:3')
  p3dv = sp.symbols('x0_1:4')

  p2d = sp.Matrix(p2dv + (1,))
  p3d = sp.Matrix(p3dv + (1,))

  equation = p2d.cross(r * p3d + t)
  loss = equation[0] ** 2 + equation[1] ** 2

  derivative = sp.derive_by_array(loss, tv + (angle,) + axis)
  fn = autowrap(derivative, tv + (angle,) + axis)
  return fn



def extract_axis_angle(rotation):
  """
  extract axis angle representation to construct quaternion
  :param rotation:
  :return:
  """
  if np.isclose(rotation, np.eye(3)).all():
    return np.array([0, 0, 1.0]), 0.0

  eigen_val, eigen_vec = np.linalg.eig(rotation)
  idx = np.where(np.isclose(np.abs(np.imag(eigen_val)), 0))[0][0]
  nidx = (idx + 1) % 3

  axis = normalize(np.real(eigen_vec[:, idx]))
  axis_x = normalize(np.real(eigen_vec[:, nidx]))
  axis_y = normalize(np.imag(eigen_vec[:, nidx]))

  det = np.cross(axis_x, axis_y).dot(axis)
  basis = np.array([axis_x, axis_y, axis]) * det
  rep = basis @ rotation @ basis.transpose()
  theta = np.arctan2(rep[1, 0], rep[0, 0])

  return axis, theta

