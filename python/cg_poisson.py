"""This demo program solves Poisson's equation

    - div grad u(x, y) = f(x, y)

on the unit square with source f given by

    f(x, y) = 100*exp(-((x - 0.5)^2 + (y - 0.5)^2) / 0.02)

and boundary conditions given by

    u(x, y) = x + 0.25*sin(2*pi*y) for x = 0 or x = 1
du/dn(x, y) = (y - 0.5)**2  for y = 0 or y = 1
"""

# Copyright (C) 2007-2011 Anders Logg
#
# This file is part of DOLFIN.
#
# DOLFIN is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DOLFIN is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
#
# First added:  2007-08-16
# Last changed: 2012-11-12

# Begin demo

from dolfin import *

set_log_level(ERROR)

def foo(n_cells):
  # Create mesh and define function space
  mesh = UnitSquareMesh(n_cells, n_cells)
  V = FunctionSpace(mesh, "Lagrange", 1)

  # Define Dirichlet boundary (x = 0 or x = 1)
  def dirichlet_boundary(x):
      return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

  # Define Neumann boundary (y = 0 or y = 1)
  class NeumanBoundary(SubDomain):
    def inside(self, x, on_boundary):
      return on_boundary and near(x[1]*(1 - x[1]), 0)

  # Define boundary condition
  u0 = Expression('0')#'x[0] + 0.25*sin(2*pi*x[1])')
  bc = DirichletBC(V, u0, dirichlet_boundary)

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)
  f = Expression("-100*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
  g = Expression("(x[1] - 0.5)*(x[1] - 0.5)")

  boundaries = FacetFunction('size_t', mesh, 0)
  NeumanBoundary().mark(boundaries, 1)

  ds = Measure('ds')[boundaries]

  a = inner(grad(u), grad(v))*dx
  L = f*v*dx + g*v*ds(1)

  # Compute solution
  u = Function(V)
  solve(a == L, u, bc)

  # Plot solution
  #plot(u, interactive=True)

  return u
