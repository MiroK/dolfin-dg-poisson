from dolfin import *

set_log_level(ERROR)

def bar(n_cells):
  # Create mesh and define function space
  mesh = UnitSquareMesh(n_cells, n_cells)
  V = FunctionSpace(mesh, "DG", 1)

  # Define Dirichlet boundary (x = 0 or x = 1)
  class DirichletBoundary(SubDomain):
    def inside(self, x, on_boundary):
      return on_boundary and near(x[0]*(1 - x[0]), 0)

  # Define Neumann boundary (y = 0 or y = 1)
  class NeumanBoundary(SubDomain):
    def inside(self, x, on_boundary):
      return on_boundary and near(x[1]*(1 - x[1]), 0)

  # Define boundary condition
  f = Expression("-100*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)")
  u0 = Expression('0.')#'x[0] + 0.25*sin(2*pi*x[1])')
  g = Expression("(x[1] - 0.5)*(x[1] - 0.5)")

  boundaries = FacetFunction('size_t', mesh, 0)
  NeumanBoundary().mark(boundaries, 2)
  DirichletBoundary().mark(boundaries, 1)

  ds = Measure('ds')[boundaries]

  n = FacetNormal(mesh)
  h = CellSize(mesh)
  h_avg = (h('+') + h('-'))/2

  # Define variational problem
  u = TrialFunction(V)
  v = TestFunction(V)

  # Define parameters
  alpha = 4.0
  gamma = 8.0

  # Define variational problem
  a = dot(grad(v), grad(u))*dx \
     - dot(avg(grad(v)), jump(u, n))*dS \
     - dot(jump(v, n), avg(grad(u)))*dS \
     + alpha/h_avg*dot(jump(v, n), jump(u, n))*dS \
     - dot(grad(v), u*n)*ds(1) \
     - dot(v*n, grad(u))*ds(1) \
     + gamma/h*v*u*ds(1)
  L = v*f*dx - u0*dot(grad(v), n)*ds(1) + gamma/h*u0*v*ds(1) + g*v*ds(2)

  # Compute solution
  u = Function(V)
  solve(a == L, u)

  # Plot solution
  #plot(u, interactive=True)

  return u
