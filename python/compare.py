from cg_poisson import foo
from dg_poisson import bar
from dolfin import *

for n_cells in [8, 16, 32, 64, 128]:
  u_cg = foo(n_cells)
  u_dg = bar(n_cells)

  plot(u_cg - u_dg, interactive=True)

  mesh = u_cg.function_space().mesh()
  print assemble(inner(u_cg - u_dg, u_cg - u_dg)*dx, mesh=mesh)
