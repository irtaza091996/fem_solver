"""Solve sin(πx)·u - div(sin(πy)·∇u) = 10 on [0,1]² with homogeneous Dirichlet BCs."""
import sys
sys.path.insert(0, '..')

import numpy as np
from fem import Mesh, P1Element, Quadrature, FESpace

a = lambda x, y: np.sin(np.pi * x)   # spatially-varying reaction
b = lambda x, y: np.sin(np.pi * y)   # spatially-varying diffusion
f = lambda x, y: 10                   # constant source

m = Mesh(10, 10, 0, 1, 0, 1)
e = P1Element()
q = Quadrature(2, 3)
fes = FESpace(m, e, q)

S, l, u = fes.assembleDR(a, b, f, dirichlet=True)
fes.plotDOFVector(u)
