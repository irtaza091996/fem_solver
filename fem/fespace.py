import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri


class FESpace:
    def __init__(self, mesh, element, quadrule):
        self.mesh = mesh
        self.element = element
        self.quadrule = quadrule

    def interpolateFunction(self, f):
        return f(self.mesh.nodes)

    def plotDOFVector(self, u):
        nodes = self.mesh.nodes
        triangulation = tri.Triangulation(nodes[:, 0], nodes[:, 1], self.mesh.cells)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(triangulation, u)
        plt.show()

    def getDOFMap(self, i, *argv):
        cells = self.mesh.cells
        if len(argv) > 0:
            T = int(argv[0])
            tmp = np.where(cells[T, :] == i)
            return tmp[0][0] if tmp[0].size > 0 else -1
        else:
            return np.where(cells == i)

    def assembleDR(self, a, b, f, dirichlet):
        """Assemble stiffness matrix and load vector for reaction-diffusion:
            a(x,y) * u - div(b(x,y) * grad(u)) = f(x,y)
        with optional homogeneous Dirichlet boundary conditions.
        Returns (S, l, U): stiffness matrix, load vector, full solution DOF vector.
        """
        nodes = self.mesh.nodes

        if dirichlet:
            bounds_low = np.array([self.mesh.omega[0], self.mesh.omega[2]])
            bounds_high = np.array([self.mesh.omega[1], self.mesh.omega[3]])
            ind = np.where(
                np.minimum(
                    np.min(np.abs(nodes - bounds_low), axis=1),
                    np.min(np.abs(nodes - bounds_high), axis=1),
                ) > 1e-6
            )[0]
        else:
            ind = np.arange(nodes.shape[0])

        nDOF = ind.shape[0]
        S = np.zeros((nDOF, nDOF))
        l = np.zeros(nDOF)

        points = self.quadrule.points
        weights = self.quadrule.weights

        dets_all = self.mesh.getTrafoDet()
        invJac_all = self.mesh.getInverseJacobian()
        traf_points_all = self.mesh.evalReferenceMap(points.T)

        basis = self.element.evalBasis(points)
        der_basis = self.element.evalDerBasis(points)

        for i in range(nDOF):
            for j in range(nDOF):
                supp_i, dmap_i = self.getDOFMap(ind[i])
                supp_j, dmap_j = self.getDOFMap(ind[j])

                supp_ij, ind_i, ind_j = np.intersect1d(
                    supp_i, supp_j, assume_unique=False, return_indices=True
                )
                dmap_i = dmap_i[ind_i]
                dmap_j = dmap_j[ind_j]

                dets = dets_all[supp_ij]
                invJacs = invJac_all[supp_ij]
                traf_points = traf_points_all[supp_ij, :, :]

                for t in range(supp_ij.shape[0]):
                    for k in range(points.shape[0]):
                        xk, yk = traf_points[t, 0, k], traf_points[t, 1, k]
                        S[i, j] += (
                            dets[t] * weights[k] * a(xk, yk)
                            * basis[dmap_i[t], k] * basis[dmap_j[t], k]
                        )
                        gradJ = np.dot(invJacs[t], der_basis[dmap_j[t], k, :])
                        gradI = np.dot(invJacs[t], der_basis[dmap_i[t], k, :])
                        S[i, j] += dets[t] * weights[k] * b(xk, yk) * np.dot(gradJ, gradI)

            supp_i, dmap_i = self.getDOFMap(ind[i])
            dets = dets_all[supp_i]
            traf_points = traf_points_all[supp_i, :, :]
            for t in range(supp_i.shape[0]):
                for k in range(points.shape[0]):
                    xk, yk = traf_points[t, 0, k], traf_points[t, 1, k]
                    l[i] += dets[t] * weights[k] * f(xk, yk) * basis[dmap_i[t], k]

        u = np.linalg.solve(S, l)
        U = np.zeros(nodes.shape[0])
        U[ind] = u
        return S, l, U
