import numpy as np
import matplotlib.pyplot as plt
import math
import time

class Mesh:
    def __init__(self, Nx, Ny, xmin, xmax, ymin, ymax):
        self.omega = np.array([xmin, xmax, ymin, ymax])
        self.nodes, self.cells = self.initializeMesh(Nx, Ny)
        self.numbers = np.array([Nx, Ny])

    def initializeMesh(self, Nx, Ny):
        x = np.linspace(self.omega[0], self.omega[1], Nx + 1)
        y = np.linspace(self.omega[2], self.omega[3], Ny + 1)

        xx, yy = np.meshgrid(x, y)
        nodes = np.stack((xx, yy), axis=2)
        nodes = nodes.reshape((Nx + 1) * (Ny + 1), 2)

        cells = np.array([[1, 2, Nx + 2], [Nx + 3, Nx + 2, 2]])
        cells = np.tile(cells, (Nx * Ny, 1))

        shift = np.arange(0, Nx * Ny).reshape(Nx * Ny, 1)
        shift = np.repeat(np.repeat(shift, 2, axis=0), 3, axis=1)
        cells = cells + shift

        shift = np.arange(0, Ny).reshape(Ny, 1)
        shift = np.repeat(np.repeat(shift, 2 * Nx, axis=0), 3, axis=1)
        cells = cells + shift - 1

        return nodes, cells

    def evalReferenceMap(self, x):
        nP = x.shape[1]

        verts = self.nodes[self.cells, :]
        verts = np.swapaxes(verts, 2, 1)

        trafoVec = verts[:, :, 0]
        trafoMat = np.stack(
            (verts[:, :, 1] - verts[:, :, 0], verts[:, :, 2] - verts[:, :, 0]),
            axis=2,
        )

        trafoVec = np.tile(trafoVec, (nP, 1, 1))
        trafoVec = np.rollaxis(trafoVec, 0, 3)

        return np.dot(trafoMat, x) + trafoVec

    def getTrafoDet(self):
        verts = self.nodes[self.cells, :]
        verts = np.swapaxes(verts, 2, 1)
        trafoMat = np.stack(
            (verts[:, :, 1] - verts[:, :, 0], verts[:, :, 2] - verts[:, :, 0]),
            axis=2,
        )
        return np.linalg.det(trafoMat)

    def getInverseJacobian(self):
        verts = self.nodes[self.cells, :]
        verts = np.swapaxes(verts, 2, 1)

        dets = self.getTrafoDet()

        firstCol = np.stack(
            (verts[:, 1, 2] - verts[:, 1, 0], verts[:, 1, 0] - verts[:, 1, 1]), axis=1
        )
        secondCol = np.stack(
            (verts[:, 0, 0] - verts[:, 0, 2], verts[:, 0, 1] - verts[:, 0, 0]), axis=1
        )
        adjMat = np.stack((firstCol, secondCol), axis=2)

        return np.divide(adjMat, dets[:, np.newaxis, np.newaxis])

    def show(self):
        plt.figure()
        triangles = []
        start = time.time()

        for i in range(self.cells.shape[0]):
            verts = self.nodes[self.cells[i].transpose(), :]
            plt.scatter(verts[:, 0], verts[:, 1], color='xkcd:blue')
            triangles.append(plt.Polygon(verts, fill=0, color='xkcd:blue'))
            plt.gca().add_patch(triangles[i])

        end = time.time()
        print(f"Mesh plotting time: {end - start:.3f}s")
        plt.show()
