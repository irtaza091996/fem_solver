import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class P1Element:

    def evalBasis(self, x):
        dim = len(x.shape)
        if dim == 1:
            phi1 = 1 - x
            phi2 = x
            return np.stack((phi1.T, phi2.T), axis=0)
        elif dim == 2:
            phi1 = 1 - x[:, 0] - x[:, 1]
            phi2 = x[:, 0]
            phi3 = x[:, 1]
            return np.stack((phi1.T, phi2.T, phi3.T), axis=0)

    def evalDerBasis(self, x):
        dim = len(x.shape)
        N = x.shape[0]
        if dim == 1:
            phi1 = -1 + 0 * x
            phi2 = 1 + 0 * x
            return np.stack((phi1.T, phi2.T), axis=0)
        elif dim == 2:
            z = 0 * x[:, 0]
            R_x = np.stack((-1 + z, 1 + z, z), axis=0)
            R_y = np.stack((-1 + z, z, 1 + z), axis=0)
            return np.stack((R_x, R_y), axis=2)

    def plotBasis(self, dim):
        fig = plt.figure()
        if dim == 1:
            x = np.linspace(0, 1, 100)
            basis = self.evalBasis(x)
            for i in range(2):
                plt.plot(x, basis[i, :])
        elif dim == 2:
            x = np.linspace(0, 1, 100)
            xx, yy = np.meshgrid(x, x)
            pts = np.stack((xx, yy), axis=2).reshape(10000, 2)
            pts[np.sum(pts, axis=1) > 1, :] = float('nan')
            basis = self.evalBasis(pts)
            ax = fig.add_subplot(111, projection='3d')
            for i in range(3):
                ax.plot_surface(xx, yy, basis[i, :].reshape(100, 100))
        plt.show()
