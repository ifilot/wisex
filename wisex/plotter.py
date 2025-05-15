import os
import numpy as np
from pyqint import PyQInt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm import tqdm

class Plotter:
    def __init__(self):
        pass

    def plot(self, nrimages, rows, cols, figsize, 
             geodesic, localizer, sz=5, dpi=144, 
             vmin=None, vmax=None, save_path=None):
        """
        Produce a series of contour plots along the geodesic path.
        Parameters:
            nrimages (int): Number of images to generate
            rows (int): Number of subplot rows
            cols (int): Number of subplot columns
            figsize (tuple): Figure size in inches
            sz (float): Spatial size for plotting grid (default: 5 a.u.)
            dpi (int): Figure resolution (default: 144)
            vmin (float or None): Minimum value for color scale (default: None)
            vmax (float or None): Maximum value for color scale (default: None)
            save_path (str or None): If set, path to save the figures
        """
        orbe = localizer.orbe
        tt = np.linspace(0, 1, nrimages)

        for i,t in tqdm(enumerate(tt), total=nrimages, desc='Generating plots'):
            U = geodesic.interpolate(t)
            C = localizer.data['orbc'] @ U

            path = os.path.join(save_path, 'path_so_%03i.png' % (i+1))
            self.produce_contour_plot(localizer.data['cgfs'], C, orbe,
                                    rows=rows,  cols=cols,  figsize=figsize, 
                                    sz=sz, vmin=vmin, vmax=vmax, save_path=path)

        tt = np.linspace(0, 1, 20)
        d = [geodesic.calculate_distance(geodesic.interpolate(t)) for t in tt]
        r2 = [localizer.calculate_fbr2(geodesic.interpolate(t), localizer.nocc) for t in tt]
        
        fig, ax = plt.subplots(1,2)
        ax[0].plot(tt, d, '-o')
        ax[1].plot(tt, r2, '-o')
        ax[0].set_xlabel('Progression coordinate []')
        ax[1].set_xlabel('Progression coordinate []')
        ax[0].set_ylabel('Distance')
        ax[1].set_ylabel(r'$\left< r^{2} \right>$')
        plt.tight_layout()

        path = os.path.join(save_path, 'curve.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()

    def produce_contour_plot(self, cgfs, orbc, orbe, rows, cols, figsize, sz=5, dpi=144, 
                             vmin=None, vmax=None, save_path=None):
        """
        Produce contour plot of the wavefunction using the specified number of rows and columns.

        Parameters:
            orbc (ndarray): Orbital coefficient matrix
            rows (int): Number of subplot rows
            cols (int): Number of subplot columns
            figsize (tuple): Figure size in inches
            sz (float): Spatial size for plotting grid (default: 5 a.u.)
            dpi (int): Figure resolution (default: 144)
            save_path (str or None): If set, path to save the figure
        """
        fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

        # Ensure ax is a 2D array, even if rows or cols == 1
        if rows == 1 and cols == 1:
            ax = np.array([[ax]])
        elif rows == 1:
            ax = np.array([ax])
        elif cols == 1:
            ax = np.array([[a] for a in ax])

        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= orbc.shape[1]:
                    break  # Avoid index overflow if more subplots than orbitals

                wf = self.__plot_wavefunction(cgfs, orbc[:, idx], sz=sz)

                if vmin is None or vmax is None:
                    limit = max(abs(np.min(wf)), abs(np.max(wf)))
                    vmin = -limit
                    vmax = limit

                x = np.linspace(-sz, sz, wf.shape[1])
                y = np.linspace(-sz, sz, wf.shape[0])
                X, Y = np.meshgrid(x, y)

                cf = ax[i, j].contourf(X, Y, wf, levels=32, cmap='PiYG', vmin=vmin, vmax=vmax)
                ax[i, j].contour(X, Y, wf, levels=32, colors='black', linewidths=0.5, vmin=vmin, vmax=vmax)
                ax[i, j].set_aspect('equal')
                ax[i, j].set_xlabel('x [a.u.]')
                ax[i, j].set_ylabel('y [a.u.]')
                ax[i,j].set_title(f'MO {idx+1}: %.4f Ht' % (orbe[idx]))

                # Create a separate colorbar using a dummy mappable
                divider = make_axes_locatable(ax[i, j])
                cax = divider.append_axes('right', size='5%', pad=0.05)

                norm = Normalize(vmin=vmin, vmax=vmax)
                sm = ScalarMappable(norm=norm, cmap='PiYG')
                sm.set_array([])  # Required for matplotlib to allow the colorbar

                fig.colorbar(sm, cax=cax, orientation='vertical')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            #print(f"Figure saved to: {save_path}")
            plt.close()
        else:
            plt.show()
    
    def __plot_wavefunction(self, cgfs, coeff, sz=5, npts=100):
        """
        Produce the scalar field associated with the many-electron wavefunction.
        Parameters:
            cgfs (ndarray): Basis set (list of CGFs)
            coeff (ndarray): Coefficient vector for the molecular orbital
            sz (float): Spatial extent for plotting grid (default: 5 a.u.)
            npts (int): Number of points in each dimension for the grid (default: 100)
        """
        # build integrator
        integrator = PyQInt()

        # build grid
        x = np.linspace(-sz, sz, npts)
        y = np.linspace(-sz, sz, npts)
        xx, yy = np.meshgrid(x,y)
        zz = np.zeros(len(x) * len(y))
        grid = np.vstack([xx.flatten(), yy.flatten(), zz]).reshape(3,-1).T
        res = integrator.plot_wavefunction(grid, coeff, cgfs).reshape((len(y), len(x)))

        return res