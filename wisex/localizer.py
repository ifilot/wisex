from pyqint import PyQInt, Molecule, GeometryOptimization, FosterBoys
import numpy as np
import pickle
import os
from .localization.fosterboys import localize_fosterboys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Localizer:
    def __init__(self, mol, basis, cachefolder):
        """
        Construct MOFlow object using completes RHF calculation
        """
        self.mol = mol
        self.basis = basis
        self.cachefolder = cachefolder
        os.makedirs(self.cachefolder, exist_ok=True)
        print("Using cache folder: ", self.cachefolder)

        # perform Geometry optimization
        self.__prepare()
        self.__build_dipole_tensor()
    
    def perform_localization(self, method='fosterboys'):
        """
        Perform localization of the MO coefficients using the specified method.
        """
        if method == 'fosterboys':
            self.orbc_opt, self.r2_opt = localize_fosterboys(self.data['orbc'], self.dipolmat, self.nocc)
        elif method == "fosterboys-pyqint":
            resfb = FosterBoys(self.data).run()
            self.orbc_opt = resfb['orbc']
            self.r2_opt = resfb['r2final']
        
        # calculate unitary transformation matrix
        self.u_opt = self.data['orbc'].T @ self.data['overlap'] @ self.orbc_opt

    def report_matrix(self):
        """
        Report the transformation matrix and its properties.
        """
        self.assess_matrix_properties(self.u_opt)

    def produce_contour_plot(self, orbc, rows, cols, figsize, sz=5, dpi=144, save_path=None):
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

                dens = self.__plot_wavefunction(self.data['cgfs'], orbc[:, idx])
                limit = max(abs(np.min(dens)), abs(np.max(dens)))

                x = np.linspace(-sz, sz, dens.shape[1])
                y = np.linspace(-sz, sz, dens.shape[0])
                X, Y = np.meshgrid(x, y)

                cf = ax[i, j].contourf(X, Y, dens, levels=100, cmap='PiYG', vmin=-limit, vmax=limit)
                ax[i, j].contour(X, Y, dens, levels=10, colors='black', linewidths=0.5)
                ax[i, j].set_aspect('equal')
                ax[i, j].set_xlabel('x [a.u.]')
                ax[i, j].set_ylabel('y [a.u.]')

                divider = make_axes_locatable(ax[i, j])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(cf, cax=cax, orientation='vertical')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        else:
            plt.show()

#------------------------------------------------------------------------------#
# PRIVATE METHODS
#------------------------------------------------------------------------------#

    def __build_dipole_tensor(self):
        """
        Build and cache a dipole tensor. Stores result in self.dipolmat.
        """
        cache_path = os.path.join(self.cachefolder, 'dipole_tensor.npy')

        if os.path.exists(cache_path):
            print("Loading cached dipole tensor. ", end="")
            self.dipolmat = np.load(cache_path)
        else:
            print("Calculating dipole matrices. ", end="")
            cgfs = self.data['cgfs']
            N = len(cgfs)
            mat = np.zeros((N, N, 3))
            integrator = PyQInt()
            for i, cgf1 in enumerate(cgfs):
                for j, cgf2 in enumerate(cgfs):
                    for k in range(3):
                        mat[i, j, k] = integrator.dipole(cgf1, cgf2, k, 0.0)

            # Save and store
            np.save(cache_path, mat)
            self.dipolmat = mat
        
        # done
        self.__print_ok()
    
    def __prepare(self):
        """
        Prepare the system for calculation - performs a Geometry Optimization
        """
        cache_path = os.path.join(self.cachefolder, 'geomopt.pkl')
        
        if os.path.exists(cache_path):
            print("Loading cached optimization result. ", end="")
            with open(cache_path, 'rb') as f:
                self.opt = pickle.load(f)
        else:
            print("Running geometry optimization. This may take a while. ", end="")
            self.opt = GeometryOptimization(verbose=False).run(self.mol, self.basis)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.opt, f)

        self.data = self.opt['data']
        self.nocc = self.data['nelec'] // 2

        # done
        self.__print_ok()
    
    def assess_matrix_properties(self, M, tol=1e-10):
        """
        Analyze and print properties of a matrix, including whether it belongs to
        U(n), SU(n), O(n), or SO(n), and highlight determinant = -1 cases.

        Parameters:
            M (ndarray): Square matrix to check
            name (str): Optional label for the matrix
            tol (float): Tolerance for numerical checks
        """
        def colored(val, color_code):
            return f"\033[{color_code}m{val}\033[0m"
    
        def colored_bool(val):
            return f"\033[92m{val}\033[0m" if val else f"\033[91m{val}\033[0m"
    
        if M.shape[0] != M.shape[1]:
            print(f"Matrix is not square — cannot classify.")
            return

        I = np.eye(M.shape[0])
        is_real = np.all(np.isreal(M))
        det = np.linalg.det(M)
        det_str = f"{det:.6f}"
        
        is_unitary = np.allclose(M.conj().T @ M, I, atol=tol)
        is_special_unitary = is_unitary and np.allclose(det, 1.0, atol=tol)
        is_orthogonal = is_real and np.allclose(M.T @ M, I, atol=tol)
        is_special_orthogonal = is_orthogonal and np.allclose(det, 1.0, atol=tol)

        print(f"\n\033[1mTransformation matrix properties\033[0m")
        print(f"  Size: {M.shape[0]} x {M.shape[1]}")
        print(f"  Determinant: {colored(det_str, '96')}")
        print(f"  Real-Valued: {colored_bool(is_real)}")

        print(f"\n  Unitary (U(n)):             {colored_bool(is_unitary)}")
        print(f"  Special Unitary (SU(n)):    {colored_bool(is_special_unitary)}")

        print(f"\n  Orthogonal (O(n)):          {colored_bool(is_orthogonal)}")
        print(f"  Special Orthogonal (SO(n)): {colored_bool(is_special_orthogonal)}")

    def __print_ok(self):
        """
        Print OK message in green color.
        """
        print("\033[92m[OK]\033[0m")

    def __plot_wavefunction(self, cgfs, coeff, sz=5, npts=100):
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