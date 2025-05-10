from pyqint import PyQInt, Molecule, GeometryOptimization, FosterBoys
import numpy as np
import pickle
import os
from .localization.fosterboys import localize_fosterboys

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
            self.orbc_opt, self.r2_opt, self.screenarr = localize_fosterboys(self.data['orbc'], self.dipolmat, self.nocc)
        elif method == "fosterboys-pyqint":
            resfb = FosterBoys(self.data).run()
            self.orbc_opt = resfb['orbc']
            self.r2_opt = resfb['r2final']
        
        # re-order the orbitals with increasing energy
        self.orbe, self.orbc_opt = self.__calculate_molecular_orbital_energies(self.orbc_opt)

        # calculate unitary transformation matrix
        self.u_opt = self.data['orbc'].T @ self.data['overlap'] @ self.orbc_opt

        # ensure that transformation has determinant of +1 by swapping the sign
        # of one of the core orbitals
        if np.linalg.det(self.u_opt) < 0:
            self.u_opt[:,0] *= -1

    def report_matrix(self):
        """
        Report the transformation matrix and its properties.
        """
        self.__assess_matrix_properties(self.u_opt)
    
    def calculate_fbr2(self, U, nocc):
        """
        Given a unitary transformation matrix, determine the R2 value
        """
        return np.sum(np.einsum('pi,qi,pql->il', 
                                (self.data['orbc'] @ U)[:,:nocc], 
                                (self.data['orbc'] @ U)[:,:nocc], 
                                self.dipolmat)**2)

    def show_jacobi_rotations(self, figsize=(16,16)):
        nsteps = len(self.screenarr)
        npairs = len(self.screenarr[0])
        nsamples = len(self.screenarr[0][0])
        theta = np.linspace(-np.pi/4, np.pi/4, nsamples)
        fig, ax = plt.subplots(nsteps,npairs)
        for j in range(nsteps):
            for i in range(npairs):
                ax[j,i].plot(theta, self.screenarr[j][i])
        plt.tight_layout()
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
    
    def __assess_matrix_properties(self, M, tol=1e-10):
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

    def __calculate_molecular_orbital_energies(self, C):
        """
        Calculate the one-electron MO energies from the Hamiltonian matrix
        and the coefficient matrix, both in their original basis

        Return *ordered* list of eigenvalue and -vector pairs
        """
        orbe = np.zeros(len(C))
        for i in range(len(C)):
            orbe[i] = C[:,i].dot(self.data['fock'].dot(C[:,i]))

        # produce list of indices for eigenvalues in ascending order
        oidx = np.argsort(orbe)

        return orbe[oidx], C[:,oidx]