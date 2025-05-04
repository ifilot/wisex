import numpy as np
import scipy.linalg

class Geodesic:
    def __init__(self, U, nocc):
        self.U = U
        self.nocc = nocc
        self.H = None

        # extract submatrix corresponding to occupied orbitals
        self.U_occ = U[:nocc, :nocc]

    def build_generator(self, group = 'su(n)'):
        """
        Build the generator of the geodesic flow.
        """
        if group == 'su(n)':
            self.H = self.__build_su_generator()
        elif group == 'so(n)':
            self.H = self.__build_so_generator()

        self.group = group

    def interpolate(self, t):
        """
        Interpolate the geodesic flow at time t.
        """
        if self.H is None:
            raise ValueError("Generator not built. Call build_generator() first.")
        
        if self.group == 'su(n)':
            U_int = self.__safe_recast_to_real(scipy.linalg.expm(1j * t * self.H))
        elif self.group == 'so(n)':
            U_int = self.__safe_recast_to_real(scipy.linalg.expm(t * self.H))
        U_full = np.eye(self.U.shape[0], dtype=U_int.dtype)
        U_full[:self.nocc, :self.nocc] = U_int

        return U_full

    def __build_su_generator(self):
        print("Building SU(n) generator.")
        e,v = self.__diagonalize_unitary_matrix(self.U_occ)
        H = v @ np.diag(np.angle(e)) @ v.conj().T
        
        Utest = scipy.linalg.expm(1j * H)
        assert np.allclose(Utest, self.U_occ, atol=1e-10), "Generator does not reproduce original matrix."

        return H

    def __build_so_generator(self):
        print("Building SO(n) generator.")
        A = scipy.linalg.logm(self.U_occ)
        A = (A - A.T) / 2
        A = A.real
        
        Utest = scipy.linalg.expm(A)
        assert np.allclose(Utest, self.U_occ, atol=1e-10), "Generator does not reproduce original matrix."

        return A

    def __diagonalize_unitary_matrix(self, U, atol=1e-8):
        """
        Given a unitary matrix U, compute eigenvectors grouped by eigenvalue
        and return an orthonormal set of eigenvectors.
        """
        e, v = np.linalg.eig(U)

        # Convert complex eigenvalues to rounded phases to group by
        angles = np.angle(e)
        eigenvalue_groups = {}

        for idx, theta in enumerate(angles):
            # Use rounded phase as key to group degenerate eigenvalues
            key = np.round(theta / atol) * atol
            eigenvalue_groups.setdefault(key, []).append(v[:, idx])

        # Orthonormalize each group
        v_orthonormal = []

        for vecs in eigenvalue_groups.values():
            block = np.column_stack(vecs)  # shape (n, deg)
            Q, _ = np.linalg.qr(block)     # QR gives orthonormal basis
            v_orthonormal.append(Q)

        # Assemble full orthonormal eigenbasis
        V_final = np.column_stack(v_orthonormal)

        return e, V_final

    def __safe_recast_to_real(self, A, tol=1e-12):
        """
        If imaginary parts of A are all below `tol`, return the real part.
        Otherwise, raise an error.
        """
        if np.all(np.abs(A.imag) < tol):
            return A.real
        else:
            return A