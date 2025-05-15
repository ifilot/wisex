import numpy as np
import scipy.linalg

class Geodesic:
    def __init__(self, U, nocc, group = 'su(n)'):
        self.U = U
        self.nocc = nocc
        self.H = None
        self.group = group
        self.U_occ = U[:nocc, :nocc]
        self.__build_generator()

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

    def project_to_so3(self, M):
        """
        Projects a nearly orthogonal matrix M onto the closest orthogonal matrix
        with determinant +1 using SVD (i.e., onto SO(3)).
        
        Parameters:
            M (numpy.ndarray): A 3x3 matrix near orthogonality.
        
        Returns:
            R (numpy.ndarray): The closest rotation matrix to M in SO(3).
        """
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt

        # Ensure det(R) = +1 (proper rotation)
        if np.linalg.det(R) < 0:
            # Flip sign of last column of U
            U[:, -1] *= -1
            R = U @ Vt

        return R

    def axis_angle_so3(self, O):
        """
        Decomposes a 3x3 rotation matrix in SO(3) into its axis-angle
        representation and Lie algebra generator.

        Parameters:
            O (numpy.ndarray): A 3x3 orthogonal matrix with determinant +1
            (i.e., an element of SO(3)).

        Returns:
            axis (numpy.ndarray): A 3-element array representing the rotation
            axis (not normalized if angle is zero). theta (float): The rotation
            angle in radians, in the range [0, π]. generator (numpy.ndarray): A
            3x3 skew-symmetric matrix in so(3), the Lie algebra of SO(3), 
                                    such that O = exp(generator).

        Raises:
            ValueError: If the input matrix is not a valid SO(3) rotation
            matrix.
        """

        # Helper function to verify the matrix is a valid rotation matrix in SO(3)
        def is_rotation_matrix(R):
            return np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)

        # Check if the input matrix is a valid rotation matrix
        if not is_rotation_matrix(O):
            raise ValueError("Input matrix is not a valid SO(3) rotation matrix")

        # Compute the rotation angle using the trace of the matrix
        theta = np.arccos(np.clip((np.trace(O) - 1) / 2.0, -1.0, 1.0))

        # Handle the special case of zero rotation
        if np.isclose(theta, 0):
            axis = np.zeros(3)              # No rotation axis for identity matrix
            generator = np.zeros((3, 3))    # Zero matrix is the generator
        else:
            # Compute the skew-symmetric part of the matrix
            skew = (O - O.T) / (2 * np.sin(theta))

            # Extract the rotation axis from the skew-symmetric matrix
            axis = np.array([skew[2,1], skew[0,2], skew[1,0]])
            axis_normalized = axis / np.linalg.norm(axis)

            # Construct the skew-symmetric matrix of the axis
            ux = np.array([
                [0, -axis_normalized[2], axis_normalized[1]],
                [axis_normalized[2], 0, -axis_normalized[0]],
                [-axis_normalized[1], axis_normalized[0], 0]
            ])

            # The Lie algebra generator is the skew matrix scaled by the angle
            generator = theta * ux

        # Return the rotation axis, angle (in radians), and the Lie algebra generator matrix
        return axis, theta, generator


    def calculate_distance(self, U):
        """
        Calculate the distance of the geodesic
        """
        if self.group == 'su(n)':
            return scipy.linalg.norm(self.__build_su_generator(U), 'fro')
        elif self.group == 'so(n)':
            return scipy.linalg.norm(self.__build_so_generator(U), 'fro')

    ###########################
    ### AUXILIARY FUNCTIONS ###
    ###########################

    def __build_su_generator(self, U):
        e,v = self.__diagonalize_unitary_matrix(U)
        H = v @ np.diag(np.angle(e)) @ v.conj().T
        
        Utest = scipy.linalg.expm(1j * H)
        assert np.allclose(Utest, U, atol=1e-10), "Generator does not reproduce original matrix."

        return H

    def __build_so_generator(self, O):
        """
        Build the generator of the geodesic flow in SO(n) using the logarithm
        of the orthogonal matrix O. The generator is skew-symmetric and real.
        The function also verifies that the input matrix is orthogonal and has
        determinant +1 (i.e., belongs to SO(n)).
        Parameters:
            O (numpy.ndarray): A square orthogonal matrix with determinant +1.
        Returns:
            A (numpy.ndarray): The generator of the geodesic flow in SO(n),
            which is a skew-symmetric matrix in the Lie algebra so(n).
        """
        assert np.allclose(O.T @ O, np.eye(O.shape[0]), atol=1e-10), "Input matrix is not orthogonal."
        det = np.linalg.det(O)

        A = scipy.linalg.logm(O)
        A = (A - A.T.conj()) / 2
        
        Utest = scipy.linalg.expm(A)
        assert np.allclose(Utest, O, atol=1e-8), "Generator does not reproduce original matrix."

        return A
    
    def __build_so_angles_basis(self, O):
        """
        An alternative strategy to build the generator of the geodesic flow in
        SO(n) using the eigenvalues and eigenvectors of the orthogonal matrix O.

        Parameters:
            O (numpy.ndarray): A square orthogonal matrix with determinant +1.
        Returns:
            angles (numpy.ndarray): The angles of the eigenvalues of O.
            basis (numpy.ndarray): The orthonormal basis of eigenvectors of O.
        """
        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eig(O)

        # Normalize eigenvectors (just in case)
        for i in range(eigenvectors.shape[1]):
            eigenvectors[:, i] /= np.linalg.norm(eigenvectors[:, i])

        self.angles, self.basis = np.angle(eigenvalues), eigenvectors
        
        # ensure det = 1
        if np.linalg.det(self.basis) < 0:
            self.basis[:, -1] *= -1

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
        
    def __build_generator(self):
        """
        Build the generator of the geodesic flow.
        """
        if self.group == 'su(n)':
            self.H = self.__build_su_generator(self.U_occ)
        elif self.group == 'so(n)':
            self.H = self.__build_so_generator(self.U_occ)
            self.__build_so_angles_basis(self.U_occ)