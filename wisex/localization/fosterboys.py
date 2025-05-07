import numpy as np
import scipy.optimize

def localize_fosterboys(orbc, dipolmat, nocc):
    threshold = 1e-6
    max_iterations = 100
    prev_r2 = None
    iteration = 0
    screennarr = [] # track mu(theta) relationship over all iterations

    while True:
        orbcopt, r2, screenarr = jacobi_sweep_with_optimizer(orbc, dipolmat, nocc)
        screennarr.append(screenarr)

        if prev_r2 is not None and abs(r2 - prev_r2) < threshold:
            #print(f"Converged after {iteration} iterations.")
            break

        if iteration >= max_iterations:
            #print("Reached maximum number of iterations without full convergence.")
            break

        prev_r2 = r2
        orbc = orbcopt
        iteration += 1
    
    return orbcopt, r2, screennarr

def screen(orbc, dipolmat, i, j):
    tt = np.linspace(-np.pi/4, np.pi/4, 100)
    mu_array = []
    for theta in tt:
        C_tmp = orbc.copy()
        c, s = np.cos(theta), np.sin(theta)
        C_tmp[:, i] =  c * orbc[:, i] + s * orbc[:, j]
        C_tmp[:, j] = -s * orbc[:, i] + c * orbc[:, j]
        mu_array.append(np.sum(np.einsum('pi,qi,pql->il', C_tmp, C_tmp, dipolmat)**2))
    return mu_array

def jacobi_sweep_with_optimizer(orbc, dipolmat, nocc):
    """
    Perform a single Jacobi sweep over occupied orbitals to maximize
    the squared dipole moment norm via pairwise rotations, using an optimizer.
    
    Parameters:
        orbc (ndarray): Molecular orbital coefficient matrix (nbasis, nocc)
        dipolmat (ndarray): Dipole tensor (nbasis, nbasis, 3)
    
    Returns:
        orbc_new (ndarray): Rotated orbital coefficients
        r2_final (float): Final squared dipole norm
    """
    orbc_new = orbc.copy()
    screennarr = []

    def compute_r2(C_occ_local):
        dipole_est = np.einsum('pi,qi,pql->il', C_occ_local, C_occ_local, dipolmat)
        return np.sum(dipole_est ** 2)

    for i in range(nocc):
        for j in range(i + 1, nocc):
            
             # store mu(theta) relation for each (i,j) pair
            screennarr.append(screen(orbc, dipolmat, i, j))

            def cost_fn(alpha):
                """Cost function: negative R2 after rotating orbitals i and j by angle alpha."""
                c, s = np.cos(alpha), np.sin(alpha)
                C_tmp = orbc_new.copy()
                C_tmp[:, i] =  c * orbc_new[:, i] + s * orbc_new[:, j]
                C_tmp[:, j] = -s * orbc_new[:, i] + c * orbc_new[:, j]
                return -compute_r2(C_tmp)

            # Optimize rotation angle alpha
            res = scipy.optimize.minimize_scalar(
                cost_fn,
                bounds=(-np.pi/4, np.pi/4),
                method='bounded',
                options={'xatol': 1e-12}
            )

            alpha_opt = res.x
            c, s = np.cos(alpha_opt), np.sin(alpha_opt)

            # Apply optimal rotation
            C_rot_i =  c * orbc_new[:, i] + s * orbc_new[:, j]
            C_rot_j = -s * orbc_new[:, i] + c * orbc_new[:, j]
            orbc_new[:, i] = C_rot_i
            orbc_new[:, j] = C_rot_j

    # Final <r2> after full sweep
    r2_final = compute_r2(orbc_new)
    print(f'Jacobi sweep (with optimizer): r² = {r2_final:.10f}')

    return orbc_new, r2_final, screennarr