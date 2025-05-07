import os, sys
from pyqint import Molecule
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wisex import Localizer, Geodesic

# construct ethylene molecule
mol = Molecule()
mol.add_atom("C", -1.2651, 0.0, 0.0)
mol.add_atom("C",  1.2651, 0.0, 0.0)
mol.add_atom("H", -2.4672, 1.1084, 0.0)
mol.add_atom("H", -2.4672, -1.1084, 0.0)
mol.add_atom("H",  2.4672, 1.1084, 0.0)
mol.add_atom("H",  2.4672, -1.1084, 0.0)

localizer = Localizer(mol, 'sto3g', cachefolder=os.path.join(os.path.dirname(__file__), 'cache', 'ethylene'))
localizer.perform_localization(method='fosterboys')
localizer.report_matrix()
#localizer.show_jacobi_rotations()

outpath = os.path.join(os.path.dirname(__file__), 'output', 'ethylene')
os.makedirs(outpath, exist_ok=True)
geodesic = Geodesic(localizer.u_opt, nocc=localizer.nocc)
tt = np.linspace(0, 1, 20)

# geodesic.build_generator(group='su(n)')
# for i,t in enumerate(tt):
#     U = geodesic.interpolate(t)
#     C = localizer.data['orbc'] @ U

#     path = os.path.join(os.path.dirname(__file__), 'output', 'ethylene', 'path_su_%d.png' % i)
#     localizer.produce_contour_plot(C, rows=2, cols=5, figsize=(16, 8), sz=2, save_path=path)

geodesic.build_generator(group='so(n)')
for i,t in enumerate(tt):
    U = geodesic.interpolate(t)
    C = localizer.data['orbc'] @ U

#     path = os.path.join(outpath, 'path_so_%d.png' % i)
#     localizer.produce_contour_plot(C, rows=2, cols=5, figsize=(16, 8), sz=2, save_path=path)

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
plt.show()