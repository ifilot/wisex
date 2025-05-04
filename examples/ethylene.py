import os, sys
from pyqint import Molecule
import numpy as np

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

geodesic = Geodesic(localizer.u_opt, nocc=localizer.nocc)
tt = np.linspace(0, 1, 10)

geodesic.build_generator(group='su(n)')
for i,t in enumerate(tt):
    U = geodesic.interpolate(t)
    C = localizer.data['orbc'] @ U

    path = os.path.join(os.path.dirname(__file__), 'output', 'ethylene', 'path_su_%d.png' % i)
    localizer.produce_contour_plot(C, rows=2, cols=5, figsize=(16, 8), sz=2, save_path=path)

geodesic.build_generator(group='so(n)')
for i,t in enumerate(tt):
    U = geodesic.interpolate(t)
    C = localizer.data['orbc'] @ U

    path = os.path.join(os.path.dirname(__file__), 'output', 'ethylene', 'path_so_%d.png' % i)
    localizer.produce_contour_plot(C, rows=2, cols=5, figsize=(16, 8), sz=2, save_path=path)