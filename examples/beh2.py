import os, sys
from pyqint import Molecule

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wisex import Localizer, Geodesic

# construct BeH2 molecule
mol = Molecule()
mol.add_atom("Be", 0, 0, 0)
mol.add_atom("H", 1, 0, 0)
mol.add_atom("H", -1, 0, 0)

localizer = Localizer(mol, 'sto3g', cachefolder=os.path.join(os.path.dirname(__file__), 'cache', 'beh2'))
localizer.perform_localization(method='fosterboys')
localizer.report_matrix()

geodesic = Geodesic(localizer.u_opt, nocc=localizer.nocc)
geodesic.build_generator(group='su(n)')
u1 = geodesic.interpolate(1.0)
geodesic.build_generator(group='so(n)')
u2 = geodesic.interpolate(1.0)

print(u1)

print(u2)