import os, sys
from pyqint import Molecule
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wisex import Localizer, Geodesic, Plotter

# construct BeH2 molecule
mol = Molecule()
mol.add_atom("Be", 0, 0, 0)
mol.add_atom("H", 1, 0, 0)
mol.add_atom("H", -1, 0, 0)

localizer = Localizer(mol, 'sto3g', cachefolder=os.path.join(os.path.dirname(__file__), 'cache', 'beh2'))
localizer.perform_localization(method='fosterboys')

outpath = os.path.join(os.path.dirname(__file__), 'output', 'beh2')
os.makedirs(outpath, exist_ok=True)
geodesic = Geodesic(localizer.u_opt, nocc=localizer.nocc, group='so(n)')

# plotter = Plotter()
# plotter.plot(nrimages=180,
#              rows=1, 
#              cols=3, figsize=(12, 8), 
#              geodesic=geodesic, 
#              localizer=localizer, 
#              vmin=-0.5,
#              vmax=0.5,
#              sz=5, 
#              dpi=144, 
#              save_path=outpath)