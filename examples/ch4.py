import os, sys
from pyqint import Molecule, MoleculeBuilder
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from wisex import Localizer, Geodesic, Plotter

# construct BeH2 molecule
mol = MoleculeBuilder().from_name("ch4")

localizer = Localizer(mol, 'p321g', cachefolder=os.path.join(os.path.dirname(__file__), 'cache', 'ch4'))
localizer.perform_localization(method='fosterboys')

outpath = os.path.join(os.path.dirname(__file__), 'output', 'ch4')
os.makedirs(outpath, exist_ok=True)
geodesic = Geodesic(localizer.u_opt, nocc=localizer.nocc, group='so(n)')

plotter = Plotter()
plotter.plot(nrimages=10,
             rows=2, 
             cols=4, figsize=(12, 8), 
             geodesic=geodesic, 
             localizer=localizer, 
             vmin=-0.5,
             vmax=0.5,
             sz=5, 
             dpi=144, 
             save_path=outpath)