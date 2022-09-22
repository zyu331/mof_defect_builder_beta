from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import pymatgen.core as mg
from pymatgen.io.cif import CifParser
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.analysis.functional_groups import FunctionalGroupExtractor

# list of SMILES
smiList = ['C']

# Create RDKit molecular objects
mols = [Chem.MolFromSmiles(m) for m in smiList]
cifFile = CifParser('linkers.cif')
structure = cifFile.get_structures()[0]
aa = mg.Molecule.from_file("linkers.cif")
FunctionalGroupExtractor(molecular)
