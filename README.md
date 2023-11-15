# OxitationStatesMetalloprotein
This script uses Machine Learning model (Decision Tree) trained on smalled molecules database collected
from Cambridge Crystallographic Data Center (CCDC) to predict the oxidation states of metals in Proteins
The script can be used as following:

python getOxidation.py 1jb0.pdb Fe Fe.csv

where:
1jb0.pdb is an example PDB file that contains Fe metal centers
Fe is the type of the metals (the script can work with Fe, Mn, Co and Cu)
Fe.csv is the database of the small Fe compounds
