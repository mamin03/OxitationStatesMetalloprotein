import Bio.PDB
import sys
import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def cleanDF(df):
    """This function clean the dataframe and replace the name of 
       the atoms with their corresponding atomic numbers.
       Please note that if you to predict oxidation states of
       metals bound to atoms except O, N, S, then you will have
       to edit this fuction by editing the values list and adding
       to it the new type of the new ligand atoms."""

    values = ['O', 'N', 'S','NE']
    df_clean = df[df.a1.isin(values) & df.a2.isin(values)&
                  df.a3.isin(values) & df.a4.isin(values)&
                  df.a5.isin(values) & df.a6.isin(values)]
    values = ['O', 'N', 'S']
    df_clean = df_clean[df_clean.a1.isin(values) & 
                        df_clean.a2.isin(values) &
                        df_clean.a3.isin(values) & 
                        df_clean.a4.isin(values)]
    df_clean = df_clean.replace("N", 7)
    df_clean = df_clean.replace("O", 8)
    df_clean = df_clean.replace("S", 16)
    df_clean = df_clean.replace("NE", 1)
    df_clean = df_clean.replace(100.000000, 10)
    return df_clean

def getMetals(model, M):
    """This function extract the metal M from the pdb file"""
    metals = []
    for chain in model:
        for res in chain:
            for atom in res:
                if (M in atom.get_name() or 
                    M in atom.element or 
                    M.upper() in atom.get_name() 
                    or M.upper() in atom.element):

                    metals.append(atom)
    return metals

def sortAtoms(metal, atoms):
    """This function sort the ligands by distances.
       Also, if the number of ligands is less than 6,
       it will add dummy ligands to complete the octahedral
       coordination shell"""

    dist = []
    elem = []
    for i in atoms:
        d = metal-i
        if d !=0 and i.element != "H":
            dist.append(d)
            elem.append(i.element)
    n = 6 - len(elem)
    if n > 0:
        for _ in range(n):
            elem.append("NE")
            dist.append(100)
    dist = np.array(dist)
    elem = np.array(elem)
    inds = dist.argsort()
    return list(elem[inds][:6])+list(dist[inds][:6])

# Creating the dataframe that carries the ligands of the metals (each row is a single metal)
df = pd.DataFrame(columns=['ID','a1','a2','a3','a4','a5','a6','d1','d2','d3','d4','d5','d6'])
c = 0
#pass the pdb file
filename = sys.argv[1]
print("Reading the PDB file")
if filename.endswith("pdb"):
        parser = Bio.PDB.PDBParser(QUIET=True) # QUIET=True avoids comments on errors in the pdb.
        structures = parser.get_structure('prot', filename)
        model = structures[0] # 'structures' may contain several proteins in this case only one.
        metals = getMetals(model, sys.argv[2]) # return a list of the metals
        atoms  = Bio.PDB.Selection.unfold_entities(model, 'A') 
        ns = Bio.PDB.NeighborSearch(atoms) 
        for i, target_atom in enumerate(metals):
            close_atoms = ns.search(target_atom.coord, 2.5) # get the atoms within 2.5Å from the metals
            row = sortAtoms(target_atom, close_atoms) # Sort them
            df.loc[i+c] = [filename]+row # Add the metal to the dataframe
        c += 1

df = cleanDF(df) # clean the dataframe (see the function)
# Read the small molecules data file Mn.csv
df_small = pd.read_csv(sys.argv[3])
print("Reading the Small Molecules Database")
df_clean = cleanDF(df_small)
# The machine learning code starts here
y = df_clean["oxd"]
X = df_clean.drop(columns=["ID","oxd"])
# build the model
print("Building the ML model")
from sklearn import tree
clfDT = tree.DecisionTreeClassifier(max_depth=5)
scaler = StandardScaler()
pipeline = Pipeline([("Standard Scaling", scaler),
                    ("SGD Regression", clfDT)])
# Evaluate the model
scores = cross_val_score(pipeline, X, y, cv=10)
print("%0.2f accuracy DT  with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
pipeline.fit(X,y)
df = df.drop(columns=["ID"])
# Predict the oxidation of the metals in the pdb file
print("Predicting the oxidation states")
oxd = pipeline.predict(df)
df["Metal"] = metals
df["Oxidation"] = oxd
# print the results
print(df)


