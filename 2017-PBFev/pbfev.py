import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


def PBFev(mol):
    '''returns an array of exit vectors for this mol'''
    # Get murcko SMILES
    murcko = MurckoScaffold.GetScaffoldForMol(mol)

    # Get PBF plane for murcko scaffold only
    confId = -1
    conf = murcko.GetConformer(confId)
    if not conf.Is3D():
        print('This mol is not 3D - all PBFev angles will be 0 degrees')
        return [0]
    pts = np.array([list(conf.GetAtomPosition(i))  # Get atom coordinates
                    for i in xrange(murcko.GetNumAtoms())])
    # GetBestFitPlane is in the RDKit Contrib directory as part of PBF
    # Plane is xyz vector with a c intercept adjustment
    plane = GetBestFitPlane(pts)

    # Map onto parent structure coords (this func adds exit vectors [*])
    murckoEv = Chem.ReplaceSidechains(mol, murcko)

    confId = -1  # embed 3D conf object with EVs (atom indices do not change)
    conf = murckoEv.GetConformer(confId)

    # Where [#0] matches exit vector SMILES [*]
    patt = Chem.MolFromSmarts('[#0]-[*]')
    matches = murckoEv.GetSubstructMatches(patt)
    if len(matches) == 0:
        return None

    # Calculate angles between exit vectors and the murcko plane of best fit
    exitVectors = np.zeros(len(matches))
    denom = np.dot(plane[:3], plane[:3])
    denom = denom**0.5
    for n, match in enumerate(matches):
        evCoords = conf.GetAtomPosition(match[0])
        anchorCoords = conf.GetAtomPosition(match[1])
        v = np.array(((evCoords[0]-anchorCoords[0]),
                      (evCoords[1]-anchorCoords[1]),
                      (evCoords[2]-anchorCoords[2])))
        angle = np.arccos((np.dot(v, plane[:3])) /
                          ((denom)*((np.dot(v, v))**0.5)))
        angle = np.abs(int(90 - np.degrees(angle)))
        exitVectors[n] = angle

    return exitVectors
