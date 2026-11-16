"""Original pPred fingerprint definition (MACCS, ErG, PubChem)."""
import numpy as np
from rdkit import Chem, DataStructs
from .patterns import smartsPatts, func_1, func_2, func_3, func_4, func_5, func_6, func_7, func_8

def InitKeys(keyList, keyDict):
    """ *Internal Use Only*
    generates SMARTS patterns for the keys, run once
    """
    assert len(keyList) == len(keyDict.keys()), 'length mismatch'
    for key in keyDict.keys():
        patt, count = keyDict[key]
        if patt != '?':
            sma = Chem.MolFromSmarts(patt)
            if not sma:
                print('SMARTS parser error for key #%d: %s' % (key, patt))
            else:
                keyList[key - 1] = sma, count
PubchemKeys = None
def calcPubChemFingerPart1(mol, **kwargs):
    """  Calculate PubChem Fingerprints （1-115; 263-881)
    **Arguments**
     - mol: the molecule to be fingerprinted
     - any extra keyword arguments are ignored
    **Returns**
      a _DataStructs.SparseBitVect_ containing the fingerprint.
    >>> m = Chem.MolFromSmiles('CNO')
    >>> bv = PubChemFingerPart1(m)
    >>> tuple(bv.GetOnBits())
    (24, 68, 69, 71, 93, 94, 102, 124, 131, 139, 151, 158, 160, 161, 164)
    >>> bv = PubChemFingerPart1(Chem.MolFromSmiles('CCC'))
    >>> tuple(bv.GetOnBits())
    (74, 114, 149, 155, 160)
    """
    global PubchemKeys
    if PubchemKeys is None:
        PubchemKeys = [(None, 0)] * len(smartsPatts.keys())
        InitKeys(PubchemKeys, smartsPatts)
    ctor = kwargs.get('ctor', DataStructs.SparseBitVect)
    res = ctor(len(PubchemKeys) + 1)
    for i, (patt, count) in enumerate(PubchemKeys):
        if patt is not None:
            if count == 0:
                res[i + 1] = mol.HasSubstructMatch(patt)
            else:
                matches = mol.GetSubstructMatches(patt)
                if len(matches) > count:
                    res[i + 1] = 1
    return res

def calcPubChemFingerPart2(mol):# 116-263
    """ *Internal Use Only*
    Calculate PubChem Fingerprints （116-263)
    """
    bits=[0]*148
    bits=func_1(mol,bits)[1]
    bits=func_2(mol,bits)[1]
    bits=func_3(mol,bits)[1]
    bits=func_4(mol,bits)[1]
    bits=func_5(mol,bits)[1]
    bits=func_6(mol,bits)[1]
    bits=func_7(mol,bits)[1]
    bits=func_8(mol,bits)

    return bits

def GetPubChemFPs(mol):
    """
    Calculate PubChem Fingerprints
    """
    mol = Chem.AddHs(mol)
    AllBits=[0]*881
    res1=list(calcPubChemFingerPart1(mol).ToBitString())
    for index, item in enumerate(res1[1:116]):
        if item == '1':
            AllBits[index] = 1
    for index2, item2 in enumerate(res1[116:734]):
        if item2 == '1':
            AllBits[index2+115+148] = 1
    res2=calcPubChemFingerPart2(mol)
    for index3, item3 in enumerate(res2):
        if item3==1:
            AllBits[index3+115]=1
    #AllBits = np.array(AllBits, dtype= np.bool)
    AllBits = np.array(AllBits)
    return AllBits
