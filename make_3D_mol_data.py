import sys
#sys.path.append('/home/jyy/LongLongCLIP/')
import os
import glob
import lmdb
import pickle
import numpy as np
from tqdm.auto import tqdm
from multiprocessing import Pool


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit import RDLogger


RDLogger.logger().setLevel(RDLogger.ERROR)

allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [True, False],
    'Acceptor': [True, False],
    'Donor': [True, False],
    'Hydrophobe': [True, False],
    'LumpedHydrophobe': [True, False],
    'PosIonizable': [True, False],
    'NegIonizable': [True, False],
    'ZnBinder': [True, False],
    'possible_bond_type_list': [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [True, False],
}


# def safe_index(l, e):
#
#     try:
#         return l.index(e)
#     except:
#         return len(l) - 1

feature_maps = {
    'atomic_num': {num: idx for idx, num in enumerate(allowable_features['possible_atomic_num_list'])},
    'chirality': {tag: idx for idx, tag in enumerate(allowable_features['possible_chirality_list'])},
    'degree': {deg: idx for idx, deg in enumerate(allowable_features['possible_degree_list'])},
    'formal_charge': {fc: idx for idx, fc in enumerate(allowable_features['possible_formal_charge_list'])},
    'numH': {nh: idx for idx, nh in enumerate(allowable_features['possible_numH_list'])},
    'radical_e': {re: idx for idx, re in enumerate(allowable_features['possible_number_radical_e_list'])},
    'hybridization': {hyb: idx for idx, hyb in enumerate(allowable_features['possible_hybridization_list'])},
    'bond_type': {bt: idx for idx, bt in enumerate(allowable_features['possible_bond_type_list'])},
    'bond_stereo': {bs: idx for idx, bs in enumerate(allowable_features['possible_bond_stereo_list'])},
    'is_conjugated': {ic: idx for idx, ic in enumerate(allowable_features['possible_is_conjugated_list'])}
}


SALT_REMOVER = SaltRemover()
def sanitize_mol(mol):
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        mol,  ## if raise error, we can use: Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
        Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY| \
        Chem.SanitizeFlags.SANITIZE_SETCONJUGATION| \
        Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION| \
        Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True
        )
    try:
        mol = SALT_REMOVER.remover.StripMol(mol)
    except:
        try:
            smi = Chem.MolToSmiles(mol)
            smi_ = sorted([(len(i), i) for i in smi.split('.')], key=lambda x:x[0])[-1][-1]
            mol = Chem.MolFromSmiles(smi_, sanitize=False)
            mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                mol,  ## if raise error, we can use: Chem.rdmolops.SanitizeFlags.SANITIZE_FINDRADICALS
                Chem.SanitizeFlags.SANITIZE_FINDRADICALS|\
                Chem.SanitizeFlags.SANITIZE_KEKULIZE|\
                Chem.SanitizeFlags.SANITIZE_SETAROMATICITY| \
                Chem.SanitizeFlags.SANITIZE_SETCONJUGATION| \
                Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION| \
                Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                catchErrors=True
                )
        except:
            print("cannot gen conf", smi)
            return None
    return mol


def gen_conformation(mol, smi, num_conf=10, num_worker=8):
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = -1
        params.useRandomCoords = True
        params.numThreads = 4
        AllChem.EmbedMultipleConfs(
            mol, numConfs=num_conf, params=params
            )
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=num_worker, maxIters=500)
        except:
            print('# Faild with AllChem.MMFFOptimizeMoleculeConfs(mol) ...')
            pass
        mol = Chem.RemoveHs(mol, sanitize=False)
    except:
        print("cannot gen conf", smi)
        return None
    if mol.GetNumConformers() == 0:
        print("cannot gen conf", smi)
        return None
    return mol


def safe_index(feature_type, value):
    return feature_maps[feature_type].get(value, len(feature_maps[feature_type]) - 1)


possible_bond_type_list = ['SINGLE','DOUBLE','TRIPLE','AROMATIC','misc']
def mol2graph(rdmol):
    edge_index, edge_attr = [], []
    for b in rdmol.GetBonds():
        row = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
        col = [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
        edge_index.extend([row, col])
        bond_feat = safe_index('bond_type', str(b.GetBondType()))
        edge_attr.extend([bond_feat, bond_feat])
    return np.array(edge_index).astype(np.int8).T, np.array(edge_attr).astype(np.int8)


def convert_3Dmol_to_data(mol, smi):
    if mol is None:
        return None
    edge_index, edge_attr = mol2graph(mol)
    coords = [
        np.array(mol.GetConformer(i).GetPositions()).astype(np.float32) for i in range(mol.GetNumConformers())
        ]
    atom_types = [a.GetSymbol() for a in mol.GetAtoms()]
    return {'coords':coords, 'atoms':atom_types, 'edge_index':edge_index,'edge_attr':edge_attr, 'smi':smi}


def get_smi_3Dmol(mol):
    smi, name = mol
    mol = sanitize_mol(smi)
    if mol is None:
        return None
    smi = Chem.MolToSmiles(mol)
    mol = gen_conformation(mol, smi, num_conf=NUM_CONF, num_worker=4)
    data_dict = convert_3Dmol_to_data(mol, smi)
    if data_dict:
        data_dict['mol_id'] = name
        return data_dict
    else:
        return None


def process_smiles(csv):
    save_name = os.path.basename(csv).replace('.csv', '')

    mol_list = []
    dedu = set()
    with open(csv) as f:
        for line in f:
            items = line.strip().split('\t')
            smi, name = items[0], items[1]
            mol = Chem.MolFromSmiles(smi,sanitize=False)
            if mol is None:
                continue

            mol = sanitize_mol(mol)
            if mol.GetNumAtoms() > 50:
                continue
            try:
                mol_id = Chem.MolToInchiKey(mol)
            except:
                mol_id = name
            if mol_id not in dedu:
                mol_list.append((mol, name))
                dedu.add(mol_id)

    with Pool(64) as pool:
        data_list = pool.map(get_smi_3Dmol, tqdm(mol_list, save_name))
        data_list = [item for item in data_list if item]
        with open(f'{SAVE_PATH}/{save_name}.pkl', 'wb') as f:
            pickle.dump(data_list, f)

if __name__ == '__main__':
    SAVE_PATH = './in_house_lib'
    NUM_CONF = 5

    csv_file = './in_house_lib/in_house_lib.csv'
    process_smiles(csv_file)


