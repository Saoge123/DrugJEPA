import os
import re
import shlex
import gzip
import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from .residues_base import RESIDUES_TOPO, RESIDUES_PROP, RESIDUES_ATOM_PROP, RESIDUES_TOPO_WITH_H
try:
    import pymol
except:
    """print('we can not compute the atoms on the surface of protein, '\
          'because pymol can not be imported')"""
    pass


substitutions = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', '5OW':'LYS', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MK8':'LEU', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

ATOM_TYPE_WITH_HYBIRD = [
    'SP3_C', 'SP2_C', 'SP_C', 'SP3_N', 'SP2_N', 'SP_N', 'SP3_O', 'SP2_O', 'SP3_F', 'SP3_P',
    'SP2_P', 'SP3D_P', 'SP3_S', 'SP2_S', 'SP3D_S', 'SP3D2_S', 'SP3_Cl', 'SP3_Br', 'SP3_I'
    ]
ATOM_MAP = [6, 6, 6, 7, 7, 7, 8, 8, 9, 15, 15, 15, 16, 16, 16, 16, 17, 35, 53]
PT = Chem.GetPeriodicTable()
BACKBONE_SYMBOL = {'N', 'CA', 'C', 'O'}
AMINO_ACID_TYPE = {
    'CYS':0, 'GLY':1, 'ALA':2, 'THR':3, 'LYS':4, 'PRO':5, 'VAL':6, 'SER':7, 'ASN':8, 'LEU':9,
    'GLN':10, 'MET':11, 'ASP':12, 'TRP':13, 'HIS':14, 'GLU':15, 'ARG':16, 'ILE':17, 'PHE':18,
    'TYR':19, 'SEC':20, 'PYL':21
    }
ESM2_VOCAB = {
    '<cls>':0, '<pad>':1, '<eos>':2, '<unk>':3, 'L':4, 'A':5, 'G':6, 'V':7, 'S':8, 'E':9,
    'R':10, 'T':11, 'I':12, 'D':13, 'P':14, 'K':15, 'Q':16, 'N':17, 'F':18, 'Y':19, 'M':20,
    'H':21, 'W':22, 'C':23, 'X':24, 'B':25, 'U':26, 'Z':27, 'O':28, '.':29, '-':30,
    '<null_1>':31, '<mask>':32
    }
NONSTANDARD_RES_MAP = {
                'MSE':'MET', 'CYX':'CYS', 'CYM':'CYS', 'HID':'HIS', 'HIE':'HIS', 'HIP':'HIS', 'ARN':'ARG',
                'ASH':'ASP', 'GLH':'GLU', 'LYN':'LYS'
                }

AMINO_ACID_MAP = {
    'ALA':'A', 'ARG':'R', 'ASN':'N', 'ASP':'D', 'CYS':'C', 'GLN':'Q', 'GLU':'E',
    'GLY':'G', 'HIS':'H', 'ILE':'I', 'LEU':'L', 'LYS':'K', 'MET':'M', 'PHE':'F',
    'PRO':'P', 'SER':'S', 'THR':'T', 'TRP':'W', 'TYR':'Y', 'VAL':'V', 'PCA':'E',
    'HID':'H', 'HIE':'H', 'HIP':'H', 'SEC':'U', 'HYP':'O', 'GLP':'E', 'ASX':'B',
    'GLX':'Z', 'XLE':'J', 'XAA':'X', #'GLH':
    }
# electronegativity
EN = {'H':2.2, 'D':2.2, 'He':0, 'Li':0.98, 'Be':1.57, 'B':2.04, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98, 'Ne':0.,
      'Na':0.93, 'Mg':1.31, 'Al':1.61, 'Si':1.98, 'P':2.19, 'S':2.58, 'Cl':3.16, 'Ar':0., 'K':0.82, 'Ca':1., 
      'Sc':1.36, 'Ti':1.54, 'V':1.63, 'Cr':1.66, 'Mn':1.55, 'Fe':1.83, 'Co':1.88, 'Ni':1.92, 'Cu':1.9, 'Zn':1.65,
      'Ga':1.81, 'Ge':2.01, 'As':2.18, 'Se':2.55, 'Br':2.96, 'I':2.66, 'X':-1.0, 'Yb':1.26}

SS_MAP = {'-1':'-', '0':'L', '1':'H', '2':'S'}
SS_TYPE = ['-', 'L', 'H', 'S']


#from Bio.PDB.MMCIF2Dict import MMCIF2Dict
def get_sequence_from_cif(cif_file, chain='A'):
    mmcif_dict = MMCIF2Dict(cif_file)
    seq_idx = [int(i) for i in mmcif_dict['_pdbx_poly_seq_scheme.pdb_seq_num']]
    seq = [AMINO_ACID_MAP[i] for i in mmcif_dict['_pdbx_poly_seq_scheme.mon_id']]
    chain_info = mmcif_dict['_pdbx_poly_seq_scheme.asym_id']
    l = sorted([i for i in zip(seq_idx, seq, chain_info) if i[-1]==chain], key=lambda x:x[0])
    return ''.join([i[1] for i in l if i[0] > 0])

def coord_norm_inverse(am, eig_m, center_of_mass, rotate_around):
    for r in rotate_around:
        am[..., r] = am[..., r] * -1
    return am @ eig_m.T + center_of_mass

def compute_surface_atoms(pdb_file, residues):
    pymol.cmd.load(pdb_file)
    path, filename = os.path.split(pdb_file)
    if path == '':
        path = './'
    sele = filename.split('.')[0]
    pymol.cmd.remove("({}) and hydro".format(sele))
    name = pymol.util.find_surface_atoms(sele=sele, _self=pymol.cmd)
    save_name = path + '/' + sele + '-surface.pdb'
    pymol.cmd.save(save_name,((name)))
    surf_protein = Protein(save_name, ignore_incomplete_res=False, compute_ss=False)
    surf_res_dict = {r.idx:r for r in surf_protein.get_residues}
    for res in residues:
        res_idx = res.idx
        if res_idx in surf_res_dict:
            for a in surf_res_dict[res_idx].get_heavy_atoms:
                res.atom_dict[a.name].is_surf = True
    os.remove(save_name)
    pymol.cmd.delete(sele)
    pymol.cmd.delete(name)

def get_resi_ss(pdb_file):
    obj_name = pdb_file.split('.')[0].split('/')[-1]
    pymol.cmd.load(pdb_file)
    pymol.cmd.remove("({}) and hydro".format(obj_name))
    pymol.stored.pairs = []
    pymol.cmd.iterate("{} and n. ca".format(obj_name), "stored.pairs.append((chain, resi, ss))")
    ss_dict = {}
    for chain, num, ss_type in pymol.stored.pairs:
        if chain not in ss_dict:
            ss_dict[chain] = {}
            ss_dict[chain][int(num)] = ss_type
        else:
            ss_dict[chain][int(num)] = ss_type
    pymol.cmd.delete(obj_name)
    #pymol.cmd.delete(name)
    return ss_dict

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


class DBREF(object):
    def __init__(self, line) -> None:
        self.id_code = line[7:11].strip()
        self.chain_id = line[12].strip()
        self.sequence_begin = line[14:18].strip()
        self.insert_begin = line[18].strip()
        self.sequence_end = line[20:24].strip()
        self.insert_end = line[24].strip()
        self.database = line[26:32].strip()
        self.db_accession = line[33:41].strip()
        self.db_id_code = line[42:54].strip()
        self.db_sequence_begin = line[55:60].strip()
        self.idbnsBeg = line[60].strip()
        self.db_sequence_end = line[62:67].strip()
        self.dbinsEnd = line[67].strip()

class DBREF1(object):
    def __init__(self, line) -> None:
        self.id_code = line[7:11].strip()
        self.chain_id = line[12].strip()
        self.sequence_begin = line[14:18].strip()
        self.insert_begin = line[18].strip()
        self.sequence_end = line[20:24].strip()
        self.insert_end = line[24].strip()
        self.database = line[26:32].strip()
        self.db_id_code = line[47:67].strip()

class DBREF2(object):
    def __init__(self, line) -> None:
        self.id_code = line[7:11].strip()
        self.chain_id = line[12].strip()
        self.db_accession = line[18:40].strip()
        self.db_sequence_begin = line[45:55].strip()
        self.db_sequence_end = line[57:67].strip()

class EXPDTA(object):
    def __init__(self, line):
        technique = line[10:].strip()
        if ';' in technique:
            self.technique = [i.strip() for i in technique.split(';')]
        else:
            self.technique = [i.strip() for i in technique.split(',')]

class Information(object):
    def __init__(self, dbrefs, expdta=None):
        self.info = self.integrate(dbrefs)
        self.exp_tech = EXPDTA(expdta) if expdta else None
        
    @staticmethod
    def integrate(dbrefs):
        info = {}

        dbref12 = []
        for i in dbrefs:
            if i.startswith('DBREF1') and dbref12 == []:
                dbref12.append(DBREF1(i))
            elif i.startswith('DBREF2') and len(dbref12) == 1:
                dbref12.append(DBREF2(i))
            if len(dbref12) == 2:
                dbref1, dbref2 = dbref12[0], dbref12[1]
                if dbref1.chain_id not in info:
                    info[dbref1.chain_id] = {
                        dbref2.db_accession:{
                            'seq_idx':[(int(dbref1.sequence_begin), int(dbref1.sequence_end))],
                            'name':dbref1.db_id_code,
                            'database':dbref1.database
                            }
                        }
                elif dbref2.db_accession not in info[dbref1.chain_id]:
                    info[dbref1.chain_id][dbref2.db_accession] = {
                            'seq_idx':[(int(dbref1.sequence_begin), int(dbref1.sequence_end))],
                            'name':dbref1.db_id_code,
                            'database':dbref1.database
                            }
                else:
                    info[dbref1.chain_id][dbref2.db_accession]['seq_idx'].append((int(dbref1.sequence_begin), int(dbref1.sequence_end)))
                dbref12 = []
            if i[0:7].strip() == 'DBREF':
                dbref = DBREF(i)
                if dbref.chain_id not in info:
                    info[dbref.chain_id] = {
                        dbref.db_accession:{
                            'seq_idx':[(int(dbref.sequence_begin), int(dbref.sequence_end))],
                            'name':dbref.db_id_code,
                            'database':dbref.database
                            }
                        }
                elif dbref.db_accession not in info[dbref.chain_id]:
                    info[dbref.chain_id][dbref.db_accession] = {
                            'seq_idx':[(int(dbref.sequence_begin), int(dbref.sequence_end))],
                            'name':dbref.db_id_code,
                            'database':dbref.database
                            }
                else:
                    info[dbref.chain_id][dbref.db_accession]['seq_idx'].append((int(dbref.sequence_begin), int(dbref.sequence_end)))
        return info
    
    def __repr__(self):
        chain_num = len(self.info)
        db_id = []
        names = []
        dbs = []
        for k,v in self.info.items():
            db_id += list(v.keys())
            for k1, v1 in v.items():
                names.append(v1['name'])
                dbs.append(v1['database'])
        db_id = tuple(set(db_id))
        names = tuple(set(names))
        dbs = tuple(set(dbs))
        tmp = '{}(NumChain={}, DB={}, DBName={}, DBId={})'
        out = tmp.format(self.__class__.__name__, chain_num, dbs, db_id, names)
        return out

class Atom(object):
    def __init__(self, atom_info, is_mae=False):
        if is_mae:
            # for Schrodinger-2023-1
            self.idx = int(atom_info['atom_index'])  # mae文件原子编号从1开始
            self.name = re.findall(r'\"(.*?)\"', atom_info['s_m_pdb_atom_name'])[0].strip()
            self.res_name = re.findall(r'\"(.*?)\"', atom_info['s_m_pdb_residue_name'])[0].strip()
            self.chain = atom_info['s_m_chain_name'].strip() if 's_m_chain_name' in atom_info else 'A'
            self.res_idx = int(atom_info['i_m_residue_number'])
            self.coord = np.array([float(atom_info['r_m_x_coord'].strip()),
                                   float(atom_info['r_m_y_coord'].strip()),
                                   float(atom_info['r_m_z_coord'].strip())])
            
            occupancy = atom_info['r_m_pdb_occupancy'].strip() if 'r_m_pdb_occupancy' in atom_info else 0.
            self.occupancy = float(occupancy) if occupancy.isdigit() else 1.0
            
            if 'r_m_pdb_tfactor' in atom_info:
                t_factor = atom_info['r_m_pdb_tfactor'].strip()
                self.temperature_factor = float(t_factor) if t_factor.isdigit() else 0.0
            else:
                self.temperature_factor = 0.0
            self.res_abbr = atom_info['s_m_mmod_res'] if 's_m_mmod_res' in atom_info else None

            self.seg_id = None
            self.element = PT.GetElementSymbol(int(atom_info['i_m_atomic_number'].strip())).capitalize()#.upper()
            self.partial_charge = float(atom_info.get('r_m_charge1', 0))
            formal_charge = int(atom_info['i_m_formal_charge'].strip())
            self.formal_charge = int(formal_charge) if formal_charge else 0
            if self.element == 'SE' and self.res_name == 'MSE':
                self.element = 'S'
                self.name = 'SD'
                self.res_name = 'MET'
            self.mass = PT.GetAtomicWeight(self.element)
            self.atomic_num = PT.GetAtomicNumber(self.element)
            self.r_vdw = PT.GetRvdw(self.atomic_num)
            self.elect_neg = EN[self.element]
            self.ss = SS_MAP[atom_info['i_m_secondary_structure'].strip()]
            if self.occupancy < 1.0:
                self.is_disorder = True
            else:
                self.is_disorder = False
            self.is_surf = False
        else:
            self.idx = int(atom_info[6:11])
            self.name = atom_info[12:16].strip()
            self.res_name = atom_info[17:20].strip()  # atom_info[17:20].strip()
            self.chain = atom_info[21:22].strip()
            self.res_idx = int(atom_info[22:26])
            self.coord = np.array([float(atom_info[30:38].strip()),
                                float(atom_info[38:46].strip()),
                                float(atom_info[46:54].strip())])
            if len(atom_info) <= 66:
                self.occupancy = 1.00
                self.partial_charge = self.occupancy
                self.temperature_factor = 1.00
                self.seg_id = ''
                self.ss = None
                self.res_abbr = None
                element = self.name[0]
                formal_charge = 0
            else:
                self.occupancy = float(atom_info[54:60])
                self.partial_charge = self.occupancy
                self.temperature_factor = float(atom_info[60:66].strip())
                self.seg_id = atom_info[72:76].strip()
                self.ss = atom_info[71:72].strip() if atom_info[71:72]!=' ' else None
                self.res_abbr = None
                element = atom_info[76:78].strip().capitalize()
                formal_charge = atom_info[78:80].strip()

            if element == 'D':
                self.element = 'H' 
            else:
                self.element = element

            try:
                self.formal_charge = int(formal_charge[::-1]) if formal_charge else 0
            except:
                self.formal_charge = int(formal_charge) if formal_charge else 0
            if self.element == 'SE' and self.res_name == 'MSE':
                self.element = 'S'
                self.name = 'SD'
                self.res_name = 'MET'
            try:
                self.mass = PT.GetAtomicWeight(self.element)
                self.atomic_num = PT.GetAtomicNumber(self.element)
                self.r_vdw = PT.GetRvdw(self.atomic_num)
            except:
                self.mass = -1.
                self.atomic_num = -1
                self.r_vdw = -1.
            try:
                self.elect_neg = EN[self.element]
            except:
                self.elect_neg = 0
            if self.occupancy < 1.0:
                self.is_disorder = True
            else:
                self.is_disorder = False

            atom_items = atom_info.split()[-1]
            self.is_surf = atom_items.strip() == 'surf'
            #self.ss = 

    @property
    def to_string(self):
        #fmt = '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}'    # https://cupnet.net/pdb-format/
        #fmt = '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}        {:<2s}{:2s}'    # https://cupnet.net/pdb-format/
        fmt = '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}     {:1s}    {:>2s}{:2s}'
        #fmt = '{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f} {:1s}         {:>2s}{:2s}'
        if self.formal_charge == 0:
            formal_charge = '  '
        elif self.formal_charge > 0:
            formal_charge = str(self.formal_charge) + '+'
        else:
            formal_charge = str(self.formal_charge)[::-1]
        if self.ss is None:
            ss = ''
        out = fmt.format('ATOM', self.idx, self.name, '', self.res_name,
                       self.chain, self.res_idx,'', self.coord[0],
                       self.coord[1], self.coord[2], 
                       self.partial_charge if self.partial_charge else self.occupancy,
                       self.temperature_factor, ss,
                       self.element.capitalize(), formal_charge)
        return out
    
    @property
    def to_dict(self):
        if self.element != 'H':
            atom_feat = RESIDUES_ATOM_PROP[self.res_name][self.name]# + [self.r_vdw, self.elect_neg]
        else:
            atom_feat = [0] * len(RESIDUES_ATOM_PROP[self.res_name]['CA'])# + [self.r_vdw, self.elect_neg]
        return {
            'element': PT.GetAtomicNumber(self.element),
            'pos': self.coord,
            'is_backbone': self.name in BACKBONE_SYMBOL,
            'atom_name': self.name,
            'atom_to_aa_type': AMINO_ACID_TYPE[self.res_name],
            'res_idx': self.res_idx,
            'atom_formal_charge':self.formal_charge,
            'atom_partial_charge':self.partial_charge,
            'atom_feat': atom_feat,
            'res_feat_oh':RESIDUES_PROP[self.res_name]['One-Hot'],
            'res_feat':RESIDUES_PROP[self.res_name]['cls']#[v for k,v in RESIDUES_PROP[self.res_name].items() if k!='One-Hot']
             }
    
    def __repr__(self):
        ss = None if 'ss' not in self.__dict__ else self.ss
        info = 'name={}, index={}, res={}, chain={}, ss={}, is_disorder={}'.format(self.name, 
                                                                 self.idx, 
                                                                 self.res_name+str(self.res_idx),
                                                                 self.chain, 
                                                                 ss,
                                                                 self.is_disorder)
        return '{}({})'.format(self.__class__.__name__, info)


class Residue(object):

    def __init__(self, res_info, ss=None, is_mae=False):
        self.res_info = res_info
        #atoms_ = [Atom(i) for i in res_info]
        self.atom_dict = {}
        disorder = []
        for i in res_info:   # 排除disorder原子
            atom = Atom(i, is_mae=is_mae)
            if not is_mae and not atom.ss and not ss:
                if atom.element != 'H':
                    atom.ss = ss
                else:
                    atom.ss = '-'
            if atom.name in self.atom_dict:
                continue
            else:
                self.atom_dict[atom.name] = atom
                disorder.append(atom.is_disorder)
            
            if atom.res_name in NONSTANDARD_RES_MAP:
                atom.res_name = NONSTANDARD_RES_MAP[atom.res_name]

        if True in disorder:
            self.is_disorder = True
        else:
            self.is_disorder = False
            
        self.idx = self.atom_dict[atom.name].res_idx
        self.chain = self.atom_dict[atom.name].chain
        self.name = self.atom_dict[atom.name].res_name
        if self.name not in RESIDUES_TOPO:
            self.is_perfect = False
            
        elif not is_mae:
            self.ss = ss
            if self.get_heavy_atom_names == set(RESIDUES_TOPO[self.name].keys()):
                self.is_perfect = True
            else:
                self.is_perfect = False
        elif is_mae:
            self.ss = self.atom_dict['CA'].ss if 'CA' in self.atom_dict else '-'
            if self.get_heavy_atom_names == set(RESIDUES_TOPO[self.name].keys()):
                self.is_perfect = True
            else:
                self.is_perfect = False

    @property
    def to_heavy_string(self):
        return '\n'.join([a.to_string for a in self.get_heavy_atoms])
    
    @property
    def to_string(self):
        return '\n'.join([a.to_string for a in self.get_atoms])
    
    @property
    def get_coords(self):
        return np.array([a.coord for a in self.get_atoms])

    @property
    def get_atoms(self):
        return list(self.atom_dict.values())
    
    @property
    def is_surf(self):
        n = sum([a.is_surf for a in self.get_atoms])
        if n > 0:
            return True
        else:
            return False
    @property
    def get_heavy_atom_names(self):
        return {a.name for a in self.atom_dict.values() if 'H' not in a.element}

    @property
    def get_heavy_atoms(self):
        return [a for a in self.atom_dict.values() if 'H' not in a.element]
    
    @property
    def get_heavy_coords(self):
        return np.array([a.coord for a in self.get_heavy_atoms])

    @property
    def get_formal_charge(self):
        return sum([a.formal_charge for a in self.atom_dict.values()])

    @property
    def center_of_mass(self):
        atom_mass = np.array([i.mass for i in self.get_atoms]).reshape(-1, 1)
        return np.sum(self.get_coords*atom_mass, axis=0)/atom_mass.sum()
    
    @property
    def center_of_mass_sidechain(self):
        atom_mass = []
        sidechain_coords = []
        for a in self.get_atoms:
            if a.name not in BACKBONE_SYMBOL:
                atom_mass.append(a.mass)
                sidechain_coords.append(a.coord)
        if atom_mass == []:
            return self.center_of_mass
        atom_mass = np.array(atom_mass).reshape(-1, 1)
        sidechain_coords = np.array(sidechain_coords)
        return np.sum(sidechain_coords*atom_mass, axis=0)/atom_mass.sum()
    
    def bond_graph(self, removeHs=True, topo_info=None):
        i, j, bt = [], [], []
        if removeHs:
            res_graph = RESIDUES_TOPO[self.name]
            atom_names = [i.name for i in self.get_heavy_atoms]
        else:
            assert isinstance(topo_info, dict) and self.idx in topo_info, 'topo_info is error'
            res_graph = topo_info[self.idx]
            atom_names = [i.name for i in self.get_atoms]
        for ix, name in enumerate(atom_names):
            for adj in res_graph[name]:
                if adj in atom_names:
                    idx_j = atom_names.index(adj)
                    i.append(ix)
                    j.append(idx_j)
                    bt.append(res_graph[name][adj])
        edge_index = np.stack([i,j]).astype(dtype=np.int64)
        bt = np.array(bt, dtype=np.int64)
        return edge_index, bt
    
    def to_dict(self):
        N_pos, Ca_pos = self.atom_dict['N'].coord, self.atom_dict['CA'].coord
        C_pos = self.atom_dict['C'].coord
        O_pos = self.atom_dict['O'].coord if 'O' in self.atom_dict else self.atom_dict['OXT'].coord
        return {
            'aa_type':AMINO_ACID_TYPE[self.name],
            'res_feature':RESIDUES_PROP[self.name]['cls'],  # or 
            'res_feature_oh':RESIDUES_PROP[self.name]['One-Hot'],
            'formal_charge':self.get_formal_charge,
            'ss':SS_TYPE.index(self.ss),
            'seq_idx':self.idx,
            'is_surf':self.is_surf,
            'bb_pos':[N_pos.tolist(), Ca_pos.tolist(), C_pos.tolist(), O_pos.tolist()],
            'sidechain_mass_center':self.center_of_mass_sidechain,
            'aa_name': self.name
            }

    @property
    def centroid(self):
        return self.get_coords.mean(axis=0)
    
    @property
    def get_prop(self):
        return RESIDUES_PROP[self.name]
    
    def __repr__(self):
        info = 'name={}, index={}, chain={}, ss={}, is_disorder={}, is_perfect={}'.format(
            self.name, self.idx, self.chain, self.ss, self.is_disorder, self.is_perfect
            )
        return '{}({})'.format(self.__class__.__name__, info)


class Chain(object):
    def __init__(self, chain_info, ignore_incomplete_res=True, pdb_file=None, ss_dict=None,
                 topo_info=None, is_mae=False):
        self.pdb_file = pdb_file
        self.topo_info = topo_info
        self.uniprot_id = chain_info['uniprot_id'] if 'uniprot_id' in chain_info else ''
        self.res_dict = {}
        if ss_dict:
            self.residues = {i:Residue(chain_info[i], ss=ss_dict[i], is_mae=is_mae) for i in chain_info if i != 'uniprot_id'}
        else:
            self.residues = {i:Residue(chain_info[i], is_mae=is_mae) for i in chain_info if i != 'uniprot_id'}
        self.chain = list(self.residues.values())[0].chain
        self.ignore_incomplete_res = ignore_incomplete_res
        self.ss_dict = ss_dict
        self.is_fusion = True if len(self.uniprot_id) > 1 else False

    def get_topo(self): 
        chain_topo = {}
        for res in self.get_residues:
            res_topo = {}
            d = {i.idx:i for i in res.get_atoms}
            for a in res.get_atoms:
                res_topo[a.name] = {}
                for a_idx, bt in self.topo_info[a.idx].items():
                    if a_idx in d:
                        res_topo[a.name][d[a_idx].name] = bt
            chain_topo[res.idx] = res_topo
        return chain_topo
    
    def get_sequence(self, without_missing=True):
        '''
        if the residue index in pdb file is accuratly map that in uniprot sequence, 
        the without_missing can be set to False
        '''
        res_items = self.get_residues
        if without_missing:
            formula_list = []
            res_idx_map = {}
            for ix, res in enumerate(res_items):
                res_idx_map[res.name+str(res.idx)] = ix
                formula_list.append(AMINO_ACID_MAP[res.name])
            seq = ''.join(formula_list)
            return seq, res_idx_map
        else:
            formula_list = []
            res_idx_list = []  # the residue idx in sequence of uniprot  
            for ix, res in enumerate(res_items):
                if res.is_perfect:
                    res_idx_list.append(res.idx)
                    formula = AMINO_ACID_MAP[res.name]
                    if ix == 0:
                        formula_list.append((res.idx - 1) * '-' + formula)
                    else:
                        formula_list.append((res.idx - res_items[ix-1].idx - 1) * '-' + formula)
            seq = ''.join(formula_list)
            return seq, res_idx_list

    def esm_input(self, without_missing=True):
        sequence, res_idx_list = self.get_sequence(without_missing=without_missing)
        esm_seq_encoding = [0]+[ESM2_VOCAB[i] for i in sequence]+[2]
        return np.array(esm_seq_encoding, dtype=np.int64)
    
    @property
    def get_incomplete_residues(self):
        return [i for i in self.residues.values() if i.is_perfect==False]

    @property
    def to_heavy_string(self):
        return '\n'.join([res.to_heavy_string for res in self.get_residues])
    
    @property
    def to_string(self):
        return '\n'.join([res.to_string for res in self.get_residues])
    
    @property
    def get_atoms(self):
        atoms = []
        for res in self.get_residues:
            atoms.extend(res.get_atoms)
        return atoms
    
    @property
    def get_residues(self):
        if self.ignore_incomplete_res:
            return [i[1] for i in self.residues.items() if i[1].is_perfect and i[0]>0]
        else:
            return list(self.residues.values())
        
    @property
    def get_heavy_atoms(self):
        atoms = []
        for res in self.get_residues:
            atoms.extend(res.get_heavy_atoms)
        return atoms
    
    @property
    def get_coords(self):
        return np.array([i.coord for i in self.get_atoms])

    @property
    def get_heavy_coords(self):
        return np.array([i.coord for i in self.get_heavy_atoms])

    @property
    def center_of_mass(self):
        atom_mass = np.array([i.mass for i in self.get_atoms]).reshape(-1, 1)
        return np.sum(self.get_coords*atom_mass, axis=0)/atom_mass.sum()

    @property
    def centroid(self):
        return self.get_coords.mean(axis=0)
        
    @property
    def get_atom_formal_charges(self):
        return np.array([a.formal_charge for a in self.get_heavy_atoms])
    
    def bond_graph(self, removeHs=True):
        res_list = self.get_residues
        bond_index = []
        bond_type = []
        N_term_list = []
        C_term_list = []
        cusum = 0
        if removeHs:
            topo_info = None
            atom_name_list = [i.name for i in self.get_heavy_atoms]
        else:
            if self.is_mae:
                topo_info = self.get_topo()
            else:
                topo_info = self.topo_info
            atom_name_list = [i.name for i in self.get_atoms]
        N_term_ix = atom_name_list.index('N')
        C_term_ix = atom_name_list.index('C')
        for ix, res in enumerate(res_list):
            e_idx, e_type = res.bond_graph(removeHs=removeHs, topo_info=topo_info)
            bond_index.append(e_idx + cusum)
            bond_type.append(e_type)
            N_term_ix_ = N_term_ix + cusum
            C_term_ix_ = C_term_ix + cusum
            N_term_list.append(N_term_ix_)
            C_term_list.append(C_term_ix_)
            if removeHs:
                cusum += res.get_heavy_coords.shape[0]
            else:
                cusum += res.get_coords.shape[0]

            if ix != 0:
                if res.idx-res_list[ix-1].idx == 1 and res.chain == res_list[ix-1].chain:
                    bond_idx_between_res = np.array(
                        [[N_term_ix_, C_term_list[ix-1]],[C_term_list[ix-1],N_term_ix_]],
                        dtype=np.int64
                    )
                    bond_index.append(bond_idx_between_res)
                    bond_type_between_res = np.array([1,1], dtype=np.int64)
                    bond_type.append(bond_type_between_res)

        bond_index = np.concatenate(bond_index, axis=1)
        bond_type = np.concatenate(bond_type)
        return bond_index, bond_type
    
    def get_atom_dict(self, removeHs=True, get_surf=False):
        atom_dict = {
            'element': [],
            'pos': [],
            'is_backbone': [],
            'res_idx': [],
            'atom_name': [],
            'atom_to_aa_type': [],
            'atom_formal_charge':[],
            'atom_partial_charge':[],
            'ss':[],
            'atom_feat':[],
            'res_feat_oh':[],
            'res_feat':[]
             }
        for a in self.get_atoms:
            if a.element == 'H' and removeHs:
                continue
            a_prop_dict = a.to_dict
            atom_dict['element'].append(a_prop_dict['element'])
            atom_dict['pos'].append(a_prop_dict['pos'])
            atom_dict['is_backbone'].append(a_prop_dict['is_backbone'])
            atom_dict['res_idx'].append(a_prop_dict['res_idx'])
            atom_dict['atom_name'].append(a_prop_dict['atom_name'])
            atom_dict['atom_to_aa_type'].append(a_prop_dict['atom_to_aa_type'])
            atom_dict['atom_formal_charge'].append(a_prop_dict['atom_formal_charge'])
            atom_dict['atom_partial_charge'].append(a_prop_dict['atom_partial_charge'])
            atom_dict['ss'].append(SS_TYPE.index(a.ss))
            atom_dict['atom_feat'].append(a_prop_dict['atom_feat'])
            atom_dict['res_feat_oh'].append(a_prop_dict['res_feat_oh'])
            atom_dict['res_feat'].append(a_prop_dict['res_feat'])
        atom_dict['element'] = np.array(atom_dict['element'], dtype=np.int64)
        atom_dict['pos'] = np.array(atom_dict['pos'], dtype=np.float32)
        atom_dict['is_backbone'] = np.array(atom_dict['is_backbone'], dtype=bool)
        atom_dict['ss'] = np.array(atom_dict['ss'], dtype=np.int64)
        atom_dict['res_idx'] = np.array(atom_dict['res_idx'], dtype=np.int64)
        atom_dict['atom_formal_charge'] = np.array(atom_dict['atom_formal_charge'], dtype=np.int64)
        atom_dict['atom_name'] = np.array(atom_dict['atom_name']).tolist()
        atom_dict['atom_partial_charge'] = np.array(atom_dict['atom_partial_charge'], dtype=np.float32)
        atom_dict['atom_feat'] = np.array(atom_dict['atom_feat'], np.int64)
        atom_dict['res_feat'] = np.array(atom_dict['res_feat'], np.int64)
        atom_dict['res_feat_oh'] = np.array(atom_dict['res_feat_oh'], np.int64)
        if get_surf:
            atom_dict['surface_mask'] = np.array([a.is_surf for a in self.get_heavy_atoms], dtype=bool)
        
        atom_dict['atom_to_aa_type'] = np.array(atom_dict['atom_to_aa_type'], dtype=np.int64)
        atom_dict['molecule_name'] = None
        protein_bond_index, protein_bond_type = self.bond_graph(removeHs=removeHs)
        atom_dict['bond_index'] = protein_bond_index
        atom_dict['bond_type'] = protein_bond_type
        atom_dict['filename'] = self.pdb_file
        return atom_dict

    def get_backbone_dict(self, removeHs=True):
        atom_dict = self.get_atom_dict(removeHs=removeHs)
        backbone_dict = {}
        backbone_dict['element'] = atom_dict['element'][atom_dict['is_backbone']]
        backbone_dict['pos'] = atom_dict['pos'][atom_dict['is_backbone']]
        backbone_dict['is_backbone'] = np.ones(atom_dict['is_backbone'].sum(), dtype=np.bool)
        backbone_dict['atom_name'] = np.array(atom_dict['atom_name'])[atom_dict['is_backbone']].tolist()
        backbone_dict['atom_to_aa_type'] = atom_dict['atom_to_aa_type'][atom_dict['is_backbone']]
        backbone_dict['ss'] = atom_dict['ss'][atom_dict['is_backbone']]
        backbone_dict['res_feat'] = atom_dict['res_feat'][atom_dict['is_backbone']]
        backbone_dict['res_feat_oh'] = atom_dict['res_feat_oh'][atom_dict['is_backbone']]
        backbone_dict['molecule_name'] = atom_dict['molecule_name']
        atom_dict['bond_index'] = np.empty([0,2], dtype=np.int64)
        atom_dict['bond_type'] = np.empty(0, dtype=np.int64)
        atom_dict['filename'] = self.pdb_file
        return backbone_dict
    
    def get_res_dict(self):
        res_dict = {
            'aa_type':[],
            'res_feature':[],
            'res_feature_oh':[],
            'res_formal_charge':[],
            'ss':[],
            'seq_idx':[],      
            'pdb_seq_idx':[], 
            'bb_pos':[],
            'sidechain_mass_center':[],
            'aa_name':[], 
            'pos_formal_charge':[],
            'formal_charges':[],
             }
        for ix, res in enumerate(self.get_residues):
            res_prop_dict = res.to_dict()
            res_dict['aa_type'].append(res_prop_dict['aa_type'])
            res_dict['res_feature'].append(res_prop_dict['res_feature'])
            res_dict['res_feature_oh'].append(res_prop_dict['res_feature_oh'])
            res_dict['res_formal_charge'].append(res_prop_dict['formal_charge'])
            res_dict['ss'].append(res_prop_dict['ss'])
            res_dict['seq_idx'].append(res_prop_dict['seq_idx'])
            res_dict['pdb_seq_idx'].append(ix)
            res_dict['bb_pos'].append(res_prop_dict['bb_pos'])
            res_dict['sidechain_mass_center'].append(res_prop_dict['sidechain_mass_center'])
            res_dict['aa_name'].append(res_prop_dict['aa_name'])

        res_dict['aa_type'] = np.array(res_dict['aa_type'], dtype=np.int64)
        res_dict['res_feature'] = np.array(res_dict['res_feature'], dtype=np.int64)
        res_dict['res_feature_oh'] = np.array(res_dict['res_feature_oh'], dtype=np.int64)
        res_dict['res_formal_charge'] = np.array(res_dict['res_formal_charge'], dtype=np.int64)
        res_dict['ss'] = np.array(res_dict['ss'], dtype=np.int64)
        res_dict['seq_idx'] = np.array(res_dict['seq_idx'], dtype=np.int64)
        res_dict['bb_pos'] = np.array(res_dict['bb_pos'], dtype=np.float32)
        res_dict['sidechain_mass_center'] = np.array(res_dict['sidechain_mass_center'], dtype=np.float32)
        res_dict['aa_name'] = res_dict['aa_name']
        for a in self.get_formal_charged_atoms:
            res_dict['pos_formal_charge'].append(a.coord)
            res_dict['formal_charges'].append(a.formal_charge)
        res_dict['pos_formal_charge'] = np.array(res_dict['pos_formal_charge'], dtype=np.float32)
        res_dict['formal_charges'] = np.array(res_dict['formal_charges'], dtype=np.int64)
        return res_dict

    @property
    def get_formal_charged_atoms(self):
        return [a for a in self.get_atoms if a.formal_charge != 0]

    @property
    def get_resi_ss(self):
        return [r.ss for r in self.get_residues]
    
    def get_atom_ss(self, hasHs=False):
        if hasHs:
            return [a.ss for a in self.get_atoms]
        else:
            return [a.ss for a in self.get_heavy_atoms]
    
    @property
    def get_backbone(self):
        atoms = []
        for res in self.get_residues:
            bkb = [a for a in res.get_atoms if a.name in BACKBONE_SYMBOL]
            atoms += bkb
        return atoms
    
    def compute_surface_atoms(self):
        path, filename = os.path.split(self.pdb_file)
        if path == '':
            path = '.'
        chain_file_name = filename.split('.')[0] + '_' + self.chain + '.pdb'
        chain_file = path+'/'+chain_file_name
        with open(chain_file, 'w') as fw:
            fw.write(self.to_heavy_string)
        compute_surface_atoms(chain_file, self.get_residues)
        self.has_surf_atom = True
        os.remove(chain_file)

    def get_surf_mask(self):
        if self.has_surf_atom is False:
            self.compute_surface_atoms()
        return np.array([a.is_surf for a in self.get_heavy_atoms], dtype=np.bool)

    def get_res_by_id(self, res_id, chain_id=None):
        return self.residues[res_id]
    
    def centrolize(self, removeHs=True):
        if removeHs:
            return self.get_heavy_coords - self.centroid
        else:
            return self.get_coords - self.centroid

    @staticmethod
    def empty_dict():
        empty_pocket_dict = {}
        empty_pocket_dict = Dict(empty_pocket_dict)
        empty_pocket_dict.element = np.empty(0, dtype=np.int64)
        empty_pocket_dict.pos = np.empty([0,3], dtype=np.float32)
        empty_pocket_dict.is_backbone = np.empty(0, dtype=bool)
        empty_pocket_dict.atom_name = []
        empty_pocket_dict.atom_to_aa_type = np.empty(0, dtype=np.int64)
        empty_pocket_dict.molecule_name = None
        empty_pocket_dict.bond_index = np.empty([2,0], dtype=np.int64)
        empty_pocket_dict.bond_type = np.empty(0, dtype=np.int64)
        empty_pocket_dict.filename = None
        return empty_pocket_dict
    
    def __repr__(self):
        tmp = 'Chain={}, NumResidues={}, NumHeavyAtoms={}, UNP={}'
        info = tmp.format(self.chain, len(self.residues), self.get_heavy_coords.shape[0], self.uniprot_id)
        return '{}({})'.format(self.__class__.__name__, info)


INFO_PATTERN = re.compile(r"\{[^}]+\}", re.DOTALL)
ATOM_INFO_PATTERN = re.compile(r"m_atom\[\d+\] \{[^}]+\}", re.DOTALL)
TOPO_INFO_PATTERN = re.compile(r"m_bond\[\d+\] \{[^}]+\}", re.DOTALL)
PROP_INFO_PATTERN = re.compile(r"f_m_ct \{[^}]+\}", re.DOTALL)
MINIMIZE_INFO_PATTERN = re.compile(r' \" \- \D+\"')

class Protein(Chain):
    def __init__(self, pdb_file, ignore_incomplete_res=True, compute_ss=False, 
                 topo_info=None):
        """
        -> pdb_file:
            .pdb or .mae file
        -> ignore_incomplete_res: 
            whether incomplete residues should be ignored, defaulf=True. If you used 
            Schrodinger's residue repair function when you generated the ".mae" file, 
            this parameter can be set to False when you read the ".mae" file.
        -> compute_ss: 
            whether compute the second structure of each residues by pymol. If you read 
            .pdb file, we recommend setting it to True, but will slow down reading. 
            If you read .mae file, it is forced to be False; because .mae file has 
            contained information of the second structure.
        -> topo_info: 
            topological information (covalent bond) between each atom. 
        """
        self.ignore_incomplete_res = ignore_incomplete_res
        self.name = pdb_file.split('/')[-1].split('.')[0]
        self.pdb_file = pdb_file
        self.compute_ss = compute_ss
        self.has_surf_atom = False
        if os.path.exists(pdb_file):
            if '.gz' in pdb_file:
                with gzip.open(pdb_file, 'rb') as fr_gz:
                    data_context = fr_gz.read().decode()
            else:
                with open(pdb_file) as fr:
                    data_context = fr.read()
        else:
            data_context = pdb_file

        if '.mae' in pdb_file:
            self.is_mae = True
            # parse .mae file only for Schrodinger-2023-1
            self.compute_ss = False
            self.has_surf_atom = False
            mae = data_context
            #mae_list = mae.split(' :::\n')
            prop_info = PROP_INFO_PATTERN.findall(mae)[0]
            self.is_minimized = True if re.findall('prepared', prop_info) else False
            bond_info = TOPO_INFO_PATTERN.findall(mae)[0]
            self.topo_info = self.extrct_topo_info(bond_info)
            atom_info = ATOM_INFO_PATTERN.findall(mae)[0]
            context = atom_info.split('\n  :::\n')
            header = context[0].split('{ \n')[1].split('\n')
            self.header = [i.strip() for i in header]
            self.header[0] = 'atom_index'
            chain_info = {}
            for line in context[1].split('\n  '):
                lex = shlex.shlex(line)
                lex.whitespace=' '
                lex.quotes='"'
                lex.whitespace_split = True
                items=list(lex)
                atom_prop = dict(zip(self.header, items))
                chain = atom_prop['s_m_chain_name'] if 's_m_chain_name' in atom_prop else 'A'
                res_idx = int(atom_prop['i_m_residue_number'])
                
                
                if chain not in chain_info:
                    chain_info[chain] = {}
                    chain_info[chain][res_idx] = [atom_prop]
                elif res_idx not in chain_info[chain]:
                    chain_info[chain][res_idx] = [atom_prop]
                else:
                    chain_info[chain][res_idx].append(atom_prop)
        else:
            lines = data_context.split('\n')
            surf_item = lines[0].strip().split()[-1]
            if surf_item in {'surf', 'inner'}:
                self.has_surf_atom = True
            self.is_mae = False
            chain_info = {}
            dbrefs_list = []
            expdta = None
            if topo_info:
                self.topo_info = eval(open(topo_info).read())
            else:
                self.topo_info = None
            for line in lines:
                if line.startswith('ATOM'):
                    line = line.strip()
                    chain = line[21:22].strip()
                    #res_idx = line[22:27].strip()
                    res_idx = line[22:26].strip()
                    try:
                        res_idx = int(res_idx)
                    except:
                        continue
                    if chain not in chain_info:
                        chain_info[chain] = {}
                        chain_info[chain][res_idx] = [line]
                    elif res_idx not in chain_info[chain]:
                        chain_info[chain][res_idx] = [line]
                    else:
                        chain_info[chain][res_idx].append(line)
                elif line.startswith('DBREF'):
                    dbrefs_list.append(line)
                elif line.startswith('EXPDTA'):
                    expdta = line
            self.info = Information(dbrefs_list, expdta=expdta)
            uniprot_info = {k:list(v.keys())[0] for k,v in self.info.info.items()}
            for c in uniprot_info:
                if c in chain_info:
                    chain_info[c]['uniprot_id'] = uniprot_info[c]
        if self.compute_ss:
            self.ss_dict = get_resi_ss(pdb_file)
            self.chains = {
                c:Chain(chain_info[c], ignore_incomplete_res=ignore_incomplete_res, pdb_file=pdb_file, ss_dict=self.ss_dict[c], is_mae=self.is_mae, topo_info=self.topo_info) for c in chain_info
                }
        else:
            self.chains = {
                c:Chain(chain_info[c], ignore_incomplete_res=ignore_incomplete_res, pdb_file=pdb_file, is_mae=self.is_mae, topo_info=self.topo_info) for c in chain_info
                }

    @staticmethod
    def extrct_topo_info(bond_info):
        topo = {}
        for line in bond_info.split(':::\n')[1].split('\n'):
            line = line.strip()
            if line:
                items = line.split()
                start = int(items[1])
                end = int(items[2])
                bond_type = int(items[3])
                if start not in topo:
                    topo[start] = {}
                    topo[start][end] = bond_type
                else:
                    topo[start][end] = bond_type
                if end not in topo:
                    topo[end] = {}
                    topo[end][start] = bond_type
                else:
                    topo[end][start] = bond_type
        return topo

    def sequence_with_chain(self, chain_id, without_missing=True):
        return self.chains[chain_id].get_sequence(without_missing=without_missing)
    
    def get_sequence(self, without_missing=True):
        sequence = ''
        for chain_id in self.chains:
            seq, _ = self.chains[chain_id].get_sequence(without_missing=without_missing)
            sequence += seq
        return sequence
    
    def esm_input(self, without_missing=True):
        sequence = self.get_sequence(without_missing=without_missing)
        esm_seq_encoding = [0]+[ESM2_VOCAB[i] for i in sequence]+[2]
        return np.array(esm_seq_encoding, dtype=np.int64)

    @property
    def get_incomplete_residues(self):
        res_list = []
        for i in self.chains:
            res_list += self.chains[i].get_incomplete_residues
        return res_list

    @property
    def get_residues(self,):
        res_list = []
        for i in self.chains:
            res_list += self.chains[i].get_residues
        return res_list
    
    def get_chain(self, chain_id):
        return self.chains.get(chain_id)
    
    def uniprot_id(self):
        return {c:self.chains[c].uniprot_id for c in self.chains}
    
    def get_res_by_id(self, res_id, chain_id='A'):
        return self.chains[chain_id].get_res_by_id(res_id)

    def compute_surface_atoms(self):
        """
        If the pdb_file is a pocket_file, the surface atoms is not correct!
        """
        compute_surface_atoms(self.pdb_file, self.get_residues)
        self.has_surf_atom = True

    def __repr__(self):
        num_res = 0
        num_atom = 0
        for i in self.chains:
            res_list = list(self.chains[i].residues.values())
            num_res += len(res_list)
            for i in res_list:
                num_atom += len(i.get_heavy_atoms)
        num_incomp = len(self.get_incomplete_residues)
        tmp = 'Name={}, NumChains={}, NumResidues={}, NumHeavyAtoms={}, NumIncompleteRes={}'
        info = tmp.format(
            self.name, len(self.chains), num_res, num_atom, num_incomp
            )
        return '{}({})'.format(self.__class__.__name__, info)

####################################################################################################################



ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable', 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {t: i for i, t in enumerate(BondType.names.values())}
BOND_NAMES = {i: t for i, t in enumerate(BondType.names.keys())}


def is_in_ring(mol):
    d = {a:np.array([], dtype=np.int64) for a in range(len(mol.GetAtoms()))}
    rings = Chem.GetSymmSSSR(mol)
    for a in d:
        for r_idx, ring in enumerate(rings):
            if a in ring:
                d[a] = np.append(d[a], r_idx+1)
            else:
                d[a] = np.append(d[a], -a)
    return d

def parse_sdf_to_dict(mol_file):
    fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    rdmol = next(iter(Chem.SDMolSupplier(mol_file, removeHs=True)))
    Chem.Kekulize(rdmol)
    ring_info = is_in_ring(rdmol)
    conformer = rdmol.GetConformer()
    feat_mat = np.zeros([rdmol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.int64)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
    
    element, pos, atom_mass = [], [], []
    for a in rdmol.GetAtoms():
        element.append(a.GetAtomicNum())
        pos.append(conformer.GetAtomPosition(a.GetIdx()))
        atom_mass.append(a.GetMass())
    element = np.array(element, dtype=np.int64)
    pos = np.array(pos, dtype=np.float32)
    atom_mass = np.array(atom_mass, np.float32)
    center_of_mass = (pos * atom_mass.reshape(-1,1)).sum(0)/atom_mass.sum()
    
    edge_index, edge_type = [], []
    for b in rdmol.GetBonds():
        row = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
        col = [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
        edge_index.extend([row, col])
        edge_type.extend([BOND_TYPES[b.GetBondType()]] * 2)
    edge_index = np.array(edge_index)
    edge_index_perm = edge_index[:,0].argsort()
    edge_index = edge_index[edge_index_perm].T
    edge_type = np.array(edge_type)[edge_index_perm]
    
    return {'element': element,
            'pos': pos,
            'bond_index': edge_index,
            'bond_type': edge_type,
            'center_of_mass': center_of_mass,
            'atom_feature': feat_mat,
            'ring_info': ring_info,
            'filename':mol_file
           }
"""
"""
class Ligand(object):
    def __init__(self, mol_file, removeHs=True, sanitize=True):
        if isinstance(mol_file, Chem.rdchem.Mol):
            mol = mol_file
            self.name = mol.GetProp('_Name')
            self.lig_file = None
        else:
            mol = Chem.MolFromMolFile(mol_file, removeHs=removeHs, sanitize=sanitize)
            if mol is None:
                mol = Chem.MolFromMolFile(mol_file, removeHs=removeHs, sanitize=False)
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
            self.name = mol_file.split('/')[-1].split('.')[0]
            self.lig_file = mol_file
            
        Chem.Kekulize(mol)
        self.mol = mol
        self.conformer = self.mol.GetConformer()
        self.num_atoms = len(self.mol.GetAtoms())
        self.normalized_coords = None
    
    def mol_block(self):
        return Chem.MolToMolBlock(self.mol)
    
    def to_dict(self):
        fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
        feat_mat = np.zeros([self.mol.GetNumAtoms(), len(ATOM_FAMILIES)], dtype=np.int64)
        for feat in factory.GetFeaturesForMol(self.mol):
            feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1
        
        element, pos, atom_mass = [], [], []
        for a in self.mol.GetAtoms():
            element.append(a.GetAtomicNum())
            pos.append(self.conformer.GetAtomPosition(a.GetIdx()))
            atom_mass.append(a.GetMass())
        element = np.array(element, dtype=np.int64)
        pos = np.array(pos, dtype=np.float32)
        atom_mass = np.array(atom_mass, np.float32)
        center_of_mass = (pos * atom_mass.reshape(-1,1)).sum(0)/atom_mass.sum()
        
        edge_index, edge_type = [], []
        for b in self.mol.GetBonds():
            row = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
            col = [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]
            edge_index.extend([row, col])
            edge_type.extend([BOND_TYPES[b.GetBondType()]] * 2)
        edge_index = np.array(edge_index, dtype=np.int64)
        edge_index_perm = edge_index[:,0].argsort()
        edge_index = edge_index[edge_index_perm].T
        edge_type = np.array(edge_type, dtype=np.int64)[edge_index_perm]
        
        return {'element': element,
                'pos': pos,
                'bond_index': edge_index,
                'bond_type': edge_type,
                'center_of_mass': center_of_mass,
                'atom_feature': feat_mat,
                'filename':self.lig_file
                }

    @property
    def mol_wt(self):
        return Descriptors.MolWt(self.mol)

    @staticmethod
    def empty_dict():
        empty_ligand_dict = {}
        empty_ligand_dict = Dict(empty_ligand_dict)
        empty_ligand_dict.element = np.empty(0, dtype=np.int64)
        empty_ligand_dict.pos = np.empty([0,3], dtype=np.float32)
        empty_ligand_dict.bond_index = np.empty([2,0], dtype=np.int64)
        empty_ligand_dict.bond_type = np.empty(0, dtype=np.int64)
        empty_ligand_dict.center_of_mass = np.empty([0,3], dtype=np.float32)
        empty_ligand_dict.atom_feature = np.empty([0,8], dtype=np.float32)
        empty_ligand_dict.ring_info = {}
        empty_ligand_dict.filename = None
        return empty_ligand_dict

    def __repr__(self):
        tmp = 'Name={}, NumAtoms={}'
        info = tmp.format(self.name, self.num_atoms)
        return '{}({})'.format(self.__class__.__name__, info)
