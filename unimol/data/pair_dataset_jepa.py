import json
import os.path

import math
from functools import lru_cache

import torch
from unicore.data import UnicoreDataset
import numpy as np
from . import data_utils
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from multiprocessing import Pool
from tqdm import tqdm

def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
    if mol is None:
        return None
    fingerprints_vect = rdFingerprintGenerator.GetCountFPs(
        [mol], fpType=rdFingerprintGenerator.MorganFP
    )[0]
    DataStructs.ConvertToNumpyArray(fingerprints_vect, fp_numpy)
    return fp_numpy

class JEPAPairDataset(UnicoreDataset):
    def __init__(self, args, pocket_dataset, mol_dataset, labels, split, sequence_dataset=None, use_cache=True, cache_dir=None):
        self.args = args
        self.pocket_dataset = pocket_dataset
        self.mol_dataset = mol_dataset
        if isinstance(sequence_dataset, dict):
            self.sequence_dataset = sequence_dataset
        else:
            self.sequence_dataset = {}
        self.labels = labels

        idx2mol_map = f"{args.data}/data_map/mol2idx_map.json"
        self.idx2mol_map = json.load(open(idx2mol_map))

        idx2pkt_map = f"{args.data}/data_map/pkt2idx_map.json"
        self.idx2pkt_map = json.load(open(idx2pkt_map))

        #uniprot_ids = [x['pocket_id'].split('-')[0] for x in labels]
        #self.uniprot_id_dict = {x:i for i,x in enumerate(set(uniprot_ids))}
        self.split = split
        if self.split == "train":
            self.max_lignum = args.max_lignum # default=16
        else:
            self.max_lignum = args.test_max_lignum # default 512

        if self.split == "train":
            trainidxmap = []
            for idx, assay_item in enumerate(self.labels):
                lig_info = assay_item["smi"] # 10
                trainidxmap += [idx]*math.ceil(len(lig_info)/max(self.max_lignum, 32))
            self.trainidxmap = trainidxmap
        self.epoch = 0

    def __len__(self):
        if self.split == "train":
            import os
            world_size = int(os.environ["WORLD_SIZE"])
            div = self.args.batch_size * world_size
            return (len(self.trainidxmap) // div) * div
        else:
            return len(self.labels)

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.pocket_dataset.set_epoch(epoch)
        self.mol_dataset.set_epoch(epoch)
        super().set_epoch(epoch)

    def collater(self, samples):
        ret_pocket = []
        ret_lig = []
        batch_list = []
        uniprot_list = []
        seq_emb_list = []

        if len(samples) == 0:
            return {}
        
        for pocket, ligs, uniprot, seq_emb in samples:
            ret_pocket.append(pocket)
            lignum_old = len(ret_lig)
            ret_lig += ligs
            batch_list.append([lignum_old, len(ret_lig)])
            #uniprot_list.append(self.uniprot_id_dict[uniprot])
            uniprot_list.append(uniprot)
            seq_emb_list.append(seq_emb)

        seq_emb_list = torch.from_numpy(np.stack(seq_emb_list, axis=0)).half()
        ret_pocket = self.pocket_dataset.collater(ret_pocket)
        ret_lig = self.mol_dataset.collater(ret_lig)
        #print(seq_emb_list.shape, seq_emb_list)
        return {"pocket": ret_pocket, "lig": ret_lig, "batch_list": batch_list, 
                "sequence_emb": seq_emb_list, "uniprot_list": uniprot_list}

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if self.split == "train":
            t_idx = self.trainidxmap[idx]
        else:
            t_idx = idx

        uniprot = self.labels[t_idx]["pocket_id"].split('-')[0]
        with data_utils.numpy_seed(2025, idx, self.epoch):
            pocket_name = np.random.choice(self.labels[t_idx]["pkt_conf"], 1, replace=False)[0]

        smiles = self.labels[t_idx]["smi"]
        seq_emb = self.sequence_dataset[uniprot]
        if len(smiles) > self.max_lignum:
            with data_utils.numpy_seed(2025, idx, self.epoch):
                smi_sele = np.random.choice(smiles, self.max_lignum, replace=False)
        else:
            smi_sele = smiles

        pocket_idx = self.idx2pkt_map[pocket_name]
        pocket_data = self.pocket_dataset[pocket_idx]
        
        lig_data = [self.mol_dataset[self.idx2mol_map[smi]] for smi in smi_sele]
        #print(lig_data)
        return pocket_data, lig_data, uniprot, seq_emb