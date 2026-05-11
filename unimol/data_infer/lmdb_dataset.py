# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import pickle
from functools import lru_cache
import logging
import numpy as np

logger = logging.getLogger(__name__)


class LMDBDataset:
    def __init__(self, db_path):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        #datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        #print(idx)
        datapoint_pickled = self.env.begin().get(f"{idx}".encode("ascii"))
        data = pickle.loads(datapoint_pickled)
        return data


from rdkit import Chem

# 获取周期表对象
ptable = Chem.GetPeriodicTable()
class PKLDataset:
    def __init__(self, db_path, atom_key='atoms', coord_key='coordinates', transform=None):
        self.db_path = db_path
        self.transform = transform
        self.ptable = Chem.GetPeriodicTable()
        self.atom_key = atom_key
        self.coord_key = coord_key
        """assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        with open(db_path, 'rb') as frb:
            self.dataset = pickle.load(frb)"""
        if isinstance(self.db_path, (list)):
            self.dataset = self.db_path
            self._keys = list(range(len(self.dataset)))
        elif isinstance(self.db_path, (dict)):
            self.dataset = self.db_path
            self._keys = list(self.dataset.keys())
        else:
            with open(self.db_path, 'rb') as frb:
                self.dataset = pickle.load(frb)
            if isinstance(self.dataset, dict):
                self._keys = list(self.dataset.keys())
            elif isinstance(self.dataset, list):
                self._keys = list(range(len(self.dataset)))
            else:
                raise ValueError(f"Unknown dataset type: {type(self.dataset)}, .pkl dataset support dict and list type")

    def __getitem__(self, index):
        data = self.dataset[self._keys[index]]
        new_data = {}
        if isinstance(data[self.atom_key], list):
            new_data['atoms'] = data[self.atom_key]
            new_data['id'] = data['mol_id']
        elif isinstance(data[self.atom_key].dtype, np.dtypes.StrDType):
            new_data['atoms'] = data[self.atom_key]
            new_data['id'] = self._keys[index]
        elif isinstance(data[self.atom_key], np.ndarray):
            new_data['atoms'] = [
                self.ptable.GetElementSymbol(int(i)+1) for i in data[self.atom_key][:,0]
                ]
            new_data['id'] = self._keys[index]
        else:
            raise ValueError(f"Unknown atoms type: {type(data[self.atom_key])}, .pkl dataset support list or np.ndarray type")
        new_data['coordinates'] = data[self.coord_key]
        new_data['smi'] = data['smi']
        if self.transform:
            new_data = self.transform(new_data)
        return new_data
    
    def __len__(self):
        return len(self._keys)