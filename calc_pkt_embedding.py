import gc
import os
import glob, h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from easydict import EasyDict
from unimol.dataset import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                                  FromNumpyDataset, NestedDictionaryDataset,
                                  PrependTokenDataset, RawArrayDataset, LMDBDataset,
                                  RawLabelDataset, RightPadDataset, RightPadDataset2D,
                                  TokenizeDataset, SortDataset, data_utils)
from unimol.data import (AffinityDataset, CroppingPocketDataset,
                               CrossDistanceDataset, DistanceDataset,
                               EdgeTypeDataset, KeyDataset, LengthDataset,
                               NormalizeDataset, NormalizeDockingPoseDataset,
                               PrependAndAppend2DDataset, RemoveHydrogenDataset,
                               RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                               RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset,
                               PKLDataset,
                               AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)

import unicore
from unicore import tasks, checkpoint_utils
from unimol.process_raw.protein.parse_pdb import Protein



def load_dictionary(vocab_path):
    dictionary = Dictionary.load(vocab_path)
    dictionary.add_symbol("[MASK]", is_special=True)
    return dictionary


def get_pocket_data(pkt_pdb):
    pocket = Protein(pkt_pdb, ignore_incomplete_res=False, compute_ss=False)
    pkt_atoms = [a.element for a in pocket.get_heavy_atoms]
    pkt_coords = np.array([a.coord for a in pocket.get_heavy_atoms], dtype=np.float32)
    pkt_resi = [f'{r.name}{r.idx}' for r in pocket.get_residues]
    return {'pkt_atoms': pkt_atoms, 'pkt_coords': pkt_coords, 'pkt_resi': pkt_resi, 'pkt_file_name': pkt_pdb}


def load_pockets_dataset(data_path, atoms, coords, vocab, **kwargs):
    dictionary = Dictionary.load(vocab)
    dictionary.add_symbol("[MASK]", is_special=True)
    #dataset = LMDBDataset(data_path)
    dataset = PKLDataset(data_path)
    dataset = AffinityPocketDataset(
        dataset,
        1,
        atoms,
        coords,
        False,
        "pkt_file_name"
    )
    

    def PrependAndAppend(dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)

    dataset = RemoveHydrogenPocketDataset(
        dataset,
        "pocket_atoms",
        "pocket_coordinates",
        True,
        True,
    )
    dataset = CroppingPocketDataset(
        dataset,
        0,
        "pocket_atoms",
        "pocket_coordinates",
        256, #2048
    )

    apo_dataset = NormalizeDataset(dataset, "pocket_coordinates")

    src_pocket_dataset = KeyDataset(apo_dataset, "pocket_atoms")
    len_dataset = LengthDataset(src_pocket_dataset)
    src_pocket_dataset = TokenizeDataset(
        src_pocket_dataset,
        dictionary,
        max_seq_len=1024,
    )
    coord_pocket_dataset = KeyDataset(apo_dataset, "pocket_coordinates")
    src_pocket_dataset = PrependAndAppend(
        src_pocket_dataset,
        dictionary.bos(),
        dictionary.eos(),
    )
    pocket_edge_type = EdgeTypeDataset(
        src_pocket_dataset, len(dictionary)
    )
    coord_pocket_dataset = FromNumpyDataset(coord_pocket_dataset)
    distance_pocket_dataset = DistanceDataset(coord_pocket_dataset)
    coord_pocket_dataset = PrependAndAppend(coord_pocket_dataset, 0.0, 0.0)
    distance_pocket_dataset = PrependAndAppend2DDataset(
        distance_pocket_dataset, 0.0
    )

    nest_dataset = NestedDictionaryDataset(
        {
            "net_input": {
                "pocket_src_tokens": RightPadDataset(
                    src_pocket_dataset,
                    pad_idx=dictionary.pad(),
                ),
                "pocket_src_distance": RightPadDataset2D(
                    distance_pocket_dataset,
                    pad_idx=0,
                ),
                "pocket_src_edge_type": RightPadDataset2D(
                    pocket_edge_type,
                    pad_idx=0,
                ),
                "pocket_src_coord": RightPadDatasetCoord(
                    coord_pocket_dataset,
                    pad_idx=0,
                ),
            },
            "pocket_len": RawArrayDataset(len_dataset),
        },
    )
    return nest_dataset


def pkt_embedding(model, pkt_loader):
    model.half()
    device = next(model.parameters()).device
    pkt_emb_list = []
    with torch.no_grad():
        model.eval()
        for pkt_batch in pkt_loader:
            pkt_batch = unicore.utils.move_to_cuda(pkt_batch)
            pkt_emb = model.pocket_forward(**pkt_batch["net_input"])
            pkt_emb_list.append(pkt_emb.detach().cpu().numpy())
    pkt_embs = np.concatenate(pkt_emb_list, axis=0)
    return pkt_embs




pocket_path = './pocket_files/TRUB1/pockets'
ckpt_path = './ckpt/checkpoint_best.pt'
device = 'cuda:0'


vocab_path = './vocab/dict_pkt.txt'
pkt_dictionary = load_dictionary(vocab_path)

pdb_list = sorted(glob.glob(f'{pocket_path}/*.pdb'))
pkt_data_list, pkt_name_list = [], []
for pkt_pdb in tqdm(pdb_list):
    pkt_data = get_pocket_data(pkt_pdb)
    pkt_data_list.append(pkt_data)
    pkt_name_list.append(pkt_data['pkt_file_name'])

pkt_dataset = load_pockets_dataset(
    pkt_data_list, 
    'pkt_atoms', 
    'pkt_coords',
    vocab_path,
    )
pkt_loader = DataLoader(
    pkt_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=pkt_dataset.collater
    )

state = torch.load(ckpt_path, weights_only=False)  #
task = tasks.setup_task(state['args'])

state['args'].finetune_mol_model = './ckpt/unimol_pretrained/mol_pre_no_h_220816.pt'
state['args'].finetune_pocket_model = './ckpt/unimol_pretrained/pocket_pre_220816.pt'

model = task.build_model(state['args'])
model.load_state_dict(state['model'], strict=True)
model.half().to(device)
model.eval()  #

device = next(model.parameters()).device
pkt_emb_list = []
with torch.no_grad():
    model.eval()
    for pkt_batch in pkt_loader:
        pkt_batch = unicore.utils.move_to_cuda(pkt_batch)
        pkt_emb = model.pocket_forward(
            pkt_batch["net_input"]['pocket_src_tokens'],
            pkt_batch["net_input"]['pocket_src_distance'],
            pkt_batch["net_input"]['pocket_src_edge_type'],
            )
        pkt_emb_list.append(pkt_emb.detach().cpu())
torch.save(
    {'embeddings': torch.cat(pkt_emb_list, dim=0), 
        'pkt_name': pkt_name_list}, 
    f'{os.path.split(pkt_pdb)[0]}/pkt_emb.pt'
    )