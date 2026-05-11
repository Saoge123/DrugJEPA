import os, h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from unimol.dataset_infer import (AppendTokenDataset, Dictionary, EpochShuffleDataset,
                                  FromNumpyDataset, NestedDictionaryDataset,
                                  PrependTokenDataset, RawArrayDataset, LMDBDataset,
                                  RawLabelDataset, RightPadDataset, RightPadDataset2D,
                                  TokenizeDataset, SortDataset, data_utils)
from unimol.data_infer import (AffinityDataset, CroppingPocketDataset,
                               CrossDistanceDataset, DistanceDataset,
                               EdgeTypeDataset, KeyDataset, LengthDataset,
                               NormalizeDataset, NormalizeDockingPoseDataset,
                               PrependAndAppend2DDataset, RemoveHydrogenDataset,
                               RemoveHydrogenPocketDataset, RightPadDatasetCoord,
                               RightPadDatasetCross2D, TTADockingPoseDataset, AffinityTestDataset, AffinityValidDataset,
                               PKLDataset,
                               AffinityMolDataset, AffinityPocketDataset, ResamplingDataset)

import unicore
from unicore import tasks


import numpy as np


def strings_to_csr(strings):
    n = len(strings)
    offset = np.empty(n + 1, dtype=np.int64)

    total = sum(len(s) for s in strings)
    buffer = bytearray(total)

    pos = 0
    offset[0] = 0

    for i, s in enumerate(strings):
        l = len(s)
        buffer[pos:pos + l] = s
        pos += l
        offset[i + 1] = pos

    data = np.frombuffer(buffer, dtype=np.uint8)
    return data, offset


def save_meta_h5(smiles, mol_ids, h5_path):
    
    smiles_data, smiles_offset = strings_to_csr(smiles)
    mol_data, mol_offset = strings_to_csr(mol_ids)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("smiles", data=smiles_data)
        f.create_dataset("smiles_offset", data=smiles_offset)

        f.create_dataset("mol_id", data=mol_data)
        f.create_dataset("mol_id_offset", data=mol_offset)


def load_dictionary(vocab_path):
    dictionary = Dictionary.load(vocab_path)
    dictionary.add_symbol("[MASK]", is_special=True)
    return dictionary


def load_mols_dataset(data_path, dictionary, atoms, coords, **kwargs):
    #

    if data_path.endswith('.pkl'):
        dataset = PKLDataset(data_path, coord_key='coords')
    elif data_path.endswith('.lmdb'):
        dataset = LMDBDataset(data_path)

    try:
        label_dataset = KeyDataset(dataset, "label")
        _ = label_dataset[0]
    except:
        label_dataset = None

    dataset = AffinityMolDataset(
        dataset, 0, atoms, coords, False,
    )

    smi_dataset = KeyDataset(dataset, "smi")
    mol_dataset = KeyDataset(dataset, "mol")
    id_dataset = KeyDataset(dataset, "id")
    if kwargs.get("load_name", False):
        name_dataset = KeyDataset(dataset, "name")

    def PrependAndAppend(dataset, pre_token, app_token):
        dataset = PrependTokenDataset(dataset, pre_token)
        return AppendTokenDataset(dataset, app_token)

    dataset = RemoveHydrogenDataset(dataset, "atoms", "coordinates", True, True)
    apo_dataset = NormalizeDataset(dataset, "coordinates")

    src_dataset = KeyDataset(apo_dataset, "atoms")
    len_dataset = LengthDataset(src_dataset)
    src_dataset = TokenizeDataset(
        src_dataset, dictionary, max_seq_len=512
    )
    coord_dataset = KeyDataset(apo_dataset, "coordinates")
    src_dataset = PrependAndAppend(
        src_dataset, dictionary.bos(), dictionary.eos()
    )
    edge_type = EdgeTypeDataset(src_dataset, len(dictionary))
    coord_dataset = FromNumpyDataset(coord_dataset)
    distance_dataset = DistanceDataset(coord_dataset)
    coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
    distance_dataset = PrependAndAppend2DDataset(distance_dataset, 0.0)

    net_input = {
        "mol_src_tokens": RightPadDataset(src_dataset, pad_idx=dictionary.pad()),
        "mol_src_distance": RightPadDataset2D(distance_dataset, pad_idx=0),
        "mol_src_edge_type": RightPadDataset2D(edge_type, pad_idx=0),
    }

    in_datasets = {
        "net_input": net_input,
        "smi_name": RawArrayDataset(smi_dataset),
        "mol_len": RawArrayDataset(len_dataset),
        "id": RawArrayDataset(id_dataset),
    }

    if label_dataset is not None:
        in_datasets["target"] = RawArrayDataset(label_dataset)
        in_datasets["mol"] = RawArrayDataset(mol_dataset)

    if kwargs.get("load_name", False):
        in_datasets["name"] = name_dataset

    nest_dataset = NestedDictionaryDataset(in_datasets)
    return nest_dataset


def lig_embedding(model, lig_loader, desc=''):
    device = next(model.parameters()).device
    embeddings, smiles, mol_id = [], [], []
    model.eval()

    with torch.inference_mode():
        for batch in tqdm(lig_loader, desc):
            batch = unicore.utils.move_to_cuda(batch, device=device)

            lig_emb = model.mol_forward(**batch["net_input"])

            embeddings.append(lig_emb.detach().cpu())
            smiles += batch['smi_name']
            mol_id += batch['id']

    if len(embeddings) > 0:
        lig_embs = torch.cat(embeddings, dim=0).half()
        return lig_embs, smiles, mol_id
    else:
        return None
    

def mol2embedding(model, pkl_file, save_path, dictionary, batch_size=128, num_workers=4):
    pkl_path, pkl_name = os.path.split(pkl_file)
    save_sub_path = pkl_path.split('/')[-1]
    full_save_dir = os.path.join(save_path, save_sub_path)
    
    os.makedirs(f'{full_save_dir}_emb/', exist_ok=True)
    os.makedirs(f'{full_save_dir}_index/', exist_ok=True)
    os.makedirs(f'{full_save_dir}_meta/', exist_ok=True)

    save_name = pkl_name.replace('.pkl', '')
    final_bin_path = f'{full_save_dir}_emb/{save_name}.bin'
    final_index_path = f'{full_save_dir}_index/{save_name}-index.bin'
    final_meta_path = f'{full_save_dir}_meta/{save_name}.h5'

    if os.path.exists(final_bin_path) and os.path.exists(final_index_path) and os.path.exists(final_meta_path):
        print(f"Skipping {save_name}, already exists.")
        return

    dataset = load_mols_dataset(
        pkl_file,
        dictionary,
        'atoms',
        'coordinates'
    )

    #
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset.collater,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    out = lig_embedding(model, data_loader, desc=f'Processing {pkl_name}')

    if out:
        lig_embs, smiles, mol_id = out

        marker = mol_id[0]
        n = 0
        index = []
        mol_id_unique, smiles_unique = [marker.encode('utf-8')], [smiles[n].encode('utf-8')]
        for ix,i in enumerate(mol_id):
            if i == marker:
                index.append(n)
            else:
                mol_id_unique.append(i.encode('utf-8'))
                smiles_unique.append(smiles[ix].encode('utf-8'))
                n += 1
                marker = i
                index.append(n)

        lig_embs.numpy().tofile(final_bin_path)
        np.array(index, dtype=np.int32).tofile(final_index_path)

        assert len(mol_id_unique) == len(smiles_unique) and max(index)+1 == len(mol_id_unique)
        save_meta_h5(smiles_unique, mol_id_unique, final_meta_path)

    del dataset
    del data_loader

if __name__ == '__main__':
    import random
    import argparse

    parser = argparse.ArgumentParser(description='Computing embeddings from .pkl files')
    parser.add_argument('--ckpt', help='checkpoint of model', default=None)
    parser.add_argument('--save_path', help='save path', default='./')
    parser.add_argument('--device', help='device', default='cuda:0')
    parser.add_argument('--mol_data', type=str, required=True, help='Directory containing .pkl files')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')  
    '''
    --ckpt ./ckpt/checkpoint_best.pt --save_path ./in_house_lib --mol_data ./in_house_lib/in_house_lib.pkl
    '''

    args = parser.parse_args()
    state = torch.load(args.ckpt, weights_only=False)  #
    task = tasks.setup_task(state['args'])

    #state['args'].finetune_mol_model = './ckpt/unimol_pretrained/mol_pre_no_h_220816.pt'
    #state['args'].finetune_pocket_model = './ckpt/unimol_pretrained/pocket_pre_220816.pt'

    model = task.build_model(state['args'])
    model.load_state_dict(state['model'], strict=True)
    model.half().to(args.device)
    model.eval()  #

    #
    vocab_path = './vocab/dict_mol.txt'
    mol_dictionary = load_dictionary(vocab_path)

    import os
    import glob
    import random
    import torch
    import shutil

    lock_root = os.path.join(args.save_path, '_locks')
    os.makedirs(lock_root, exist_ok=True)

    mol_data_file = args.mol_data
    
    mol2embedding(model, mol_data_file, args.save_path, mol_dictionary, batch_size=128, num_workers=0)
