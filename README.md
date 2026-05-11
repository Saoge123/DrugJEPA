## Overview of DrugJEPA

Ultra-large accessible chemical spaces (~100 billion molecules) offer unprecedented opportunities for discovering novel chemotypes in drug development, yet conventional screening remains prohibitively costly and inaccessible to most drug developers. Here we present DrugJEPA, a deep learning framework integrating contrastive learning, Joint Embedding Predictive Architecture (JEPA) and Mixture-of-Experts (MoE), achieving state-of-the-art computational benchmark accuracy. We built the ZEUS-3D virtual library with 100-billion-scalle compounds; DrugJEPA completed its full screening in 28 hours on 4 GPUs, a 5 to 7 orders-of-magnitude speedup over standard molecular docking. Wet-lab validation on three distinct targets (kinase TYK2, GPCR TAAR1, epigenetic PUS1) identified nanomolar-active compounds with high hit rates (40%, 20%, 30%). This pioneering work enables affordable ultrafast virtual screening against ultra-large chemical libraries for routine drug development.

## Dependencies

- Python 3.8+
- PyTorch 1.10+
- RDKit 2022+
- NumPy, SciPy, scikit-learn
- Unicore (internal framework)


## Training

```bash
# Training command example
sh train_moe.sh
```

## Testing

Use the `test_moe.sh` script for testing:

```bash
sh test_moe.sh DUDE drug_jepa_moe alternating ./ckpt/checkpoint_last.pt 0
```

## Core Modules

### Model Tasks (`unimol/tasks/`)
- `drugjepa_moe_train_task.py`: MoE training task
- `test_task.py`: Test task

### Utility Scripts
- `calc_mol_embedding.py`: Calculate molecular embeddings
- `calc_pkt_embedding.py`: Calculate protein pocket embeddings

## Pretrained Models
Pretrained models are from the Unimol repository, located in `ckpt/unimol_pretrained/`:
- `mol_pre_no_h_220816.pt`: Molecular pretrained model, https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/mol_pre_no_h_220816.pt
- `pocket_pre_220816.pt`: Pocket pretrained model, https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/pocket_pre_220816.pt


## License

This project is released under the MIT License. See the LICENSE file for details.
