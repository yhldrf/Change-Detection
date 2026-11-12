# Point-cloud change detection: checkpoint release

This package contains the training, sampling, model, loss and evaluation code for the two supplied checkpoints. The original model mathematics and state-dict names are retained. Local paths have been replaced and the required ME-CPT/Pointcept backbone is bundled under `vendor/ME-CPT`. No adjacent source checkout is required for these two architectures.


## Environment

The locally inspected environment uses Python 3.8, PyTorch 1.10.1, torchvision 0.11.2, CUDA 11.3 wheels, spconv-cu113 2.3.6, torch-scatter 2.0.9, torch-cluster 1.6.0 and torch-geometric 2.0.4. This is an observed environment, not a newly tested clean installation. Training and evaluation require a working NVIDIA GPU; the release audit could not access CUDA.

Example installation matching that environment (CUDA extensions must match PyTorch/CUDA):

```bash
conda create -n change-detection python=3.8 -y
conda activate change-detection
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter==2.0.9 torch-cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install spconv-cu113==2.3.6
pip install -r requirements.txt
```


## Data

Datasets are not included. Set `DATA_ROOT` to an absolute dataset path. The loaders build `.pt` caches automatically. Delete the associated caches when changing datasets, splits or preprocessing. Cached files are not portable evidence of a particular split unless their provenance is known.

SLPCCD expects `train.txt`, `val.txt`, `test.txt`, each containing two relative point-cloud paths per line. Point files have two header lines and columns `X Y Z Rf Gf Bf label`. Place files under the corresponding `train/`, `val/`, `test/` directory or the dataset root as resolved by `cd_siamese.py`. `splits/slpccd/` contains the currently available local split lists for reference; the historical logs do not establish that these lists are identical to the original training splits. Original binary changed labels become removed for T0 and added for T1.

Urb3DCD expects:

```text
data_root/1-Lidar05/
  Train/<scene>/pointCloud0.ply
  Train/<scene>/pointCloud1.ply
  Val/<scene>/pointCloud{0,1}.ply
  Test/<scene>/pointCloud{0,1}.ply
```

The binary PLY reader uses xyz, optional RGB, `label_ch` in T1, and `label_mono` in both epochs. Seven change classes: unchanged, new building, demolition, new vegetation, vegetation growth, missing vegetation, mobile objects.



```bash
DATA_ROOT=/path/to/SLPCCD/SHREC2020-CD bash scripts/test_slpccd.sh
DATA_ROOT=/path/to/IEEE_Dataset_V2_Lid05_MS bash scripts/test_urb3dcd.sh
```

`PYTHON` overrides the Python executable; `CHECKPOINT` overrides the evaluated weights. Logs go to `outputs/`. SLPCCD caches go to `cache/`; the Urb3DCD entry point creates `urb3dcd_v2_1-Lidar05_all_*.pt` in this directory.

## Train

```bash
DATA_ROOT=/path/to/SLPCCD/SHREC2020-CD bash scripts/train_slpccd.sh
DATA_ROOT=/path/to/IEEE_Dataset_V2_Lid05_MS bash scripts/train_urb3dcd.sh


