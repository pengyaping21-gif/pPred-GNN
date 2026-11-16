# pPred Usage Guide

## Running the Models

### Get the Code

```bash
git clone https://github.com/pengyaping21-gif/pPred-GNN.git
cd pPred-GNN
```

### Download the Checkpoints

Before the first run, download the checkpoint archive from the [v0.2.0 Release](https://github.com/pengyaping21-gif/pPred-GNN/releases/tag/v0.2.0) and extract it from the project root:

```bash
mkdir -p .cache/checkpoints
curl -fL --retry 3 \
  https://github.com/pengyaping21-gif/pPred-GNN/releases/download/v0.2.0/ppred-checkpoints-v0.2.0.tar.gz \
  -o .cache/checkpoints/ppred-checkpoints-v0.2.0.tar.gz
curl -fL --retry 3 \
  https://github.com/pengyaping21-gif/pPred-GNN/releases/download/v0.2.0/ppred-checkpoints-v0.2.0.tar.gz.sha256 \
  -o .cache/checkpoints/ppred-checkpoints-v0.2.0.tar.gz.sha256
(cd .cache/checkpoints && sha256sum -c ppred-checkpoints-v0.2.0.tar.gz.sha256)
tar -xzf .cache/checkpoints/ppred-checkpoints-v0.2.0.tar.gz
```

The three weights will be available at `checkpoints/single_rt.pth`, `checkpoints/crosslink_rt.pth`, and `checkpoints/ms2.pth`.

### Run with Docker (Recommended)

Clone the repository on a Linux x86_64 host, enter its root directory, and download the checkpoints above first. The complete `torch1` image includes Python 3.8, PyTorch 2.4.0 + CUDA 12.1, PyG 2.6.1, RDKit 2024.3.5, and the remaining dependencies from the original environment. GPU execution requires an NVIDIA host driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Download the complete image from the [torch1 runtime Release](https://github.com/pengyaping21-gif/pPred-GNN/releases/tag/torch1-runtime-20261116). The script verifies and joins the parts:

```bash
bash scripts/download_runtime.sh docker
docker load -i .cache/runtime/ppred-torch1-linux-amd64.docker.tar

# Enter the container; /app and all outputs map to the host project directory.
docker run --rm -it --gpus all \
  --user "$(id -u):$(id -g)" \
  -v "$PWD:/app" -w /app ppred-gnn:torch1 bash

# Run inside the container.
python scripts/verify_bundle.py
```

For CPU execution, remove `--gpus all` and use `--device cpu` below. Checkpoints, data, and `outputs/` persist in the host project directory after the container exits. Run the download script with Bash and allow at least 25 GB for downloading, joining, and importing the image.

To rebuild the image with the current source, run from the host project root:

```bash
bash scripts/download_runtime.sh conda
docker buildx build --platform linux/amd64 \
  --build-context torch1_snapshot=.cache/runtime \
  --load -t ppred-gnn:torch1 .
```

### Run with Conda (Optional)

If the environment is already installed, use `conda activate XXX/torch1`. To restore the complete backup on a Linux x86_64 host, run from the project root:

```bash
bash scripts/download_runtime.sh conda
mkdir -p .cache/envs/torch1
tar -xzf .cache/runtime/torch1-conda-linux-amd64.tar.gz -C .cache/envs/torch1
.cache/envs/torch1/bin/python .cache/envs/torch1/bin/conda-unpack
source .cache/envs/torch1/bin/activate
python scripts/verify_bundle.py
```

Keep the restored environment at the same path after running `conda-unpack`; extract the archive again to use a different path. Run the remaining commands from the project root in the active environment. Use `--device cpu` if no GPU is available.

### Run the Demo

```bash
# All three tasks: evaluate → finetune → test → predict
python scripts/run_demo.py --task all --device cuda:0 --epochs 1

# Run a single task
python scripts/run_demo.py --task single_rt --device cuda:0 --epochs 1
python scripts/run_demo.py --task crosslink_rt --device cuda:0 --epochs 1
python scripts/run_demo.py --task ms2 --device cuda:0 --epochs 1
```

Results are saved to `outputs/demo_<timestamp>/`. The finetuned checkpoint is saved as `finetune/best.pth` within each task directory.

### Finetune

```bash
# Single-peptide RT
python -m ppred finetune --task single_rt \
  --checkpoint checkpoints/single_rt.pth \
  --train demo/single_rt/train.rtdata \
  --val demo/single_rt/val.rtdata \
  --test demo/single_rt/test.rtdata \
  --epochs 3 --lr 1e-5 --batch-size 8 \
  --device cuda:0 --output outputs/single_rt

# Crosslinked-peptide RT
python -m ppred finetune --task crosslink_rt \
  --checkpoint checkpoints/crosslink_rt.pth \
  --train demo/crosslink_rt/train.rtdata \
  --val demo/crosslink_rt/val.rtdata \
  --test demo/crosslink_rt/test.rtdata \
  --epochs 3 --lr 1e-5 --batch-size 8 \
  --device cuda:0 --output outputs/crosslink_rt

# Single-peptide MS2
python -m ppred finetune --task ms2 \
  --checkpoint checkpoints/ms2.pth \
  --train demo/ms2/train.mgf \
  --val demo/ms2/val.mgf \
  --test demo/ms2/test.mgf \
  --epochs 3 --lr 1e-5 --batch-size 8 \
  --device cuda:0 --output outputs/ms2
```

To use your own data, replace the paths passed to `--train`, `--val`, and `--test`. The `--test` argument is optional. Peptide backbones must not overlap between the training, validation, and test splits. Variants of the same sequence with different modifications, charge states, or NCE values must stay in the same split. Swapping the two chains of a crosslinked peptide also counts as the same group.

After finetuning, use `<output>/best.pth`. For RT finetuning on the same task, the checkpoint's normalization scale is retained by default. When adapting to a new experimental domain, add `--rt-normalization train` to fit the scale using the new training set.

### Evaluate

```bash
python -m ppred evaluate --task single_rt \
  --checkpoint outputs/single_rt/best.pth \
  --input demo/single_rt/test.rtdata \
  --device cuda:0 --output outputs/single_rt/evaluation.json

python -m ppred evaluate --task crosslink_rt \
  --checkpoint outputs/crosslink_rt/best.pth \
  --input demo/crosslink_rt/test.rtdata \
  --device cuda:0 --output outputs/crosslink_rt/evaluation.json

python -m ppred evaluate --task ms2 \
  --checkpoint outputs/ms2/best.pth \
  --input demo/ms2/test.mgf \
  --device cuda:0 --output outputs/ms2/evaluation.json
```

### Predict

```bash
python -m ppred predict --task single_rt \
  --checkpoint checkpoints/single_rt.pth \
  --input demo/single_rt/test.rtdata \
  --device cuda:0 --output outputs/single_rt_predictions.csv

python -m ppred predict --task crosslink_rt \
  --checkpoint checkpoints/crosslink_rt.pth \
  --input demo/crosslink_rt/test.rtdata \
  --device cuda:0 --output outputs/crosslink_rt_predictions.csv

python -m ppred predict --task ms2 \
  --checkpoint checkpoints/ms2.pth \
  --input demo/ms2/test.mgf \
  --device cuda:0 --output outputs/ms2_predictions.jsonl
```

To use a finetuned model, set `--checkpoint` to the corresponding `outputs/<task>/best.pth`. RT predictions are saved as CSV files, and MS2 predictions as JSONL files. Add `--overwrite` to replace existing results.

## Dataset Input Formats

### Single-Peptide RT

Use a tab-separated `.rtdata` file with column names in the first row:

```text
seq	mod	rt
ACDK	2,Carbamidomethyl[C];	45.2
PEPTIDEK		62.8
```

- `seq`: A sequence containing only the 20 standard amino acids, up to 60 residues long.
- `mod`: Modifications, such as `2,Carbamidomethyl[C];`. Residue positions are 1-based; position 0 denotes the N-terminus. Separate multiple modifications with semicolons. Leave this field empty for an unmodified peptide.
- `rt`: A numeric RT label. Use consistent units within a dataset. This column can be omitted for prediction.

### Crosslinked-Peptide RT

Use a tab-separated `.rtdata` file:

```text
seq	mod	linker	rt
AAKAA(3)-ACKAA(3)	10,Carbamidomethyl[C];	DSSO	75.3
AAKAA(3)-AAKAA(3)		DSSO	68.1
```

- `seq`: Use `ALPHA(link_position)-BETA(link_position)`, with 1-based link positions. The combined length of both chains must not exceed 60 residues.
- `mod`: Supports `position,Name;` or `Name(position);`. Leave empty if there are no modifications. For the alpha chain, use the position within that chain. For the beta chain, use `len(alpha) + 3 + position_within_beta`. In the example above, the modification on the second residue, C, of the beta chain has position 10.
- `linker`: Use `DSSO`, with K-K crosslinking sites.
- `rt`: A numeric RT label. This column can be omitted for prediction.

Modification names must match the [modification definitions](ppred/resources/modification_smiles_new.ini), and modification positions must match the corresponding residues.

### Single-Peptide MS2

Use an MGF file. Each spectrum starts with `BEGIN IONS` and ends with `END IONS`. The following example illustrates the field format:

```text
BEGIN IONS
TITLE=example_1
SEQ=ACDK@2,Carbamidomethyl[C];@2@27
CHARGE=2+
NCE=27
INSTRUMENT_NAME=QE
ACTIVATION_TYPE=HCD
INSTRUMENT_TYPE=FT
100.0 25.0 b1-z1
200.0 50.0 y2-z2
250.0 100.0 Unknown
END IONS
```

| Field | Format or allowed values |
| --- | --- |
| `SEQ` | `sequence@modifications@charge@NCE`; leave the modification segment empty if unmodified; peptide length: 2–60 |
| `CHARGE` | Integer charge, optionally followed by `+`; range: 1–10 |
| `NCE` | Normalized collision energy; range: 0–100 |
| `INSTRUMENT_NAME` | `QE`, `Velos`, `Elite`, `Fusion`, or `Lumos` |
| `ACTIVATION_TYPE` | `HCD` or `CID` |
| `INSTRUMENT_TYPE` | `FT` or `IT` |
| Peak line | `m/z intensity annotation` |

Peak annotations support forms such as `b1-z1`, `b2-z2`, `y1-z1`, and `y2-z2`, as well as modification-loss annotations such as `b2-ModLoss-z1`. Unmatched peaks can be labeled `Unknown`. Training and evaluation require positive-intensity peaks with b/y annotations. Experimental peaks may be omitted for prediction, but the sequence, modification field, and instrument conditions must still be supplied.

Complete input examples are available in `demo/single_rt/`, `demo/crosslink_rt/`, and `demo/ms2/`.
