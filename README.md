# MMF: Interpretable Multi-scale Multi-modal Framework for Few-shot WSI Classification

## Install

```bash
pip install -r requirements.txt        # pip freeze of our training environment
pip install git+https://github.com/mahmoodlab/CONCH.git    # conch.*, not on PyPI
```

## Data

Features are pre-extracted CONCH embeddings: one `.pt` per slide and scale, a `[N, 512]` float
tensor. They are expected under `--data_root` as:

```
{data_root}/
├── rcc/feature/{conch_2048, conch}/
├── c16/feature/{conch_2048, conch}/
├── c17/feature/{conch_2048, conch}/
├── nsclc/feature/{conch_2048, conch}/
└── ubc_ocean/feature/{conch_2048, conch_512}/
```

- **Features** — the 5x branch reads `{data_root}/<subdir>/feature/conch_2048`, the 20x branch
  reads `conch` (`conch_512` for UBC-OCEAN).
- **Labels** — `dataset_csv/{name}.csv` with columns `slide_id, case_id, label`.
- **Splits** — `splits/{dataset}_{shots}shots_5folds/splits_{i}.csv`, columns `train, val, test`
  holding slide ids, `i = 0 ... k-1`.
- **Prompts** — `text_prompt/*.csv`, `2C` rows: first `C` low-resolution, last `C` high-resolution.

## Run

```bash
python main.py <dataset> <shots> --data_root /path/to/features --conch_ckpt /path/to/conch.bin
#   dataset: rcc | nsclc | ubc_ocean | camelyon | camelyon17
#   shots:   16 | 8 | 4

python main.py camelyon17 4 --data_root /data/RCB-img --conch_ckpt ckpts/conch.bin
```

Other options: `--results_root` (default `./results/tmp`), `--k` (default 5), `--max_epochs`,
`--lr`, `--reg`, `--k_start` / `--k_end`.

### Patch-level evaluation

`--camelyon_patch` additionally reports patch-level AUC / ACC / F1 on C16 and C17. The released code
evaluates the **5x** patches (the `conch_2048` branch, `out_alpha_s` in `MMF.forward`). It requires
`--patch_label_dir` holding one `{slide_id}_label.pt` per slide: a 1-D tensor of 0/1 with one entry
per patch, **in the same order as the patches of that slide's `conch_2048` file**. A patch is
labelled positive when at least 50% of its area overlaps an annotated tumor region. The metrics are
printed and, when `--patch_result_csv` is given, appended to that csv.

To evaluate the **20x** patches instead, return `out_alpha_l` / `alpha_l` in the `camelyon_patch`
branch of `MMF.forward`, and point `--patch_label_dir` to labels built for the `conch` features
(one entry per 20x patch, same order).

## Output

```
results/tmp/
├── result.csv                          # mean/std of each run
└── {dataset}/{shots}/{dataset}_{shots}shots/
    ├── summary.csv                     # per-fold test AUC / ACC / F1
    ├── experiment_*.txt                # settings
    ├── s_{i}_checkpoint.pt
    └── split_{i}_results.pkl
```

## Acknowledgments

This code builds on [CLAM](https://github.com/mahmoodlab/CLAM),
[ViLa-MIL](https://github.com/Jiangbo-Shi/ViLa-MIL),
[CONCH](https://github.com/mahmoodlab/CONCH) and
[FOCUS](https://github.com/dddavid4real/FOCUS). Please follow their licenses.
CONCH model weights are not included in this repository.
