# Bayesian ICA for Transcriptome Analysis

This repository contains documents that describe how we want to use Bayesian
statistical models to describe transcriptome data.

## Reproducibility

[Stan](https://mc-stan.org/), python 3.9 and pip (or [uv](https://docs.astral.sh/uv/)) are required.

```bash
pip install .
# or uv sync
```


> [!IMPORTANT]  
> `cmdstanpy` is listed as a dev dependency but it is required to run the scripts!
> Check [the Stan documentation](https://mc-stan.org/install/index.html) to check how to install it for your platform.

The scripts are to be run from the [`scripts`](./scripts) directory:

```bash
# z in operons Eq (S.12)
python3 infer_cmd.py 0.3 --beta-v 0.5 --suffix _o --whiten-data
# z in genes, p0=3 Eq (S.13)
python3 infer_cmd.py 0.3 --beta-v 0.5 --suffix _zaft --whiten-data --n-p0 3
# z in genes, p0 ≅ TF
python3 infer_cmd.py 0.3 --beta-v 0.5 --suffix _zaft --whiten-data --design-p0
# z in genes, kappa penalisation (Eq S.15)
python3 infer_cmd.py 0.3 --beta-v 0.5 --suffix _kappalikpois --design-p0 --whiten-data
# z in genes, kappa penalisation (Eq S.15) + Variance hyperprior (Eq S.18)
python3 infer_cmd.py 0.3 --beta-v 0.5 --suffix _kappalikpois_dAm --design-p0 --whiten-data
# same with removed conditions
python3 infer_cmd.py 0.3 --beta-v 0.5 --suffix _kappalikpois --design-p0 --whiten-data --remove-condition "pal" --remove-condition "42c" --remove-condition "efeU"
# Hierarchical ALE Eq S.19
python3 infer_cmd_ale.py 0.3 --beta-v 0.5 --rho 0.1 --suffix _kappalikpois --design-p0 --whiten-data --groupout-conditions "pal" --groupout-conditions "42c"  --groupout-conditions "efeU"
# two datasets: RNAseq from PRECISE-1 + RNAseq (Eq S.15 + S.16)
python3 infer_2Dz_cmd.py 0.3 --beta-v 0.5 --design-p0 --whiten-data --suffix _diris
# two datasets: RNAseq from PRECISE-1 + proteomics (Eq S.15 + S.16)
python3 infer_2Dz_cmd.py 0.3 --beta-v 0.5 --design-p0 --whiten-data --suffix _diris --comp-data-path ../data/principles/proteomics_fractions.tsv --design-excel-path ../data/principles/science.abk2066_table_s4.xlsx
```
