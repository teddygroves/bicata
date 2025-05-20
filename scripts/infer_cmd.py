import arviz as az
from pathlib import Path
from typing import Optional
import sys
import cmdstanpy
import numpy as np
import pandas as pd

from typer import Argument, Option, run
from typing_extensions import Annotated
from spbica import prepare_p0_prior, whiten
from spbica.alignment import align_to_trn
from spbica.plotting import gather_kappa


def run_cmd(
    sigma: Annotated[float, Argument(help="measurement model sigma.")],
    beta_v: Annotated[
        float,
        Option(help="beta parameter of Inverse gamma prior for the Activity matrix."),
    ] = 1.0,
    suffix: Annotated[
        Optional[str], Option(help="suffix, will decide the Stan model to use.")
    ] = None,
    design_p0: Annotated[
        bool, Option(help="whether to use a prior vector for tau.")
    ] = False,
    n_p0: Annotated[
        int, Option(help="prior for number of operons per modulon (if `design_p0` is False)")
    ] = 3,
    iter_warmup: int = 250,
    iter_sampling: int = 100,
    whiten_data: bool = True,
    test: Annotated[bool, Option(help="will sample once if True")] = False,
    n_modulons: int = 92,
    remove_condition: Optional[list[str]] = None,
):
    """Sample sparse ICA."""
    if suffix == None:
        suffix = ""

    log_tpm = pd.read_csv("../data/log_tpm_norm.csv", index_col=0)
    if remove_condition is not None:
        for condition in remove_condition:
            assert log_tpm.columns.str.startswith(
                condition
            ).any(), f"Remove condition {condition} was passed as argument but was not found in the dataset"
            log_tpm = log_tpm.loc[:, ~log_tpm.columns.str.startswith(condition)]
    log_tpm_matrix = log_tpm.to_numpy()
    log_tpm_matrix_white = (
        whiten(log_tpm.to_numpy()).T if whiten_data else log_tpm_matrix
    )

    biocyc = pd.read_csv("../data/gene_info.csv", index_col=0)
    # map from gene to operon (1-indexed)
    gxo_df = pd.get_dummies(biocyc[["operon"]])
    gxo_df = gxo_df[~gxo_df.index.duplicated()].reindex(log_tpm.index)
    gxo_df = gxo_df.loc[:, gxo_df.sum(axis=0) != 0]
    gxo = gxo_df.apply(np.argmax, axis=1).to_numpy().astype(int) + 1

    TRUTH = {}

    # map from runs to conditions (1-indexed)
    runs = log_tpm.columns.to_list()
    conditions = sorted(list(set([run[:-1] for run in runs])))
    design = np.array([conditions.index(run[:-1]) + 1 for run in runs]).astype(int)
    control_index = conditions.index("control__wt_glc__") + 1

    assert len(design) == len(runs)
    assert all(design <= len(conditions))

    if design_p0:
        suffix += "_p0prior"
        if "all" in suffix:
            # we expect that the modulon matrix is relates to genes instead of operons
            p0_dataset = pd.read_csv("../data/TRN.csv")
            # this is just so that everything is taken into account in the prepare_p0_prior
            p0_dataset["class"] = -1
            p0_dataset["tf_gene"] = p0_dataset["regulator"]
        else:
            p0_dataset = pd.read_csv("../data/PPT-seq/dataset.tsv", sep="\t")
        p0 = prepare_p0_prior(p0_dataset, n_modulons)
        print(f"p0 prior (all={'all' in suffix}): {p0}", file=sys.stderr)
    else:
        p0 = n_p0

    model_reghorse_sparse_nov = cmdstanpy.CmdStanModel(
        stan_file=f"sparse_orthogonal_posredux_design_sigmas{suffix}.stan"
    )

    TRUTH = TRUTH | {
        "N_gene": log_tpm_matrix.shape[0],
        "N_ope": gxo.max(),
        "N_mode": n_modulons,
        "N_cond": len(conditions),
        "N_run": log_tpm_matrix.shape[1],
        "nu": 1,
        # sparse
        "p0": p0,
        "nu_global": 1,
        "nu_local": 1,
        "slab_scale": 4,
        "slab_df": 4,
        "ope_design": gxo,
        "exp_design": design,
        "WT_IDX": control_index,
    }

    print(
        "Fitting RNAseq matrix: [N_gene, N_run] -> "
        f"{log_tpm_matrix_white.shape}; suffix={suffix}; σ={sigma}; ß={beta_v}; whiten={whiten_data}"
    )

    whiten_suffix = "" if whiten_data else "_nowhiten"
    if remove_condition is not None:
        suffix += "".join([f"_noC{cond}" for cond in remove_condition])
    result_dir = (
        f"../res/test/orthogonal_design_sigma{sigma}v{beta_v}{suffix}{whiten_suffix}w{iter_warmup}s{iter_sampling}p0{n_p0}"
        if test
        else f"/work3/jcamu/bicata_res/orthogonal_design_sigma{sigma}v{beta_v}{suffix}{whiten_suffix}w{iter_warmup}s{iter_sampling}p0{n_p0}"
    )

    mcmc_fit_data = model_reghorse_sparse_nov.sample(
        # the median RNAseq std between replicates is ~0.075 and the p75 is ~0.16
        # the prior for the std of the Activity matrix was set to a InvGamma with
        # similar quantiles and also sigma = 0.1 to account for greater error given
        # that genes in the an operon also vary
        data=TRUTH
        | {
            "Y": log_tpm_matrix_white,
            "sigma": sigma,
            "beta_v": beta_v,
            "likelihood": 1,
        },
        chains=4,
        iter_warmup=iter_warmup if not test else 1,
        iter_sampling=iter_sampling if not test else 1,
        step_size=0.000655964,
        refresh=10,
        output_dir=result_dir,
    )
    # report TRN performance for a quick lookup
    idata = az.from_cmdstanpy(mcmc_fit_data)
    trn = pd.read_csv("../data/TRN.csv")
    trn_enrichment = align_to_trn(
        gather_kappa(idata.posterior), trn, True, 2, 0, log_tpm, biocyc
    )
    # return best match per modulon
    top_trn_enrichment = (
        trn_enrichment.groupby("modulon")
        .apply(lambda x: x.sort_values("f1score").tail(1))
        .sort_values("f1score")
    )
    top_trn_enrichment.to_csv(Path(result_dir) / "trn_enrichment_chain0.tsv", sep="\t")
    top_trn_enrichment.f1score.describe().to_csv(
        Path(result_dir) / "trn_enrichment_f1score_chain0.tsv", sep="\t"
    )


if __name__ == "__main__":
    run(run_cmd)
