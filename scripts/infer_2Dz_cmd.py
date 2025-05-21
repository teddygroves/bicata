"""Run inference for two datasets at the same time."""

import sys
from glob import glob
from pathlib import Path
from typing import Optional

import arviz as az
import cmdstanpy
import numpy as np
import pandas as pd
from typer import Argument, Option, run
from typing_extensions import Annotated
from spbica import prepare_p0_prior, whiten
from spbica.alignment import align_to_trn
from spbica.plotting import gather_kappa
from spbica.processing import expand_operons


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
    iter_warmup: int = 250,
    iter_sampling: int = 100,
    whiten_data: bool = True,
    test: Annotated[bool, Option(help="will sample once if True")] = False,
    comp_data_path: Path = Path("../data/principles/transcriptomics_fractions.tsv"),
    design_excel_path: Annotated[
        Path, "path to excel file with the description of the conditions associated with `comp_data_path`"
    ] = Path("../data/principles/science.abk2066_table_s3.xlsx"),
):
    """Sample sparse ICA."""
    if suffix == None:
        suffix = ""

    log_tpm = pd.read_csv("../data/log_tpm_norm.csv", index_col=0)
    log_tpm = log_tpm.sort_index()
    log_tpm_matrix = log_tpm.to_numpy()
    log_tpm_matrix_white = (
        whiten(log_tpm.to_numpy()).T if whiten_data else log_tpm_matrix
    )
    rna_fracs = pd.read_csv(comp_data_path, sep="\t", index_col=1)
    print(
        f"Will remove {rna_fracs[~rna_fracs.index.str.startswith('b')].shape[0]} genes"
    )
    # remove inserts in the same locus across conditions
    rna_fracs = rna_fracs[rna_fracs.index.str.startswith("b")].sort_index()
    gene_lengths = rna_fracs["gene length (nt)"].to_numpy()
    rna_fracs = rna_fracs.iloc[:, 2:]
    # the values of this matrix are handled by stan
    rna_fracs_matrix = rna_fracs.to_numpy()

    # build the gxo mapping
    biocyc = pd.read_csv("../data/gene_info.csv", index_col=0)
    gxo_df = pd.get_dummies(biocyc[["operon"]])
    gxo_df = gxo_df[~gxo_df.index.duplicated()]
    # remove genes that are not in any of the datasets
    all_index = sorted(list(set(rna_fracs.index) | set(log_tpm.index)))
    gxo_df = gxo_df.reindex(all_index)
    # remove empty columns
    gxo_df = gxo_df.loc[:, gxo_df.sum(axis=0) != 0]
    gxo = gxo_df.apply(np.argmax, axis=1)
    assert all(log_tpm.index.isin(gxo.index))
    assert all(rna_fracs.index.isin(gxo.index))
    # np.argmax returns -1 if it could not find a match
    gxo[gxo == -1] = max(gxo) + np.arange(1, sum(gxo == -1) + 1, dtype=int)
    assert all(gxo >= 0)
    # 1-based stan indexing
    gxo_stan = gxo.to_numpy().astype(int) + 1

    # mapping from common genes to their particular dataset
    gxo_df_f = pd.DataFrame(gxo).reset_index()
    gxo_df_f.columns = ["locus", "operon_number"]
    gxg_1 = (
        gxo_df_f.index[gxo_df_f["locus"].isin(log_tpm.index)].to_numpy().astype(int) + 1
    )
    gxg_2 = (
        gxo_df_f.index[gxo_df_f["locus"].isin(rna_fracs.index)].to_numpy().astype(int)
        + 1
    )

    # map from runs to conditions (1-indexed)
    design_rna_df = pd.read_excel(
        design_excel_path,
        sheet_name="1 - Proteomics (description)" if "proteomics" in str(comp_data_path) else "1 - RNAseq-ss (description)",
        index_col=None,
    )
    runs = log_tpm.columns.to_list()
    conditions = list(set([run[:-1] for run in runs]))
    design = np.array([conditions.index(run[:-1]) + 1 for run in runs]).astype(int)
    design_rna_df["condition"] = design_rna_df["Strain"] + design_rna_df["media_tag"]
    runs = rna_fracs.columns.to_list()
    conditions2 = list(set(design_rna_df["condition"].tolist()))
    design2 = np.array(
        [
            conditions2.index(
                design_rna_df[
                    run.replace(".1", "") == design_rna_df["Sample ID"]
                ].condition.values[0]
            )
            + 1
            for run in runs
        ]
    ).astype(int)
    # calculate sigma as the within condition p75 std
    rna_sigma = pd.melt(
        rna_fracs.reset_index(),
        id_vars="locus",
        var_name="Sample ID",
        value_name="frac",
    )
    rna_sigma = pd.merge(
        rna_sigma, design_rna_df[["Sample ID", "condition"]], on="Sample ID"
    )
    # sigmas_2 = rna_sigma.groupby(["locus", "condition"])["frac"].std()
    # sigmas_2 = sigmas_2[sigmas_2.notnull() & (sigmas_2 != 0)]
    # sigma2 = np.percentile(sigmas_2, 0.75)
    # log scale
    sigma2 = 0.2

    assert len(design) == len(log_tpm.columns)
    assert all(design <= len(conditions))

    TRUTH = {}

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
        p0 = prepare_p0_prior(p0_dataset, 92)
        print(f"p0 prior (all={'all' in suffix}): {p0}", file=sys.stderr)
    else:
        p0 = 3

    model_reghorse_sparse_nov = cmdstanpy.CmdStanModel(
        stan_file=f"spbica_2Dz{suffix}.stan"
    )

    TRUTH = TRUTH | {
        "N_gene": log_tpm_matrix.shape[0],
        "N_gene_2": rna_fracs_matrix.shape[0],
        "N_gene_total": len(gxo_stan),
        "N_ope": gxo_stan.max(),
        "N_mode": 92,
        "N_cond": len(conditions),
        "N_cond_2": len(conditions2),
        "N_run": log_tpm_matrix.shape[1],
        "N_run_2": rna_fracs_matrix.shape[1],
        "nu": 1,
        # sparse
        "p0": p0,
        "nu_global": 1,
        "nu_local": 1,
        "slab_scale": 4,
        "slab_df": 4,
        "ope_design": gxo_stan,
        "gxg_1": gxg_1,
        "gxg_2": gxg_2,
        "exp_design": design,
        "exp_design_2": design2,
        "gene_lengths": gene_lengths,
    }

    print(
        "Fitting RNAseq matrix: [N_gene, N_run] -> "
        f"{log_tpm_matrix_white.shape}; [N_gene_2, N_run2] -> {rna_fracs_matrix.shape}"
        f"suffix={suffix}; σ={sigma}; σ_2={sigma2}; ß={beta_v}; whiten={whiten_data}"
    )

    whiten_suffix = "" if whiten_data else "_nowhiten"
    prot_suffix = "_prot" if "prot" in str(comp_data_path) else ""
    result_dir = (
        f"../res/test/2Dz{sigma}v{beta_v}{suffix}{prot_suffix}{whiten_suffix}"
        if test
        else f"/work3/jcamu/bicata_res/2Dz{sigma}v{beta_v}{suffix}{prot_suffix}{whiten_suffix}"
    )

    # add pseudocounts per run
    def close_comp(x: pd.Series):
        pseudocount = x[x != 0].min()
        x[x == 0] = pseudocount
        return x

    rna_fracs_matrix = rna_fracs.apply(close_comp).to_numpy()

    _ = model_reghorse_sparse_nov.sample(
        # the median RNAseq std between replicates is ~0.075 and the p75 is ~0.16
        # the prior for the std of the Activity matrix was set to a InvGamma with
        # similar quantiles and also sigma = 0.1 to account for greater error given
        # that genes in the an operon also vary
        data=TRUTH
        | {
            "Y": log_tpm_matrix_white,
            "Y2": rna_fracs_matrix,
            "sigma": sigma,
            "sigma2": sigma2,
            "beta_v": beta_v,
            "likelihood": 1,
        },
        chains=4,
        iter_warmup=iter_warmup if not test else 1,
        iter_sampling=iter_sampling if not test else 1,
        step_size=0.0116065,
        refresh=10,
        output_dir=result_dir,
        max_treedepth=10,
    )
    # report TRN performance for a quick lookup
    # idata = az.from_cmdstanpy(mcmc_fit_data)
    idata = az.from_cmdstan(glob(str(Path(result_dir) / "*1.csv")))
    trn = pd.read_csv("../data/TRN.csv")
    kappa = gather_kappa(idata.posterior)
    kappa = expand_operons(kappa, gxo)
    # analysis only of those in the datasets
    # TODO(carrascomj): should I also remove kappa genes not in trn?
    trn = trn.loc[trn.gene_id.isin(all_index), :]
    trn_enrichment = align_to_trn(kappa, trn, False, 2, 0, None, None)
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
