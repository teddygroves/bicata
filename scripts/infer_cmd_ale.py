import arviz as az
from pathlib import Path
from typing import Optional
from functools import reduce
import sys
import cmdstanpy
import numpy as np
import pandas as pd

from typer import Argument, Option, run
from typing_extensions import Annotated
from spbica import prepare_p0_prior, whiten
from spbica.processing import build_ko
from spbica.alignment import align_to_trn
from spbica.plotting import gather_kappa


def run_cmd(
    sigma: Annotated[float, Argument(help="measurement model sigma.")],
    beta_v: Annotated[
        float,
        Option(help="beta parameter of Inverse gamma prior for the Activity matrix."),
    ] = 1.0,
    rho: Annotated[
        float,
        Option(help="std parameter of hierarchical prior"),
    ] = 0.15,
    suffix: Annotated[
        Optional[str], Option(help="suffix, will decide the Stan model to use.")
    ] = None,
    design_p0: Annotated[
        bool, Option(help="whether to use a prior vector for tau.")
    ] = False,
    pptp_sigma: Annotated[
        float, Option(help="measurement model sigma for the PPTP data.")
    ] = 0.2,
    iter_warmup: int = 250,
    iter_sampling: int = 100,
    whiten_data: bool = True,
    ko: Annotated[
        bool, Option(help="whether to zero KOs in their respective conditions.")
    ] = False,
    test: Annotated[bool, Option(help="will sample once if True")] = False,
    groupout_conditions: Annotated[
        list[str],
        Option(
            help="list of suffixes to group in different hierarchical gene covariates (columns must contain `_ale`)"
        ),
    ] = ["glu", "42c", "ssw", "pgi", "fps", "pal", "efeU"],
):
    """Sample sparse ICA."""
    if suffix == None:
        suffix = ""

    log_tpm = pd.read_csv("../data/log_tpm_norm.csv", index_col=0)
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
    n_groups = 1
    runs = log_tpm.columns.to_list()
    conditions = sorted(list(set([run[:-1] for run in runs])))
    for groupout_condition in groupout_conditions:
        n_groups += 1
        this_runs = log_tpm.columns[
            log_tpm.columns.str.startswith(groupout_condition)
            & log_tpm.columns.str.contains("_ale")
        ].to_list()
        design_2 = np.array(
            [conditions.index(run[:-1]) + 1 for run in runs if run in this_runs]
        ).astype(int)
        yindex = [i for i, run in enumerate(log_tpm.columns) if run in this_runs]
        TRUTH |= {
            f"N_run_{n_groups}": len(design_2),
            f"exp_design_{n_groups}": design_2,
            f"y{n_groups}_index": yindex,
        }
    suffix += f"_{n_groups}out"

    # the index of the rest of conditions
    rest_of_runs = log_tpm.columns[
        ~(
            log_tpm.columns.str.startswith(tuple(groupout_conditions))
            & log_tpm.columns.str.contains("_ale")
        )
    ].to_list()
    TRUTH |= {
        "y1_index": [
            i + 1 for i, col in enumerate(log_tpm.columns) if col in rest_of_runs
        ]
    }
    assert (
        len(
            reduce(
                lambda a, b: a & b,
                [
                    set(indices)
                    for key, indices in TRUTH.items()
                    if key.endswith("_index")
                ],
            )
        )
        == 0
    )
    design = np.array([conditions.index(run[:-1]) + 1 for run in rest_of_runs]).astype(
        int
    )
    control_index = conditions.index("control__wt_glc__") + 1

    assert len(design) == len(rest_of_runs)
    assert all(design <= len(conditions))

    pptp_indicator = ""

    if "PPTP" in suffix:
        operons = [col.replace("operon_", "") for col in gxo_df.columns]
        pptp_fc_posgenes = pd.read_csv(
            "../data/PPT-seq/internal/reduced_posgenes_trn.tsv", sep="\t", index_col=0
        )
        pptp_matrix = pptp_fc_posgenes.to_numpy()
        ope_index = [operons.index(i) + 1 for i in pptp_fc_posgenes.index]
        tf_index = [operons.index(i) + 1 for i in pptp_fc_posgenes.columns]
        pptp_indicator = f"p{pptp_sigma}"
        TRUTH = TRUTH | {
            "T": pptp_matrix,
            "ope_index": ope_index,
            "tf_index": tf_index,
            "N_tf": len(tf_index),
            "N_ope_tf": len(ope_index),
            "pptp_sigma": pptp_sigma,
        }

    if ko:
        suffix += "_ko"
        kos = build_ko(log_tpm, biocyc).to_numpy().astype(float)
        assert (
            kos.shape == log_tpm_matrix.shape
        ), "KOs must be able to perform a Hadamard product with log TPM."
        TRUTH = TRUTH | {
            "ko": kos,
        }

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
        p0 = prepare_p0_prior(p0_dataset, len(tf_index) if "tfPPTP" in suffix else 92)
        print(f"p0 prior (all={'all' in suffix}): {p0}", file=sys.stderr)
    else:
        p0 = 3

    stan_file = f"sparse_orthogonal_posredux_design_sigmas{suffix}.stan"
    if not Path(stan_file).exists:
        raise NotImplementedError(
            f"Hierarchical gene covariates not implemented for {n_groups} groups"
        )

    model_reghorse_sparse_nov = cmdstanpy.CmdStanModel(
        stan_file=f"sparse_orthogonal_posredux_design_sigmas{suffix}.stan"
    )

    TRUTH = TRUTH | {
        "N_gene": log_tpm_matrix.shape[0],
        "N_ope": gxo.max(),
        "N_mode": 92,
        "N_cond": len(conditions),
        "N_run": log_tpm_matrix.shape[1],
        "nu": 1,
        "rho_std": rho,
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
    assert (
        len(
            reduce(
                lambda a, b: a & b,
                [
                    set(indices)
                    for key, indices in TRUTH.items()
                    if key.startswith("exp_design")
                ],
            )
        )
        == 0
    )
    metagroups = pd.DataFrame(
        {
            "h_group": ["rest"] + [cond for cond in groupout_conditions],
            "y_index": list(range(1, len(groupout_conditions) + 2)),
            "num_runs": [len(rest_of_runs)]
            + [TRUTH[f"N_run_{g}"] for g in range(2, len(groupout_conditions) + 2)],
        }
    )
    print("Groups descriptions")
    print(metagroups, end="\n\n")
    print(
        "Fitting RNAseq matrix: [N_gene, N_run] -> "
        f"{log_tpm_matrix_white.shape}; suffix={suffix}; σ={sigma}; ß={beta_v}; whiten={whiten_data}"
    )

    whiten_suffix = "" if whiten_data else "_nowhiten"
    result_dir = (
        f"../res/test/orthogonal_design_sigma{sigma}v{beta_v}r{rho}{pptp_indicator}{suffix}{whiten_suffix}"
        if test
        else f"/work3/jcamu/bicata_res/orthogonal_design_sigma{sigma}v{beta_v}{pptp_indicator}{suffix}{whiten_suffix}"
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
    # save metadata about groups (we are not saving an InferenceData coords because it gets too big)
    metagroups.to_csv((Path(result_dir) / "groupout_conditions.tsv"), sep="\t")

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
