"""Functions for processing the result of a sparse Bayesian ICA."""

from typing import Optional, Union
from dataclasses import dataclass

import arviz as az
import pandas as pd
import numpy as np
import xarray as xr
from pymodulon.enrichment import compute_trn_enrichment, contingency

from .plotting import gather_kappa
from .processing import expand_operons, gene_to_operon, handle_kappa, ok_or


@dataclass
class ModulonMatch:
    fscore: float
    size: int
    ref: int
    target: int


@dataclass
class ModulonAlignment:
    """Result of aligning modulons."""

    reference_chain: int
    target_chain: int
    fscores: list[float]
    """F1-score between each aligned modulon."""
    sizes: list[int]
    """size of each modulon in the reference modulon."""
    ref_modulons: list[Optional[int]]
    """reference modulon index."""
    target_modulons: list[int]
    """closest modulon index to the reference."""

    def __getitem__(self, ref_mod: int) -> ModulonMatch:
        idx = self.ref_modulons.index(ref_mod)
        return ModulonMatch(
            self.fscores[idx],
            self.sizes[idx],
            self.ref_modulons[idx],
            self.target_modulons[idx],
        )


def f1score(gene_set: set, target_genes: set, all_genes: set) -> float:
    ((tp, fp), (fn, _)) = contingency(gene_set, target_genes, all_genes)
    # edge cases
    if tp == 0:
        f1score = 0.0
    elif fp == 0 and fn == 0:
        f1score = 1.0
    else:
        recall = np.true_divide(tp, tp + fn)
        precision = np.true_divide(tp, tp + fp)
        f1score = (2 * precision * recall) / (precision + recall)
    return f1score


def compute_best_hits(
    ref_kappa: pd.DataFrame, kappa: pd.DataFrame, reference_chain: int, chain: int
) -> ModulonAlignment:
    """Compute best match in `kappa` for each modulon in `ref_kappa`."""
    result = ModulonAlignment(reference_chain, chain, [], [], [], [])
    all_genes = set(pd.unique(ref_kappa.gene).tolist())
    for modulon in pd.unique(ref_kappa.modulon):
        ref_genes = set(
            ref_kappa.loc[
                (ref_kappa.modulon == modulon) & (ref_kappa.draw < 0.5), "gene"
            ].tolist()
        )
        fscores = np.array(
            [
                f1score(
                    ref_genes,
                    set(
                        kappa.loc[
                            (kappa.modulon == mod) & (kappa.draw < 0.5), "gene"
                        ].tolist()
                    ),
                    all_genes,
                )
                for mod in pd.unique(kappa.modulon)
            ]
        )
        target_modulon = np.argmax(fscores)
        result.fscores.append(fscores.max())
        result.ref_modulons.append(modulon)
        result.target_modulons.append(int(target_modulon))
        result.sizes.append(len(ref_genes))
    return result


def align_modulons(
    idata: az.InferenceData, reference_chain: int = 0
) -> list[ModulonAlignment]:
    """Align the kappa values in each chain to a reference chain.

    Returns
    -------
    alignment: list[ModulonAlignment]
        each element corresponds to one chained mapped to the reference chain.

    Raises
    ------
    If `idata` does not contain posterior with a "kappa" variable.
    """
    assert hasattr(idata, "posterior")
    assert hasattr(idata.posterior, "kappa")
    ref_kappa = gather_kappa(idata.posterior, reference_chain)
    kappas = [
        (chain, gather_kappa(idata.posterior, chain))
        for chain in idata.posterior.chain
        if chain != reference_chain
    ]
    alignments = [
        compute_best_hits(ref_kappa, kappa, reference_chain, chain)
        for chain, kappa in kappas
    ]
    return alignments


def align_to_trn(
    idata: Union[az.InferenceData, pd.DataFrame],
    trn: pd.DataFrame,
    should_expand_operons: bool = False,
    max_regs: int = 1,
    chain: int = 0,
    log_tpm: Optional[pd.DataFrame] = None,
    biocyc: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Align an inference object to a TRN using pymodulon enrichment.

    Parameter
    ---------
    trn: pd.DataFrame
        Transcriptional Regulatory Network with colums [regulator, gene_id, effect],
        as expected by pymodulon.
    should_expand_operons: bool, default=False
        whether to treat the "gene" dimension in `kappa` as operons, mapping
        them to their corresponding genes. Requires `log_tpm` and `biocyc`.
    max_regs: int, default=1
        max number of combination of regulators that should be tested per modulon.
    log_tpm: Optional[pd.DataFrame]
        indexed by genes.
    biocyc: Optional[pd.DataFrame]
        indexed by genes, dataframe from biocyc with an "operon" column
    """
    if isinstance(idata, az.InferenceData):
        kappa = handle_kappa(idata, chain)
    else:
        # idata is a dataframe, kappa has already been processed
        kappa = idata
    if log_tpm is not None:
        # if the log tpms are passed, we remove non present genes as in the pymodulon impl
        trn = trn.loc[trn.gene_id.isin(log_tpm.index), :]
    if should_expand_operons:
        assert log_tpm is not None and biocyc is not None
        gxo: pd.Series = ok_or(gene_to_operon(log_tpm, biocyc, False), pd.Series)
        kappa = expand_operons(kappa, gxo)
    all_genes = set(pd.unique(kappa.gene).tolist())
    enrichments = []
    for modulon in pd.unique(kappa.modulon):
        modulon_genes = set(
            kappa.loc[(kappa.modulon == modulon) & (kappa.draw < 0.5), "gene"].tolist()
        )
        enrichment = compute_trn_enrichment(modulon_genes, all_genes, trn, max_regs)
        enrichment["modulon"] = modulon
        enrichments.append(enrichment)
    return pd.concat(enrichments).sort_values("f1score", ascending=False)


def sort_idata_modulons(
    idata: az.InferenceData,
    alignments: list[ModulonAlignment],
    gxo: Optional[np.ndarray],
):
    """Sort idata across modulon dimension according to an alignment.

    This operation is **in-place**. Assumes that the reference chain is 0.

    Parameters
    ----------
    gxo: np.array
        map from genes to operons
    """
    N_mod = idata.posterior.V.shape[-1]
    betas = [idata.posterior.beta.sel(chain=0).to_numpy()]
    activities = [idata.posterior.V.sel(chain=0).to_numpy()]
    sel_kappa = {} if gxo is None else {"kappa_dim_0": gxo.tolist()}
    kappas = [idata.posterior.kappa.sel(sel_kappa | {"chain": 0}).to_numpy()]

    for i, alignment in enumerate(alignments):
        chain = i + 1
        modulons_0_to_1 = {
            alignment[i].ref: alignment[i].target for i in range(len(alignment.fscores))
        }
        chain_1_order = [modulons_0_to_1[i] for i in range(N_mod)]
        betas.append(
            idata.posterior.beta.sel(chain=chain).to_numpy()[:, :, chain_1_order]
        )
        activities.append(
            idata.posterior.V.sel(chain=chain).to_numpy()[:, :, chain_1_order]
        )
        kappas.append(
            idata.posterior.kappa.sel(sel_kappa | {"chain": chain}).to_numpy()[
                :, :, chain_1_order
            ]
        )
    idata.posterior["beta"] = xr.DataArray(
        np.stack(betas), coords=idata.posterior.beta.coords
    )
    idata.posterior["V"] = xr.DataArray(
        np.stack(activities), coords=idata.posterior.V.coords
    )

    if gxo is not None:
        idata.posterior = idata.posterior.rename({"kappa": "kappa_ope"})
        idata.posterior["kappa_ope"] = idata.posterior.kappa_ope.rename(
            {"kappa_dim_0": "kappa_ope_dim"}
        )
        idata.posterior = idata.posterior.rename({"kappa_dim_0": "kappa_gene_dim_0"})
        idata.posterior["kappa"] = xr.DataArray(
            np.stack(kappas), dims=["chain", "draw", "kappa_dim_0", "kappa_dim_1"]
        )
    else:
        idata.posterior["kappa"] = xr.DataArray(
            np.stack(kappas), coords=idata.posterior.kappa.coords
        )
