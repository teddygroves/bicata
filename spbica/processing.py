"""Functions for processing the result of a sparse Bayesian ICA."""

import re
from typing import Any, Optional, TypeVar, Union
from warnings import warn

from scipy.linalg import svd
import arviz as az
import pandas as pd
import numpy as np
from numpy.typing import NDArray

from .plotting import gather_kappa


T = TypeVar("T", covariant=True)


def ok_or(
    nullable: Union[T, Any, None], type_h: type[T], default: Optional[T] = None
) -> T:
    """Unwrap a possibly None value with a `default` fallback."""
    if isinstance(nullable, type_h) and nullable is not None:
        return nullable
    if default is not None:
        return default
    else:
        # if this happens, the caller was not implemented correctly
        raise ValueError(f"Value expected to be not None, found type {type(nullable)}.")


def handle_kappa(
    idata: az.InferenceData,
    chain: int,
) -> pd.DataFrame:
    assert hasattr(idata, "posterior")
    assert hasattr(idata.posterior, "kappa")
    kappa = gather_kappa(idata.posterior, chain)
    return kappa


def gene_to_operon(
    log_tpm: pd.DataFrame, biocyc: pd.DataFrame, as_array=True
) -> Union[np.ndarray, pd.DataFrame]:
    """Build vector Gene(1) -> Operon(n).

    This should be summed +1 if passed to Stan.

    Parameters
    ----------
    log_tpm: pd.DataFrame
        indexed by genes.
    biocyc: pd.DataFrame
        indexed by genes, dataframe from biocyc with an "operon" column
    """
    gxo = pd.get_dummies(biocyc[["operon"]])
    gxo = gxo[~gxo.index.duplicated()].reindex(log_tpm.index)
    gxo = gxo.loc[:, gxo.sum(axis=0) != 0]
    gxo = gxo.apply(np.argmax, axis=1)
    if as_array:
        gxo = gxo.to_numpy().astype(int)
    return gxo


def expand_operons(
    df: pd.DataFrame, gxo: Union[pd.DataFrame, pd.Series]
) -> pd.DataFrame:
    """Map operons to their corresponding gene mapping.

    This should be summed +1 if passed to Stan.

    Parameters
    ----------
    df: pd.DataFrame
        with a "gene" column
    gxo: np.ndarray
        Output of gene_to_operon.
    """
    df_expanded = df.rename(columns={"gene": "operon"})
    operon_to_gene = (
        pd.DataFrame({"gene": gxo.index, "operon": gxo})
        .groupby("operon")
        .agg(list)
        .reset_index()
    )
    return pd.merge(df_expanded, operon_to_gene, on="operon").explode("gene")


def whiten(x: NDArray) -> NDArray:
    """Whiten the data.

    Centering is assumed, since in imodulons, the centering is usually done
    for a condition, similar to ILR.

    Adapted from https://github.com/scikit-learn/scikit-learn/blob/093e0cf14/sklearn/decomposition/_fastica.py#L355

    Parameters
    ----------
    x: NDArray, features x samples
    """
    u, d = svd(x.T, full_matrices=False, check_finite=False)[:2]
    # Give consistent eigenvectors
    u *= np.sign(u[0])
    K = (u / d).T
    del u, d
    return np.dot(K, x.T) * np.sqrt(x.shape[1])


def prepare_p0_prior(df: pd.DataFrame, n_modulons: int = 92):
    """Compute p0 guesses of number of non-sparse params.

    This expects a dataset like the one generated in Han et al., 2023.
    """
    promoter_per_tf = df[df["class"].isin([1, -1])]["tf_gene"].value_counts()
    x = np.arange(0, len(promoter_per_tf))
    x_star = np.linspace(0, len(promoter_per_tf), n_modulons)
    # xp must increasing
    assert np.all(np.diff(x) > 0)
    p0 = np.interp(x_star, x, promoter_per_tf.to_numpy()).astype(int)
    p0.sort()
    return p0


def build_ko(tpm: pd.DataFrame, gene_info: pd.DataFrame) -> pd.DataFrame:
    """Build [Knock-out, Run] matrix."""
    gene_info = gene_info.reset_index().rename(columns={"index": "gene_id"})
    gene_info["gene_name"] = gene_info.gene_name.str.lower()
    pat = re.compile(r".*del([^_]+)_.*")
    conditions = tpm.columns.to_list()
    knock_outs = [
        pat.search(condition)[1] if "del" in condition else None
        for condition in conditions
    ]
    ko_mat = tpm.copy()
    ko_mat.iloc[:, :] = 1
    for cond, ko in zip(conditions, knock_outs):
        if ko is not None:
            matched_genes = gene_info.loc[gene_info.gene_name == ko.lower(), "gene_id"]
            n_genes = matched_genes.shape[0]
            if n_genes >= 1:
                if n_genes > 1:
                    warn(
                        f"More than one gene could be matched for {ko}: {matched_genes}"
                    )
                gene_id = matched_genes.iloc[0]
                ko_mat.loc[gene_id, cond] = 0
            else:
                warn(f"No gene name found for {ko}. Skipping...")
    return ko_mat
