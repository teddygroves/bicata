"""Utils for plotting bayesian ICA."""

import numpy as np
import pandas as pd
import plotnine as p9
from arviz import InferenceData
from tqdm import tqdm


def prepare_matrix(
    post: InferenceData,
    var: str = "U_tilde",
    chain: int = 0,
    filter_low_kappa: bool = False,
) -> pd.DataFrame:
    """Build a matrix to then be plotted."""
    other_var = "condition" if ("V" in var or "theta" in var) else "gene"
    # try with the transpose since some times we may not generate it to save disk space
    var = var if var in post else f"{var}_t"
    post_var = post[var]
    if var.endswith("_t"):
        dim_1 = post_var.shape[-2]
    else:
        dim_1 = post_var.shape[-1]
    all_us = []
    dfs = []

    # melt(id=modulon) + groupby(modulon, other_var) + apply(quantiles)
    # is infeasible memory-wise. Thus, first reduce the df and then melt + pivot
    quantiles = post_var.sel({"chain": chain}).quantile(q=[0.25, 0.5, 0.75], dim="draw")
    for i in range(dim_1):
        u = quantiles.sel({f"{var}_dim_1": i}).to_pandas()
        u["modulon"] = i
        dfs.append(u)

    all_us = (
        pd.melt(
            # melt the other_var columns but don't drop the quantiles
            pd.concat(dfs),
            id_vars="modulon",
            var_name=other_var,
            ignore_index=False,
        )
        .reset_index()
        # turn the index quantiles into respective columns
        .pivot(index=["modulon", other_var], columns="quantile", values="value")
        .rename(columns={0.25: "down", 0.75: "up", 0.5: "median"})
        .reset_index()
    )
    del dfs

    if "kappa" in post:
        if post_var.shape == post["kappa"].shape:
            kappas = gather_kappa(post, chain)
            all_us = pd.merge(all_us, kappas, on=["modulon", "gene"]).rename(
                columns={"draw": "kappa"}
            )
            if filter_low_kappa:
                all_us = all_us[all_us.kappa < 0.6]
    if "kappa" not in all_us.columns:
        all_us["uran"] = (all_us["up"] - all_us["down"]).abs()
    return all_us


def plot_modulons(
    post: InferenceData,
    var: str = "U_tilde",
    chain: int = 0,
    filter_low_kappa: bool = False,
) -> p9.ggplot:
    """Plot variable `var` over imodulons as line ranges with points as median.

    If the variable can be arranged to a `kappa` variable found in `post`,
    the color of the line ranges is the median of the kappa values, where values
    close to 0 indicate true parameters (membership to modulon) and close to 1
    indicate a non-parameter.

    Parameters
    ----------
    post: arviz.InferenceData.posterior
        it accepts directly the posterior in case the user wants
        to select some dimensions first.
    var: str
        if V or theta, it will assume that it is a condition x modulon matrix.
    chain: int
    filter_low_kappa: bool, default=False
        whether to only plot values with associated kappa < 0.6.
    """
    other_var = "condition" if ("V" in var or "theta" in var) else "gene"
    all_us = prepare_matrix(post, var, chain, filter_low_kappa)
    return (
        p9.ggplot(
            all_us,
            p9.aes(
                ymin="down",
                ymax="up",
                y="median",
                x=other_var,
                color="uran" if "uran" in all_us.columns else "kappa",
            ),
        )
        + p9.geom_errorbar()
        + p9.geom_point()
        + p9.facet_wrap("~modulon", scales="free_y", dir="v", ncol=3)
        + p9.theme(figure_size=(20, 40))
        + p9.ylab("modulon activity")
    )


def gather_kappa(post: InferenceData, chain: int = 0, agg: bool = True) -> pd.DataFrame:
    """Wrangle kappa draws per modulon and gene."""
    kappas = []
    dim_1 = post.kappa.shape[-1]
    for i in range(dim_1):
        kappa = post.kappa.sel({"chain": chain, "kappa_dim_1": i}).to_pandas()
        kappa["modulon"] = i
        kappas.append(kappa)
    kappas = pd.melt(
        pd.concat(kappas), id_vars=["modulon"], var_name="gene", value_name="draw"
    )
    return (
        (
            kappas.groupby(["modulon", "gene"])
            .agg(lambda x: (x > 0.5).sum() / len(x))
            .reset_index()
        )
        if agg
        else kappas
    )


def gather_sigv(
    post: InferenceData, chain: int = 0, prior_a: float = 1.5, prior_b=0.92
) -> pd.DataFrame:
    """Wrangle kappa draws per modulon and gene."""
    df = pd.DataFrame(post["sigv" if "sigv" in post else "sig2"].sel({"chain": chain}).to_pandas(), columns=["draw"])
    df["source"] = "Posterior"
    inv_gamma = lambda alpha, beta: 1 / np.random.gamma(
        alpha, 1 / beta, size=df.shape[0]
    )
    df_prior = pd.DataFrame({"draw": inv_gamma(prior_a, prior_b), "source": "Prior"})
    return pd.concat([df, df_prior])


def plot_kappa(post: InferenceData, chain: int = 0, ncol: int = 3) -> p9.ggplot:
    """Plot kappa draws per modulon and gene."""
    kappas = gather_kappa(post, chain)
    return (
        p9.ggplot(kappas, p9.aes(x="draw", fill="factor(gene)"))
        + p9.geom_histogram(bins=120, position="identity", alpha=0.8)
        + p9.facet_wrap("~modulon", ncol=ncol)
        + p9.theme_tufte()
        + p9.theme(figure_size=(20, 20))
    )


def prepare_tf_matrix(
    post: InferenceData,
    var: str = "trn_link",
    chain: int = 0,
) -> pd.DataFrame:
    """Build a TF-modulon matrix to be plotted."""
    other_var = "modulon"
    dfs = []
    dim_1 = post[var].shape[-1]
    quantiles = (
        post[var].sel({"chain": chain}).quantile(q=[0.25, 0.5, 0.75], dim="draw")
    )
    for i in tqdm(range(dim_1), total=dim_1, desc="Gathering data"):
        u = quantiles.sel({f"{var}_dim_1": i}).to_pandas()
        u["tf"] = i
        dfs.append(u)
    all_tfs = (
        pd.melt(
            # melt the other_var columns but don't drop the quantiles
            pd.concat(dfs),
            id_vars="tf",
            var_name=other_var,
            ignore_index=False,
        )
        .reset_index()
        # turn the index qunatiles into respective columns
        .pivot(index=["tf", other_var], columns="quantile", values="value")
        .rename(columns={0.25: "down", 0.75: "up", 0.5: "median"})
        .reset_index()
    )
    all_tfs["uran"] = (all_tfs["up"] - all_tfs["down"]).abs()
    return all_tfs


def plot_tf_link(
    post: InferenceData,
    var: str = "trn_link",
    chain: int = 0,
) -> p9.ggplot:
    r"""Plot median of variable `var` as heatmap.

    Parameters
    ----------
    post: arviz.InferenceData.posterior
        it accepts directly the posterior in case the user wants
        to select some dimensions first.
    var: str
    chain: int
    """
    all_tf = prepare_tf_matrix(post, var, chain)
    return (
        p9.ggplot(
            all_tf,
            p9.aes(fill="median", color="uran", y=f"factor(modulon)", x="factor(tf)"),
        )
        + p9.geom_tile()
        + p9.theme(figure_size=(20, 18))
        + p9.scale_fill_cmap(cmap_name="inferno")
        + p9.ylab("modulon")
        + p9.xlab("TF")
    )
