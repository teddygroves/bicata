"""Dashboard with panel."""

import sys
from os import path
from pathlib import Path

import arviz as az
import pandas as pd
import plotly.express as px
import panel as pn
import xarray as xr
from spbica.plotting import gather_kappa, gather_sigv, prepare_matrix, prepare_tf_matrix
from spbica.processing import expand_operons, gene_to_operon
from spbica.alignment import align_to_trn


pn.extension("plotly")
pn.extension("floatpanel", design="material", template="fast")
pn.state.template.param.update(title="Modulon viewer")
config = {"headerControls": {"close": "remove", "maximize": "remove"}}

data, results, csv = Path("data"), Path(sys.argv[1]).parent, Path(sys.argv[1])


if not path.exists(results / "kappa.csv"):
    idata = az.from_cmdstan([csv])
    if idata.posterior.kappa.shape != idata.posterior.beta.shape:
        # we have to expand kappa
        log_tpm = pd.read_csv(data / "log_tpm_norm.csv", index_col=0)
        biocyc = pd.read_csv(data / "gene_info.csv", index_col=0)
        gxo = gene_to_operon(log_tpm, biocyc, True)
        kappa_np = idata.posterior.kappa.sel(kappa_dim_0=gxo.tolist()).to_numpy()
        idata.posterior = idata.posterior.rename({"kappa": "kappa_ope"})
        idata.posterior["kappa_ope"] = idata.posterior.kappa_ope.rename(
            {"kappa_dim_0": "kappa_ope_dim"}
        )
        idata.posterior = idata.posterior.rename({"kappa_dim_0": "kappa_gene_dim_0"})
        idata.posterior["kappa"] = xr.DataArray(
            kappa_np, dims=["chain", "draw", "kappa_dim_0", "kappa_dim_1"]
        )
    modulons = prepare_matrix(idata.posterior, "beta")
    activities = prepare_matrix(idata.posterior, "V")
    kappa = gather_kappa(idata.posterior)
    act_sig = None
    modulons.to_csv(results / "modulons.csv", index=False)
    activities.to_csv(results / "activities.csv", index=False)
    kappa.to_csv(results / "kappa.csv", index=False)
    if "sig2" in idata.posterior or "sigv" in idata.posterior:
        act_sig = gather_sigv(idata.posterior, 0, 1.5, 0.92)
        act_sig.to_csv(results / "act_sig.csv", index=False)
    if "trn_link" in idata.posterior:
        tf_link = prepare_tf_matrix(idata.posterior)
        tf_link.to_csv(results / "tf_link.csv", index=False)
    else:
        tf_link = None
else:
    modulons = pd.read_csv(results / "modulons.csv")
    activities = pd.read_csv(results / "activities.csv")
    kappa = pd.read_csv(results / "kappa.csv")
    tf_link = (
        pd.read_csv(results / "tf_link.csv")
        if path.exists(results / "tf_link.csv")
        else None
    )
    act_sig = (
        pd.read_csv(results / "act_sig.csv")
        if path.exists(results / "act_sig.csv")
        else None
    )

# expand modulons from operon to genes and add annotation
log_tpm = pd.read_csv(data / "log_tpm_norm.csv", index_col=0)
biocyc = pd.read_csv(data / "gene_info.csv", index_col=0)
gxo = gene_to_operon(log_tpm, biocyc, False)
should_expand_operons = True
if len(pd.unique(modulons["gene"])) == len(pd.unique(gxo.index)):
    gene_map = {i: gene for i, gene in enumerate(log_tpm.index)}
    modulons["gene"] = modulons["gene"].apply(lambda x: gene_map[x])
    print("Genes named!", file=sys.stderr)
    # this is later passed to the TRN enrichment function
    should_expand_operons = False
else:
    modulons = expand_operons(modulons, gxo)
    print("Operons expanded!", file=sys.stderr)
modulons = pd.merge(
    modulons, biocyc.reset_index().rename(columns={"index": "gene"}), on="gene"
).rename(columns={"operon_y": "operon_name", "operon_x": "operon_index"})

# name the activities
conditions = sorted(list(set([run[:-1] for run in log_tpm.columns.to_list()])))
cond_map = {i: cond for i, cond in enumerate(conditions)}
activities["condition"] = activities["condition"].apply(lambda x: cond_map[x])

# results of TRN enrichment
if path.exists(results / "top_trn_enrich.tsv"):
    top_trn_enrichment = pd.read_csv(
        results / "top_trn_enrich.tsv", sep="\t", index_col=0
    )
else:
    print("Computing TRN enrichment...", file=sys.stderr)
    trn = pd.read_csv(data / "TRN.csv")
    if not should_expand_operons:
        kappa["gene"] = kappa["gene"].apply(lambda x: gene_map[x])
    trn_enrichment = align_to_trn(
        kappa, trn, should_expand_operons, 2, 0, log_tpm, biocyc
    )
    # return best match per modulon
    top_trn_enrichment = (
        trn_enrichment.groupby("modulon")
        .apply(lambda x: x.sort_values("f1score").tail(1))
        .sort_values("f1score", ascending=False)
    )
    print("Enriched!", file=sys.stderr)
    top_trn_enrichment.to_csv(results / "top_trn_enrich.tsv", sep="\t")

# Use st.selectbox to let the user choose an option
choose_imod_widget = pn.widgets.Select(
    name="imodulon", value=34, options=pd.unique(modulons.modulon).tolist()
)


def plot_matrix_fn(matrix, other_var):
    if "error_up" not in matrix.columns:
        matrix["error_up"] = matrix["up"] - matrix["median"]
        matrix["error_down"] = matrix["median"] - matrix["down"]
    hover_name = "gene_name" if "gene_name" in matrix.columns else "condition"
    return px.scatter(
        matrix,
        x=other_var,
        hover_name=hover_name,
        y="median",
        color=None if "uran" in matrix.columns else "kappa",
        # size="errors",
        error_y="error_up",
        error_y_minus="error_down",
        range_color=None if "uran" in matrix.columns else [0, 1],
    )


def plot_activities_fn(imodulon):
    return pn.pane.Plotly(
        plot_matrix_fn(activities[activities.modulon == imodulon], "condition"),
        dpi=90,
        tight=True,
    )


def plot_modulons_fn(imodulon):
    return pn.pane.Plotly(
        plot_matrix_fn(modulons[modulons.modulon == imodulon], "gene"),
        dpi=90,
        tight=True,
    )


def plot_tfs(imodulon):
    return pn.pane.Plotly(
        plot_matrix_fn(tf_link[tf_link.modulon == imodulon], "tf"),
        dpi=90,
        tight=True,
    )


def choose_enrichment(imodulon):
    choice = imodulon
    if any(imodulon == top_trn_enrichment["modulon.1"]):
        choice = top_trn_enrichment.loc[top_trn_enrichment["modulon.1"] == imodulon, :].iloc[0, 0]
    return f"# Modulon {choice}"


def plot_kappas(imodulon):
    return pn.pane.Plotly(
        px.histogram(
            kappa[kappa.modulon == imodulon], y="kappa", color="gene", opacity=0.8
        ),
        dpi=90,
        tight=True,
    )


def plot_actsig():
    return pn.pane.Plotly(
        px.histogram(act_sig, x="draw", color="source"),
        dpi=90,
        tight=True,
    )


def show_genes(imodulon):
    mod = modulons.loc[
        (modulons.modulon == imodulon) & (modulons.kappa < 0.5), :
    ].sort_values("kappa")
    return pn.panel(mod)


bound_activities = pn.bind(plot_activities_fn, imodulon=choose_imod_widget)
bound_weights = pn.bind(plot_modulons_fn, imodulon=choose_imod_widget)
if tf_link is not None:
    bound_tfs = pn.bind(plot_tfs, imodulon=choose_imod_widget)
if act_sig is not None:
    bound_sig = plot_actsig()
chosen_tf = pn.bind(choose_enrichment, imodulon=choose_imod_widget)
# bound_kappas = pn.bind(plot_kappas, imodulon=choose_imod_widget)
bound_genes = pn.bind(show_genes, imodulon=choose_imod_widget)

settings = pn.layout.FloatPanel(
    choose_imod_widget,
    contained=False,
    name="Choose imodulon",
    offsetx=10,
    position="center",
    config=config,
)
pn.Column(
    settings,
    pn.Row(chosen_tf),
    pn.Row(
        pn.Column(f"### Gene weight", bound_weights),
        pn.Column("### Activities", bound_activities),
    ),
    pn.Row(
        pn.Column("### TF", bound_tfs if tf_link is not None else "None"),
        pn.Column("### Sig", bound_sig if act_sig is not None else "None"),
        # pn.Column("### Kappas", bound_kappas),
    ),
    "## Genes",
    bound_genes,
    "## Enrichments",
    pn.panel(top_trn_enrichment),
    "## F1score",
    pn.panel(top_trn_enrichment.f1score.describe()),
).servable(target="main")
