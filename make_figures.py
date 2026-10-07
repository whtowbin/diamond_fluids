"""Make the README figures and the summary table behind them.

Run from the repo root: uv run python make_figures.py
Writes figures/*.png and figures/results.csv. Takes about 3 minutes, most of it the
Gaussian process classifier that the old pipeline used.
"""

import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from hdf_compilation import HDF_TYPES, LABEL, concentration, has_trace_data, load_compilation, log_ratios, make_classifier

OUT = Path("figures")
OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
TYPE_STYLE = {  # same colour and marker for a fluid type in every figure
    "saline": ("#0072B2", "o"),
    "silicic": ("#E69F00", "s"),
    "silicic - low-Mg carbonatitic": ("#CC79A7", "D"),
    "low-Mg carbonatitic": ("#009E73", "^"),
    "high-Mg carbonatitic": ("#D55E00", "v"),
}
MODEL_COLOR, BASELINE_COLOR, FOLD_COLOR = "#0072B2", "#E69F00", "0.7"
WIDTH = 6.5  # inches: a GitHub README column at 200 dpi
OLD_RATIOS = ["Nb/La", "Zr/Sr", "Zr/La", "Rb/Sr", "Ba/Sr", "U/Sr", "Ba/Ce", "La/Ce", "Pb/U", "Y/La", "Zr/Y", "Zr/Nb", "Nb/La"]

mpl.rcParams.update(
    {
        "figure.constrained_layout.use": True,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.6,
        "axes.prop_cycle": cycler(color=OKABE_ITO[1:] + OKABE_ITO[:1]),
        "legend.frameon": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
    }
)


def short_type(name):
    return name.replace("carbonatitic", "carb.").replace(" - ", " –\n")


def stratified_scores(model, X, y):
    """Balanced accuracy on each of 15 folds (5-fold CV repeated 3 times)."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    scores = []
    for train, test in cv.split(X, y):
        model.fit(X[train], y[train])
        scores.append(balanced_accuracy_score(y[test], model.predict(X[test])))
    return np.array(scores)


def held_out_paper_predictions(model, X, y, groups):
    return cross_val_predict(model, X, y, cv=GroupKFold(5), groups=groups)


def score_row(name, model, X, y, groups):
    folds = stratified_scores(model, X, y)
    y_group = held_out_paper_predictions(model, X, y, groups) if groups is not None else None
    held_out = balanced_accuracy_score(y, y_group) if y_group is not None else np.nan
    print(f"{name:55s} n={len(y):3d} stratified {folds.mean():.3f}±{folds.std():.3f} held-out papers {held_out:.3f}")
    return {"setup": name, "n": len(y), "classes": len(set(y)), "stratified_mean": folds.mean(),
            "stratified_sd": folds.std(), "held_out_papers": held_out, "folds": folds, "y_group": y_group}


def old_pipeline_data(df):
    """The inputs the previous script used: Excel ratio columns, rows complete in all of them."""
    ratios = df.loc[df[LABEL].notna(), OLD_RATIOS].dropna()
    return ratios.to_numpy(), df.loc[ratios.index, LABEL].to_numpy(), df.loc[ratios.index, "Reference"].fillna("unknown").to_numpy()


def type_results(df):
    features = log_ratios(df)
    keep = df[LABEL].notna() & has_trace_data(features)
    X, y = features[keep].to_numpy(), df.loc[keep, LABEL].to_numpy()
    groups = df.loc[keep, "Reference"].fillna("unknown").to_numpy()
    X_old, y_old, g_old = old_pipeline_data(df)
    def log1p_scaled(m):
        return make_pipeline(FunctionTransformer(np.log1p), StandardScaler(), m)

    gpc = GaussianProcessClassifier(kernel=1.0 * RBF(np.ones(X_old.shape[1])), random_state=0)
    rows = [
        score_row("Previous: Excel ratios, log(x+1), Gaussian process", log1p_scaled(gpc), X_old, y_old, g_old),
        score_row("Excel ratios, log(x+1), logistic regression", log1p_scaled(make_classifier()[-1]), X_old, y_old, g_old),
        score_row("Log ratios from concentrations, logistic regression", make_classifier(), X, y, groups),
    ]
    model = make_classifier().fit(X, y)
    coefficients = pd.DataFrame(model[-1].coef_, index=model[-1].classes_, columns=features.columns).loc[HDF_TYPES]
    return rows, y, coefficients


def location_data(df, target, min_rows=10, multi_paper_only=False):
    features = log_ratios(df)
    keep = has_trace_data(features) & df[target].notna()
    counts = df.loc[keep, target].value_counts()
    keep &= df[target].isin(counts[counts >= min_rows].index)
    if multi_paper_only:
        papers = df[keep].groupby(target)["Reference"].nunique()
        keep &= df[target].isin(papers[papers > 1].index)
    y = df.loc[keep, target].str.strip().to_numpy()
    groups = df.loc[keep, "Reference"].fillna("unknown").to_numpy()
    return features[keep].to_numpy(), df.loc[keep, [LABEL]].fillna("unknown").to_numpy(), y, groups


def location_results(df):
    baseline = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(class_weight="balanced", max_iter=5000))
    rows = []
    for name, target, multi, by_paper in [
        ("Craton", "Craton", False, True),
        ("Locality", "Locality ", False, False),
        ("Locality, multi-paper only", "Locality ", True, True),
    ]:
        X, fluid_type, y, groups = location_data(df, target, multi_paper_only=multi)
        g = groups if by_paper else None
        trace = score_row(f"{name}: trace-element ratios", make_classifier(), X, y, g)
        fluid = score_row(f"{name}: fluid type only", baseline, fluid_type, y, g)
        rows.append({"target": name, "trace": trace, "fluid": fluid})
    return rows


def fig_model_comparison(rows):
    fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
    labels = [textwrap.fill(r["setup"], 30) + f"\n(n = {r['n']})" for r in rows]
    ypos = np.arange(len(rows))[::-1]
    for yp, r in zip(ypos, rows):
        ax.scatter(r["folds"], np.full(len(r["folds"]), yp + 0.12), s=10, color=FOLD_COLOR, zorder=2, linewidths=0)
        ax.scatter(r["stratified_mean"], yp + 0.12, s=45, color=MODEL_COLOR, zorder=3)
        ax.scatter(r["held_out_papers"], yp - 0.15, s=50, marker="D", facecolor="white", edgecolor=MODEL_COLOR, linewidth=1.5, zorder=3)
    chance = 1 / rows[0]["classes"]
    ax.axvline(chance, color="0.4", linestyle="--", linewidth=1)
    ax.annotate("chance", (chance, ypos[0] + 0.45), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color="0.3")
    ax.set_yticks(ypos, labels)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("Balanced accuracy over the five fluid types")
    ax.grid(axis="y", visible=False)
    handles = [
        plt.Line2D([], [], marker="o", color=FOLD_COLOR, linestyle="", markersize=4, label="one CV fold"),
        plt.Line2D([], [], marker="o", color=MODEL_COLOR, linestyle="", markersize=7, label="mean of 15 folds"),
        plt.Line2D([], [], marker="D", markerfacecolor="white", markeredgecolor=MODEL_COLOR, linestyle="", markersize=7, label="papers held out"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    old, new = rows[0]["held_out_papers"], rows[-1]["held_out_papers"]
    ax.set_title(f"With whole papers held out, balanced accuracy rises\nfrom {old:.2f} to {new:.2f} (chance {chance:.2f})")
    fig.savefig(OUT / "model_comparison.png")
    plt.close(fig)


def fig_confusion(y, y_pred):
    counts = confusion_matrix(y, y_pred, labels=HDF_TYPES)
    share = counts / counts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.4))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(HDF_TYPES)):
        for j in range(len(HDF_TYPES)):
            ax.text(j, i, f"{counts[i, j]}", ha="center", va="center", fontsize=9, color="white" if share[i, j] > 0.55 else "0.15")
    names = [short_type(t) for t in HDF_TYPES]
    ax.set_xticks(range(len(HDF_TYPES)), [textwrap.fill(n.replace("\n", " "), 8) for n in names])
    ax.set_yticks(range(len(HDF_TYPES)), [f"{n.replace(chr(10), ' ')} ({c})" for n, c in zip(names, counts.sum(axis=1))])
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("Actual type (number of diamonds)")
    ax.grid(False)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Share of the actual type")
    recall = np.diag(share)
    best, worst = int(recall.argmax()), int(recall.argmin())
    ax.set_title(
        f"Papers held out: {recall[best]:.0%} of {HDF_TYPES[best]} fluids are recognised,\n"
        f"{recall[worst]:.0%} of {HDF_TYPES[worst]}"
    )
    fig.savefig(OUT / "confusion_held_out_papers.png")
    plt.close(fig)
    return pd.DataFrame(counts, index=HDF_TYPES, columns=HDF_TYPES)


def fig_coefficients(coefficients):
    data = coefficients.rename(columns=lambda c: c.removeprefix("log "))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    limit = np.abs(data.to_numpy()).max()
    im = ax.imshow(data, cmap="RdBu_r", norm=TwoSlopeNorm(0, -limit, limit), aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.iat[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color="white" if abs(v) > 0.6 * limit else "0.15")
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.set_yticks(range(data.shape[0]), [short_type(t).replace("\n", " ") for t in data.index])
    ax.set_xlabel("log ratio (each scaled to unit variance)")
    ax.grid(False)
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label="Coefficient")
    top_saline = data.loc["saline"].nlargest(2).index
    low_hmg = data.loc["high-Mg carbonatitic"].nsmallest(2).index
    ax.set_title(f"High {top_saline[0]} and {top_saline[1]} point to saline fluids;\nlow {low_hmg[0]} and {low_hmg[1]} to high-Mg carbonatitic")
    fig.savefig(OUT / "coefficients.png")
    plt.close(fig)


def fig_ba_la(df):
    ratios = pd.DataFrame({
        "La/Ce": concentration(df, "La") / concentration(df, "Ce"),
        "Ba/Ce": concentration(df, "Ba") / concentration(df, "Ce"),
        "type": df[LABEL],
    }).dropna()
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    for t in HDF_TYPES[::-1]:  # saline drawn last, on top
        sub = ratios[ratios["type"] == t]
        color, marker = TYPE_STYLE[t]
        ax.scatter(sub["Ba/Ce"], sub["La/Ce"], s=22, color=color, marker=marker, alpha=0.8, linewidths=0, label=f"{short_type(t).replace(chr(10), ' ')} ({len(sub)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plain)
        axis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v))) in (2, 5) else ""))
    ax.set_xlabel("Ba/Ce (log scale)")
    ax.set_ylabel("La/Ce (log scale)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper left", bbox_to_anchor=(1.01, 1), title="Fluid type (n)")
    saline = ratios[ratios["type"] == "saline"]
    others = ratios[ratios["type"] != "saline"]
    ax.set_title(
        f"Saline fluids have higher Ba/Ce and La/Ce\n(medians {saline['Ba/Ce'].median():.0f} and {saline['La/Ce'].median():.2f}"
        f" against {others['Ba/Ce'].median():.0f} and {others['La/Ce'].median():.2f})"
    )
    fig.savefig(OUT / "ba_ce_vs_la_ce.png")
    plt.close(fig)


def fig_location(rows):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    ypos = np.arange(len(rows))[::-1]
    for ax, key, title in [(axes[0], "stratified_mean", "Same papers in training and test"), (axes[1], "held_out_papers", "Whole papers held out")]:
        for yp, r in zip(ypos, rows):
            if np.isnan(r["trace"][key]):
                ax.annotate("not testable:\none paper per locality", (0.5, yp), ha="center", va="center", fontsize=7.5, color="0.4")
                continue
            chance = 1 / r["trace"]["classes"]
            ax.plot([chance, chance], [yp - 0.3, yp + 0.3], color="0.4", linewidth=1.2)
            ax.scatter(r["fluid"][key], yp, s=45, marker="s", facecolor="white", edgecolor=BASELINE_COLOR, linewidth=1.5, zorder=3)
            ax.scatter(r["trace"][key], yp, s=45, color=MODEL_COLOR, zorder=3)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Balanced accuracy")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(ypos, [f"{r['target']}\n({r['trace']['classes']} classes, n = {r['trace']['n']})" for r in rows])
    handles = [
        plt.Line2D([], [], marker="o", color=MODEL_COLOR, linestyle="", markersize=7, label="trace-element ratios"),
        plt.Line2D([], [], marker="s", markerfacecolor="white", markeredgecolor=BASELINE_COLOR, linestyle="", markersize=7, label="fluid type only"),
        plt.Line2D([0, 0], [0, 1], color="0.4", linewidth=1.2, label="chance"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    craton = rows[0]["trace"]
    fig.suptitle(
        f"Craton accuracy falls from {craton['stratified_mean']:.2f} to {craton['held_out_papers']:.2f} when whole papers are held out",
        x=0.01, ha="left", fontweight="bold", fontsize=10,
    )
    fig.savefig(OUT / "location.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df = load_compilation()
    rows, y, coefficients = type_results(df)
    loc_rows = location_results(df)
    fig_model_comparison(rows)
    confusion = fig_confusion(y, rows[-1]["y_group"])
    fig_coefficients(coefficients)
    fig_ba_la(df)
    fig_location(loc_rows)
    flat = rows + [r[k] for r in loc_rows for k in ("trace", "fluid")]
    summary = pd.DataFrame(flat).drop(columns=["folds", "y_group"])
    summary.round(3).to_csv(OUT / "results.csv", index=False)
    confusion.to_csv(OUT / "confusion_held_out_papers.csv")
    coefficients.round(3).to_csv(OUT / "coefficients.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
