import argparse
import shelve
import dbm
import os.path
import pandas as pd
import tikzplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
import pprint
from datetime import timedelta
from pathlib import Path
import numpy as np

LOG_PATH = "/Users/madsadrian/Springfield/ChangeDetection/logs/"
LOG_PATH = "/Users/madsadrian/OneDrive - UiT Office 365/tensorboard-logs/"

MODEL_SORT_ORDER = {key: i for i, key in enumerate(["--X", "-CX", "A-X", "ACX"])}
model_sorter = lambda tuple_: MODEL_SORT_ORDER[tuple_[0]]

PRIOR_BASELINE = {"Texas": 0.6516, "California": 0.2038}


def td_print(td):
    """ Print formatter for pd.timedelta """
    comps = td.round("1s").components
    retval = ""
    for i, s in enumerate(["days", "hours", "min", "sec"]):
        if comps[i] > 0:
            retval += f"{comps[i]}{s} "
    if retval == "":
        retval = "$\\cdot$"
    return retval


def experiment_one_boxplot(epoch=99, dataset="Texas"):
    """ Make boxplot for experiment one """
    path_ = os.path.join(LOG_PATH, "Final/Experiment 01", dataset)
    df = pd.read_csv(os.path.join(path_, "metrics.csv"))
    df = df[df["epoch"] == epoch]
    df["training time"] = pd.to_timedelta(df["training time"], unit="s")

    order = {
        "A--": [0],
        "-C-": [1],
        "--X": [2],
        "AC-": [3],
        "A-X": [4],
        "-CX": [5],
        "ACX": [6],
    }

    if ARGS.to_tikz:
        df["model"] = df["model"].apply(lambda s: f"\\m{{{s}}}")
        order = {f"\\m{{{key}}}": value for key, value in order.items()}

    print("Model", "Average", "$\\sigma$", "Max", sep="\t& ", end="\t\\\\\n")
    fig, ax = plt.subplots()
    for name, gdf in df.groupby("model"):
        ax.boxplot(gdf["cohens kappa"], positions=order[name], widths=0.8)
        d = gdf["training time"].describe()
        print(
            name,
            td_print(d["mean"]),
            td_print(d["std"]),
            td_print(d["max"]),
            sep="\t& ",
            end="\t\\\\\n",
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order.keys())
    ax.set_xlabel("Model")
    ax.set_ylabel("Cohens $\\kappa$")
    ax.grid(axis="y")
    ax.axhline(PRIOR_BASELINE[dataset], c="red")

    if ARGS.to_tikz:
        tikzplotlib.save(
            figure=fig,
            filepath=os.path.join(path_, f"E1-{dataset}-cohens-kappa-boxplot.tex"),
            figureheight=".4\\textheight",
            figurewidth=".95\\textwidth",
        )
    else:
        plt.show()


def color_box_plot(bp, edge_color=None, fill_color=None, hatch=None, label=None):
    if edge_color is not None:
        for element in ["boxes", "whiskers", "fliers", "means", "medians", "caps"]:
            plt.setp(bp[element], color=edge_color)
    if fill_color is not None:
        for patch in bp["boxes"]:
            patch.set(facecolor=fill_color)
            if hatch is not None:
                patch.set_hatch(hatch)
            if label is not None:
                patch.set_label(label)
            return patch


def recurse(obj, prefix=""):
    print(prefix, type(obj), "\t", obj)
    for child in obj.get_children():
        recurse(child, prefix + "\t")


COLORS = {"A": "yellow", "B": "cyan", "C": "magenta", "D": "orange"}
PLT_HATCHES = {"A": r"////", "B": r"\\\\", "C": r"O", "D": r"."}
TIKZ_HATCHES = {
    "A": "north east lines",
    "B": "north west lines",
    "C": "crosshatch",
    "D": "dots",
}
Y_LIMS = {"Texas": (0.64, 0.89), "California": (0.19, 0.56)}


def experiment_two_paired_boxplot(
    epoch=99,
    dataset="Texas",
    pairs=(("A", "C"), ("B", "D")),
    metric="cohens kappa",
    labels=("Fixed Learning Rate", "Decay Learning Rate"),
    y_label="Cohens $\\kappa$",
    training_times=False,
):
    """ Make boxplot for experiment one """
    assert dataset in ["Texas", "California"]
    e_num = 2 if dataset == "Texas" else 3

    order = {"A-X": np.array([3]), "-CX": np.array([1]), "ACX": np.array([5])}
    if ARGS.to_tikz:
        order = {f"\\m{{{key}}}": value for key, value in order.items()}

    path_ = LOG_PATH + "/Resultater/"

    figs, axes = [], []
    for pair in pairs:
        fig, ax = plt.subplots()
        figs.append(fig)
        axes.append(ax)
        legend_hs = []
        for name, x_mod, label in zip(pair, [-0.25, 0.25], labels):
            fill_color, hatch = COLORS[name], PLT_HATCHES[name]
            df = pd.read_csv(path_ + f"{name}-{dataset}-metrics.csv")
            df = df[df["epoch"] == epoch]
            df["training time"] = pd.to_timedelta(df["training time"], unit="s")

            df["model"] = df["model"].apply(lambda s: s.strip("p"))

            if ARGS.to_tikz:
                df["model"] = df["model"].apply(lambda s: f"\\m{{{s}}}")

            if training_times:
                print(f"\n\n{dataset} {name} Training times")
                print(
                    "Model",
                    "Average",
                    "$\\sigma$",
                    "Max",
                    sep="\t& ",
                    end="\t\\\\\\hline\n",
                )

            for name, gdf in df.groupby("model"):
                bp = ax.boxplot(
                    gdf[metric],
                    whis=[5, 95],
                    positions=order[name] + x_mod,
                    widths=0.4,
                    patch_artist=True,
                )
                lh = color_box_plot(bp, fill_color=fill_color, hatch=hatch, label=label)

                if training_times:
                    d = gdf["training time"].describe()
                    print(
                        name,
                        td_print(d["mean"]),
                        td_print(d["std"]),
                        td_print(d["max"]),
                        sep="\t& ",
                        end="\t\\\\\n",
                    )

            if label is not None:
                legend_hs.append(lh)

        ax.axhline(PRIOR_BASELINE[dataset], c="red")
        ax.set_xticks(np.concatenate(list(order.values())))
        ax.set_xticklabels(order.keys())
        ax.set_xlabel("Model")
        ax.set_ylabel(y_label)
        ax.set_ylim(Y_LIMS[dataset])
        ax.grid(axis="y")
        ax.legend(handles=legend_hs, loc="lower right")

    # for pair, fig, ax in zip(pairs, figs, axes):
    #     recurse(fig)

    if ARGS.to_tikz:
        for pair, fig in zip(pairs, figs):
            code = tikzplotlib.get_tikz_code(
                figure=fig, figureheight=".4\\textheight", figurewidth=".95\\textwidth"
            )

            legend_str = "\n"
            colors = ("color1", "color2")
            directions = ("east", "west")
            for label, name, col in zip(labels, pair, colors):
                legend_str += "\\addlegendentry{" + label + "};\n"
                old = f"fill={col}"
                new = f"pattern={TIKZ_HATCHES[name]}, pattern color={col}"
                code = code.replace(old, new)
                legend_str += (
                    "\\addlegendimage{draw=black, area legend, " + new + "};\n"
                )
            if metric == "cohens kappa":
                code = code.replace("\\end{axis}", legend_str + "\n\\end{axis}")

            tag = metric if metric != "cohens kappa" else "K"
            filepath = os.path.join(LOG_PATH, f"{dataset}-{pair[0]}{pair[1]}-{tag}.tex")
            with open(filepath, "w") as f:
                f.write(code)
    else:
        plt.show()


def min_sec_formatter(value, pos):
    td = timedelta(seconds=value * 1e-9)
    return td


def boxplot_training_time(df):
    ax = df.boxplot(column="training time", by="model")
    ax.set_title(f"Training time after epoch {epoch}")
    ax.set_xlabel("Method")
    ax.set_ylabel("Training time [hh:mm:ss]")
    ax.ticklabel_format(axis="y", style="plain")
    ymajor_formatter = FuncFormatter(min_sec_formatter)
    ax.yaxis.set_major_formatter(ymajor_formatter)

    if ARGS.to_tikz:
        tikzplotlib.save(os.path.join(LOG_PATH, TAG, "training_time.tex"))
    else:
        plt.show()


def boxplot_metrics(df):

    # for epoch, edf in df.groupby(by="epoch"):
    if True:
        edf = df
        epoch = 50
        for model, mdf in edf.groupby("model"):
            print(f"{model} average training time", mdf["training time"].mean())

        ax = edf.boxplot(column="cohens kappa", by="model", grid=False)
        ax.set_title(f"Cohens $\\kappa$ after epoch {epoch}")
        ax.set_xlabel("Method")
        ax.set_ylabel("Cohens $\\kappa$")
        if ARGS.to_tikz:
            tikzplotlib.save(os.path.join(LOG_PATH, TAG, "cohens_kappa.tex"))
        else:
            plt.show()


def labeled_subplots(nrows, ncols, row_labs, col_labs, pad=5):
    """
        fig
        col_labs
        row_labs
        pad=5 in points
    """
    assert len(col_labs) == ncols
    assert len(row_labs) == nrows
    # cols = ["Column {}".format(col) for col in range(1, 4)]
    # rows = ["Row {}".format(row) for row in ["A", "B", "C", "D"]]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
    plt.setp(axes.flat, xlabel="X-label", ylabel="Y-label")

    for ax, col in zip(axes[0], col_labs):
        ax.annotate(
            col,
            xy=(0.5, 1),
            xytext=(0, pad),
            xycoords="axes fraction",
            textcoords="offset points",
            size="large",
            ha="center",
            va="baseline",
        )

    for ax, row in zip(axes[:, 0], row_labs):
        ax.annotate(
            row,
            xy=(0, 0.5),
            xytext=(-ax.yaxis.labelpad - pad, 0),
            xycoords=ax.yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
        )

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)
    return fig, axes


def tensor_string_to_float(s):
    return float(s[10:].split(",")[0])


def plot_experiment_losses(model_dir, metrics=None, ax=None):

    convs = {
        "reg": tensor_string_to_float,
        "cross": tensor_string_to_float,
        "aff": tensor_string_to_float,
        "cycle": tensor_string_to_float,
    }
    metrics = metrics or {"x_loss": "fx_losses.csv", "y_loss": "fy_losses.csv"}
    dfs = {key: [] for key in metrics.keys()}
    print(model_dir.path, model_dir.name)
    with os.scandir(model_dir.path) as it:
        for entry in it:
            # iterate timestamp level
            if entry.name == ".DS_Store":
                continue
            print(entry.name, model_dir.name)
            for key, file_name in metrics.items():
                df = pd.read_csv(os.path.join(entry.path, file_name), converters=convs)
                # df["model"] = model_dir.name
                # df["timestamp"] = entry.name
                dfs[key].append(df)

    colors = {"reg": "y", "cross": "r", "aff": "b", "cycle": "g"}
    fig, axes = plt.subplots(1, 2)
    fig.set_title()
    for ax, (key, df_list) in zip(axes.flatten(), dfs.items()):
        df = pd.concat(df_list)
        df2 = df.groupby(["epoch"], as_index=False).agg(["mean", "std"])

        ax.set_yscale("log")
        ax.grid(axis="y", which="both")
        ax.set_title(key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss value")

        for loss in df2.columns.get_level_values(0).unique():
            ax.plot(df2[loss, "mean"], color=colors[loss], label=loss)
            ax.fill_between(
                df2.index,
                df2[loss, "mean"] - 2 * df2[loss, "std"],
                df2[loss, "mean"] + 2 * df2[loss, "std"],
                color=colors[loss],
                alpha=0.2,
            )
        ax.legend()

    plt.show()


def experiment_loss_plot(experiment="Experiment 02b"):
    path_ = os.path.join(
        "/Users/madsadrian/OneDrive - UiT Office 365/tensorboard-logs/keep", experiment
    )
    runs = {
        # "a": "20191201-035206-Texas-A",
        # "b": "20191201-035213-Texas-B/",
        # "c": "20191201-035221-Texas-C/",
        "d": "20191201-035234-Texas-D/"
    }
    for key, folder in runs.items():  # iterate experiment level
        with os.scandir(os.path.join(path_, folder)) as it:
            # iterate model level
            for entry in it:
                if not entry.is_dir() or entry.name == "shelf":
                    continue
                plot_experiment_losses(entry)
            return


def experiment_three_boxplot(epoch=99):
    runs = {
        "a": "Experiment 03/3A/",
        "b": "Experiment 03/3B/",
        "c": "Experiment 03/3C/",
        "d": "Experiment 03/3D/",
    }
    note = {
        "a": "lrd=False, aff_patch=32, ps=128, bs=1, #b=12",
        "b": "lrd=False, aff_patch=False, ps=50/100, bs=2, #b=40/10",
        "c": "lrd=True, aff_patch=32, ps=128, bs=1, #b=12",
        "d": "lrd=True, aff_patch=False, ps=50/100, bs=2, #b=40/10",
    }

    dfs = {}
    for key, folder in runs.items():
        df = pd.read_csv(os.path.join(LOG_PATH, folder, "metrics.csv"))
        df = df[df["epoch"] == epoch]
        df["training time"] = pd.to_timedelta(df["training time"], unit="s")
        dfs[key] = df

    _, axes = plt.subplots(2, 2, sharey=True)
    model_order = {"-CX": [0], "A-X": [1], "A-Xp": [1], "ACX": [2], "ACXp": [2]}

    if ARGS.to_tikz:
        for key, df in dfs.items():
            dfs[key]["model"] = df["model"].apply(lambda s: f"\\m{{{s}}}")
        model_order = {f"\\m{{{key}}}": value for key, value in model_order.items()}

    for ax, (key, df) in zip(axes.flatten(), dfs.items()):
        for name, gdf in df.groupby("model"):
            ax.boxplot(gdf["cohens kappa"], positions=model_order[name])
            print(key, name, gdf["training time"].mean())
        ax.set_xticks(range(3))
        ax.set_xticklabels(["-CX", "A-X", "ACX"])
        ax.set_xlabel(note[key])
        ax.set_ylabel("Cohens $\\kappa$")
        ax.grid(axis="y")
        ax.axhline(0.2038, c="red")

    if ARGS.to_tikz:
        print("tikz not configured")
        exit()
        tikzplotlib.save(
            figure=fig50,
            filepath=os.path.join(
                LOG_PATH, "Experiment 02", "cohens-kappa-boxplot-a.tex"
            ),
            figureheight="4in",
            figurewidth="4in",
            # strict=True,
        )
        tikzplotlib.save(
            figure=fig32,
            filepath=os.path.join(
                LOG_PATH, "Experiment 02", "cohens-kappa-boxplot-a.tex"
            ),
            figureheight="4in",
            figurewidth="4in",
            # strict=True,
        )
    else:
        plt.show()


def experiment_two_b_boxplot(epoch=99, wrong_one=False):
    runs = {
        "a": "Experiment 02/2A/",
        "b": "Experiment 02/2B/",
        "c": "Experiment 02/2C/",
        "d": "Experiment 02/2D/",
    }
    note = {
        "a": "lrd=False, aff_patch=32, ps=128, bs=1, #b=12",
        "b": "lrd=False, aff_patch=False, ps=50/100, bs=2, #b=40/10",
        "c": "lrd=True, aff_patch=32, ps=128, bs=1, #b=12",
        "d": "lrd=True, aff_patch=False, ps=50/100, bs=2, #b=40/10",
    }
    if wrong_one:
        runs = {
            "a": "errors/bs nb wrong/20191201-035213-Texas-B",
            "b": "Experiment 02/2B",
            "c": "errors/bs nb wrong/20191201-035234-Texas-D",
            "d": "Experiment 02/2D",
        }
        note = {
            "a": "lrd=False, ps=50/100, bs=2/10, #b=40/2",
            "b": "lrd=False, ps=50/100, bs=2, #b=40/10",
            "c": "lrd=True, ps=50/100, bs=2/10, #b=40/2",
            "d": "lrd=True, ps=50/100, bs=2, #b=40/10",
        }

    dfs = {}
    for key, folder in runs.items():
        df = pd.read_csv(os.path.join(LOG_PATH, folder, "metrics.csv"))
        df = df[df["epoch"] == epoch]
        df["training time"] = pd.to_timedelta(df["training time"], unit="s")
        dfs[key] = df

    fig, axes = plt.subplots(2, 2, sharey=True)
    # fig, axes = labeled_subplots(
    #     2,
    #     2,
    #     ["lrd=False", "lrd=True"],
    #     [
    #         "aff_patch=32, ps=128, bs=1, #b=12",
    #         "aff_patch=False, ps=50/100, bs=10, #b=2",
    #     ],
    # )
    model_order = {
        "--X": [0],
        "A-X": [1],
        "A-Xp": [1],
        "-CX": [2],
        "ACX": [3],
        "ACXp": [3],
    }

    if ARGS.to_tikz:
        for key, df in dfs.items():
            dfs[key]["model"] = df["model"].apply(lambda s: f"\\m{{{s}}}")
        model_order = {f"\\m{{{key}}}": value for key, value in model_order.items()}
    lpos = 0
    for ax, (key, df) in zip(axes.flatten(), dfs.items()):
        for name, gdf in df.groupby("model"):
            ax.boxplot(gdf["cohens kappa"], positions=model_order[name])
        ax.set_xticks(range(4))
        ax.set_xticklabels(["--X", "A-X", "-CX", "ACX"])
        ax.set_xlabel(note[key], x=lpos)
        ax.set_ylabel("Cohens $\\kappa$")
        ax.grid(axis="y")
        ax.axhline(0.6516, c="red")

    if ARGS.to_tikz:
        print("tikz not configured")
        exit()
        tikzplotlib.save(
            figure=fig50,
            filepath=os.path.join(
                LOG_PATH, "Experiment 02", "cohens-kappa-boxplot-a.tex"
            ),
            figureheight="4in",
            figurewidth="4in",
            # strict=True,
        )
        tikzplotlib.save(
            figure=fig32,
            filepath=os.path.join(
                LOG_PATH, "Experiment 02", "cohens-kappa-boxplot-a.tex"
            ),
            figureheight="4in",
            figurewidth="4in",
            # strict=True,
        )
    else:
        plt.show()


def experiment_two_boxplot(epoch=49):
    path_ = (
        "/Users/madsadrian/OneDrive - UiT Office 365/tensorboard-logs/keep/"
        + "Experiment 02/"
    )
    runs = {"ps50": "20191129-191422-Texas/", "ps32": "20191130-185159-Texas/"}

    df50 = pd.read_csv(path_ + runs["ps50"] + "metrics.csv")
    df50 = df50[df50["epoch"] == epoch]
    df50["training time"] = pd.to_timedelta(df50["training time"], unit="s")

    df32 = pd.read_csv(path_ + runs["ps32"] + "metrics.csv")
    df32 = df32[df32["epoch"] == epoch]
    df32["training time"] = pd.to_timedelta(df32["training time"], unit="s")

    # fig, (ax50, ax32) = plt.subplots(1, 2, sharey=True)
    fig50, ax50 = plt.subplots()
    fig32, ax32 = plt.subplots()

    model_order50 = {"--X": [0], "A-X": [1], "-CX": [2], "ACX": [3]}
    model_order32 = {"A-X": 1, "ACX": 3}

    if ARGS.to_tikz:
        df50["model"] = df50["model"].apply(lambda s: f"\\m{{{s}}}")
        df32["model"] = df32["model"].apply(lambda s: f"\\m{{{s}}}")
        model_order50 = {f"\\m{{{key}}}": value for key, value in model_order50.items()}
        # model_order32 = [f"\\m{{{key}}}" for key in model_order32]

    for name, gdf in df50.groupby("model"):
        ax50.boxplot(gdf["cohens kappa"], positions=model_order50[name], widths=0.8)
    ax50.set_xticks(range(len(model_order50)))
    ax50.set_xticklabels(model_order50.keys())
    ax50.set_xlabel("affinity patch size of 50")
    ax50.set_ylabel("Cohens $\\kappa$")
    ax50.grid(axis="y")
    xmin, xmax = ax50.get_xlim()

    ax32.set_xlim(xmin + 0.5, xmax + 0.5)
    ax32.set_ylim(ax50.get_ylim())
    for name, gdf in df32.groupby("model"):
        ax32.boxplot(gdf["cohens kappa"], positions=model_order50[name], widths=0.8)
    ax32.set_xticks(list(model_order32.values()))
    ax32.set_xticklabels(model_order32.keys())
    ax32.set_xlabel("affinity patch size of 32")
    ax32.grid(axis="y")
    ax32.tick_params(
        axis="y",
        which="major",
        labelleft=False,  # bottom=False, top=False, labelbottom=False
    )

    if ARGS.to_tikz:
        tikzplotlib.save(
            figure=fig50,
            filepath=os.path.join(path_, "cohens-kappa-boxplot-a.tex"),
            figureheight="4in",
            figurewidth="4in",
            # strict=True,
        )
        tikzplotlib.save(
            figure=fig32,
            filepath=os.path.join(path_, "cohens-kappa-boxplot-b.tex"),
            figureheight="4in",
            figurewidth="4in",
            # strict=True,
        )
    else:
        plt.show()


EXPERIMENTS = {"one": "keep/Experiment 01/20191128-153016-Texas"}


def parse_args():
    """ Parse commandline arguments """
    parser = argparse.ArgumentParser(description="Make boxplots")
    parser.add_argument(
        "--to-tikz",
        "-t",
        type=bool,
        default=False,
        help="Supress plt.show() and make .tex",
    )
    parser.add_argument("--dataset", "-ds", type=str, default="Texas")
    return parser.parse_args()


def join_metrics():
    """ Join metrics dataframe for experiments split over several jobs """
    join = {
        # f"{LOG_PATH}/Experiment 02/2D": [1, 2, 3, 4, 5],
        # f"{LOG_PATH}/Experiment 02/2B": [1, 2]
    }
    for path_, fnames in join.items():
        dfl = []
        for i in fnames:
            df = pd.read_csv(os.path.join(path_, f"metrics{i}.csv"))
            # if i == 5:
            #     df = df[df.model != "--X"]
            #     df = df[df.model != "-CX"]
            #     print(df["model"].nunique())
            print(df)
            dfl.append(df)
        df = pd.concat(dfl)
        print(df)
        df.to_csv(os.path.join(path_, "metrics.csv"))


def join_metrics_shelves(path_):
    """ Join metrics from shelves for experiments split over several jobs """

    with os.scandir(path_) as level1:
        # Taverse main folder
        for entry1 in level1:
            if not entry1.is_dir() or entry1.name == ".DS_Store":
                continue
            with os.scandir(entry1) as level2:
                # Taverse subfolder
                for entry2 in level2:
                    if not entry2.name == "shelf copy":
                        continue
                    print(entry2.path)
                    shelf_path = os.path.join(entry2.path, "/experiment.db")
                    print(dbm.whichdb(shelf_path))


def fix_dir_mess():
    path_ = LOG_PATH + "/Resultater/"

    for l in ["A", "B", "C", "D"]:
        df2 = pd.read_csv(path_ + f"2{l}metrics.csv")
        df3 = pd.read_csv(path_ + f"3{l}metrics.csv")
        df4 = pd.read_csv(path_ + f"{l}-Texas-add10-metrics.csv")

        print(l)
        print("2", df2.columns, df2["dataset"].unique())
        print("3", df3.columns, df3["dataset"].unique())
        print("4", df4.columns, df4["dataset"].unique())
        print("\n\n")

        df = pd.concat([df2, df3, df4], axis=0)
        for dataset, gdf in df.groupby("dataset"):
            # print(df)
            gdf.to_csv(path_ + f"{l}-{dataset}-metrics.csv")


E2PLOTS = {
    "AC/BD-K": {  # E3
        "epoch": 99,
        "pairs": (("A", "C"), ("B", "D")),
        "metric": "cohens kappa",
        "labels": ("Fixed Learning Rate", "Decay Learning Rate"),
        "y_label": "Cohens $\\kappa$",
    },
    "AB-K": {  # E2
        "epoch": 99,
        "pairs": (("A", "B"),),
        "metric": "cohens kappa",
        "labels": ("Patched Affinity Computation", "Unpatched Affinity Computation"),
        "y_label": "Cohens $\\kappa$",
    },
    "AB-MCC": {  # E2
        "epoch": 99,
        "pairs": (("A", "B"),),
        "metric": "MCC",
        "labels": ("Patched Affinity Computation", "Unpatched Affinity Computation"),
        "y_label": "MCC",
    },
    "AB-F1": {  # E2
        "epoch": 99,
        "pairs": (("A", "B"),),
        "metric": "F1",
        "labels": ("Patched Affinity Computation", "Unpatched Affinity Computation"),
        "y_label": "F1 Score",
    },
}

if __name__ == "__main__":
    ARGS = parse_args()

    # fix_dir_mess()
    # exit()

    for dataset in ["Texas", "California"]:  # E2
        break
        for metric in ["K", "MCC", "F1"]:
            experiment_two_paired_boxplot(dataset=dataset, **E2PLOTS[f"AB-{metric}"])

    for dataset in ["Texas", "California"]:  # E3
        experiment_two_paired_boxplot(dataset=dataset, **E2PLOTS[f"AC/BD-K"])
    exit()

    # experiment_one_boxplot()
    # exit()
    # for key, value in E2PLOTS.items():
    #     experiment_two_paired_boxplot(dataset=ARGS.dataset, **value)
    # join_metrics()
    # exit()
    # experiment_one_boxplot()
    # exit()
    # experiment_loss_plot()
    # exit()
    # experiment_two_b_boxplot(wrong_one=True)
    # exit()
    experiment_three_boxplot()
    exit()

    TAG = "keep/Experiment 01/20191128-153016-Texas"
    FILEPATH = os.path.join(LOG_PATH, TAG, "metrics.csv")

    DF = pd.read_csv(FILEPATH)  # , index_col=[0, 1

    DF["model"] = DF["model"].apply(lambda s: f"\\m{{{s}}}")

    # DF.index.names = ["model", "timestamp"]
    DF["training time"] = pd.to_timedelta(DF["training time"], unit="s")

    boxplot_metrics(DF)
    plt.show()
