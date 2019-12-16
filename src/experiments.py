""" Run experiments for master thesis """
import os
import os.path
import errno
import argparse
import json
import shelve

from itertools import count
from datetime import datetime
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay

import datasets
import config

from filtering import decorated_median_filter, decorated_gaussian_filter

from affinity_guided_change_detector import AffinityGuidedChangeDetector
from affinity_guided_cross_change_detector import (
    UnpatchedAffinityGuidedCrossChangeDetector,
    PatchedAffinityGuidedCrossChangeDetector,
)
from affinity_guided_cyclic_change_detector import (
    PatchedAffinityGuidedCyclicChangeDetector,
    UnpatchedAffinityGuidedCyclicChangeDetector,
)
from cross_change_detector import CrossChangeDetector
from cyclic_change_detector import CyclicChangeDetector


NONAFFINITY_MODELS = {
    # Comment to keep black from squishing
    "-C-": CyclicChangeDetector,
    "--X": CrossChangeDetector,
    "-CX": CyclicChangeDetector,
}
AFFINITY_MODELS = {
    "A--": AffinityGuidedChangeDetector,
    "AC-": UnpatchedAffinityGuidedCrossChangeDetector,
    "A-X": UnpatchedAffinityGuidedCrossChangeDetector,
    "ACX": UnpatchedAffinityGuidedCyclicChangeDetector,
    "A-Xp": PatchedAffinityGuidedCrossChangeDetector,
    "ACXp": PatchedAffinityGuidedCyclicChangeDetector,
}

LAMBDAS = {
    "--X": {"cross_lambda": 1, "reg_lambda": 1e-5},
    "-CX": {"cross_lambda": 0.8, "cycle_lambda": 1, "reg_lambda": 1e-5},
    "A-X": {"cross_lambda": 1, "aff_lambda": 1, "reg_lambda": 1e-5},
    "A-Xp": {"cross_lambda": 1, "aff_lambda": 1, "reg_lambda": 1e-5},
    "ACX": {
        "cross_lambda": 0.8,
        "cycle_lambda": 1,
        "reg_lambda": 1e-5,
        "aff_lambda": 0.8,
    },
    "ACXp": {
        "cross_lambda": 0.8,
        "cycle_lambda": 1,
        "reg_lambda": 1e-5,
        "aff_lambda": 0.8,
    },
    "A--": {"aff_lambda": 1, "reg_lambda": 1e-5},
    "-C-": {"cross_lambda": 0, "cycle_lambda": 1, "reg_lambda": 1e-5},
    "AC-": {
        "cross_lambda": 0,
        "cycle_lambda": 1,
        "reg_lambda": 1e-5,
        "aff_lambda": 0.8,
    },
}

METRICS_COLUMNS = [
    "dataset",
    "model",
    "run",
    "timestamp",
    "epoch",
    "training time",
    "AUC",
    "ACC",
    "F1",
    "MCC",
    "cohens kappa",
]


def append_to_csv(data_dict):
    """ The keys of data_dict should correspond to METRICS_COLUMNS """
    df = pd.DataFrame([data_dict], columns=METRICS_COLUMNS)
    with open(METRICS, "a") as f:
        df.to_csv(f, header=(f.tell() == 0))


def lambdas_table():
    """ Create latex table from LABMDAS dict """
    df = pd.DataFrame(LAMBDAS)
    print(df.to_latex(na_rep="$\\cdot$", bold_rows=True, float_format="{:0.1f}".format))


def models_run(models):
    """ Perform RUNS runs of the models in models """
    tr_data, ev_data, _ = datasets.fetch(DATASET, CONFIG["patch_size"])
    for name, model in models.items():
        m_logdir = os.path.join(LOGDIR, name)
        for run in range(ARGS.runs):
            if CONFIG["decay_learning_rate"]:
                CONFIG["learning_rate"] = ExponentialDecay(
                    CONFIG["initial_learning_rate"],
                    decay_steps=10000,
                    decay_rate=0.96,
                    staircase=True,
                )
            tf.print(f"Run {run+1} of {name}:")
            total_training_time = 0
            change_detector = model(
                translation_spec=TRANSLATION_SPEC,
                logdir=m_logdir,
                **LAMBDAS[name],
                **CONFIG,
            )
            change_detector._to_tensorboard.assign(False)
            for epochs in CONFIG["epochs_list"]:
                training_time, epoch = change_detector.train(
                    tr_data, evaluation_dataset=ev_data, epochs=epochs, **CONFIG
                )
                total_training_time += training_time
                metrics = change_detector.final_evaluate(ev_data, **CONFIG)

                append_to_csv(
                    {
                        "dataset": DATASET,
                        "model": name,
                        "run": run,
                        "timestamp": change_detector.timestamp,
                        "epoch": epoch,
                        "training time": total_training_time,
                        **metrics,
                    }
                )
            change_detector.write_metric_history()
            tf.keras.backend.clear_session()
            tf.print(
                f"Trained {epoch} epochs in {total_training_time}",
                f"Final kappa = {metrics['cohens kappa']}",
                sep="\n\t",
                end="\n\n### ### ###\n",
            )


def make_dir(path):
    """ Make directory if it does not exist """
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise
    return path


def config_to_shelf():
    """ Write the experiment configuration to the shelf """
    exclude = ["filter_", "final_filter", "logdir"]
    out = {
        "dataset": DATASET,
        "timestamp": TIMESTAMP,
        "runs": ARGS.runs,
        **{key: value for key, value in CONFIG.items() if key not in exclude},
        **{key: value for key, value in LAMBDAS.items() if key in MODELS},
        "nonaffinity_config": NM_CONFIG,
        "affinity_config": AM_CONFIG,
        "translation_spec": TRANSLATION_SPEC,
    }

    with shelve.open(SHELVE_PATH) as shelf:
        shelf["config"] = out


def config_to_json():
    """ Write the config to .json"""
    with shelve.open(SHELVE_PATH) as shelf:
        for i in count():
            try:
                with open(LOGDIR + f"config{i}.json", "x") as file:
                    json.dump(shelf["config"], file, indent=4)
            except FileExistsError as error:
                pass
            else:
                tf.print(f"Wrote {LOGDIR}config{i}.json")
                break


def write_note(note):
    """ Write experiment note to shelf and file """
    with shelve.open(SHELVE_PATH, writeback=True) as shelf:
        shelf["config"]["note"] = note
    with open(LOGDIR + "notes.txt", "a") as file:
        file.write(note)


def shelf_setup():
    """ Create shelf dir and initialize 'evaluations' key  """
    path = make_dir(LOGDIR + "shelf/") + "experiment"
    with shelve.open(path, writeback=True) as shelf:
        shelf["evaluations"] = []
    return path


def write_dataframe():
    """ Write shelf['evaluations'] to metrics.csv """
    with shelve.open(SHELVE_PATH) as shelf:
        df = pd.DataFrame(shelf["evaluations"])
    tf.print(df)
    with open(LOGDIR + "metrics.csv", "x") as file:
        df.to_csv(file, index=False)


def parse_arguments():
    """ Parse commandline arguments """
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="Texas",
        help=f"Dataset to use. str in {datasets.DATASETS.keys()}",
    )
    parser.add_argument("--tag", required=True, help="configtag A, B, C or D")
    parser.add_argument("--runs", "-r", type=int, default=1, help="Number of runs")
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        nargs="+",
        required=True,
        help="Number of epochs, sequence of ints",
    )
    parser.add_argument("--logdir", "-l", type=str, help="Experiment logdir")

    parser.add_argument(
        "--debug", type=bool, default=False, help="reduce dimensions if True"
    )
    parser.add_argument("--notes", "-n", type=str, help="Note to save in config")
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_arguments()

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    DATASET = ARGS.dataset
    if ARGS.logdir is not None:
        LOGDIR = make_dir(os.path.join("logs", ARGS.logdir)) + "/"
    else:
        LOGDIR = make_dir(f"logs/{TIMESTAMP}-{DATASET}-{ARGS.tag}/")

    MODELS, E_CONFIG, NM_CONFIG, AM_CONFIG = config.experiment_config(ARGS.tag)
    CONFIG = {
        "clipnorm": 1,  # threshold for gradient clipping
        "save_images": True,  # bool, wheter to store images after training
        "filter_": decorated_median_filter("z_median_filtered_diff"),
        "final_filter": decorated_gaussian_filter("z_gaussian_filtered_diff"),
        "epochs_list": ARGS.epochs,
        **E_CONFIG,
    }

    tf.print(
        f"Performing experiment with {ARGS.runs} runs",
        f"of {CONFIG['epochs_list']} epochs:\n",
        f"\tThe logdir is {LOGDIR}",
        end="\n\n",
    )

    SHELVE_PATH = shelf_setup()
    METRICS = LOGDIR + "metrics.csv"

    TRANSLATION_SPEC = config.get_ccd_like_translation_spec(DATASET, debug=ARGS.debug)
    config_to_shelf()
    if ARGS.notes is not None:
        write_note(ARGS.notes)
    config_to_json()

    TMP = set(MODELS).intersection(set(NONAFFINITY_MODELS.keys()))
    if TMP:
        CONFIG.update(NM_CONFIG)
        tf.print(
            f"Running {TMP} on {DATASET} data with",
            f"patch size {CONFIG['patch_size']}",
            f"batch_size {CONFIG['batch_size']}",
            f"batches {CONFIG['batches']}",
            f"affinity_patch_size {CONFIG['affinity_patch_size']}",
        )
        TO_RUN = {
            key: value for key, value in NONAFFINITY_MODELS.items() if key in MODELS
        }
        models_run(TO_RUN)

    TMP = set(MODELS).intersection(set(AFFINITY_MODELS.keys()))
    if TMP:
        CONFIG.update(AM_CONFIG)
        tf.print(
            f"Running {TMP} on {DATASET} data with",
            f"patch size {CONFIG['patch_size']}",
            f"batch_size {CONFIG['batch_size']}",
            f"batches {CONFIG['batches']}",
            f"affinity_patch_size {CONFIG['affinity_patch_size']}",
        )
        TO_RUN = {key: value for key, value in AFFINITY_MODELS.items() if key in MODELS}
        models_run(TO_RUN)

    # write_dataframe()
    tf.print(
        f"### Experiment {ARGS.tag} on {DATASET} started at {TIMESTAMP}",
        f"completed at {datetime.now().strftime('%Y%m%d-%H%M%S')} ###",
    )
