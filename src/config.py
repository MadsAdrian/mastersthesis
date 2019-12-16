from filtering import decorated_median_filter, decorated_gaussian_filter
import tensorflow as tf
from datasets import channels


def get_config(dataset_name, debug=False):
    CONFIG = {
        "clipnorm": 1,  # threshold for gradient clipping
        "logdir": f"logs/{dataset_name}/",  # logdir for tensorboard. Can be None
        "save_images": True,  # bool, wheter to store images after training
        "filter_": decorated_median_filter("z_median_filtered_diff"),
        "final_filter": decorated_gaussian_filter("z_gaussian_filtered_diff"),
    }

    if not debug and tf.test.is_gpu_available():
        CONFIG.update(
            {
                "batches": 10,  # number of batches per epoch
                "batch_size": 10,  # number of samples per batch
                "patch_size": 100,  # square size of patches extracted for training
                "evaluation_frequency": 0,
                "affinity_patch_size": 32,  # patch size for affinity grid
            }
        )
        affinity_config = {"patch_size": 50, "batch_size": 2, "batches": 40}
        nonaffinity_config = {"patch_size": 128, "batch_size": 1, "batches": 5}
        affinity_config = nonaffinity_config = {
            "patch_size": 128,
            "batch_size": 1,
            "batches": 12,
        }

    else:
        CONFIG.update(
            {
                "batches": 2,  # number of batches per epoch
                "batch_size": 2,  # number of samples per batch
                "patch_size": 10,  # square size of patches extracted for training
                "affinity_patch_size": 4,  # patch size for affinity grid
            }
        )
        affinity_config = nonaffinity_config = {
            "patch_size": 12,
            "batch_size": 2,
            "batches": 2,
        }

    return CONFIG, affinity_config, nonaffinity_config


def get_prior(debug=False):
    prior_computation_config = {}
    if not debug and tf.test.is_gpu_available():
        prior_computation_config = {
            "affinity_batch_size": 250,  # batch size for prior computation
            "affinity_patch_size": 20,  # patch size for prior computation
            "affinity_stride": 5,  # stride for prior computation
        }
    else:
        prior_computation_config = {
            "affinity_batch_size": 10,  # batch size for prior computation
            "affinity_patch_size": 20,  # patch size for prior computation
            "affinity_stride": 5,  # stride for prior computation
        }
    return prior_computation_config


def _experiment_A():
    models = ["-CX", "A-Xp", "ACXp"]
    config = {
        "evaluation_frequency": 0,
        "affinity_patch_size": 32,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "decay_learning_rate": False,
    }
    am_config = nm_config = {"patch_size": 128, "batches": 8}  # 524288 pixels/epoch

    am_pixels = (
        (am_config["patch_size"] ** 2) * config["batch_size"] * am_config["batches"]
    )
    nm_pixels = (
        (nm_config["patch_size"] ** 2) * config["batch_size"] * nm_config["batches"]
    )
    assert am_pixels == nm_pixels == 2 ** 18

    return models, config, nm_config, am_config


def _experiment_B():
    models = ["-CX", "A-X", "ACX"]
    config = {
        "evaluation_frequency": 0,
        "affinity_patch_size": False,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "decay_learning_rate": False,
    }
    am_config = {"patch_size": 64, "batches": 32}  # 524288 pixels/epoch
    nm_config = {"patch_size": 128, "batches": 8}  # 524288 pixels/epoch

    am_pixels = (
        (am_config["patch_size"] ** 2) * config["batch_size"] * am_config["batches"]
    )
    nm_pixels = (
        (nm_config["patch_size"] ** 2) * config["batch_size"] * nm_config["batches"]
    )
    assert am_pixels == nm_pixels == 2 ** 18
    return models, config, nm_config, am_config


def _experiment_C():
    models = ["-CX", "A-Xp", "ACXp"]
    config = {
        "evaluation_frequency": 0,
        "affinity_patch_size": 32,
        "batch_size": 2,
        "initial_learning_rate": 1e-4,
        "decay_learning_rate": True,
    }
    am_config = nm_config = {"patch_size": 128, "batches": 8}  # 524288 pixels/epoch

    am_pixels = (
        (am_config["patch_size"] ** 2) * config["batch_size"] * am_config["batches"]
    )
    nm_pixels = (
        (nm_config["patch_size"] ** 2) * config["batch_size"] * nm_config["batches"]
    )
    assert am_pixels == nm_pixels == 2 ** 18
    return models, config, nm_config, am_config


def _experiment_D():
    models = ["-CX", "A-X", "ACX"]
    config = {
        "evaluation_frequency": 0,
        "affinity_patch_size": False,
        "batch_size": 2,
        "initial_learning_rate": 1e-4,
        "decay_learning_rate": True,
    }
    am_config = {"patch_size": 64, "batches": 32}  # 524288 pixels/epoch
    nm_config = {"patch_size": 128, "batches": 8}  # 524288 pixels/epoch

    am_pixels = (
        (am_config["patch_size"] ** 2) * config["batch_size"] * am_config["batches"]
    )
    nm_pixels = (
        (nm_config["patch_size"] ** 2) * config["batch_size"] * nm_config["batches"]
    )
    assert am_pixels == nm_pixels == 2 ** 18
    return models, config, nm_config, am_config


EXPERIMENTS = {
    "A": _experiment_A,
    "B": _experiment_B,
    "C": _experiment_C,
    "D": _experiment_D,
}


def experiment_config(name):
    return EXPERIMENTS[name]()


def get_ccd_like_translation_spec(dataset, debug=False):
    c_x, c_y = channels(dataset)
    if not debug and tf.test.is_gpu_available():
        translation_spec = {
            "f_X": {"input_chs": c_x, "filter_spec": [100, 50, 20, c_y]},
            "f_Y": {"input_chs": c_y, "filter_spec": [100, 50, 20, c_x]},
        }
    else:
        translation_spec = {
            "f_X": {"input_chs": c_x, "filter_spec": [c_y]},
            "f_Y": {"input_chs": c_y, "filter_spec": [c_x]},
        }
    return translation_spec
