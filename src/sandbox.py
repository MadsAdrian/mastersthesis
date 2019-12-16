import numpy as np
import tensorflow as tf
from itertools import count
import datasets
import decorators


def gen(patch_size, dataset_size):
    def gen_():
        for i in range(dataset_size):
            yield np.arange(i * patch_size, (i + 1) * patch_size)

    return gen_


def foo(pre_cast):
    tensor = tf.constant(
        [
            [[False, False], [False, False]],
            [[True, False], [False, True]],
            [[True, True], [True, True]],
        ],
        dtype=tf.bool,
    )
    tensor = tf.expand_dims(tensor, -1)
    print(tensor)
    writer = tf.summary.create_file_writer("logs/test/")

    if pre_cast:
        tensor = tf.cast(tensor, tf.float32)
    with writer.as_default():
        tf.summary.image("bool tensor", tensor, step=0)


def test_image_patch_extraction():
    ch = 2
    patch_size = 7

    im1 = tf.reshape(tf.range(400), [2, 10, 10, ch])
    # print(im1)
    out = tf.image.extract_patches(
        images=im1,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    print(out.shape)

    im2 = tf.reshape(out, [-1, patch_size, patch_size, ch])
    k = 0
    for i in [0, 5]:
        for j in [0, 5]:
            if k < 5:
                print(im1[0, i : i + 5, j : j + 5, :] == im2[k])
            else:
                print(im1[1, i : i + 5, j : j + 5, :] == im2[k])
            k += 1

    # WORKS!
    # im2 = tf.reshape(out, [8, patch_size, patch_size, ch])
    # k = 0
    # for i in [0, 5]:
    #     for j in [0, 5]:
    #         if k < 5:
    #             print(im1[0, i : i + 5, j : j + 5, :] == im2[k])
    #         else:
    #             print(im1[1, i : i + 5, j : j + 5, :] == im2[k])
    #         k += 1


from filtering import threshold_otsu, _dense_gaussian_filtering


def icm_confusion_map(x, y, tcm, icm):
    # Filter difference image
    di = _dense_gaussian_filtering(
        tf.expand_dims(x, 0), tf.expand_dims(y, 0), tf.expand_dims(icm, 0)
    )

    # Compute change map
    tmp = tf.cast(di * 255, tf.int32)
    threshold = tf.cast(threshold_otsu(tmp) / 255, tf.float32)
    cm = di >= threshold

    # Compute confusion map
    tcm = tf.expand_dims(tcm, 0)
    conf_map = tf.concat([tcm, cm, tf.math.logical_and(tcm, cm)], axis=-1)

    return {"gauss_filtered_icm": di[0], "icm_confusion_map": conf_map[0]}


def save_input_images(dataset):
    x, y, target_cm = datasets.DATASETS[dataset](datasets.prepare_data[dataset])
    initial_cm = datasets.load_prior(dataset, x.shape[:2])
    data = {"x": x, "y": y, "tcm": target_cm, "icm": initial_cm}
    data.update(icm_confusion_map(**data))
    for name, image in data.items():
        print(image.dtype)
        if image.shape[-1] > 3:
            image = image[..., 1:4]
        if image.dtype == tf.bool:
            image = tf.cast(image, tf.float32)
        with tf.device("cpu:0"):
            image = decorators._change_image_range(image)
            image = tf.cast(255 * image, tf.uint8)
            contents = tf.image.encode_png(image)
            tf.io.write_file(f"logs/{dataset}/{name}.png", contents)


if __name__ == "__main__":
    # PATCH_SIZE, DATASET_SIZE, BATCH_SIZE = 2, 10, 4
    #
    # g = gen(PATCH_SIZE, DATASET_SIZE)
    # ds = tf.data.Dataset.from_generator(g, (tf.int32))
    #
    # for d in ds.batch(BATCH_SIZE):
    #     print(d)
    for dataset in ["California"]:  # datasets.DATASETS.keys():
        print(dataset, "to .png")
        save_input_images(dataset)
