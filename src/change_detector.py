import os.path
from datetime import datetime

from tqdm import trange
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa


from image_translation import ImageTranslationNetwork
from filtering import threshold_otsu
from decorators import image_to_tensorboard, timed
from metrics import CohenKappa, MatthewsCorrelationCoefficient as MCC

# instead of "from tensorflow_addons.metrics import CohenKappa" due to
# https://github.com/tensorflow/addons/pull/675


class ChangeDetector:
    """docstring for ChangeDetector."""

    def __init__(
        self,
        learning_rate,
        clipnorm=None,
        logdir=None,
        evaluation_frequency=1,
        **kwargs,
    ):
        """
            Input:
                learning_rate=1e-5 - float, initial learning rate for
                                     ExponentialDecay
                clipnorm=None - gradient norm clip value, passed to
                                tf.clip_by_global_norm if not None
                logdir=None - path to log directory. If provided, tensorboard
                              logging of training and evaluation is set up at
                              'logdir/timestamp/' + 'train' and 'evaluation'
                evaluation_frequency=1 - int, rate at which images are sent to
                                         TensorBoard. If <= 0, the images are never
                                         sent.
        """
        self._optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.clipnorm = clipnorm

        # To keep a history for a specific training_metrics,
        # add `self.metrics_history[name] = []` in subclass __init__
        self.train_metrics = {}
        self.difference_img_metrics = {"AUC": tf.keras.metrics.AUC()}
        self.change_map_metrics = {
            "ACC": tf.keras.metrics.Accuracy(),
            "F1": tfa.metrics.F1Score(num_classes=1, average=None),
            "MCC": MCC(num_classes=1),
            "cohens kappa": CohenKappa(num_classes=2),
        }
        assert not set(self.difference_img_metrics) & set(self.change_map_metrics)
        # If the metric dictionaries shares keys, the history will not work
        self.metrics_history = {
            **{key: [] for key in self.change_map_metrics},
            **{key: [] for key in self.difference_img_metrics},
        }

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Flag used in image_to_tensorboard decorator
        self._save_images = tf.Variable(False, trainable=False)
        self._to_tensorboard = tf.Variable(True, trainable=False)

        if logdir is not None:
            self.log_path = os.path.join(logdir, self.timestamp)
            self.tb_writer = tf.summary.create_file_writer(self.log_path)
            self._image_dir = tf.constant(os.path.join(self.log_path, "images"))
        else:
            self.tb_writer = tf.summary.create_noop_writer()

        self.evaluation_frequency = tf.constant(evaluation_frequency, dtype=tf.int64)
        self.epoch = tf.Variable(0, dtype=tf.int64)

    @image_to_tensorboard(static_name=None)
    # @tf.function
    def _domain_difference_img(
        self, original, transformed, bandwidth=tf.constant(3, dtype=tf.float32)
    ):
        """
            Compute difference image in one domain between original image
            in that domain and the transformed image from the other domain.
            Bandwidth governs the norm difference clipping threshold
        """
        d = tf.norm(original - transformed, ord=2, axis=-1)
        threshold = tf.math.reduce_mean(d) + bandwidth * tf.math.reduce_std(d)
        d = tf.where(d < threshold, d, threshold)

        return tf.expand_dims(d / tf.reduce_max(d), -1)

    # @tf.function
    def _difference_img(self, x, y, x_hat, y_hat):
        """
        Should compute the two possible change maps and do the 5 method
        ensamble to output a final change-map?
        """
        assert x.shape[0] == y.shape[0] == 1, "Can not handle batch size > 1"

        d_x = self._domain_difference_img(x, x_hat, name="x_ut_diff")
        d_y = self._domain_difference_img(y, y_hat, name="y_ut_diff")

        # Weighted average based on the number of estimated channels
        c_x, c_y = x.shape[-1], y.shape[-1]
        d = (c_y * d_x + c_x * d_y) / (c_x + c_y)

        # Return expanded dims (rank = 4)?
        return d

    # @tf.function
    def _change_map(self, difference_img):
        """
            Input:
                difference_img - tensor of shape (h, w), (1, h, w)
                                 or (1, h, w, 1) in [0,1]
            Output:
                change_map - tensor with same shape as input, bool
        """
        tmp = tf.cast(difference_img * 255, tf.int32)
        threshold = tf.cast(threshold_otsu(tmp) / 255, tf.float32)

        return difference_img >= threshold

    @image_to_tensorboard(static_name="z_Confusion_map")
    # @tf.function
    def _confusion_map(self, target_change_map, change_map):
        """
            Compute RGB confusion map for the change map.
                True positive   - White  [1,1,1]
                True negative   - Black  [0,0,0]
                False positive  - Green  [0,1,0]
                False negative  - Red    [1,0,0]
        """
        conf_map = tf.concat(
            [
                target_change_map,
                change_map,
                tf.math.logical_and(target_change_map, change_map),
            ],
            axis=-1,
            name="confusion map",
        )

        return tf.cast(conf_map, tf.float32)

    def early_stopping_criterion(self):
        """
            To be implemented in subclasses.

            Called for each epoch epoch in training. If it returns True, training will
            be terminated.

            To keep a history for a metric in self.training_metrics,
            add `self.metrics_history[name] = []` in subclass __init__
        """
        return False

    @timed
    # @tf.function
    def train(
        self,
        training_dataset,
        epochs,
        batches,
        batch_size,
        evaluation_dataset=None,
        filter_=None,
        final_filter=None,
        **kwargs,
    ):
        """
            Inputs:
                training_dataset - tf.data.Dataset with tensors x, y, p
                    x - image of size (patch_size, patch_size, c_x)
                    y - image of size (patch_size, patch_size, c_y)
                    p - change prior of size (patch_size, patch_size, 1)
                epochs - int, number of training epochs
                batches - int, number of batches per epoch
                batch_size - int, number of samples per batch
                evaluation_dataset=None - tf.data.Dataset with tensors x, y, cm
                    x - image of size (h, w, c_x)
                    y - image of size (h, w, c_y)
                    cm - change map of size (h, w, 1)
                filter_=None - passed to evaluate if evaluation data is provided
                               Can be decorated with image_to_tensorboard
        """

        for epoch in trange(self.epoch.numpy(), epochs + self.epoch.numpy()):
            self.epoch.assign(epoch)
            # tf.summary.experimental.set_step(self.epoch)

            for _, batch in zip(range(batches), training_dataset.batch(batch_size)):
                self._train_step(*batch)
            self._track_training_metrics()

            if evaluation_dataset is not None:
                for eval_data in evaluation_dataset.batch(1):
                    ev_res = self.evaluate(*eval_data, filter_)

            # tf.summary.flush(self.tb_writer)
            # if self.early_stopping_criterion():
            #     break

        return self.epoch.numpy()

    def _track_training_metrics(self):
        # device is a workaround for github.com/tensorflow/tensorflow/issues/28007
        with tf.device("cpu:0"):
            with self.tb_writer.as_default():
                for name, metric in self.train_metrics.items():
                    tf.summary.scalar(name, metric.result())
                    try:
                        self.metrics_history[name].append(metric.result().numpy())
                    except KeyError as e:
                        pass
                    metric.reset_states()

    def evaluate(self, x, y, target_change_map, filter_=None, to_history=True):
        """
              Evaluate performance of the change detection scheme based on the
              image regressors. The metrics are computed for both an unfiltered
              and a filtered version of the produced change map.
              Input:
                  x - image tensor (h, w, c_x)
                  y - image tensor (h, w, c_y)
                  target_change_map - binary tensor (h, w). Ground truth
                                      indicating changes across the images
                  filter_=None - if provided, callable(self, x, y, difference_img)
                  to_history=True - whether to write the computed metrics to
                                    self.metrics_history and tensorboard
              Output:
                  change_map - image tensor (1, h, w, 1)
        """
        x_hat, y_hat, difference_img = self((x, y))
        if filter_ is not None:
            difference_img = filter_(self, x, y, difference_img)

        di_metrics = self._compute_metrics(
            target_change_map,
            difference_img,
            self.difference_img_metrics,
            to_history=to_history,
        )

        change_map = self._change_map(difference_img)
        cm_metrics = self._compute_metrics(
            target_change_map,
            change_map,
            self.change_map_metrics,
            to_history=to_history,
        )

        confusion_map = self._confusion_map(target_change_map, change_map)

        return confusion_map, {**di_metrics, **cm_metrics}

    def final_evaluate(self, evaluation_dataset, save_images, final_filter, **kwargs):
        """
            Call evaluate method wrapped with image saving logic

            Inputs:
                evaluation_dataset - tf.data.Dataset with tensors x, y, tcm
                    x - image of size (h, w, c_x)
                    y - image of size (h, w, c_y)
                    target_change_map - change map of size (h, w, 1)
                save_images=True - bool, wheter to store images after training
                final_filter - passed to evaluate. Can be None
                               Can be decorated with image_to_tensorboard
        """
        self.epoch.assign_add(1)
        tf.summary.experimental.set_step(self.epoch)
        self._save_images.assign(save_images)
        self._to_tensorboard.assign(False)
        for eval_data in evaluation_dataset.batch(1):
            _, metrics = self.evaluate(*eval_data, final_filter, to_history=False)
        self._save_images.assign(False)
        self._to_tensorboard.assign(True)
        tf.summary.flush(self.tb_writer)
        return metrics

    def _compute_metrics(self, y_true, y_pred, metrics, to_history=True):
        """
            Compute the metrics specified in metrics.
            Write results to self.tb_writer
            Input:
                y_true - tensor (n, )
                y_pred - tensor (n, )
                metrics - dict {name: tf.metrics class instance}
                to_history=True - whether to write the computed metrics to
                                  self.metrics_history and tensorboard
            Output:
                dict, contains the computed metrics
        """
        categoricals = ["F1", "MCC"]
        retval = {}

        for name, metric in metrics.items():
            if name in categoricals:
                y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1, 1])
                y_pred = tf.reshape(tf.cast(y_pred, tf.int32), [-1, 1])
            else:
                y_true, y_pred = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
            metric.update_state(y_true, y_pred)
            value = metric.result()
            if name in categoricals:
                value = value[0]
            retval[name] = value.numpy()

            if to_history:
                self.metrics_history[name].append(value.numpy())
            if self._to_tensorboard and to_history:
                # device is a workaround for github.com/tensorflow/tensorflow/issues/28007
                with tf.device("cpu:0"):
                    with self.tb_writer.as_default():
                        tf.summary.scalar(name, value)

            metric.reset_states()

        return retval

    def write_metric_history(self):
        """ Write the contents of metrics_history to file """
        for name, history in self.metrics_history.items():
            with open(self.log_path + "/" + name + ".txt", "w") as f:
                f.write(str(history))

    def save_model(self):
        print("ChangeDetector.save_model() is not implemented")


class ChangeDetectorMkII(ChangeDetector):
    """docstring for ChangeDetectorMkII."""

    def __init__(self, translation_spec, **kwargs):
        """
            Input:
                translation_spec - dict with keys 'f_X', 'f_Y'.
                                   Values are passed as kwargs to the
                                   respective ImageTranslationNetwork's
                learning_rate=1e-5 - float, initial learning rate for
                                     ExponentialDecay
                clipnorm=None - gradient norm clip value, passed to
                                tf.clip_by_global_norm if not None
                logdir=None - path to log directory. If provided, tensorboard
                              logging of training and evaluation is set up at
                              'logdir/timestamp/' + 'train' and 'evaluation'
        """
        super(ChangeDetectorMkII, self).__init__(**kwargs)
        self._fx = ImageTranslationNetwork(
            **translation_spec["f_X"], name="f_X", l2_lambda=1
        )
        self._fy = ImageTranslationNetwork(
            **translation_spec["f_Y"], name="f_Y", l2_lambda=1
        )

        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.fx_metrics = {key: tf.keras.metrics.Sum() for key in self.lambdas}
        self.fy_metrics = {key: tf.keras.metrics.Sum() for key in self.lambdas}

        self.metrics_history["fx_losses"] = []
        self.metrics_history["fy_losses"] = []

        self.writers = {
            key: tf.summary.create_file_writer(self.log_path + "/" + key)
            for key in self.lambdas
        }

    @image_to_tensorboard(static_name="y_hat")
    def fx(self, inputs, training=False):
        """ Wrapper for ImageTranslationNetwork.__call__ to TensorBoard it """
        return self._fx(inputs, training)

    @image_to_tensorboard(static_name="x_hat")
    def fy(self, inputs, training=False):
        """ Wrapper for ImageTranslationNetwork.__call__ to TensorBoard it """
        return self._fy(inputs, training)

    def _track_training_metrics(self):
        # device is a workaround for github.com/tensorflow/tensorflow/issues/28007
        # with tf.device("cpu:0"):
        #     for name, writer in self.writers.items():
        #         with writer.as_default():
        #             tf.summary.scalar("fx loss", self.fx_metrics[name].result())
        #             tf.summary.scalar("fy loss", self.fy_metrics[name].result())

        self.metrics_history["fx_losses"].append(
            {name: metric.result() for name, metric in self.fx_metrics.items()}
        )
        for name, metric in self.fx_metrics.items():
            metric.reset_states()

        self.metrics_history["fy_losses"].append(
            {name: metric.result() for name, metric in self.fy_metrics.items()}
        )
        for name, metric in self.fy_metrics.items():
            metric.reset_states()

    def write_metric_history(self):
        """ Write the contents of metrics_history to file """
        for name, history in self.metrics_history.items():
            df = pd.DataFrame(history)
            if df.shape[1] == 1:
                df = df.rename(columns={0: name})
            with open(self.log_path + "/" + name + ".csv", "w") as f:
                df.to_csv(f, index_label="epoch")
