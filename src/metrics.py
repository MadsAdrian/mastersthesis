# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
    Code is copy-pasted from tensorflow_addons
    https://github.com/tensorflow/addons/tree/master/tensorflow_addons/metrics
    as changes (#483 and #675) from the nightly branch was needed.
"""
import numpy as np
import tensorflow.keras.backend as K
from tensorflow_addons.utils import keras_utils
import tensorflow as tf


class CohenKappa(tf.keras.metrics.Metric):
    """Computes Kappa score between two raters.
    The score lies in the range [-1, 1]. A score of -1 represents
    complete disagreement between two raters whereas a score of 1
    represents complete agreement between the two raters.
    A score of 0 means agreement by chance.
    Note: As of now, this implementation considers all labels
    while calculating the Cohen's Kappa score.
    Usage:
    ```python
    actuals = np.array([4, 4, 3, 4, 2, 4, 1, 1], dtype=np.int32)
    preds = np.array([4, 4, 3, 4, 4, 2, 1, 1], dtype=np.int32)
    weights = np.array([1, 1, 2, 5, 10, 2, 3, 3], dtype=np.int32)

    m = tfa.metrics.CohenKappa(num_classes=5)
    m.update_state(actuals, preds)
    print('Final result: ', m.result().numpy()) # Result: 0.61904764

    # To use this with weights, sample_weight argument can be used.
    m = tfa.metrics.CohenKappa(num_classes=5)
    m.update_state(actuals, preds, sample_weight=weights)
    print('Final result: ', m.result().numpy()) # Result: 0.37209308
    ```
    Usage with tf.keras API:
    ```python
    model = keras.models.Model(inputs, outputs)
    model.add_metric(tfa.metrics.CohenKappa(num_classes=5)(outputs))
    model.compile('sgd', loss='mse')
    ```
    Args:
      num_classes : Number of unique classes in your dataset
      weightage   : Weighting to be considered for calculating
                    kappa statistics. A valid value is one of
                    [None, 'linear', 'quadratic']. Defaults to None.
    Returns:
      kappa_score : float
        The kappa statistic, which is a number between -1 and 1. The maximum
        value means complete agreement; zero or lower means chance agreement.
    Raises:
      ValueError: If the value passed for `weightage` is invalid
        i.e. not any one of [None, 'linear', 'quadratic']
    """

    def __init__(
        self, num_classes, name="cohen_kappa", weightage=None, dtype=tf.float32
    ):
        super(CohenKappa, self).__init__(name=name, dtype=dtype)

        if weightage not in (None, "linear", "quadratic"):
            raise ValueError("Unknown kappa weighting type.")
        else:
            self.weightage = weightage

        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(
            "conf_mtx",
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.int64,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix condition statistics.
        Args:
          y_true : array, shape = [n_samples]
                   Labels assigned by the first annotator.
          y_pred : array, shape = [n_samples]
                   Labels assigned by the second annotator. The kappa statistic
                   is symmetric, so swapping ``y_true`` and ``y_pred`` doesn't
                   change the value.
          sample_weight(optional) : for weighting labels in confusion matrix
                   Default is None. The dtype for weights should be the same
                   as the dtype for confusion matrix. For more details,
                   please check tf.math.confusion_matrix.
        Returns:
          Update op.
        """
        y_true = tf.cast(y_true, dtype=tf.int64)
        y_pred = tf.cast(y_pred, dtype=tf.int64)

        if y_true.shape != y_pred.shape:
            raise ValueError("Number of samples in y_true and y_pred are different")

        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            dtype=tf.dtypes.int64,
            weights=sample_weight,
        )

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)

    def result(self):
        nb_ratings = tf.shape(self.conf_mtx)[0]
        weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.int64)

        # 2. Create a weight matrix
        if self.weightage is None:
            diagonal = tf.zeros([nb_ratings], dtype=tf.int64)
            weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

        else:
            weight_mtx += tf.range(nb_ratings, dtype=tf.int64)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

            if self.weightage == "linear":
                weight_mtx = tf.abs(weight_mtx - tf.transpose(weight_mtx))
            else:
                weight_mtx = tf.pow((weight_mtx - tf.transpose(weight_mtx)), 2)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

        # 3. Get counts
        actual_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=1)
        pred_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=0)

        # 4. Get the outer product
        out_prod = pred_ratings_hist[..., None] * actual_ratings_hist[None, ...]

        # 5. Normalize the confusion matrix and outer product
        conf_mtx = self.conf_mtx / tf.reduce_sum(self.conf_mtx)
        out_prod = out_prod / tf.reduce_sum(out_prod)

        conf_mtx = tf.cast(conf_mtx, dtype=tf.float32)
        out_prod = tf.cast(out_prod, dtype=tf.float32)

        # 6. Calculate Kappa score
        numerator = tf.reduce_sum(conf_mtx * weight_mtx)
        denominator = tf.reduce_sum(out_prod * weight_mtx)
        kp = 1 - (numerator / denominator)
        return kp

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {"num_classes": self.num_classes, "weightage": self.weightage}
        base_config = super(CohenKappa, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        """Resets all of the metric state variables."""

        for v in self.variables:
            K.set_value(v, np.zeros((self.num_classes, self.num_classes), np.int64))


class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    """Computes the Matthews Correlation Coefficient.
    The statistic is also known as the phi coefficient.
    The Matthews correlation coefficient (MCC) is used in
    machine learning as a measure of the quality of binary
    and multiclass classifications. It takes into account
    true and false positives and negatives and is generally
    regarded as a balanced measure which can be used even
    if the classes are of very different sizes. The correlation
    coefficient value of MCC is between -1 and +1. A
    coefficient of +1 represents a perfect prediction,
    0 an average random prediction and -1 an inverse
    prediction. The statistic is also known as
    the phi coefficient.
    MCC = (TP * TN) - (FP * FN) /
          ((TP + FP) * (TP + FN) * (TN + FP ) * (TN + FN))^(1/2)
    Usage:
    ```python
    actuals = tf.constant([[1.0], [1.0], [1.0], [0.0]],
             dtype=tf.float32)
    preds = tf.constant([[1.0], [0.0], [1.0], [1.0]],
             dtype=tf.float32)
    # Matthews correlation coefficient
    mcc = MatthewsCorrelationCoefficient(num_classes=1)
    mcc.update_state(actuals, preds)
    print('Matthews correlation coefficient is:',
    mcc.result().numpy())
    # Matthews correlation coefficient is : -0.33333334
    ```
    """

    def __init__(
        self, num_classes=None, name="MatthewsCorrelationCoefficient", dtype=tf.float32
    ):
        """Creates a Matthews Correlation Coefficient instanse.
        Args:
            num_classes : Number of unique classes in the dataset.
            name: (Optional) String name of the metric instance.
            dtype: (Optional) Data type of the metric result.
            Defaults to `tf.float32`.
        """
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.true_positives = self.add_weight(
            "true_positives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.false_negatives = self.add_weight(
            "false_negatives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.true_negatives = self.add_weight(
            "true_negatives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )

    # TODO: sample_weights
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=self.dtype)
        y_pred = tf.cast(y_pred, dtype=self.dtype)

        true_positive = tf.math.count_nonzero(y_true * y_pred, 0)
        # true_negative
        y_true_negative = tf.math.not_equal(y_true, 1.0)
        y_pred_negative = tf.math.not_equal(y_pred, 1.0)
        true_negative = tf.math.count_nonzero(
            tf.math.logical_and(y_true_negative, y_pred_negative), axis=0
        )
        # predicted sum
        pred_sum = tf.math.count_nonzero(y_pred, 0)
        # Ground truth label sum
        true_sum = tf.math.count_nonzero(y_true, 0)
        false_positive = pred_sum - true_positive
        false_negative = true_sum - true_positive

        # true positive state_update
        self.true_positives.assign_add(tf.cast(true_positive, self.dtype))
        # false positive state_update
        self.false_positives.assign_add(tf.cast(false_positive, self.dtype))
        # false negative state_update
        self.false_negatives.assign_add(tf.cast(false_negative, self.dtype))
        # true negative state_update
        self.true_negatives.assign_add(tf.cast(true_negative, self.dtype))

    def result(self):
        # numerator
        numerator1 = self.true_positives * self.true_negatives
        numerator2 = self.false_positives * self.false_negatives
        numerator = numerator1 - numerator2
        # denominator
        denominator1 = self.true_positives + self.false_positives
        denominator2 = self.true_positives + self.false_negatives
        denominator3 = self.true_negatives + self.false_positives
        denominator4 = self.true_negatives + self.false_negatives
        denominator = tf.math.sqrt(
            denominator1 * denominator2 * denominator3 * denominator4
        )
        mcc = tf.math.divide_no_nan(numerator, denominator)
        return mcc

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {"num_classes": self.num_classes}
        base_config = super(MatthewsCorrelationCoefficient, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        """Resets all of the metric state variables."""
        self.true_positives.assign(tf.zeros((self.num_classes), self.dtype))
        self.false_positives.assign(tf.zeros((self.num_classes), self.dtype))
        self.false_negatives.assign(tf.zeros((self.num_classes), self.dtype))
        self.true_negatives.assign(tf.zeros((self.num_classes), self.dtype))
