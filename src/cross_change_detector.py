import tensorflow as tf
from change_detector import ChangeDetectorMkII


class CrossChangeDetector(ChangeDetectorMkII):
    def __init__(self, translation_spec, cross_lambda, reg_lambda, **kwargs):
        """
            Input:
                translation_spec - dict with keys 'f_X', 'f_Y'.
                                   Values are passed as kwargs to the
                                   respective ImageTranslationNetwork's
                cross_lambda - float, loss term scale weight
                reg_lambda - float, loss term scale weight

                kwargs as ChangeDetectorMkII
        """
        self.lambdas = {
            "cross": tf.constant(cross_lambda, dtype=tf.float32),
            "reg": tf.constant(reg_lambda, dtype=tf.float32),
        }
        super(CrossChangeDetector, self).__init__(translation_spec, **kwargs)

    # @tf.function
    def __call__(self, inputs, training=False):
        """
            Input:
                inputs - list or tuple (x, y)
                    x - tensor (None,h,w,c_x); image x
                    y - tensor (None,h,w,c_y); image y
                training=False - bool; indicates whether training face or
                                 evaluation face. As in tf.keras.Model docs.
            Output:
                when training=False:
                    outputs - list with
                        x_hat - tensor (None,h,w,c_x); transformed y
                        y_hat - tensor (None,h,w,c_y); transformed x
                        difference_img - tensor (None,h,w); pseudo probability map
                when training=True:
                    outputs - list with
                        x_hat - tensor (None,h,w,c_x); transformed y
                        y_hat - tensor (None,h,w,c_y); transformed x
        """
        x, y = inputs

        if training:
            x_hat, y_hat = self._fy(y, training), self._fx(x, training)

            retval = [x_hat, y_hat]
        else:
            x_hat, y_hat = self.fy(y), self.fx(x)
            difference_img = self._difference_img(x, y, x_hat, y_hat)

            retval = [x_hat, y_hat, difference_img]

        return retval

    # @tf.function
    def _train_step(self, x, y, clw):
        """
            Input:
                x - tensor of shape (bs, ps_h, ps_w, c_x)
                y - tensor of shape (bs, ps_h, ps_w, c_y)
                clw - cross_loss_weight, tensor of shape (bs, ps_h, ps_w, 1)
        """
        with tf.GradientTape() as tape:
            x_hat, y_hat = self((x, y), training=True)

            fx_loss = {
                "cross": self.loss_object(y, y_hat, clw),
                "reg": sum(self._fx.losses),  # regularization from submodel
            }
            fx_loss = {key: self.lambdas[key] * value for key, value in fx_loss.items()}
            fx_total_loss = sum(fx_loss.values())

            fy_loss = {
                "cross": self.loss_object(x, x_hat, clw),
                "reg": sum(self._fy.losses),  # regularization from submodel
            }
            fy_loss = {key: self.lambdas[key] * value for key, value in fy_loss.items()}
            fy_total_loss = sum(fy_loss.values())

            loss_value = fx_total_loss + fy_total_loss

        gradient_targets = self._fx.trainable_variables + self._fy.trainable_variables

        gradients = tape.gradient(loss_value, gradient_targets)
        if self.clipnorm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clipnorm)
        self._optimizer.apply_gradients(zip(gradients, gradient_targets))

        for key, value in fx_loss.items():
            self.fx_metrics[key].update_state(value)
        for key, value in fy_loss.items():
            self.fy_metrics[key].update_state(value)


if __name__ == "__main__":
    """
    1. Fetch data (x, y, change_map)
    2. Compute/estimate A_x and A_y (for patches)
    3. Compute change_prior
    4. Define dataset with (x, A_x, y, A_y, p). Choose patch size compatible
       with affinity computations.
    5. Train CrossCyclicImageTransformer unsupervised
        a. Evaluate the image transformations in some way?
    6. Evaluate the change detection scheme
        a. change_map = threshold [(x - f_y(y))/2 + (y - f_x(x))/2]
    """
    import datasets
    from config import get_config

    DATASET = "Texas"
    CONFIG = get_config(DATASET)
    LAMBDAS = {
        "cross_lambda": 1,  # weight for cross loss term
        "reg_lambda": 10e-4,  # weight for the reconstruction term
    }

    print(f"Loading {DATASET} data")
    TRAIN, EVALUATE, (C_X, C_Y) = datasets.fetch(DATASET, CONFIG["patch_size"])
    TRANSLATION_SPEC = config.get_ccd_like_translation_spec(C_X, C_Y)

    print("Initializing CrossChangeDetector")
    cd = CrossChangeDetector(TRANSLATION_SPEC, **LAMBDAS, **CONFIG)
    print("Training")
    cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)
    cd.final_evaluate(EVALUATE, **CONFIG)
