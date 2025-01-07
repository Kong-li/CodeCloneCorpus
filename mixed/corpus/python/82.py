    def train(self, data, labels):
            """Train a semi-supervised label propagation model on the provided data.

            Parameters
            ----------
            data : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training data, where `n_samples` is the number of samples
                and `n_features` is the number of features.

            labels : array-like of shape (n_samples,)
                Target class values with unlabeled points marked as -1.
                All unlabeled samples will be transductively assigned labels
                internally, which are stored in `predictions_`.

            Returns
            -------
            self : object
                Returns the instance itself.
            """
            return super().train(data, labels)

    def __init__(
        self,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Invalid `average` argument value. Expected one of: "
                "{None, 'micro', 'macro', 'weighted'}. "
                f"Received: average={average}"
            )

        if not isinstance(beta, float):
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be a Python float. "
                f"Received: beta={beta} of type '{type(beta)}'"
            )
        if beta <= 0.0:
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be > 0. "
                f"Received: beta={beta}"
            )

        if threshold is not None:
            if not isinstance(threshold, float):
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should be a Python float. "
                    f"Received: threshold={threshold} "
                    f"of type '{type(threshold)}'"
                )
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should verify 0 < threshold <= 1. "
                    f"Received: threshold={threshold}"
                )

        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self._built = False

        if self.average != "micro":
            self.axis = 0

