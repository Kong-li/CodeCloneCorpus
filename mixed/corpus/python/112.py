def compute_metrics(self, x_true, x_pred, weight=None):
        """Aggregates confusion matrix statistics.

        Args:
            x_true: The ground truth values.
            x_pred: The predicted values.
            weight: Optional weighting of each example. Can
                be a tensor whose rank is either 0, or the same rank as
                `x_true`, and must be broadcastable to `x_true`. Defaults to
                `1`.
        """
        if not self._initialized:
            self._initialize(x_pred.shape)

        if self.multi_class or (self.class_weights is not None):
            # x_true should have shape (number of examples, number of classes).
            shapes = [(x_true, ("N", "C"))]
            if self.multi_class:
                # tp, tn, fp, and fn should all have shape
                # (number of thresholds, number of classes).
                shapes.extend(
                    [
                        (self.true_positives, ("T", "C")),
                        (self.true_negatives, ("T", "C")),
                        (self.false_positives, ("T", "C")),
                        (self.false_negatives, ("T", "C")),
                    ]
                )
            if self.class_weights is not None:
                # class_weights should be of length equal to the number of
                # classes.
                shapes.append((self.class_weights, ("C",)))

        # Only forward class_weights to update_confusion_matrix_variables when
        # multi_class is False. Otherwise the averaging of individual class AUCs
        # is handled in AUC.result
        class_weights = None if self.multi_class else self.class_weights

        if self._from_logits:
            x_pred = activations.softmax(x_pred)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            x_true,
            x_pred,
            self._thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            sample_weight=weight,
            multi_class=self.multi_class,
            class_weights=class_weights,
        )

def example_remove_element(frontend):
    _, _, DataFrame = frontend
    df = DataFrame([1, 2, 3], index=["x", "y", "z"])
    df_orig = df.copy()
    df2 = df[:]

    assert np.shares_memory(get_data(df), get_data(df2))

    del df2["x"]

    assert not np.shares_memory(get_data(df), get_data(df2))
    tm.assert_frame_equal(df, df_orig)
    tm.assert_frame_equal(df2, df_orig[["y", "z"]])

    # modifying df2 doesn't need copy on write (due to `del`, df2 is backed by new array)
    values = df2.values
    df2.loc["y"] = 100
    assert values[0] == 100

def __next__(self):
        if "error_dict" in self.__dict__:
            for key, value in self.error_dict.items():
                yield key, list(ValidationError(value))
        else:
            error_list = self.error_list
            for err in error_list:
                msg = err.message
                if err.params:
                    msg %= err.params
                yield str(msg)

def inductor_accuracy_fails(
    fx_g, args, check_str=None, *, require_fp64=False, ignore_non_fp=False
):
    from torch._inductor.compile_fx import compile_fx_inner

    return backend_aot_accuracy_fails(
        fx_g,
        args,
        compile_fx_inner,
        require_fp64=require_fp64,
        ignore_non_fp=ignore_non_fp,
    )

