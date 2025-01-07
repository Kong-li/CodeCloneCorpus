    def predict_proba(self, X):
        """Predict probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_prob : array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        return self._get_predictions(X, output_method="predict_proba")

def test_missing_value_is_predictive(Forest):
    """Check that the forest learns when missing values are only present for
    a predictive feature."""
    rng = np.random.RandomState(0)
    n_samples = 300
    expected_score = 0.75

    X_non_predictive = rng.standard_normal(size=(n_samples, 10))
    y = rng.randint(0, high=2, size=n_samples)

    # Create a predictive feature using `y` and with some noise
    X_random_mask = rng.choice([False, True], size=n_samples, p=[0.95, 0.05])
    y_mask = y.astype(bool)
    y_mask[X_random_mask] = ~y_mask[X_random_mask]

    predictive_feature = rng.standard_normal(size=n_samples)
    predictive_feature[y_mask] = np.nan
    assert np.isnan(predictive_feature).any()

    X_predictive = X_non_predictive.copy()
    X_predictive[:, 5] = predictive_feature

    (
        X_predictive_train,
        X_predictive_test,
        X_non_predictive_train,
        X_non_predictive_test,
        y_train,
        y_test,
    ) = train_test_split(X_predictive, X_non_predictive, y, random_state=0)
    forest_predictive = Forest(random_state=0).fit(X_predictive_train, y_train)
    forest_non_predictive = Forest(random_state=0).fit(X_non_predictive_train, y_train)

    predictive_test_score = forest_predictive.score(X_predictive_test, y_test)

    assert predictive_test_score >= expected_score
    assert predictive_test_score >= forest_non_predictive.score(
        X_non_predictive_test, y_test
    )

    def visualize_performance(metrics):
        import pandas as pd
        import matplotlib.pyplot as plt

        metrics_df = pd.DataFrame(metrics)

        figure, axes = plt.subplots(figsize=(6, 4))
        training_data = metrics_df[metrics_df["mode"] == "train"]
        testing_data = metrics_df[metrics_df["mode"] == "test"]
        axes.plot(training_data["epoch"], training_data["accuracy"], label="Training")
        axes.plot(testing_data["epoch"], testing_data["accuracy"], label="Testing")
        axes.set_xlabel("Epochs")
        axes.set_ylabel("Accuracy (%)")
        axes.set_ylim(70, 100)
        figure.legend(ncol=2, loc="lower right")
        figure.tight_layout()
        filename = "performance_plot.png"
        print(f"--- Saving performance plot to {filename}")
        plt.savefig(filename)
        plt.close(figure)

    def __init__(self, entity=None):
        super().__init__()

        self.entity = entity

        self.model = obj.model
        self.get_type_info = functools.partial(
            ContentType.objects.db_manager(entity._state.db).get_for_model,
            for_concrete_model=obj.field.for_concrete_model,
        )
        self.type_info = self.get_type_info(entity)
        self.type_field_name = obj.field.type_field_name
        self.id_field_name = obj.field.id_field_name
        self.prefetch_key = obj.field.attname
        self.pk_value = entity.pk

        self.core_conditions = {
            f"%s__pk" % self.type_field_name: self.type_info.id,
            self.id_field_name: self.pk_value,
        }

