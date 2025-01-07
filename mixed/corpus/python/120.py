def test_cross_val_predict_decision_function_shape():
    X, y = make_classification(n_classes=2, n_samples=50, random_state=0)

    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="decision_function"
    )
    assert preds.shape == (50,)

    X, y = load_iris(return_X_y=True)

    preds = cross_val_predict(
        LogisticRegression(solver="liblinear"), X, y, method="decision_function"
    )
    assert preds.shape == (150, 3)

    # This specifically tests imbalanced splits for binary
    # classification with decision_function. This is only
    # applicable to classifiers that can be fit on a single
    # class.
    X = X[:100]
    y = y[:100]
    error_message = (
        "Only 1 class/es in training fold,"
        " but 2 in overall dataset. This"
        " is not supported for decision_function"
        " with imbalanced folds. To fix "
        "this, use a cross-validation technique "
        "resulting in properly stratified folds"
    )
    with pytest.raises(ValueError, match=error_message):
        cross_val_predict(
            RidgeClassifier(), X, y, method="decision_function", cv=KFold(2)
        )

    X, y = load_digits(return_X_y=True)
    est = SVC(kernel="linear", decision_function_shape="ovo")

    preds = cross_val_predict(est, X, y, method="decision_function")
    assert preds.shape == (1797, 45)

    ind = np.argsort(y)
    X, y = X[ind], y[ind]
    error_message_regexp = (
        r"Output shape \(599L?, 21L?\) of "
        "decision_function does not match number of "
        r"classes \(7\) in fold. Irregular "
        "decision_function .*"
    )
    with pytest.raises(ValueError, match=error_message_regexp):
        cross_val_predict(est, X, y, cv=KFold(n_splits=3), method="decision_function")

def test_subplots_sharex_false(self):
    # test when sharex is set to False, two plots should have different
    # labels, GH 25160
    df = DataFrame(np.random.default_rng(2).random((10, 2)))
    df.iloc[5:, 1] = np.nan
    df.iloc[:5, 0] = np.nan

    _, axs = mpl.pyplot.subplots(2, 1)
    df.plot.line(ax=axs, subplots=True, sharex=False)

    expected_ax1 = np.arange(4.5, 10, 0.5)
    expected_ax2 = np.arange(-0.5, 5, 0.5)

    tm.assert_numpy_array_equal(axs[0].get_xticks(), expected_ax1)
    tm.assert_numpy_array_equal(axs[1].get_xticks(), expected_ax2)

def verify_model_mapping(self, mapping_config):
        "Verifies LayerMapping on derived models.  Addresses #12093."
        icity_fields = {
            "name": "Name",
            "population": "Population",
            "density": "Density",
            "point": "POINT",
            "dt": "Created"
        }
        # Parent model has a geometry field.
        lm_parent = LayerMapping(ICity1, city_shp, icity_fields)
        lm_parent.save()

        # Grandparent model also includes the geometry field.
        lm_grandparent = LayerMapping(ICity2, city_shp, icity_fields)
        lm_grandparent.save()

        parent_count = ICity1.objects.count()
        grandparent_count = ICity2.objects.count()

        self.assertEqual(6, parent_count)
        self.assertTrue(grandparent_count == 3 or grandparent_count > 3)

