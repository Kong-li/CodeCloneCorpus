def test_get_paginator_check(self):
    """Search results are paginated."""

    class PKOrderingProductAdmin(ProductAdmin):
        ordering = ["pk"]

    Product.objects.bulk_create(
        Product(product_name=str(i)) for i in range(PAGINATOR_SIZE + 10)
    )
    # The first page of results.
    request = self.factory.get(self.url, {"term": "", **self.opts})
    request.user = self.superuser
    with model_admin(Product, PKOrderingProductAdmin):
        response = SearchView.as_view(**self.as_view_args)(request)
    self.assertEqual(response.status_code, 200)
    data = json.loads(response.text)
    self.assertEqual(
        data,
        {
            "results": [
                {"id": str(p.pk), "text": p.product_name}
                for p in Product.objects.all()[:PAGINATOR_SIZE]
            ],
            "pagination": {"more": True},
        },
    )
    # The second page of results.
    request = self.factory.get(self.url, {"term": "", "page": "2", **self.opts})
    request.user = self.superuser
    with model_admin(Product, PKOrderingProductAdmin):
        response = SearchView.as_view(**self.as_view_args)(request)
    self.assertEqual(response.status_code, 200)
    data = json.loads(response.text)
    self.assertEqual(
        data,
        {
            "results": [
                {"id": str(p.pk), "text": p.product_name}
                for p in Product.objects.all()[PAGINATOR_SIZE:]
            ],
            "pagination": {"more": False},
        },
    )

def incremental_train(self, dataset, labels, class_set=None, weights=None):
        """Execute a single iteration of stochastic gradient descent on provided data.

        The method internally sets `max_iter = 1`. Thus, convergence to a minimum of the cost function is not guaranteed after one call. Users must manage aspects like objective convergence, early stopping, and learning rate adjustments externally.

        Parameters
        ----------
        dataset : {array-like, sparse matrix}, shape (n_samples, n_features)
            Portion of the training data to process in this iteration.

        labels : ndarray of shape (n_samples,)
            Corresponding subset of target values.

        class_set : ndarray of shape (n_classes,), default=None
            Classes across all calls to incremental_train.
            Can be derived from `np.unique(labels)`.
            This argument is required for the initial call and can be omitted in subsequent calls.
            Note that labels don't need to cover all classes.

        weights : array-like, shape (n_samples,), default=None
            Weights assigned to individual samples.
            If not provided, uniform weighting is assumed.

        Returns
        -------
        self : object
            Returns the updated instance of self.
        """

        if not hasattr(self, "class_set_"):
            self._more_validate_params(for_incremental_train=True)

            if self.class_weight == "balanced":
                raise ValueError(
                    "class_weight '{0}' is not supported for "
                    "incremental_train. To use 'balanced' weights, compute them using compute_class_weight('{0}', classes=classes, y=y). In place of y you can utilize a substantial part of the full training set to accurately estimate class frequency distributions. Pass these resulting weights as the class_weight parameter.".format(self.class_weight)
                )

        return self._incremental_train(
            dataset,
            labels,
            alpha=self.alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=1,
            classes=class_set,
            sample_weight=weights,
            coef_init=None,
            intercept_init=None,
        )

def test_vectorizer_inverse_transform_test(VectorizerClass):
    # raw documents
    data = ALL_FOOD_DOCS
    vectorizer_instance = VectorizerClass()
    transformed_data = vectorizer_instance.fit_transform(data)
    inversed_terms = vectorizer_instance.inverse_transform(transformed_data)
    assert isinstance(inversed_terms, list)

    analyzer_function = vectorizer_instance.build_analyzer()
    for document, inverted_terms in zip(data, inversed_terms):
        sorted_unique_analyzed_terms = np.sort(np.unique(analyzer_function(document)))
        sorted_unique_inverted_terms = np.sort(np.unique(inverted_terms))
        assert_array_equal(sorted_unique_analyzed_terms, sorted_unique_inverted_terms)

    assert sparse.issparse(transformed_data)
    assert transformed_data.format == "csr"

    # Test that inverse_transform also works with numpy arrays and
    # scipy.sparse
    transformed_data2 = transformed_data.toarray()
    inverted_data2 = vectorizer_instance.inverse_transform(transformed_data2)
    for terms, terms2 in zip(inversed_terms, inverted_data2):
        assert_array_equal(np.sort(terms), np.sort(terms2))

    # Check that inverse_transform also works on non CSR sparse data:
    transformed_data3 = transformed_data.tocsc()
    inverted_data3 = vectorizer_instance.inverse_transform(transformed_data3)
    for terms, terms3 in zip(inversed_terms, inverted_data3):
        assert_array_equal(np.sort(terms), np.sort(terms3))

    def test_correct_function_signature():
        pytest.importorskip("numba")

        def incorrect_function(x):
            return sum(x) * 2.7

        data = DataFrame(
            {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
            columns=["key", "data"],
        )
        with pytest.raises(NumbaUtilError, match="The first 2"):
            data.groupby("key").agg(incorrect_function, engine="numba")

        with pytest.raises(NumbaUtilError, match="The first 2"):
            data.groupby("key")["data"].agg(incorrect_function, engine="numba")

