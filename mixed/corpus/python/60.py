    def test_initially_immediate_database_constraint(self):
        obj_1 = UniqueConstraintDeferrable.objects.create(name="p1", shelf="front")
        obj_2 = UniqueConstraintDeferrable.objects.create(name="p2", shelf="back")
        obj_1.shelf, obj_2.shelf = obj_2.shelf, obj_1.shelf
        with self.assertRaises(IntegrityError), atomic():
            obj_1.save()
        # Behavior can be changed with SET CONSTRAINTS.
        with connection.cursor() as cursor:
            constraint_name = connection.ops.quote_name("sheld_init_immediate_uniq")
            cursor.execute("SET CONSTRAINTS %s DEFERRED" % constraint_name)
            obj_1.save()
            obj_2.save()

    def test_astype_mixed_type(self):
        # mixed casting
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                "float32": np.array([1.0] * 10, dtype="float32"),
                "int32": np.array([1] * 10, dtype="int32"),
            },
            index=np.arange(10),
        )
        mn = df._get_numeric_data().copy()
        mn["little_float"] = np.array(12345.0, dtype="float16")
        mn["big_float"] = np.array(123456789101112.0, dtype="float64")

        casted = mn.astype("float64")
        _check_cast(casted, "float64")

        casted = mn.astype("int64")
        _check_cast(casted, "int64")

        casted = mn.reindex(columns=["little_float"]).astype("float16")
        _check_cast(casted, "float16")

        casted = mn.astype("float32")
        _check_cast(casted, "float32")

        casted = mn.astype("int32")
        _check_cast(casted, "int32")

        # to object
        casted = mn.astype("O")
        _check_cast(casted, "object")

    def test_knn_parallel_settings(algorithm):
        features, labels = datasets.make_classification(n_samples=30, n_features=5, n_redundant=0, random_state=0)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

        knn_classifier = neighbors.KNeighborsClassifier(algorithm=algorithm, n_neighbors=3)
        knn_classifier.fit(train_features, train_labels)
        predictions = knn_classifier.predict(test_features)
        distances, indices = knn_classifier.kneighbors(test_features)
        graph = knn_classifier.kneighbors_graph(test_features, mode="distance").toarray()

        knn_classifier.set_params(n_jobs=3)
        knn_classifier.fit(train_features, train_labels)
        parallel_predictions = knn_classifier.predict(test_features)
        parallel_distances, parallel_indices = knn_classifier.kneighbors(test_features)
        graph_parallel = knn_classifier.kneighbors_graph(test_features, mode="distance").toarray()

        assert_array_equal(predictions, parallel_predictions)
        assert_allclose(distances, parallel_distances)
        assert_array_equal(indices, parallel_indices)
        assert_allclose(graph, graph_parallel)

    def convert_to_geometry(self, data):
            """Transform the value to a Geometry object."""
            if data in self.null_values:
                return None

            if not isinstance(data, GeoPoint):
                if hasattr(self.form, "unserialize"):
                    try:
                        data = self.form.unserialize(data)
                    except GDALException:
                        data = None
                else:
                    try:
                        data = GeoPoint(data)
                    except (GEOSException, ValueError, TypeError):
                        data = None
                if data is None:
                    raise ValidationError(
                        self.error_messages["invalid_point"], code="invalid_point"
                    )

            # Try to set the srid
            if not data.srid:
                try:
                    data.srid = self.form.default_srid
                except AttributeError:
                    if self.srid:
                        data.srid = self.srid
            return data

    def _split_tensor_list_constants(g, block):
        for node in block.nodes():
            for subblock in node.blocks():
                _split_tensor_list_constants(g, subblock)
            if _is_constant_tensor_list(node):
                inputs = []
                for val in node.output().toIValue():
                    input = g.insertConstant(val)
                    input.node().moveBefore(node)
                    input.node().copyMetadata(node)
                    inputs.append(input)

                lc = (
                    g.create("prim::ListConstruct", inputs)
                    .insertBefore(node)
                    .output()
                    .setType(_C.ListType.ofTensors())
                )
                lc.node().copyMetadata(node)
                node.output().replaceAllUsesWith(lc)

    def example_knn_neighbors_predict_scores():
        for index in range(4):
            A, B = samples.generate_classification(
                n_samples=40,
                n_features=6,
                n_informative=3,
                n_classes=2,
                random_state=index,
            )
            C, D, E, F = train_test_split(A, B, random_state=1)
            G = int(1 - index)
            H = neighbors.KNeighborsClassifier(n_neighbors=2, outlier_label=G)
            H.fit(C, E)
            I = H.predict(D)
            J = H.predict_proba(D)
            K = np.argmax(J, axis=1)
            K = np.where(np.sum(J, axis=1) == 0, G, K)
            assert_array_equal(I, K)

