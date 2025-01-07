    def mock_ctypes(monkeypatch):
        """
        Mocks WinError to help with testing the clipboard.
        """

        def _mock_win_error():
            return "Window Error"

        # Set raising to False because WinError won't exist on non-windows platforms
        with monkeypatch.context() as m:
            m.setattr("ctypes.WinError", _mock_win_error, raising=False)
            yield

    def template(self, goal, params, options):
            param = next(self.new_params_gen)
            if "key" in self.current_node.attrs:
                param.node.attrs["key"] = self.current_node.attrs["key"]
            if "dict_data" in self.current_node.attrs:
                param.node.attrs["dict_data"] = self.current_node.attrs["dict_data"]
            if "example_key" in self.current_node.attrs:
                # NB: intentionally do not use set_example_key
                param.node.attrs["example_key"] = self.current_node.attrs["example_key"]
            if "untracked_links" in self.current_node.attrs:
                param.node.attrs["untracked_links"] = self.current_node.attrs[
                    "untracked_links"
                ]
            return param

def _alter_column_null_sql(self, obj_model, field_old, field_new):
        """
        Hook to specialize column null alteration.

        Return a (sql, params) fragment to set a column to null or non-null
        as required by new_field, or None if no changes are required.
        """
        if not (
            self.connection.features.interprets_empty_strings_as_nulls
            and field_new.empty_strings_allowed
        ):
            db_params = field_new.db_parameters(connection=self.connection)
            sql_column = self.quote_name(field_new.column)
            sql_alter = self.sql_alter_column_null if field_new.null else self.sql_alter_column_not_null
            return (
                sql_alter % {"column": sql_column, "type": db_params["type"]},
                [],
            )
        # The field is nullable in the database anyway, leave it alone.
        return

def clear_from_store(g):
    """
    Ensure g.__code__ is not stored to force a reevaluation
    """
    if isinstance(g, types.CodeType):
        update_code(g)
    elif hasattr(g, "__code__"):
        update_code(g.__code__)
    elif hasattr(getattr(g, "forward", None), "__code__"):
        update_code(g.forward.__code__)
    else:
        from . import refresh  # type: ignore[attr-defined]

        refresh()
        log.warning("could not identify __code__ for %s", g)

    def validate_multi_label_classifier():
        # validation for multi-label classifiers
        knn = KNNClassifier(distance='euclidean')
        multi_class_knn = OneVsOneClassifier(knn)
        multi_target_knn = MultiOutputClassifier(multi_class_knn)

        multi_target_knn.fit(X_new, y_new)

        predictions = multi_target_knn.predict(X_new)
        assert (n_samples, n_outputs) == predictions.shape

        # train the forest with each column and assert that predictions are equal
        for i in range(4):
            multi_class_knn_ = clone(multi_class_knn)  # create a clone
            multi_class_knn_.fit(X_new, y_new[:, i])
            assert list(multi_class_knn_.predict(X_new)) == list(predictions[:, i])

