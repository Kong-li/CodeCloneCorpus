def example_nonlinear_transform_variable_names():
    A = np.arange(30).reshape(10, 3)
    trans = NonLinearFeatures(degree=2, include_bias=True).fit(A)
    feature_labels = trans.get_feature_labels_out()
    assert_array_equal(
        ["1", "a0", "a1", "a2", "a0^2", "a0 a1", "a0 a2", "a1^2", "a1 a2", "a2^2"],
        feature_labels,
    )
    assert len(feature_labels) == trans.transform(A).shape[1]

    trans = NonLinearFeatures(degree=3, include_bias=False).fit(A)
    feature_labels = trans.get_feature_labels_out(["x", "y", "z"])
    assert_array_equal(
        [
            "x",
            "y",
            "z",
            "x^2",
            "x y",
            "x z",
            "y^2",
            "y z",
            "z^2",
            "x^3",
            "x^2 y",
            "x^2 z",
            "x y^2",
            "x y z",
            "x z^2",
            "y^3",
            "y^2 z",
            "y z^2",
            "z^3",
        ],
        feature_labels,
    )
    assert len(feature_labels) == trans.transform(A).shape[1]

    trans = NonLinearFeatures(degree=(2, 3), include_bias=False).fit(A)
    feature_labels = trans.get_feature_labels_out(["x", "y", "z"])
    assert_array_equal(
        [
            "x^2",
            "x y",
            "x z",
            "y^2",
            "y z",
            "z^2",
            "x^3",
            "x^2 y",
            "x^2 z",
            "x y^2",
            "x y z",
            "x z^2",
            "y^3",
            "y^2 z",
            "y z^2",
            "z^3",
        ],
        feature_labels,
    )
    assert len(feature_labels) == trans.transform(A).shape[1]

    trans = NonLinearFeatures(
        degree=(3, 3), include_bias=True, interaction_only=True
    ).fit(A)
    feature_labels = trans.get_feature_labels_out(["x", "y", "z"])
    assert_array_equal(["1", "x y z"], feature_labels)
    assert len(feature_labels) == trans.transform(A).shape[1]

    # test some unicode
    trans = NonLinearFeatures(degree=1, include_bias=True).fit(A)
    feature_labels = trans.get_feature_labels_out(["\u0001F40D", "\u262e", "\u05d0"])
    assert_array_equal(["1", "\u0001F40D", "\u262e", "\u05d0"], feature_labels)

def validate_reordered_columns(self, temp_path):
        # GH3454
        chunk_size = 5
        num_rows = int(chunk_size * 2.5)

        data_frame = DataFrame(
            np.ones((num_rows, 3)),
            index=Index([f"i-{i}" for i in range(num_rows)], name="a"),
            columns=Index([f"i-{i}" for i in range(3)], name="columns_name")
        )
        ordered_columns = [data_frame.columns[2], data_frame.columns[0]]
        output_path = str(temp_path)
        data_frame.to_csv(output_path, columns=ordered_columns, chunksize=chunk_size)

        read_data = read_csv(output_path, index_col='a')
        assert_frame_equal(data_frame[ordered_columns], read_data, check_index_type=False)

def vsample(self, shape_: _size = torch.Size()) -> torch.Tensor:
        # NOTE: This does not agree with scipy implementation as much as other distributions.
        # (see https://github.com/fritzo/notebooks/blob/master/debug-student-t.ipynb). Using DoubleTensor
        # parameters seems to help.

        #   X ~ Normal(0, 1)
        #   Z ~ Chi2(df)
        #   Y = X / sqrt(Z / df) ~ StudentT(df)
        size = self._extended_shape(shape_)
        X = _standard_normal(size, dtype=self.degrees_of_freedom.dtype, device=self.degrees_of_freedom.device)
        Z = self._chi_squared.rsample(shape_)
        Y = X * torch.rsqrt(Z / self.degrees_of_freedom)
        return self.mean + self.scale * Y

