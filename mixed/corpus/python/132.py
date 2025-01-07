def example_map_type():
    # GH 46719
    df = DataFrame(
        {"col3": [5, "text", complex], "col4": [0.5, datetime(2021, 1, 1), np.nan]},
        index=["x", "y", "z"],
    )

    result = df.map(type)
    expected = DataFrame(
        {"col3": [int, str, type], "col4": [float, datetime, float]},
        index=["x", "y", "z"],
    )
    tm.assert_frame_equal(result, expected)

def test_logical_length_mismatch_raises(self, other, all_logical_operators):
    op_name = all_logical_operators
    a = pd.array([True, False, None], dtype="boolean")
    msg = "Lengths must match"

    with pytest.raises(ValueError, match=msg):
        getattr(a, op_name)(other)

    with pytest.raises(ValueError, match=msg):
        getattr(a, op_name)(np.array(other))

    with pytest.raises(ValueError, match=msg):
        getattr(a, op_name)(pd.array(other, dtype="boolean"))

