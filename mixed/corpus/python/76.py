    def verify_json_api_output(self, request_path):
            response = self.client.get(request_path)
            content_type = "application/json"
            expected_status_code = 200
            parsed_response = json.loads(response.text)

            assert response.status_code == expected_status_code
            assert response.headers["content-type"] == content_type
            assert parsed_response == {
                "a": [1, 2, 3],
                "foo": {"bar": "baz"},
                "timestamp": "2013-05-19T20:00:00",
                "value": "3.14",
            }

    def test_closed_fixed(closed, arithmetic_win_operators):
        # GH 34315
        func_name = arithmetic_win_operators
        df_fixed = DataFrame({"A": [0, 1, 2, 3, 4]})
        df_time = DataFrame({"A": [0, 1, 2, 3, 4]}, index=date_range("2020", periods=5))

        result = getattr(
            df_fixed.rolling(2, closed=closed, min_periods=1),
            func_name,
        )()
        expected = getattr(
            df_time.rolling("2D", closed=closed, min_periods=1),
            func_name,
        )().reset_index(drop=True)

        tm.assert_frame_equal(result, expected)

def test_groups_support(Est):
    # Check if ValueError (when groups is None) propagates to
    # HalvingGridSearchCV and HalvingRandomSearchCV
    # And also check if groups is correctly passed to the cv object
    rng = np.random.RandomState(0)

    X, y = make_classification(n_samples=50, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 50)

    clf = LinearSVC(random_state=0)
    grid = {"C": [1]}

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(random_state=0),
    ]
    error_msg = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        gs = Est(clf, grid, cv=cv, random_state=0)
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        gs.fit(X, y, groups=groups)

    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit(random_state=0)]
    for cv in non_group_cvs:
        gs = Est(clf, grid, cv=cv)
        # Should not raise an error
        gs.fit(X, y)

    def is_similar(ca: torch.fx.node.Node, aot: torch.fx.node.Node):
        # 1. comparing using target (for aten ops)
        target_match = ca.target == aot.target
        if not target_match:
            # 2. comparing using name (for HOPs)
            target_match = (
                hasattr(ca.target, "__name__")
                and hasattr(aot.target, "__name__")
                and ca.target.__name__ == aot.target.__name__
            )
        if (
            not target_match
            and hasattr(ca.target, "name")
            and hasattr(aot.target, "name")
            and aot.target.name() == "aten::reshape"
            and hasattr(aot.meta.get("original_aten"), "name")
        ):
            # 3. undo view_to_reshape post grad pass
            target_match = ca.target.name() == aot.meta["original_aten"].name()

        return (
            target_match
            and ca.op == aot.op
            and ca.type == aot.type
            and len(ca.all_input_nodes) == len(aot.all_input_nodes)
        )

