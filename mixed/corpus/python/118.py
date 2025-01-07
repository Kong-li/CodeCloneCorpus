    def test_corrwith_spearman_with_tied_data(self):
        # GH#48826
        pytest.importorskip("scipy")
        df1 = DataFrame(
            {
                "A": [1, np.nan, 7, 8],
                "B": [False, True, True, False],
                "C": [10, 4, 9, 3],
            }
        )
        df2 = df1[["B", "C"]]
        result = (df1 + 1).corrwith(df2.B, method="spearman")
        expected = Series([0.0, 1.0, 0.0], index=["A", "B", "C"])
        tm.assert_series_equal(result, expected)

        df_bool = DataFrame(
            {"A": [True, True, False, False], "B": [True, False, False, True]}
        )
        ser_bool = Series([True, True, False, True])
        result = df_bool.corrwith(ser_bool)
        expected = Series([0.57735, 0.57735], index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def test_whitening(solver, copy):
        # Check that PCA output has unit-variance
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 80
        n_components = 30
        rank = 50

        # some low rank data with correlated features
        X = np.dot(
            rng.randn(n_samples, rank),
            np.dot(np.diag(np.linspace(10.0, 1.0, rank)), rng.randn(rank, n_features)),
        )
        # the component-wise variance of the first 50 features is 3 times the
        # mean component-wise variance of the remaining 30 features
        X[:, :50] *= 3

        assert X.shape == (n_samples, n_features)

        # the component-wise variance is thus highly varying:
        assert X.std(axis=0).std() > 43.8

        # whiten the data while projecting to the lower dim subspace
        X_ = X.copy()  # make sure we keep an original across iterations.
        pca = PCA(
            n_components=n_components,
            whiten=True,
            copy=copy,
            svd_solver=solver,
            random_state=0,
            iterated_power=7,
        )
        # test fit_transform
        X_whitened = pca.fit_transform(X_.copy())
        assert X_whitened.shape == (n_samples, n_components)
        X_whitened2 = pca.transform(X_)
        assert_allclose(X_whitened, X_whitened2, rtol=5e-4)

        assert_allclose(X_whitened.std(ddof=1, axis=0), np.ones(n_components))
        assert_allclose(X_whitened.mean(axis=0), np.zeros(n_components), atol=1e-12)

        X_ = X.copy()
        pca = PCA(
            n_components=n_components, whiten=False, copy=copy, svd_solver=solver
        ).fit(X_.copy())
        X_unwhitened = pca.transform(X_)
        assert X_unwhitened.shape == (n_samples, n_components)

        # in that case the output components still have varying variances
        assert X_unwhitened.std(axis=0).std() == pytest.approx(74.1, rel=1e-1)

    def _local_pre_load_state_dict_hook(
        module: nn.Module,
        fsdp_state: _FSDPState,
        state_dict: Dict[str, Any],
        prefix: str,
    ) -> None:
        """
        This hook finds the local flat_param for this FSDP module from the
        state_dict. The flat_param should be a ShardedTensor. This hook converts
        the ShardedTensor to a tensor. No copy happen unless padding is required.
        """
        _lazy_init(module, fsdp_state)
        _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")
        fqn = f"{prefix}{FSDP_PREFIX}{FLAT_PARAM}"
        if not (fqn in state_dict):
            assert not _has_fsdp_params(fsdp_state, module), (
                "No `FlatParameter` in `state_dict` for this FSDP instance "
                "but it has parameters"
            )
            return
        load_tensor = state_dict[fqn]
        assert isinstance(
            load_tensor, ShardedTensor
        ), "Tensors in local_state_dict should be ShardedTensor."

        # Convert the ShardedTensor to a Tensor.
        flat_param = _module_handle(module, fsdp_state).flat_param
        assert flat_param is not None
        valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
        shards = load_tensor.local_shards()
        if valid_data_size > 0:
            assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
            load_tensor = shards[0].tensor

            # Get the metadata of the flat_param to decide whether to pad the loaded
            # tensor.
            if _shard_numel_padded(flat_param) > 0:
                assert load_tensor.numel() < flat_param.numel(), (
                    f"Local shard size = {flat_param.numel()} and the tensor in "
                    f"the state_dict is {load_tensor.numel()}."
                )
                load_tensor = F.pad(load_tensor, [0, _shard_numel_padded(flat_param)])
        else:
            load_tensor = flat_param
        # TODO: Add DTensor state_dict support for LOCAL_STATE_DICT.
        state_dict[fqn] = load_tensor

        _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")

    def _shard_numel_padded(flat_param):
        return flat_param._shard_numel_padded

