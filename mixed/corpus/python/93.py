def _broadcast_processed_optimizer_state(
    fsdp_state: _FSDPState,
    optimizer_state: Dict[str, Any],
    group: Optional[dist.ProcessGroup],
) -> Any:
    objects = [None]
    if dist.get_rank(group) != 0:
        result = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),  # type: ignore[union-attr]
            optimizer_state
        )
        objects[0] = result
    dist.broadcast_object_list(objects, src=0, group=group)
    return objects[0] if dist.get_rank(group) != 0 else optimizer_state

    def _gather_all_orig_param_state(
        fsdp_param_info: FSDPParamInfo,
        input_states: Dict[str, Any],
        shard_state: bool,
        to_save: bool,
        cpu_offload: bool,
    ) -> Dict[str, Any]:
        """
        Given a optimizer state dict, ``input_states``, which the keys are FQNs to the
        original parameters (not FlatParameters nor parmeter ID), gather all the
        states and unflatten them to the original dimensions. Note that all the
        params referred by the ``input_states`` must be managed by FSDP.
        """
        fsdp_state = fsdp_param_info.state
        if (
            fsdp_state.world_size == 1
            or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
        ):
            return input_states if to_save else {}

        with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
            with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER_OBJ):
                gathered_state_info = _allgather_state_info(fsdp_state, input_states)
            output_states = _allgather_orig_param_states(
                fsdp_param_info,
                gathered_state_info,
                input_states,
                shard_state,
                to_save,
                cpu_offload,
            )
        if to_save:
            for key, idx in fsdp_param_info.param_indices.items():
                if key in output_states:
                    continue
                if not fsdp_param_info.param_requires_grad[idx]:
                    continue

                raise RuntimeError(
                    f"{key} is not in the output state. "
                    "The FSDPParamInfo has the param keys "
                    f"{sorted(fsdp_param_info.param_indices.keys())} while "
                    "the output_states has the param keys "
                    f"{sorted(output_states.keys())}."
                )
            return output_states
        else:
            return {}

    def example_check_():
        si = SingleIndex.from_tuples(zip(range(10), range(10)))
        assert si.is_(si)
        assert si.is_(si.view())
        assert si.is_(si.view().view().view().view())
        si2 = si.view()
        # names are metadata, they don't change id
        si2.names = ["X", "Y"]
        assert si2.is_(si)
        assert si.is_(si2)

        assert not si.is_(si.set_names(["Z", "W"]))
        # levels are inherent properties, they change identity
        si3 = si2.set_levels([list(range(10)), list(range(10))])
        assert not si3.is_(si2)
        # shouldn't change
        assert si2.is_(si)
        si4 = si3.view()

        # GH 17464 - Remove duplicate SingleIndex levels
        si4 = si4.set_levels([list(range(10)), list(range(10))])
        assert not si4.is_(si3)
        si5 = si.view()
        si5 = si5.set_levels(si5.levels)
        assert not si5.is_(si)

    def test_astype_object(self):
        idx = PeriodIndex([], freq="M")

        exp = np.array([], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        idx = PeriodIndex(["2011-01", NaT], freq="M")

        exp = np.array([Period("2011-01", freq="M"), NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        exp = np.array([Period("2011-01-01", freq="D"), NaT], dtype=object)
        idx = PeriodIndex(["2011-01-01", NaT], freq="D")
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

    def validate_encode_with_custom_behavior():
        from numpy import array
        with pytest.raises(ValueError, match="y contains previously unseen labels"):
            custom_uniques = array([1, 2, 3])
            custom_values = array([1, 2, 3, 4])
            _encode(custom_values, uniques=custom_uniques, check_unknown=True)

        # dont raise error if False
        no_error_uniques = array(["a", "b", "c"], dtype='O')
        no_error_values = array(["a", "b", "c", "d"], dtype='O')
        _encode(no_error_values, uniques=no_error_uniques, check_unknown=False)

        # parameter is ignored for object dtype
        with pytest.raises(ValueError, match="y contains previously unseen labels"):
            custom_uniques_obj = array(["x", "y", "z"], dtype=object)
            custom_values_obj = array(["x", "y", "z", "w"], dtype=object)
            _encode(custom_values_obj, uniques=custom_uniques_obj, check_unknown=False)

        def _encode(values, uniques, check_unknown):
            if not check_unknown and values.dtype != 'O':
                raise ValueError("y contains previously unseen labels")

    def explore(proc, element, forward=True):
        # From https://github.com/google/jax/pull/19695
        def explore_children():
            kids, typedef = optree.tree_flatten(
                element,
                is_leaf=lambda x: x is not element,
                none_is_leaf=True,
                namespace="tensorflow",
            )
            if typedef.num_nodes == 1 and typedef.num_leaves == 1:
                return element
            else:
                return optree.tree_unflatten(
                    typedef,
                    [explore(proc, kid, forward=forward) for kid in kids],
                )

        if forward:
            res = proc(element)
            if res is None:
                return explore_children()
        else:
            explored_element = explore_children()
            res = proc(explored_element)
            if res is None:
                return explored_element
        # Detect MAP_TO_NONE without tree_api import to avoid circular import.
        if isinstance(res, type) and res.__name__ == "MAP_TO_NONE":
            return None
        return res

def _flatten_optim_state_dict_v2(
    optim_state_dict: Dict[str, Any],
    model_module: nn.Module,
    use_original_params: bool = False,
    optimizer_instance: Optional[torch.optim.Optimizer] = None,
    rank_zero_only: bool = True,
    process_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If `use_original_params` is set to `True`, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optimizer instance, states for those empty parameters won't be
    initialized. So, when aggregating the fully-qualified names (FQNs) across ranks,
    no assert will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP knows how to aggregate them. However, FSDP has to ignore handling those
    parameters that are not managed by FSDP and do not exist on the local rank -- it is
    managed by other parallelism and FSDP does not know how to handle/aggregate them.

    Note that `_flatten_tensor_optim_state` doesn't need `optimizer_instance` to flatten/shard the state.
    However, NamedOptimizer and KeyedOptimizer require all the states even if the corresponding parameters are empty.
    To this end, `optimizer_instance` will be used to get the initial state of the empty parameters.
    `optimizer_instance` should only be non-None if the optimizer is KeyedOptimizer or NamedOptimizer.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
    SimpleProfiler.reset()

    unflattened_osd = optim_state_dict
    param_to_fqns_map = _get_param_to_fqns(model_module)
    if not rank_zero_only:
        process_group = None

    if "param_groups" in unflattened_osd and optimizer_instance is not None:
        param_groups_copy = copy.deepcopy(unflattened_osd["param_groups"])
    else:
        param_groups_copy = None

    flattened_state_dict = {}
    for param, fqns in param_to_fqns_map.items():
        if len(fqns) == 1:
            key = _OptimStateKey(fqns[0], True)
            user_state = unflattened_osd.get(fqns[0], None)
            if isinstance(user_state, torch.Tensor) and rank_zero_only and use_original_params:
                user_state = _broadcast_state(process_group, user_state)
            flattened_state_dict[key] = copy.copy(user_state)

        for fqn in fqns:
            param_state = unflattened_osd.get(fqn, None)
            if not use_original_params or (fqn in flattened_state_dict):
                key = _OptimStateKey(fqn, False)
                flattened_state_dict[key] = copy.deepcopy(param_state) if param_state else {}
            elif process_group and fqn:
                del unflattened_osd[fqn]
                user_state_cpu = _broadcast_state(process_group, param_state.cpu())
                flattened_state_dict[fqn] = copy.copy(user_state_cpu)

    SimpleProfiler.dump_and_reset("FSDP _flatten_optim_state_dict_v2 profiling: ")
    return {"state": flattened_state_dict, "param_groups": param_groups_copy}

