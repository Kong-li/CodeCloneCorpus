    def decorator(optimizer):
            # Adjust TritonKernel's XBLOCK parameter if it is not a function argument.
            # This ensures coordinate descent tuning does not attempt to tune it.
            #
            # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.
            import inspect

            fn = optimizer.fn
            configs = optimizer.configs
            inductor_meta = optimizer.inductor_meta
            triton_meta = optimizer.triton_meta
            autotune_cache = optimizer.autotune_cache

            if "XBLOCK" not in inspect.signature(fn).parameters:
                for tconfig in configs:
                    if "XBLOCK" in tconfig.kwargs:
                        assert tconfig.kwargs["XBLOCK"] == 1, "Unexpected XBLOCK value"
                        del tconfig.kwargs["XBLOCK"]

            mutated_arg_names = optimizer.mutated_arg_names
            reset_to_zero_arg_names = optimizer.reset_to_zero_arg_names
            optimize_mem = optimizer.optimize_mem
            heuristic_type = optimizer.heuristic_type
            size_hints = optimizer.size_hints
            custom_kernel = optimizer.custom_kernel
            filename = optimizer.filename

            if inductor_meta.get("profile_bandwidth"):
                return DebugAutotuner(
                    fn,
                    triton_meta=triton_meta,
                    inductor_meta=inductor_meta,
                    regex_filter=inductor_meta["profile_bandwidth_regex"],
                    with_profiler=inductor_meta[
                        "profile_bandwidth_with_do_bench_using_profiling"
                    ],
                    configs=configs,
                    save_cache_hook=autotune_cache and autotune_cache.save,
                    mutated_arg_names=mutated_arg_names,
                    reset_to_zero_arg_names=reset_to_zero_arg_names,
                    optimize_mem=optimize_mem,
                    heuristic_type=heuristic_type,
                    size_hints=size_hints,
                    custom_kernel=custom_kernel,
                    filename=filename,
                    with_bandwidth_info=True,
                )
            else:
                return CachingAutotuner(
                    fn,
                    triton_meta=triton_meta,
                    inductor_meta=inductor_meta,
                    configs=configs,
                    save_cache_hook=autotune_cache and autotune_cache.save,
                    mutated_arg_names=mutated_arg_names,
                    reset_to_zero_arg_names=reset_to_zero_arg_names,
                    optimize_mem=optimize_mem,
                    heuristic_type=heuristic_type,
                    size_hints=size_hints,
                    custom_kernel=custom_kernel,
                    filename=filename,
                )

    def test_prevent_change_outer_model_and_create_invalid_data(self):
        author = Author.objects.create(name="Charles")
        other_author = Author.objects.create(name="Walt")
        AuthorFormSet = modelformset_factory(Author, fields="__all__")
        data = {
            "form-TOTAL_FORMS": "2",
            "form-INITIAL_FORMS": "2",
            "form-MAX_NUM_FORMS": "",
            "form-0-id": str(author.id),
            "form-0-name": "Charles",
            "form-1-id": str(other_author.id),  # A model not in the formset's queryset.
            "form-1-name": "Changed name",
        }
        # This formset is only for Walt Whitman and shouldn't accept data for
        # other_author.
        formset = AuthorFormSet(
            data=data, queryset=Author.objects.filter(id__in=(author.id,))
        )
        self.assertTrue(formset.is_valid())
        formset.save()
        # The name of other_author shouldn't be changed and new models aren't
        # created.
        self.assertSequenceEqual(Author.objects.all(), [author, other_author])

def call_method(
    self,
    tx,
    name,
    args: "List[VariableTracker]",
    kwargs: "Dict[str, VariableTracker]",
) -> "VariableTracker":
    # NB - Both key and value are LazyVariableTrackers in the beginning. So,
    # we have to insert guards when a dict method is accessed. For this to
    # be simple, we are conservative and overguard. We skip guard only for
    # get/__getitem__ because the key guard will be inserted by the
    # corresponding value VT. For __contains__, we add a DICT_CONTAINS
    # guard. But for all the other methods, we insert the DICT_KEYS_MATCH
    # guard to be conservative.
    from . import BuiltinVariable, ConstantVariable, TupleVariable

    Hashable = ConstDictVariable._HashableTracker

    arg_hashable = args and is_hashable(args[0])

    if name == "__init__":
        temp_dict_vt = variables.BuiltinVariable(dict).call_dict(
            tx, *args, **kwargs
        )
        tx.output.side_effects.mutation(self)
        self.items.update(temp_dict_vt.items)
        return ConstantVariable.create(None)
    elif name == "__getitem__":
        # Key guarding - Nothing to do. LazyVT for value will take care.
        assert len(args) == 1
        return self.getitem_const_raise_exception_if_absent(tx, args[0])
    elif name == "items":
        assert not (args or kwargs)
        self.install_dict_keys_match_guard()
        if self.source:
            tx.output.guard_on_key_order.add(self.source.name())
        return TupleVariable(
            [TupleVariable([k.vt, v]) for k, v in self.items.items()]
        )
    elif name == "keys":
        self.install_dict_keys_match_guard()
        if self.source:
            tx.output.guard_on_key_order.add(self.source.name())
        assert not (args or kwargs)
        return DictKeysVariable(self)
    elif name == "values":
        self.install_dict_keys_match_guard()
        if self.source:
            tx.output.guard_on_key_order.add(self.source.name())
        assert not (args or kwargs)
        return DictValuesVariable(self)
    elif name == "copy":
        self.install_dict_keys_match_guard()
        assert not (args or kwargs)
        return self.clone(
            items=self.items.copy(), mutation_type=ValueMutationNew(), source=None
        )
    elif name == "__len__":
        assert not (args or kwargs)
        self.install_dict_keys_match_guard()
        return ConstantVariable.create(len(self.items))
    elif name == "__setitem__" and arg_hashable and self.is_mutable():
        self.install_dict_keys_match_guard()
        assert not kwargs and len(args) == 2
        tx.output.side_effects.mutation(self)
        self.items[Hashable(args[0])] = args[1]
        return ConstantVariable.create(None)
    elif name == "__delitem__" and arg_hashable and self.is_mutable():
        self.install_dict_keys_match_guard()
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.__delitem__(Hashable(args[0]))
        return ConstantVariable.create(None)
    elif name in ("pop", "get") and len(args) in (1, 2) and args[0] not in self:
        # missing item, return the default value. Install no DICT_CONTAINS guard.
        self.install_dict_contains_guard(tx, args)
        if len(args) == 1:
            return ConstantVariable(None)
        else:
            return args[1]
    elif name == "pop" and arg_hashable and self.is_mutable():
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        return self.items.pop(Hashable(args[0]))
    elif name == "clear":
        self.should_reconstruct_all = True
        tx.output.side_effects.mutation(self)
        self.items.clear()
        return ConstantVariable.create(None)
    elif name == "update" and self.is_mutable():
        # In general, this call looks like `a.update(b, x=1, y=2, ...)`.
        # Either `b` or the kwargs is omittable, but not both.
        self.install_dict_keys_match_guard()
        has_arg = len(args) == 1
        has_kwargs = len(kwargs) > 0
        if has_arg or has_kwargs:
            tx.output.side_effects.mutation(self)
            if has_arg:
                if isinstance(args[0], ConstDictVariable):
                    dict_vt = args[0]
                else:
                    dict_vt = BuiltinVariable.call_custom_dict(tx, dict, args[0])
                self.items.update(dict_vt.items)
            if has_kwargs:
                # Handle kwargs
                kwargs = {
                    Hashable(ConstantVariable.create(k)): v
                    for k, v in kwargs.items()
                }
                self.items.update(kwargs)
            return ConstantVariable.create(None)
        else:
            return super().call_method(tx, name, args, kwargs)
    elif name in ("get", "__getattr__") and args[0] in self:
        # Key guarding - Nothing to do.
        return self.getitem_const(tx, args[0])
    elif name == "__contains__" and len(args) == 1:
        self.install_dict_contains_guard(tx, args)
        contains = args[0] in self
        return ConstantVariable.create(contains)
    elif name == "setdefault" and arg_hashable and self.is_mutable():
        self.install_dict_keys_match_guard()
        assert not kwargs
        assert len(args) <= 2
        value = self.maybe_getitem_const(args[0])
        if value is not None:
            return value
        else:
            if len(args) == 1:
                x = ConstantVariable.create(None)
            else:
                x = args[1]
            tx.output.side_effects.mutation(self)
            self.items[Hashable(args[0])] = x
            return x
    else:
        return super().call_method(tx, name, args, kwargs)

def initialize_to_zero_params(self, *args, **kwargs):
    if not self.initialize_to_zero_param_names:
        return
    for i, arg in enumerate(args):
        if self.fn.param_names[i] in self.initialize_to_zero_param_names:
            assert isinstance(
                arg,
                torch.Tensor,
            ), "self.initialize_to_zero_param_names should only contain valid argument names"
            arg.zero_()

    for name, arg in kwargs.items():
        if name in self.initialize_to_zero_param_names:
            assert isinstance(
                arg,
                torch.Tensor,
            ), "self.initialize_to_zero_param_names should only contain valid argument names"
            arg.zero_()

    def test_validation_with_invalid_id(self):
        AuthorFormSet = modelformset_factory(Author, fields="__all__")
        data = {
            "form-TOTAL_FORMS": "1",
            "form-INITIAL_FORMS": "1",
            "form-MAX_NUM_FORMS": "",
            "form-0-id": "abc",
            "form-0-name": "Charles",
        }
        formset = AuthorFormSet(data)
        self.assertEqual(
            formset.errors,
            [
                {
                    "id": [
                        "Select a valid choice. That choice is not one of the "
                        "available choices."
                    ]
                }
            ],
        )

    def validate_tril_triu_dtypes():
        # Issue 4916
        # tril and triu should return the same dtype as input
        for c in np.typecodes('All'):
            if c == 'V':
                continue
            arr = np.zeros((3, 3), dtype=c)
            dtype_check = lambda x: assert_equal(x.dtype, arr.dtype)
            dtype_check(np.triu(arr))
            dtype_check(np.tril(arr))

        # check special cases
        arr = np.array([['2001-01-01T12:00', '2002-03-03T13:56'],
                        ['2004-01-01T12:00', '2003-01-03T13:45']], dtype='datetime64')
        assert_equal(np.triu(arr).dtype, arr.dtype)
        assert_equal(np.tril(arr).dtype, arr.dtype)

        arr = np.zeros((3, 3), dtype=('f4', 'f4'))
        assert_equal(np.triu(arr).dtype, arr.dtype)
        assert_equal(np.tril(arr).dtype, arr.dtype)

    def fetch_table_data(self, max_depth=None):
            if max_depth is None:
                max_depth = self.depth
            if max_depth is None:
                max_depth = 999999

            import tabulate
            tabulate.PRESERVE_WHITESPACE = True
            header = ["Module", "FLOP", "% Total"]
            values = []
            global_flops = self.calculate_total_flops()
            global_suffix = get_suffix_str(global_flops)
            is_global_included = False

            def format_module(mod_name, depth):
                nonlocal is_global_included

                total_flops = sum(self.flop_counts[mod_name].values())

                is_global_included |= total_flops >= global_flops

                padding = " " * depth
                row_data = [
                    padding + mod_name,
                    convert_num_with_suffix(total_flops, global_suffix),
                    convert_to_percent_str(total_flops, global_flops)
                ]
                for k, v in self.flop_counts[mod_name].items():
                    values.append([
                        padding + " - " + str(k),
                        convert_num_with_suffix(v, global_suffix),
                        convert_to_percent_str(v, global_flops)
                    ])
                return row_data

            for mod in sorted(self.flop_counts.keys()):
                if mod == 'Global':
                    continue
                depth_level = mod.count(".") + 1
                if depth_level > max_depth:
                    continue

                cur_values = format_module(mod, depth_level - 1)
                values.append(cur_values)

            # 处理全局模块的输出逻辑
            if 'Global' in self.flop_counts and not is_global_included:
                for value in values[1:]:
                    value[0] = " " + value[0]

                values.insert(0, format_module('Global', 0))

            if len(values) == 0:
                values.append(["Global", "0", "0%"])

            return tabulate.tabulate(values, headers=header, colalign=("left", "right", "right"))

