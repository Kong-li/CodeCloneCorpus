    def checker(ts, nanos, unit):
        # First check that we do raise in cases where we should
        if nanos == 1:
            pass
        else:
            div, mod = divmod(ts._value, nanos)
            diff = int(nanos - mod)
            lb = ts._value - mod
            assert lb <= ts._value  # i.e. no overflows with python ints
            ub = ts._value + diff
            assert ub > ts._value  # i.e. no overflows with python ints

            msg = "without overflow"
            if mod == 0:
                # We should never be raising in this
                pass
            elif method is cls.ceil:
                if ub > cls.max._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return
            elif method is cls.floor:
                if lb < cls.min._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return
            elif mod >= diff:
                if ub > cls.max._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return
            elif lb < cls.min._value:
                with pytest.raises(err_cls, match=msg):
                    method(ts, unit)
                return

        res = method(ts, unit)

        td = res - ts
        diff = abs(td._value)
        assert diff < nanos
        assert res._value % nanos == 0

        if method is cls.round:
            assert diff <= nanos / 2
        elif method is cls.floor:
            assert res <= ts
        elif method is cls.ceil:
            assert res >= ts

def check_missing_field(self):
        class InlineValidationTest(TabularInline):
            model = InlineValidationTestModel
            fk_name = "non_existent_field"

        class ModelAdminConfig(ModelAdmin):
            inlines = [InlineValidationTest]

        self.assertIsInvalid(
            ModelAdminConfig,
            ValidationTestModel,
            "'modeladmin.InlineValidationTestModel' has no field named "
            "'non_existent_field'.",
            "admin.E202",
            invalid_obj=InlineValidationTest,
        )

def validate_invalid_inlines(self):
        class RandomCallable:
            pass

        inlines = [RandomCallable()]

        test_model_admin = ModelAdmin()
        test_model_admin.inlines = inlines

        self.assertIsInvalidRegexp(
            test_model_admin,
            ValidationTestModel,
            r"'.*\.RandomCallable' must inherit from 'InlineModelAdmin'\.",
            "admin.E104",
        )

    def _process_lowerdim_multi_index_row(self, index: tuple):
        # we have a row multi-index, process or raise
        axis = self.axis_or_0
        try:
            # fast path for series or for index devoid of slices
            return self._find_label(index, axis=axis)

        except KeyError as ekey:
            # raise KeyError if number of indexers match
            # else IndexingError will be raised
            if self.dimension < len(index) <= self.data.index.levels_count():
                raise ekey
            raise IndexError("No label returned") from ekey

    def check_suggestions_c42(self):
            class User(Profile):
                suggestions_fields = ("username",)

            self.assertIsInvalid(
                User,
                CustomUserModel,
                msg=(
                    "The value of 'suggestions_fields[0]' must be a foreign "
                    "key or a many-to-many field."
                ),
                id="profile.E042",
                invalid_obj=User,
            )

    def __enter__(self, *args, **kwargs):
        import unittest.mock as mock

        import torch._inductor.codecache

        _compile_method_orig = torch._inductor.codecache.CUDACodeCache.compile

        def my_compile(source_code, dst_file_ext):
            self.sources.append(source_code)
            return _compile_method_orig(source_code, dst_file_ext)

        self._compile_patch = mock.patch(
            "torch._inductor.codecache.CUDACodeCache.compile", my_compile
        )
        return self._compile_patch.__enter__(*args, **kwargs)  # type: ignore[union-attr]

    def bn2d_infer_rule(node: Node, module_ins):
        """
        Given a BatchNorm2D instance and a node check the following conditions:
        - the input type can be expanded to a size 4 tensor: t = (x_1, x_2, x_3, x_4)
        - the current node type can be expanded to a size 4 tensor: t' = (x_1', x_2', x_3', x_4')
        - t is consistent with t'
        - x_2 is consistent with the module's num_features
        - x_2' is consistent with the module's num_features
        output type: the more precise type of t and t'
        """
        assert isinstance(node.args[0], Node)
        node.args[0].type = expand_to_tensor_dim(node.args[0].type, 4)
        arg_type = node.args[0].type
        node.type = expand_to_tensor_dim(node.type, 4)

        # we check the conditions on the incoming argument and any existing annotation
        # we also check for consistency between both annotations
        if (
            is_consistent(arg_type.__args__[1], module_ins.num_features)
            and is_consistent(node.type.__args__[1], module_ins.num_features)
            and is_consistent(arg_type, node.type)
        ):
            # choose the more precise type to be the node's type
            # so if an incoming argument has more type information,
            # we set this node's type to be the argument type
            node.type = get_greatest_upper_bound(arg_type, node.type)
            return node.type
        else:
            raise TypeError(
                f"Cannot apply {module_ins} with input type {arg_type} and existing type {node.type} on {node}"
            )

