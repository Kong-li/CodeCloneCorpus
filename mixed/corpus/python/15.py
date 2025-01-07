def verify_field_output_check(self):
        class Entity(models.Model):
            amount = models.DecimalField(max_digits=5, decimal_places=2)
            result = models.GeneratedField(
                expression=models.F("amount") * 2,
                output_field=models.DecimalField(max_digits=-1, decimal_places=-1),
                db_persist=True
            )

        expected_warnings = [
            Error(
                message="GeneratedField.output_field has errors:"
                "\n    'decimal_places' must be a non-negative integer. (fields.E131)"
                "\n    'max_digits' must be a positive integer. (fields.E133)",
                obj=Entity._meta.get_field("result"),
                id="fields.E223"
            )
        ]
        self.assertListEqual(
            list(Model._meta.get_field("field").check(databases={"default"})),
            expected_warnings
        )

def test_github_linkcode_resolve_link_to_module_older_version(self):
    info = {
        "module": "tests.sphinx.testdata.module",
        "fullname": "MyModule",
    }
    self.assertEqual(
        github_links.github_linkcode_resolve(
            "py", info, version="2.0", next_version="3.0"
        ),
        "https://github.com/django/django/blob/stable/2.0.x/tests/sphinx/"
        "testdata/module.py#L15",
    )

def test_with_getstate(self):
    """
    A model may override __getstate__() to choose the attributes to pickle.
    """

    class PickledModel(models.Model):
        def __getstate__(self):
            state = super().__getstate__().copy()
            del state["dont_pickle"]
            return state

    m = PickledModel()
    m.dont_pickle = 1
    dumped = pickle.dumps(m)
    self.assertEqual(m.dont_pickle, 1)
    reloaded = pickle.loads(dumped)
    self.assertFalse(hasattr(reloaded, "dont_pickle"))

def test_append_empty_tz_frame_with_datetime64ns_check(self):
        df = DataFrame(columns=["col_a"]).astype("datetime64[ns, UTC]")

        result = df._append({"col_a": pd.NaT}, ignore_index=True)
        expected = DataFrame({"col_a": [pd.NaT]}, dtype=object)
        tm.assert_frame_equal(result, expected)

        df = DataFrame(columns=["col_a"]).astype("datetime64[ns, UTC]")
        other = Series({"col_a": pd.NaT}, dtype="datetime64[ns]")
        result = df._append(other, ignore_index=True)
        tm.assert_frame_equal(result, expected)

        # mismatched tz
        other = Series({"col_b": pd.NaT}, dtype="datetime64[ns, US/Pacific]")
        result = df._append(other, ignore_index=True)
        expected = DataFrame({"col_a": [pd.NaT]}).astype(object)
        tm.assert_frame_equal(result, expected)

def _verify_input_requirements_for_model(
    input_nodes: List[torch.fx.Node], flat_args_with_paths, dimension_limits
) -> None:
    def derive_description(key_path: KeyPath) -> str:
        """For a given index into the flat_args, return a human readable string
        describing how to access it, e.g. "*args['foo'][0].bar"
        """
        # Prefix the keypath with "*args" or "**kwargs" to make it clearer where
        # the arguments come from. Ultimately we ought to serialize the
        # original arg names for the best error message here.
        args_kwargs_key_path = key_path[0]
        assert isinstance(args_kwargs_key_path, SequenceKey)
        if args_kwargs_key_path.idx == 0:
            return f"*args{description(key_path[1:])}"
        else:
            kwarg_key = key_path[1]
            assert isinstance(kwarg_key, MappingKey)
            name = str(kwarg_key)[1:-1]  # get rid of the enclosing []
            return f"{name}{description(key_path[2:])}"

    import sympy

    from torch._export.passes.add_runtime_assertions_for_requirements_pass import (
        _translate_range_to_int,
    )
    from torch.utils._sympy.solve import attempt_solve

    if len(flat_args_with_paths) != len(input_nodes):
        raise RuntimeError(
            "Unexpected number of inputs "
            f"(expected {len(input_nodes)}, got {len(flat_args_with_paths)})"
        )
    # NOTE: export already guarantees that the same symbol is used in metadata
    # for all InputDims related by equality constraints, so we can just unify
    # symbols with given input dimension values to check equality constraints.
    unification_map: Dict[sympy.Symbol, Any] = {}
    for (key_path, arg), node in zip(flat_args_with_paths, input_nodes):
        node_val = node.meta.get("val")
        if isinstance(node_val, FakeTensor):
            if not isinstance(arg, torch.Tensor):
                raise RuntimeError(
                    f"Expected input at {derive_description(key_path)} to be a tensor, but got {type(arg)}"
                )

            if len(node_val.shape) != len(arg.shape):
                raise RuntimeError(
                    f"Unexpected number of dimensions in input at {derive_description(key_path)}.shape "
                    f"(expected {node_val.shape}, got {arg.shape})"
                )

            for j, (arg_dim, node_dim) in enumerate(zip(arg.shape, node_val.shape)):
                if (
                    isinstance(arg_dim, torch.SymInt)
                    and not arg_dim.node.expr.is_number
                ):
                    # This can happen when, say, arg is a fake tensor.
                    # We do not run checks on symbolic shapes of fake inputs as
                    # such checks can affect the shape env.
                    continue
                if (
                    isinstance(node_dim, torch.SymInt)
                    and len(node_dim.node.expr.free_symbols) == 1
                ):
                    symbol = next(iter(node_dim.node.expr.free_symbols))
                    if symbol in unification_map:
                        existing_dim = node_dim.node.expr.subs(unification_map)
                        if arg_dim != existing_dim:
                            raise RuntimeError(
                                f"Expected input at {derive_description(key_path)}.shape[{j}] to be >= "
                                f"{existing_dim}, but got {arg_dim}"
                            )
                    else:
                        unification_map[symbol] = arg_dim
                elif (
                    isinstance(node_dim, torch.SymInt)
                    and not node_dim.node.expr.is_number
                ):
                    # this means we deferred a guard from export analysis to runtime, let this pass
                    # we'll add a runtime assert checking equality to this replacement expression
                    continue
                elif arg_dim != node_dim:
                    raise RuntimeError(
                        f"Expected input at {derive_description(key_path)}.shape[{j}] to be equal to "
                        f"{node_dim}, but got {arg_dim}"
                    )
        elif isinstance(node_val, (int, float, str)):
            if type(arg) != type(node_val) or arg != node_val:
                raise RuntimeError(
                    f"Expected input at {derive_description(key_path)} to be equal to {node_val}, but got {arg}"
                )

def test_check_default_value(self):
        class Config(models.Model):
            date_field = models.DateField(default=datetime.now())
            date_only = models.DateField(default=datetime.now().date())
            current_time = models.DateTimeField(default=datetime.now)

        date_field = Config._meta.get_field("date_field")
        date_only = Config._meta.get_field("date_only")
        current_time = Config._meta.get_field("current_time")
        warnings = date_field.check()
        warnings.extend(date_only.check())
        warnings.extend(current_time.check())  # doesn't raise a warning
        self.assertEqual(
            warnings,
            [
                DjangoWarning(
                    "Fixed default value provided.",
                    hint="It seems you set a fixed date / time / datetime "
                    "value as default for this field. This may not be "
                    "what you want. If you want to have the current date "
                    "as default, use `django.utils.timezone.now`",
                    obj=date_field,
                    id="fields.W161",
                ),
                DjangoWarning(
                    "Fixed default value provided.",
                    hint="It seems you set a fixed date / time / datetime "
                    "value as default for this field. This may not be "
                    "what you want. If you want to have the current date "
                    "as default, use `django.utils.timezone.now`",
                    obj=date_only,
                    id="fields.W161",
                ),
            ],
        )

def validate_field_choices_single_value(self):
        """Single value isn't a valid choice for the field."""

        class ExampleModel(models.Model):
            example_field = models.CharField(max_length=10, choices=("ab",))

        model_instance = ExampleModel._meta.get_field("example_field")
        check_results = model_instance.check()
        self.assertEqual(
            check_results,
            [
                Error(
                    "'choices' must be a mapping of actual values to human readable "
                    "names or an iterable containing (actual value, human readable "
                    "name) tuples.",
                    obj=model_instance,
                    id="fields.E005",
                ),
            ],
        )

