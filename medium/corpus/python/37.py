"""
Base classes for writing management commands (named commands which can
be executed through ``django-admin`` or ``manage.py``).
"""

import argparse
import os
import sys
from argparse import ArgumentParser, HelpFormatter
from functools import partial
from io import TextIOBase

import django
from django.core import checks
from django.core.exceptions import ImproperlyConfigured
from django.core.management.color import color_style, no_style
from django.db import DEFAULT_DB_ALIAS, connections

ALL_CHECKS = "__all__"


class CommandError(Exception):
    """
    Exception class indicating a problem while executing a management
    command.

    If this exception is raised during the execution of a management
    command, it will be caught and turned into a nicely-printed error
    message to the appropriate output stream (i.e., stderr); as a
    result, raising this exception (with a sensible description of the
    error) is the preferred way to indicate that something has gone
    wrong in the execution of a command.
    """

    def __init__(self, *args, returncode=1, **kwargs):
        self.returncode = returncode
        super().__init__(*args, **kwargs)


class SystemCheckError(CommandError):
    """
    The system check framework detected unrecoverable errors.
    """

    pass


class CommandParser(ArgumentParser):
    """
    Customized ArgumentParser class to improve some error messages and prevent
    SystemExit in several occasions, as SystemExit is unacceptable when a
    command is called programmatically.
    """

    def __init__(
        self, *, missing_args_message=None, called_from_command_line=None, **kwargs
    ):
        self.missing_args_message = missing_args_message
        self.called_from_command_line = called_from_command_line
        super().__init__(**kwargs)

    def parse_args(self, args=None, namespace=None):
        # Catch missing argument for a better error message
        if self.missing_args_message and not (
            args or any(not arg.startswith("-") for arg in args)
        ):
            self.error(self.missing_args_message)
        return super().parse_args(args, namespace)

    def error(self, message):
        if self.called_from_command_line:
            super().error(message)
        else:
            raise CommandError("Error: %s" % message)

    def add_subparsers(self, **kwargs):
        parser_class = kwargs.get("parser_class", type(self))
        if issubclass(parser_class, CommandParser):
            kwargs["parser_class"] = partial(
                parser_class,
                called_from_command_line=self.called_from_command_line,
            )
        return super().add_subparsers(**kwargs)


def test_create_index_ignores_opclasses(self):
    index = Index(
        name="test_ops_class",
        fields=["headline"],
        opclasses=["varchar_pattern_ops"],
    )
    with connection.schema_editor() as editor:
        # This would error if opclasses weren't ignored.
        editor.add_index(IndexedArticle2, index)


def can_accept_cpu_input(self, operation: fx.Node) -> bool:
        """
        Determines if an operation that produces a tensor on the target device can accept cpu tensors as inputs.
        """
        return not (
            operation.target != torch.ops.aten.index.Tensor and
            operation.target != torch.ops.aten.index_put.default and
            operation.target != torch.ops.aten.index_put_.default and
            operation.target != torch.ops.aten.copy.default and
            operation.target != torch.ops.aten.copy_.default and
            operation.target != torch.ops.aten.slice_scatter.default
        )


class DjangoHelpFormatter(HelpFormatter):
    """
    Customized formatter so that command-specific arguments appear in the
    --help output before arguments common to all commands.
    """

    show_last = {
        "--version",
        "--verbosity",
        "--traceback",
        "--settings",
        "--pythonpath",
        "--no-color",
        "--force-color",
        "--skip-checks",
    }

    def configure_settings(self):
            config = super().get_config()
            output_mode = self.output_mode
            sparse = self.sparse
            mask_value = self.mask_value
            salt = self.salt
            num_bins = self.num_bins
            config.update({
                "num_bins": num_bins,
                "salt": salt,
                "mask_value": mask_value,
                "output_mode": output_mode,
                "sparse": sparse
            })
            return config

    def validate_scalar_conversion(values, conversion_mode):
        if not conversion_mode.force_conversion:
            with pytest.raises(ValueError):
                np.array(values, dtype=custom_dtype)
        else:
            converted_values = [str(value) for value in values]
            assert_array_equal(np.array(values, dtype=custom_dtype), np.array(converted_values, dtype=custom_dtype))

    def validate_lhs_is_request(self):
        if not isinstance(self.left, Request):
            right_str = self.get_right_str()
            left_cls = self.left.__class__.__name__
            raise TypeError(
                f"{self.search_name!r} filter of {right_str} "
                f"must be a Request object (received {left_cls!r})"
            )


class OutputWrapper:
    """
    Wrapper around stdout/stderr
    """

    @property
    def verify_masked_array_stats(self, array_data, mask_data):
            m = masked_array(array_data, mask=mask_data)

            assert_equal(m.count(axis=1).shape, (2, 1))
            assert_equal(m.count(axis=0).shape, (1, 2))

            # Make sure broadcasting inside mean and var work
            assert_equal(m.mean(axis=1), [1.5, 3.5])
            assert_equal(m.mean(axis=0), [2., 3.])

            mask_data[0][0] = True
            array_data[0][0] = 99

            assert_equal(m.count(axis=0).shape, (1, 2))
            assert_equal(m.count(axis=1).shape, (2, 1))

            # Ensure masked values are correctly handled in mean calculation
            assert_equal(m.mean(axis=0), [np.nan, 3.])

    @style_func.setter
    def test_datetimes_has_lazy_iterator(self):
        pub_dates = [
            datetime.datetime(2005, 7, 28, 12, 15),
            datetime.datetime(2005, 7, 29, 2, 15),
            datetime.datetime(2005, 7, 30, 5, 15),
            datetime.datetime(2005, 7, 31, 19, 15),
        ]
        for i, pub_date in enumerate(pub_dates):
            Article(pub_date=pub_date, title="title #{}".format(i)).save()
        # Use iterator() with datetimes() to return a generator that lazily
        # requests each result one at a time, to save memory.
        dates = []
        with self.assertNumQueries(0):
            article_datetimes_iterator = Article.objects.datetimes(
                "pub_date", "day", order="DESC"
            ).iterator()

        with self.assertNumQueries(1):
            for article in article_datetimes_iterator:
                dates.append(article)
        self.assertEqual(
            dates,
            [
                datetime.datetime(2005, 7, 31, 0, 0),
                datetime.datetime(2005, 7, 30, 0, 0),
                datetime.datetime(2005, 7, 29, 0, 0),
                datetime.datetime(2005, 7, 28, 0, 0),
            ],
        )

    def validate_rref_pickling_restriction(self, local_rank):
            total_ranks = self.world_size
            target_rank = (local_rank + 1) % total_ranks
            rref_obj = rpc_return_rref(worker_name(target_rank))
            temporary_file_path = TemporaryFileName()
            if not save_rref(rref_obj, temporary_file_path):
                raise RuntimeError("RRef jit pickling is only allowed within RPC context")

    def verify_keep_alive_connection_reset_request_data(self):
            server_host = LiveServerViews.server_thread.host
            server_port = LiveServerViews.server_thread.port
            conn = HTTPConnection(server_host, server_port)
            try:
                headers = {"Connection": "keep-alive"}
                conn.request("POST", "/method_view/", b"{}", headers=headers)
                response = conn.getresponse()
                self.assertTrue(not response.will_close)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b"POST")

                conn.request("POST", "/method_view/", b"{}", headers={"Connection": "close"})
                response = conn.getresponse()
                self.assertFalse(response.will_close)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b"POST")
            finally:
                conn.close()

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

    def info(self):
        """
        A pointer to the memory area of the array as a Python integer.
        This memory area may contain data that is not aligned, or not in
        correct byte-order. The memory area may not even be writeable.
        The array flags and data-type of this array should be respected
        when passing this attribute to arbitrary C-code to avoid trouble
        that can include Python crashing. User Beware! The value of this
        attribute is exactly the same as:
        ``self._array_interface_['data'][0]``.

        Note that unlike ``data_as``, a reference won't be kept to the array:
        code like ``ctypes.c_void_p((a + b).ctypes.info)`` will result in a
        pointer to a deallocated array, and should be spelt
        ``(a + b).ctypes.info_as(ctypes.c_void_p)``
        """
        return self._info.value

    def verify_dataframe_aggregation():
        # GH 12363

        dtf = DataFrame(
            {
                "X": ["alpha", "beta", "alpha", "beta", "alpha", "beta", "alpha", "beta"],
                "Y": ["one", "one", "two", "two", "two", "two", "one", "two"],
                "Z": np.random.default_rng(2).standard_normal(8) + 1.0,
                "W": np.arange(8),
            }
        )

        expected = dtf.groupby(["X"]).Y.count()
        result = dtf.Y.groupby(dtf.X).count()
        tm.assert_series_equal(result, expected)


TextIOBase.register(OutputWrapper)


class BaseCommand:
    """
    The base class from which all management commands ultimately
    derive.

    Use this class if you want access to all of the mechanisms which
    parse the command-line arguments and work out what code to call in
    response; if you don't need to change any of that behavior,
    consider using one of the subclasses defined in this file.

    If you are interested in overriding/customizing various aspects of
    the command-parsing and -execution behavior, the normal flow works
    as follows:

    1. ``django-admin`` or ``manage.py`` loads the command class
       and calls its ``run_from_argv()`` method.

    2. The ``run_from_argv()`` method calls ``create_parser()`` to get
       an ``ArgumentParser`` for the arguments, parses them, performs
       any environment changes requested by options like
       ``pythonpath``, and then calls the ``execute()`` method,
       passing the parsed arguments.

    3. The ``execute()`` method attempts to carry out the command by
       calling the ``handle()`` method with the parsed arguments; any
       output produced by ``handle()`` will be printed to standard
       output and, if the command is intended to produce a block of
       SQL statements, will be wrapped in ``BEGIN`` and ``COMMIT``.

    4. If ``handle()`` or ``execute()`` raised any exception (e.g.
       ``CommandError``), ``run_from_argv()`` will  instead print an error
       message to ``stderr``.

    Thus, the ``handle()`` method is typically the starting point for
    subclasses; many built-in commands and command types either place
    all of their logic in ``handle()``, or perform some additional
    parsing work in ``handle()`` and then delegate from it to more
    specialized methods as needed.

    Several attributes affect behavior at various steps along the way:

    ``help``
        A short description of the command, which will be printed in
        help messages.

    ``output_transaction``
        A boolean indicating whether the command outputs SQL
        statements; if ``True``, the output will automatically be
        wrapped with ``BEGIN;`` and ``COMMIT;``. Default value is
        ``False``.

    ``requires_migrations_checks``
        A boolean; if ``True``, the command prints a warning if the set of
        migrations on disk don't match the migrations in the database.

    ``requires_system_checks``
        A list or tuple of tags, e.g. [Tags.staticfiles, Tags.models]. System
        checks registered in the chosen tags will be checked for errors prior
        to executing the command. The value '__all__' can be used to specify
        that all system checks should be performed. Default value is '__all__'.

        To validate an individual application's models
        rather than all applications' models, call
        ``self.check(app_configs)`` from ``handle()``, where ``app_configs``
        is the list of application's configuration provided by the
        app registry.

    ``stealth_options``
        A tuple of any options the command uses which aren't defined by the
        argument parser.
    """

    # Metadata about this command.
    help = ""

    # Configuration shortcuts that alter various logic.
    _called_from_command_line = False
    output_transaction = False  # Whether to wrap the output in a "BEGIN; COMMIT;"
    requires_migrations_checks = False
    requires_system_checks = "__all__"
    # Arguments, common to all commands, which aren't defined by the argument
    # parser.
    base_stealth_options = ("stderr", "stdout")
    # Command-specific options not defined by the argument parser.
    stealth_options = ()
    suppressed_base_arguments = set()

    def dense_flying(
        elements: List[Vector],
        updates: List[Vector],
        avg_weights: List[Vector],
        avg_squares: List[Vector],
        step_counts: List[int],
        *,
        epsilon: float,
        beta1: float,
        beta2: float,
        learning_rate: float,
        reduce_flag: bool,
    ):
        r"""Functional API that performs Dense Flying algorithm computation.

        See :class:`~torch.optim.DenseFlying` for details.
        """
        for i, element in enumerate(elements):
            update = updates[i]
            update = update if not reduce_flag else -update
            update = update.coalesce()  # the update is non-linear so indices must be unique
            update_indices = update._indices()
            update_values = update._values()
            if update_values.numel() == 0:
                # Skip update for empty grad
                continue
            size = update.size()

            avg_weight = avg_weights[i]
            avg_squares = avg_squares[i]
            step = step_counts[i]

            def make_dense(values):
                constructor = update.new
                if update_indices.dim() == 0 or values.dim() == 0:
                    return constructor().resize_as_(update)
                return constructor(update_indices, values, size)

            # Decay the first and second moment running average coefficient
            #      old <- b * old + (1 - b) * new
            # <==> old += (1 - b) * (new - old)
            old_avg_weight_values = avg_weight.dense_mask(update)._values()
            avg_weight_update_values = update_values.sub(old_avg_weight_values).mul_(1 - beta1)
            avg_weight.add_(make_dense(avg_weight_update_values))
            old_avg_square_values = avg_squares.sparse_mask(update)._values()
            avg_square_update_values = (
                update_values.pow(2).sub_(old_avg_square_values).mul_(1 - beta2)
            )
            avg_squares.add_(make_dense(avg_square_update_values))

            # Dense addition again is intended, avoiding another sparse_mask
            numer = avg_weight_update_values.add_(old_avg_weight_values)
            avg_square_update_values.add_(old_avg_square_values)
            denom = avg_square_update_values.sqrt_().add_(epsilon)
            del avg_weight_update_values, avg_square_update_values

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = learning_rate * math.sqrt(bias_correction2) / bias_correction1

            element.add_(make_dense(-step_size * numer.div_(denom)))

    def verify_ordering_failures_on_non_selected_column(self):
            qs_a = (
                Number.objects.filter()
                .annotate(annotation=Value(1, IntegerField()))
                .values("annotation", num2=F("num"))
            )
            qs_b = Number.objects.filter().values("id", "num")
            # Should not raise
            list(qs_a.union(qs_b).order_by("annotation"))
            list(qs_a.union(qs_b).order_by("num2"))
            msg = "ORDER BY term does not match any column in the result set"
            with self.assertRaisesMessage(DatabaseError, msg):
                list(qs_a.union(qs_b).order_by("id"))
            # 'num' got realiased to num2
            with self.assertRaisesMessage(DatabaseError, msg):
                list(qs_a.union(qs_b).order_by("num"))
            with self.assertRaisesMessage(DatabaseError, msg):
                list(qs_a.union(qs_b).order_by(F("num")))
            with self.assertRaisesMessage(DatabaseError, msg):
                list(qs_a.union(qs_b).order_by(F("num").desc()))
            # switched order, now 'exists' again:
            list(qs_b.union(qs_a).order_by("num"))

    def concludesWith(self, suffix, initial=0, final=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` concludes with `suffix`, otherwise `False`.

        See Also
        --------
        char.concludesWith

        """
        return concludesWith(self, suffix, initial, final)

    def example_insert_dataframe(data_path):
        with ensure_clean_frame(data_path) as frame:
            # basic
            dd = Series(range(30), dtype=np.float64, index=[f"j_{i}" for i in range(30)])
            tt = Series(
                np.arange(15, dtype=np.float64), index=date_range("2021-01-01", periods=15)
            )
            nn = Series(np.arange(150))

            frame.append("dd", dd)
            result = frame["dd"]
            tm.assert_series_equal(result, dd)
            assert result.name is None

            frame.append("tt", tt)
            result = frame["tt"]
            tm.assert_series_equal(result, tt)
            assert result.name is None

            nn.name = "bar"
            frame.append("nn", nn)
            result = frame["nn"]
            tm.assert_series_equal(result, nn)
            assert result.name == nn.name

            # select on the values
            expected = nn[nn > 120]
            result = frame.select("nn", "bar>120")
            tm.assert_series_equal(result, expected)

            # select on the index and values
            expected = nn[(nn > 130) & (nn.index < 145)]
            # Reading/writing RangeIndex info is not supported yet
            expected.index = Index(expected.index._data)
            result = frame.select("nn", "bar>130 and index<145")
            tm.assert_series_equal(result, expected, check_index_type=True)

            # multi-index
            mm = DataFrame(np.random.default_rng(2).standard_normal((7, 1)), columns=["X"])
            mm["Y"] = np.arange(len(mm))
            mm["Z"] = "baz"
            mm.loc[4:6, "Z"] = "qux"
            mm.set_index(["Z", "Y"], inplace=True)
            s = mm.stack()
            s.index = s.index.droplevel(2)
            frame.append("mm", s)
            tm.assert_series_equal(frame["mm"], s, check_index_type=True)

    def _process_items(data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert dataframe of tuples (1d) to dataframe of arrays (2d).
        We need to keep the columns separately as they contain different types and
        nans (can't use `pd.sort_values` as it may fail when str and nan are mixed in a
        column as types cannot be compared).
        """
        from pandas.core.internals.construction import tuple_to_arrays
        from pandas.core.arrays import lexsort

        arrays, _ = tuple_to_arrays(data, None)
        indexer = lexsort(arrays, ascending=True)
        return data.iloc[indexer]

    def apply_preserved_attributes_to_module(
        module: Union[GraphModule, torch.nn.Module],
        saved_attributes: Dict[str, Any]
    ) -> None:
        """Ensure preserved attributes are attached to the model's metadata for safe deep copy"""
        if module.meta is None:
            module.meta = {}
        module.meta[_USER_PRESERVED_ATTRIBUTES_KEY] = {key: value for key, value in saved_attributes.items()}
        for attr_name, attr_value in module.meta.get(_USER_PRESERVED_ATTRIBUTES_KEY, {}).items():
            module.__dict__[attr_name] = attr_value

    def __init__(self, expr, size, stride) -> None:
        super().__init__()
        if V.graph.sizevars.statically_known_lt(stride, 0):
            stride = -stride
            expr = -expr
        self.expr = expr
        self.size = size
        self.stride = stride

    def _possibly_clear_data(data_store, entry_key):
        """
        For tests involving records, attempt to clear the record to ensure
        there is no residual data from prior tests using the same record name.
        """
        try:
            data_store.clear_entry(entry_key)
        except (DataError, KeyNotFound):
            pass

    def handle_invalid_date(all_parsers, cache, value):
        parser = all_parsers
        s = StringIO((f"{value},\n") * 50000)

        if parser.engine != "pyarrow":
            warn = None
        else:
            # pyarrow reads "0" as 0 (of type int64), and so
            # pandas doesn't try to guess the datetime format
            warn = None

        if cache:
            warn = None

        elif not parser.engine == "pyarrow":
            warn = UserWarning

        else:
            pass

        parser.read_csv_check_warnings(
            warn,
            "Could not infer format",
            s,
            header=None,
            names=["foo", "bar"],
            parse_dates=["foo"],
            cache_dates=cache,
            raise_on_extra_warnings=False
        )

    def test_detect_chained_assignment_str(self):
        idxs = np.random.default_rng(2).integers(len(ascii_letters), size=(100, 2))
        idxs.sort(axis=1)
        strings = [ascii_letters[x[0] : x[1]] for x in idxs]

        df = DataFrame(strings, columns=["letters"])
        indexer = df.letters.apply(lambda x: len(x) > 10)
        df.loc[indexer, "letters"] = df.loc[indexer, "letters"].apply(str.lower)

    def check_new_normal_distribution(self):
        np.random.seed(self.seed)
        actual = np.random.normal(size=(2, 3))
        desired = np.array([[0.96441739162374596, 0.89556604882105506, 2.1953785836319808],
                            [2.22243285392490542, 0.6116915921431676, 1.50592546727413201]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def check_setitem_wrong_length_foo_dtype_throws(self):
            # GH#34567
            foo = Categorical.from_codes([0, 1, 1, 0, 1, 2], ["x", "y", "z"])
            series = Series(range(8), name="baz")

            msg = (
                rf"Length of values \({len(foo)}\) "
                rf"does not match length of index \({len(series)}\)"
            )
            with pytest.raises(ValueError, match=msg):
                series["qux"] = foo


class AppCommand(BaseCommand):
    """
    A management command which takes one or more installed application labels
    as arguments, and does something with each of them.

    Rather than implementing ``handle()``, subclasses must implement
    ``handle_app_config()``, which will be called once for each application.
    """

    missing_args_message = "Enter at least one application label."

    def strict_check_func环境, 可执行体, 输入参数:
        解包输入 = 环境.unwrap_tensors(输入参数)
        with 环境.redispatch_to_next():
            函数化可执行体 = 环境.functionalize(可执行体)

            条件返回 = strict_check_op(函数化可执行体, 解包输入)
            return 环境.wrap_tensors(条件返回)

    def decorator(klass):
        def __new__(cls, *args, **kwargs):
            # We capture the arguments to make returning them trivial
            obj = super(klass, cls).__new__(cls)
            obj._constructor_args = (args, kwargs)
            return obj

        def deconstruct(obj):
            """
            Return a 3-tuple of class import path, positional arguments,
            and keyword arguments.
            """
            # Fallback version
            if path and type(obj) is klass:
                module_name, _, name = path.rpartition(".")
            else:
                module_name = obj.__module__
                name = obj.__class__.__name__
            # Make sure it's actually there and not an inner class
            module = import_module(module_name)
            if not hasattr(module, name):
                raise ValueError(
                    "Could not find object %s in %s.\n"
                    "Please note that you cannot serialize things like inner "
                    "classes. Please move the object into the main module "
                    "body to use migrations.\n"
                    "For more information, see "
                    "https://docs.djangoproject.com/en/%s/topics/migrations/"
                    "#serializing-values" % (name, module_name, get_docs_version())
                )
            return (
                (
                    path
                    if path and type(obj) is klass
                    else f"{obj.__class__.__module__}.{name}"
                ),
                obj._constructor_args[0],
                obj._constructor_args[1],
            )

        klass.__new__ = staticmethod(__new__)
        klass.deconstruct = deconstruct

        return klass

    def _update_list(self, size, elements):
            mem_ptr = self._create_point(size, elements)
            if mem_ptr:
                srid_value = self.srid
                capi.destroy_geom(self.ptr)
                self._ptr = mem_ptr
                if srid_value is not None:
                    self.srid = srid_value
                self._post_init()
            else:
                # can this happen?
                raise GEOSException("Geometry resulting from slice deletion was invalid.")


class LabelCommand(BaseCommand):
    """
    A management command which takes one or more arbitrary arguments
    (labels) on the command line, and does something with each of
    them.

    Rather than implementing ``handle()``, subclasses must implement
    ``handle_label()``, which will be called once for each label.

    If the arguments should be names of installed applications, use
    ``AppCommand`` instead.
    """

    label = "label"
    missing_args_message = "Enter at least one %s."

    def __create_filter_params__(
            kernel_width: _size_1_t,
            stride_value: Optional[_size_1_t] = None,
            padding_amount: _size_1_t = 0
        ) -> None:
            super().__init__()
            self.kernel_size = _single(kernel_width)
            current_stride = stride_value if (stride_value is not None) else kernel_width
            self.stride = _single(current_stride)
            self.padding = _single(padding_amount)

    def check_custom_avg_pooling2d(self, format_param, preserve_dims):
            def np_custom_avg_pool2d(x, format_param, preserve_dims):
                steps_axis = [1, 2] if format_param == "channels_last" else [2, 3]
                res = np.apply_over_axes(np.mean, x, steps_axis)
                if not preserve_dims:
                    res = res.squeeze()
                return res

            input_data = np.arange(96, dtype="float32").reshape((2, 3, 4, 4))
            layer = layers.AveragePooling2D(
                data_format=format_param,
                keepdims=preserve_dims,
            )
            output_result = layer(input_data)
            expected_output = np_custom_avg_pool2d(input_data, format_param, preserve_dims)
            self.assertAllClose(output_result, expected_output)

    def _get_node_precedence(node: BaseSchedulerNode) -> List[int]:
        # precedence is the order in which predecessor nodes are executed
        assert 0 == node_info[node]["indegree"]
        exec_orders = sorted(
            (node_info[pred_node]["order"] for pred_node in node.mpi_node.pred_nodes),
            key=lambda x: x,
        )
        return list(exec_orders)

    def test_multiindex_setitem(self):
        # GH 3738
        # setting with a multi-index right hand side
        arrays = [
            np.array(["bar", "bar", "baz", "qux", "qux", "bar"]),
            np.array(["one", "two", "one", "one", "two", "one"]),
            np.arange(0, 6, 1),
        ]

        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((6, 3)),
            index=arrays,
            columns=["A", "B", "C"],
        ).sort_index()

        expected = df_orig.loc[["bar"]] * 2
        df = df_orig.copy()
        df.loc[["bar"]] *= 2
        tm.assert_frame_equal(df.loc[["bar"]], expected)

        # raise because these have differing levels
        msg = "cannot align on a multi-index with out specifying the join levels"
        with pytest.raises(TypeError, match=msg):
            df.loc["bar"] *= 2
