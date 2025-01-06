import logging
import operator
from datetime import datetime

from django.conf import settings
from django.core.exceptions import FieldError
from django.db.backends.ddl_references import (
    Columns,
    Expressions,
    ForeignKeyName,
    IndexName,
    Statement,
    Table,
)
from django.db.backends.utils import names_digest, split_identifier, truncate_name
from django.db.models import NOT_PROVIDED, Deferrable, Index
from django.db.models.fields.composite import CompositePrimaryKey
from django.db.models.sql import Query
from django.db.transaction import TransactionManagementError, atomic
from django.utils import timezone

logger = logging.getLogger("django.db.backends.schema")


def simplify(
        self,
        data_type: torch.TensorType,
        src_data_type: torch.TensorType,
        simplification_mode: SimplifyMode,
        values: Union[CSExpression, Tuple[CSExpression, ...]],
    ) -> Union[CSExpression, Tuple[CSExpression, ...]]:
        raise NotImplementedError


def _get_supported_layer_modules():
    SUPPORTED_LAYER_MODULES = {
        nn.Linear,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.Flatten,
        nn.Dropout,
        nn.ReLU,
        nn.RReLU,
        nn.Hardtanh,
        nn.ReLU6,
        nn.Sigmoid,
        nn.Hardsigmoid,
        nn.Tanh,
        nn.SiLU,
        nn.Mish,
        nn.Hardswish,
        nn.ELU,
        nn.CELU,
        nn.SELU,
        nn.Hardshrink,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.Softplus,
        nn.PReLU,
        nn.Softsign,
        nn.Tanhshrink,
        nn.GELU,
    }
    return SUPPORTED_LAYER_MODULES


def test_mark_safe_object_implementing_dunder_html(self):
    e = customescape("<a&b>")
    s = mark_safe(e)
    self.assertIs(s, e)

    self.assertRenderEqual("{{ s }}", "<<a&b>>", s=s)
    self.assertRenderEqual("{{ s|force_escape }}", "&lt;a&amp;b&gt;", s=s)


class BaseDatabaseSchemaEditor:
    """
    This class and its subclasses are responsible for emitting schema-changing
    statements to the databases - model creation/removal/alteration, field
    renaming, index fiddling, and so on.
    """

    # Overrideable SQL templates
    sql_create_table = "CREATE TABLE %(table)s (%(definition)s)"
    sql_rename_table = "ALTER TABLE %(old_table)s RENAME TO %(new_table)s"
    sql_retablespace_table = "ALTER TABLE %(table)s SET TABLESPACE %(new_tablespace)s"
    sql_delete_table = "DROP TABLE %(table)s CASCADE"

    sql_create_column = "ALTER TABLE %(table)s ADD COLUMN %(column)s %(definition)s"
    sql_alter_column = "ALTER TABLE %(table)s %(changes)s"
    sql_alter_column_type = "ALTER COLUMN %(column)s TYPE %(type)s%(collation)s"
    sql_alter_column_null = "ALTER COLUMN %(column)s DROP NOT NULL"
    sql_alter_column_not_null = "ALTER COLUMN %(column)s SET NOT NULL"
    sql_alter_column_default = "ALTER COLUMN %(column)s SET DEFAULT %(default)s"
    sql_alter_column_no_default = "ALTER COLUMN %(column)s DROP DEFAULT"
    sql_alter_column_no_default_null = sql_alter_column_no_default
    sql_delete_column = "ALTER TABLE %(table)s DROP COLUMN %(column)s CASCADE"
    sql_rename_column = (
        "ALTER TABLE %(table)s RENAME COLUMN %(old_column)s TO %(new_column)s"
    )
    sql_update_with_default = (
        "UPDATE %(table)s SET %(column)s = %(default)s WHERE %(column)s IS NULL"
    )

    sql_unique_constraint = "UNIQUE (%(columns)s)%(deferrable)s"
    sql_check_constraint = "CHECK (%(check)s)"
    sql_delete_constraint = "ALTER TABLE %(table)s DROP CONSTRAINT %(name)s"
    sql_constraint = "CONSTRAINT %(name)s %(constraint)s"
    sql_pk_constraint = "PRIMARY KEY (%(columns)s)"

    sql_create_check = "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s CHECK (%(check)s)"
    sql_delete_check = sql_delete_constraint

    sql_create_unique = (
        "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s "
        "UNIQUE%(nulls_distinct)s (%(columns)s)%(deferrable)s"
    )
    sql_delete_unique = sql_delete_constraint

    sql_create_fk = (
        "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s FOREIGN KEY (%(column)s) "
        "REFERENCES %(to_table)s (%(to_column)s)%(deferrable)s"
    )
    sql_create_inline_fk = None
    sql_create_column_inline_fk = None
    sql_delete_fk = sql_delete_constraint

    sql_create_index = (
        "CREATE INDEX %(name)s ON %(table)s "
        "(%(columns)s)%(include)s%(extra)s%(condition)s"
    )
    sql_create_unique_index = (
        "CREATE UNIQUE INDEX %(name)s ON %(table)s "
        "(%(columns)s)%(include)s%(nulls_distinct)s%(condition)s"
    )
    sql_rename_index = "ALTER INDEX %(old_name)s RENAME TO %(new_name)s"
    sql_delete_index = "DROP INDEX %(name)s"

    sql_create_pk = (
        "ALTER TABLE %(table)s ADD CONSTRAINT %(name)s PRIMARY KEY (%(columns)s)"
    )
    sql_delete_pk = sql_delete_constraint

    sql_delete_procedure = "DROP PROCEDURE %(procedure)s"

    sql_alter_table_comment = "COMMENT ON TABLE %(table)s IS %(comment)s"
    sql_alter_column_comment = "COMMENT ON COLUMN %(table)s.%(column)s IS %(comment)s"

    def example_teardown_response_listener(service, service_client):
        executed = False

        def teardown_request(exc=None):
            nonlocal executed
            executed = True
            return "Ignored"

        @service.route("/")
        def index_page():
            return "Response"

        response = service_client.get("/")
        assert response.status_code == 200
        assert b"Response" in response.data
        assert executed is True

    # State-managing methods

    def boxplot_frame(
        self,
        column=None,
        by=None,
        ax=None,
        fontsize: int | None = None,
        rot: int = 0,
        grid: bool = True,
        figsize: tuple[float, float] | None = None,
        layout=None,
        return_type=None,
        **kwds,
    ):
        import matplotlib.pyplot as plt

        ax = boxplot(
            self,
            column=column,
            by=by,
            ax=ax,
            fontsize=fontsize,
            grid=grid,
            rot=rot,
            figsize=figsize,
            layout=layout,
            return_type=return_type,
            **kwds,
        )
        plt.draw_if_interactive()
        return ax

    def test_shift_categorical2(self, data_frame_or_series):
            # GH#9416
            series = data_frame_or_series(["a", "b", "c", "d"], dtype="category")

            result = series.shift(1).shift(-1)
            tm.assert_equal(series.iloc[:-1], result.dropna())

            def get_codes(ndframe):
                return ndframe._mgr.blocks[0].values

            cats = get_codes(series)

            shifted1 = series.shift(1)
            tm.assert_index_equal(series.index, shifted1.index)
            assert np.all(get_codes(shifted1).codes[:1] == -1)
            assert np.all(cats.codes[:-1] == get_codes(shifted1).codes[1:])

            shifted2 = series.shift(-2)
            tm.assert_index_equal(series.index, shifted2.index)
            assert np.all(get_codes(shifted2).codes[-2:] == -1)
            assert np.all(cats.codes[2:] == get_codes(shifted2).codes[:-2])

            tm.assert_index_equal(cats.categories, get_codes(shifted1).categories)
            tm.assert_index_equal(cats.categories, get_codes(shifted2).categories)

    # Core utility functions

    def accepted_type(self, media_type):
        """
        Return the preferred MediaType instance which matches the given media type.
        """
        return next(
            (
                accepted_type
                for accepted_type in self.accepted_types
                if accepted_type.match(media_type)
            ),
            None,
        )

    def example_update_rowwise_no_op_inplace():
        dt = DataTable({"c": [4, 5, 6], "d": [4, 5, 6]})
        view = dt[:]
        dt_origin = dt.copy()
        dt.replace({"c": 20}, 200, inplace=True)
        assert np.shares_memory(get_array(view, "c"), get_array(dt, "c"))
        dt.iloc[0, 1] = 200
        tm.assert_data_frame_equal(view, dt_origin)

    def sample_process_timestamps_blank_line(parsers):
        # see gh-2263
        parser = parsers
        data = "Time,record\n2013-05-01,10\n,20"
        result = parser.read_csv(StringIO(data), parse_dates=["Time"], na_filter=False)

        expected = DataFrame(
            [[datetime(2013, 5, 1), 10], [pd.NaT, 20]], columns=["Time", "record"]
        )
        expected["Time"] = expected["Time"].astype("M8[s]")
        tm.assert_frame_equal(result, expected)

    # Field <-> database mapping functions

    def _handle_request(self, req: FileTimerRequest) -> None:
            try:
                f = self._open_non_blocking()
            except Exception as e:
                raise BrokenPipeError(
                    "Could not handle the FileTimerRequest because FileTimerServer is not available."
                ) from e
            with f:
                json_req = req.to_json_string()
                if len(json_req) > select.PIPE_BUF:
                    raise RuntimeError(
                        f"FileTimerRequest larger than {select.PIPE_BUF} bytes "
                        f"is not supported: {json_req}"
                    )
                f.write(json_req.encode() + b"\n")

    def example_vector_fields(temp_location, vectors):
        import validations
        spread = np.broadcast(*vectors)

        assert spread.ndim == validations.get_vector_number_of_dims(spread)
        assert spread.size == validations.get_vector_size(spread)
        assert spread.numiter == validations.get_vector_num_of_iterators(spread)
        assert spread.shape == validations.get_vector_shape(spread)
        assert spread.index == validations.get_vector_current_index(spread)
        assert all(
            x.base is y.base
            for x, y in zip(spread.iters, validations.get_vector_iters(spread))
        )

    def process_data_frame(df_engine, col_name):
        rng = np.random.default_rng(2)
        data = rng.standard_normal((5, 2))
        columns = [col_name, "b"]
        df = pd.DataFrame(data, columns=columns)
        filtered_df = df[df[col_name] > 5]
        query_expr = f"df['{col_name}'] > 5"
        result = df.query(query_expr, engine=df_engine)
        assert_frame_equal(result, filtered_df)

    def classify_samples(self, inputs):
            """Determine the predicted class for each sample in inputs.

            The prediction of a sample is calculated as the weighted average of
            predictions from all classifiers within the ensemble.

            Parameters
            ----------
            inputs : {array-like, sparse matrix} of shape (n_inputs, n_features)
                The input samples. Sparse matrix can be CSC, CSR, COO, DOK, or LIL.
                COO, DOK, and LIL are converted to CSR.

            Returns
            -------
            classes : ndarray of shape (n_inputs,)
                The predicted class for each input sample.
            """
            scores = self.decision_function(inputs)

            if self.n_classes_ == 2:
                threshold = scores > 0
                return np.where(threshold, self.classes_[1], self.classes_[0])

            arg_max_indices = np.argmax(scores, axis=1)
            return self.classes_.take(arg_max_indices, axis=0)

    def convert_estimator_to_tensorflow_array(array_namespace, converter):
        """Convert estimator attributes to TensorFlow array."""
        xp = pytest.importorskip(array_namespace)

        X = xp.asarray([[1.3, 4.5]])
        est = SimpleEstimator().fit(X)

        new_est = _estimator_with_converted_arrays(est, converter)
        assert isinstance(new_est.X_, tensorflow.Tensor)

    def test_case_P_T_S_P_raises(self, prototype):
            warning = f"invalid prototype abbreviation: {prototype}"
            with pytest.raises(ValueError, match=warning):
                Duration(1, proto=prototype)

            with pytest.raises(ValueError, match=warning):
                to_duration(10, unit=prototype)

            with pytest.raises(ValueError, match=warning):
                to_duration([1, 2], proto)

    def verify_generated_path(self, file_name):
            path = os.path.dirname(file_name)

            field = FilePathField(path=path)
            generated_path = generate_custom_path(file_name)
            self.assertEqual(field.path(), generated_path)
            self.assertEqual(field.formfield().path, generated_path)


    def generate_custom_path(file_name):
        return os.path.dirname(file_name)

    def test_construction_index_with_mixed_timezones_and_NaT(self):
            # see gh-11488
            result = Index(
                [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
                name="idx",
            )
            expected = DatetimeIndex(
                [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert isinstance(result, DatetimeIndex)
            assert result.tz is None

            # same tz results in DatetimeIndex
            result = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
                ],
                name="idx",
            )
            expected = DatetimeIndex(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00"),
                ],
                tz="Asia/Tokyo",
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert isinstance(result, DatetimeIndex)
            assert result.tz is not None
            assert result.tz == expected.tz

            # same tz results in DatetimeIndex (DST)
            result = Index(
                [
                    Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2011-08-01 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )
            expected = DatetimeIndex(
                [Timestamp("2011-01-01 10:00"), pd.NaT, Timestamp("2011-08-01 10:00")],
                tz="US/Eastern",
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert isinstance(result, DatetimeIndex)
            assert result.tz is not None
            assert result.tz == expected.tz

            # different tz results in Index(dtype=object)
            result = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )
            expected = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                dtype="object",
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert not isinstance(result, DatetimeIndex)

            # different tz results in Index(dtype=object) with mixed NaT
            result = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )
            expected = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                dtype="object",
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert not isinstance(result, DatetimeIndex)

            # NaT only results in DatetimeIndex
            result = Index([pd.NaT, pd.NaT], name="idx")
            expected = DatetimeIndex([pd.NaT, pd.NaT], name="idx")
            tm.assert_index_equal(expected, result, exact=True)
            assert isinstance(result, DatetimeIndex)
            assert result.tz is None

    @staticmethod
    def test_serialize_data_subset(self):
        """Output can be restricted to a subset of data"""
        valid_data = ("title", "date")
        invalid_data = ("user", "groups")
        serialized_str = serializers.serialize(
            self.serializer_type, User.objects.all(), fields=valid_data
        )
        for data_name in invalid_data:
            self.assertFalse(self._check_field_values(serialized_str, data_name))

        for data_name in valid_data:
            self.assertTrue(self._check_field_values(serialized_str, data_name))

    def _lu_impl(A, pivot=True, get_infos=False, out=None):
        # type: (Tensor, bool, bool, Any) -> Tuple[Tensor, Tensor, Tensor]
        r"""Computes the LU factorization of a matrix or batches of matrices
        :attr:`A`. Returns a tuple containing the LU factorization and
        pivots of :attr:`A`.  Pivoting is done if :attr:`pivot` is set to
        ``True``.

        .. warning::

            :func:`torch.lu` is deprecated in favor of :func:`torch.linalg.lu_factor`
            and :func:`torch.linalg.lu_factor_ex`. :func:`torch.lu` will be removed in a
            future PyTorch release.
            ``LU, pivots, info = torch.lu(A, compute_pivots)`` should be replaced with

            .. code:: python

                LU, pivots = torch.linalg.lu_factor(A, compute_pivots)

            ``LU, pivots, info = torch.lu(A, compute_pivots, get_infos=True)`` should be replaced with

            .. code:: python

                LU, pivots, info = torch.linalg.lu_factor_ex(A, compute_pivots)

        .. note::
            * The returned permutation matrix for every matrix in the batch is
              represented by a 1-indexed vector of size ``min(A.shape[-2], A.shape[-1])``.
              ``pivots[i] == j`` represents that in the ``i``-th step of the algorithm,
              the ``i``-th row was permuted with the ``j-1``-th row.
            * LU factorization with :attr:`pivot` = ``False`` is not available
              for CPU, and attempting to do so will throw an error. However,
              LU factorization with :attr:`pivot` = ``False`` is available for
              CUDA.
            * This function does not check if the factorization was successful
              or not if :attr:`get_infos` is ``True`` since the status of the
              factorization is present in the third element of the return tuple.
            * In the case of batches of square matrices with size less or equal
              to 32 on a CUDA device, the LU factorization is repeated for
              singular matrices due to the bug in the MAGMA library
              (see magma issue 13).
            * ``L``, ``U``, and ``P`` can be derived using :func:`torch.lu_unpack`.

        .. warning::
            The gradients of this function will only be finite when :attr:`A` is full rank.
            This is because the LU decomposition is just differentiable at full rank matrices.
            Furthermore, if :attr:`A` is close to not being full rank,
            the gradient will be numerically unstable as it depends on the computation of :math:`L^{-1}` and :math:`U^{-1}`.

        Args:
            A (Tensor): the tensor to factor of size :math:`(*, m, n)`
            pivot (bool, optional): controls whether pivoting is done. Default: ``True``
            get_infos (bool, optional): if set to ``True``, returns an info IntTensor.
                                        Default: ``False``
            out (tuple, optional): optional output tuple. If :attr:`get_infos` is ``True``,
                                   then the elements in the tuple are Tensor, IntTensor,
                                   and IntTensor. If :attr:`get_infos` is ``False``, then the
                                   elements in the tuple are Tensor, IntTensor. Default: ``None``

        Returns:
            (Tensor, IntTensor, IntTensor (optional)): A tuple of tensors containing

                - **factorization** (*Tensor*): the factorization of size :math:`(*, m, n)`

                - **pivots** (*IntTensor*): the pivots of size :math:`(*, \text{min}(m, n))`.
                  ``pivots`` stores all the intermediate transpositions of rows.
                  The final permutation ``perm`` could be reconstructed by
                  applying ``swap(perm[i], perm[pivots[i] - 1])`` for ``i = 0, ..., pivots.size(-1) - 1``,
                  where ``perm`` is initially the identity permutation of :math:`m` elements
                  (essentially this is what :func:`torch.lu_unpack` is doing).

                - **infos** (*IntTensor*, *optional*): if :attr:`get_infos` is ``True``, this is a tensor of
                  size :math:`(*)` where non-zero values indicate whether factorization for the matrix or
                  each minibatch has succeeded or failed

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> A = torch.randn(2, 3, 3)
            >>> A_LU, pivots = torch.lu(A)
            >>> A_LU
            tensor([[[ 1.3506,  2.5558, -0.0816],
                     [ 0.1684,  1.1551,  0.1940],
                     [ 0.1193,  0.6189, -0.5497]],

                    [[ 0.4526,  1.2526, -0.3285],
                     [-0.7988,  0.7175, -0.9701],
                     [ 0.2634, -0.9255, -0.3459]]])
            >>> pivots
            tensor([[ 3,  3,  3],
                    [ 3,  3,  3]], dtype=torch.int32)
            >>> A_LU, pivots, info = torch.lu(A, get_infos=True)
            >>> if info.nonzero().size(0) == 0:
            ...     print('LU factorization succeeded for all samples!')
            LU factorization succeeded for all samples!
        """
        # If get_infos is True, then we don't need to check for errors and vice versa
        return torch._lu_with_info(A, pivot=pivot, check_errors=(not get_infos))

    def _convert_numeric_vals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert numeric dtypes to float64 for operations that only support this.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        data : pd.DataFrame
        """
        method = self.method

        if method in ["median", "std", "sem", "skew"]:
            # median only has a float64 implementation
            # We should only get here with is_numeric, as non-numeric cases
            #  should raise in _convert_cython_function
            data = ensure_float64(data)

        elif data.dtypes.kind in "iu":
            if method in ["var", "mean"] or (
                self.type == "transform" and self.has_missing_values
            ):
                # has_dropped_na check needed for test_null_group_str_transformer
                # result may still include NaN, so we have to cast
                data = ensure_float64(data)

            elif method in ["sum", "ohlc", "prod", "cumsum", "cumprod"]:
                # Avoid overflow during group operation
                if data.dtypes.kind == "i":
                    data = ensure_int64(data)
                else:
                    data = ensure_uint64(data)

        return data

    # Actions

    def test_date_range_business_year_end_year(self, unit):
        # see GH#9313
        rng = date_range("1/1/2013", "7/1/2017", freq="BYE", unit=unit)
        exp = DatetimeIndex(
            ["2013-12-31", "2014-12-31", "2015-12-31", "2016-12-30"],
            dtype=f"M8[{unit}]",
            freq="BYE",
        )
        tm.assert_index_equal(rng, exp)

    def show_versions(as_json: str | bool = False) -> None:
        """
        Provide useful information, important for bug reports.

        It comprises info about hosting operation system, pandas version,
        and versions of other installed relative packages.

        Parameters
        ----------
        as_json : str or bool, default False
            * If False, outputs info in a human readable form to the console.
            * If str, it will be considered as a path to a file.
              Info will be written to that file in JSON format.
            * If True, outputs info in JSON format to the console.

        See Also
        --------
        get_option : Retrieve the value of the specified option.
        set_option : Set the value of the specified option or options.

        Examples
        --------
        >>> pd.show_versions()  # doctest: +SKIP
        Your output may look something like this:
        INSTALLED VERSIONS
        ------------------
        commit           : 37ea63d540fd27274cad6585082c91b1283f963d
        python           : 3.10.6.final.0
        python-bits      : 64
        OS               : Linux
        OS-release       : 5.10.102.1-microsoft-standard-WSL2
        Version          : #1 SMP Wed Mar 2 00:30:59 UTC 2022
        machine          : x86_64
        processor        : x86_64
        byteorder        : little
        LC_ALL           : None
        LANG             : en_GB.UTF-8
        LOCALE           : en_GB.UTF-8
        pandas           : 2.0.1
        numpy            : 1.24.3
        ...
        """
        sys_info = _get_sys_info()
        deps = _get_dependency_info()

        if as_json:
            j = {"system": sys_info, "dependencies": deps}

            if as_json is True:
                sys.stdout.writelines(json.dumps(j, indent=2))
            else:
                assert isinstance(as_json, str)  # needed for mypy
                with codecs.open(as_json, "wb", encoding="utf8") as f:
                    json.dump(j, f, indent=2)

        else:
            assert isinstance(sys_info["LOCALE"], dict)  # needed for mypy
            language_code = sys_info["LOCALE"]["language-code"]
            encoding = sys_info["LOCALE"]["encoding"]
            sys_info["LOCALE"] = f"{language_code}.{encoding}"

            maxlen = max(len(x) for x in deps)
            print("\nINSTALLED VERSIONS")
            print("------------------")
            for k, v in sys_info.items():
                print(f"{k:<{maxlen}}: {v}")
            print("")
            for k, v in deps.items():
                print(f"{k:<{maxlen}}: {v}")

    def test_steps_and_mode_interactions(self, steps_per_exec, mode):
            dataset_size = 100
            batch_sz = 16
            epochs_cnt = 2

            exec_indices = list(range(0, dataset_size, steps_per_exec * batch_sz))

            data_x = np.ones((dataset_size, 4))
            data_y = np.ones((dataset_size, 1))

            model_instance = ExampleModel(units=1)
            model_instance.compile(
                loss="mse",
                optimizer="sgd",
                steps_per_execution=steps_per_exec,
                run_eagerly=(mode == "eager"),
                jit_compile=(mode != "jit"),
            )
            step_counter = StepCount(exec_indices, batch_sz)

            fit_history = model_instance.fit(
                x=data_x,
                y=data_y,
                batch_size=batch_sz,
                epochs=epochs_cnt,
                callbacks=[step_counter],
                verbose=0,
            )

            self.assertEqual(step_counter.begin_count, len(exec_indices))
            self.assertEqual(step_counter.end_count, step_counter.begin_count)
            self.assertEqual(step_counter.epoch_begin_count, epochs_cnt)
            self.assertEqual(
                step_counter.epoch_end_count, step_counter.epoch_begin_count
            )

            model_second = ExampleModel(units=1)
            model_second.compile(
                loss="mse",
                optimizer="sgd",
                steps_per_execution=1,
                run_eagerly=(mode == "eager"),
                jit_compile=(mode != "jit"),
            )
            fit_history_2 = model_second.fit(
                x=data_x, y=data_y, batch_size=batch_sz, epochs=epochs_cnt, verbose=0
            )

            self.assertAllClose(fit_history.history["loss"], fit_history_2.history["loss"])
            self.assertAllClose(model_instance.get_weights(), model_second.get_weights())
            self.assertAllClose(
                model_instance.predict(data_x, batch_size=batch_sz),
                model_second.predict(data_x, batch_size=batch_sz),
            )
            self.assertAllClose(model_instance.evaluate(data_x, data_y), model_second.evaluate(data_x, data_y))

    def test_custom_optimizer(kernel):
        # Test that GPR can use externally defined optimizers.
        # Define a dummy optimizer that simply tests 50 random hyperparameters
        def optimizer(obj_func, initial_theta, bounds):
            rng = np.random.RandomState(0)
            theta_opt, func_min = initial_theta, obj_func(
                initial_theta, eval_gradient=False
            )
            for _ in range(50):
                theta = np.atleast_1d(
                    rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1]))
                )
                f = obj_func(theta, eval_gradient=False)
                if f < func_min:
                    theta_opt, func_min = theta, f
            return theta_opt, func_min

        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer)
        gpr.fit(X, y)
        # Checks that optimizer improved marginal likelihood
        assert gpr.log_marginal_likelihood(gpr.kernel_.theta) > gpr.log_marginal_likelihood(
            gpr.kernel.theta
        )

    def test_initialization_for_int8(
        self, source_name, expected_compute_dtype, expected_variable_dtype
    ):
        name = f"int8_from_{source_name}"
        policy = QuantizedDTypePolicy(mode="int8", source_name=source_name)
        self.assertEqual(policy.name, name)
        self.assertEqual(policy.compute_dtype, expected_compute_dtype)
        self.assertEqual(policy.variable_dtype, expected_variable_dtype)
        self.assertEqual(repr(policy), f'<QuantizedDTypePolicy "{name}">')

    def _read_config_imp(filenames, dirs=None):
        def _read_config(f):
            meta, vars, sections, reqs = parse_config(f, dirs)
            # recursively add sections and variables of required libraries
            for rname, rvalue in reqs.items():
                nmeta, nvars, nsections, nreqs = _read_config(pkg_to_filename(rvalue))

                # Update var dict for variables not in 'top' config file
                for k, v in nvars.items():
                    if not k in vars:
                        vars[k] = v

                # Update sec dict
                for oname, ovalue in nsections[rname].items():
                    if ovalue:
                        sections[rname][oname] += ' %s' % ovalue

            return meta, vars, sections, reqs

        meta, vars, sections, reqs = _read_config(filenames)

        # FIXME: document this. If pkgname is defined in the variables section, and
        # there is no pkgdir variable defined, pkgdir is automatically defined to
        # the path of pkgname. This requires the package to be imported to work
        if not 'pkgdir' in vars and "pkgname" in vars:
            pkgname = vars["pkgname"]
            if not pkgname in sys.modules:
                raise ValueError("You should import %s to get information on %s" %
                                 (pkgname, meta["name"]))

            mod = sys.modules[pkgname]
            vars["pkgdir"] = _escape_backslash(os.path.dirname(mod.__file__))

        return LibraryInfo(name=meta["name"], description=meta["description"],
                version=meta["version"], sections=sections, vars=VariableSet(vars))

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

    def modify(self):
        result_sm = super().modify()
        if "schema_dynamic_name_to_original_fqn" in self.module.info:  # type: ignore[operator]
            result_sm.info["schema_dynamic_name_to_original_fqn"] = self.module.info[  # type: ignore[index]
                "schema_dynamic_name_to_original_fqn"  # type: ignore[index]
            ]
        if "schema_compile_id" in self.module.info:  # type: ignore[operator]
            result_sm.info["schema_compile_id"] = self.module.info["schema_compile_id"]  # type: ignore[index]
        return result_sm

    def depth_first_collect_leaf_values(node_idx):
        node = nodes[node_idx]
        if node["is_leaf"]:
            values.append(node["value"])
            return
        depth_first_collect_leaf_values(node["left"])
        depth_first_collect_leaf_values(node["right"])

    def _deferrable_constraint_sql(self, deferrable):
        if deferrable is None:
            return ""
        if deferrable == Deferrable.DEFERRED:
            return " DEFERRABLE INITIALLY DEFERRED"
        if deferrable == Deferrable.IMMEDIATE:
            return " DEFERRABLE INITIALLY IMMEDIATE"

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

    def __init__(self, entity=None):
        super().__init__()

        self.entity = entity

        self.model = obj.model
        self.get_type_info = functools.partial(
            ContentType.objects.db_manager(entity._state.db).get_for_model,
            for_concrete_model=obj.field.for_concrete_model,
        )
        self.type_info = self.get_type_info(entity)
        self.type_field_name = obj.field.type_field_name
        self.id_field_name = obj.field.id_field_name
        self.prefetch_key = obj.field.attname
        self.pk_value = entity.pk

        self.core_conditions = {
            f"%s__pk" % self.type_field_name: self.type_info.id,
            self.id_field_name: self.pk_value,
        }

    def example_div_error(self, error, numeric_pos):
            pos = numeric_pos

            expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
            # We only adjust for Index, because Series does not yet apply
            #  the adjustment correctly.
            expected2 = modify_negative_error(error, expected)

            result = pos / error
            tm.assert_index_equal(result, expected2)
            ser_compat = Series(expected).astype("i8") / np.array(error).astype("i8")
            tm.assert_series_equal(ser_compat, Series(expected))

    def validate_unfitted_model(name):
        err_msg = (
            f"The {name} model is not fitted yet. Ensure to call 'fit' before accessing the feature_importances_ attribute."
        )
        with pytest.raises(NotFittedError, match=err_msg):
            FOREST_ESTIMATORS.get(name, lambda: None)().feature_importances_

    def predict_proba(self, X):
        """Predict probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        Y_prob : array-like of shape (n_samples, n_classes)
            The predicted probabilities.
        """
        return self._get_predictions(X, output_method="predict_proba")

    def visualize_performance(metrics):
        import pandas as pd
        import matplotlib.pyplot as plt

        metrics_df = pd.DataFrame(metrics)

        figure, axes = plt.subplots(figsize=(6, 4))
        training_data = metrics_df[metrics_df["mode"] == "train"]
        testing_data = metrics_df[metrics_df["mode"] == "test"]
        axes.plot(training_data["epoch"], training_data["accuracy"], label="Training")
        axes.plot(testing_data["epoch"], testing_data["accuracy"], label="Testing")
        axes.set_xlabel("Epochs")
        axes.set_ylabel("Accuracy (%)")
        axes.set_ylim(70, 100)
        figure.legend(ncol=2, loc="lower right")
        figure.tight_layout()
        filename = "performance_plot.png"
        print(f"--- Saving performance plot to {filename}")
        plt.savefig(filename)
        plt.close(figure)

    def initialize(self, tire, fabricator, pic_path) -> None:
        self._tire = tire
        self._fabricator = fabricator
        self._pic_directory = pic_path
        self._extracted = not False
        __dist_info_var = None
        self._console_entry_points = None

    def verify_qr_code_scanned_with_new_data(self):
            """QR codes are updated when new data is detected."""
            Path("utils.py.tmp").rename("utils.py")
            _, qr_contents = self._generate_qrcode()
            self.assertNotEqual(qr_contents, self.original_qr_contents)
            self.assertQrCode(
                "This is a previously unknown scannable code.",
                qr_contents,
            )

    def matrixFlip(x, transposeAxes=None):
        if isinstance(x, tf.SparseTensor):
            import keras.src.ops.operation_utils as utils
            output_shape = utils.compute_transpose_output_shape(x.shape, transposeAxes)
            output = tf.sparse.transpose(x, perm=transposeAxes)
            output.set_shape(output_shape)
            return output
        else:
            return tf.transpose(x, perm=transposeAxes)

    def modify_param_structure(param_set, target_key, value_to_set, current_index=None):
        """
        This method provides a basic reverse JMESPath implementation that
        lets you go from a JMESPath-like string to a possibly deeply nested
        object. The `param_set` are mutated in-place, so subsequent calls
        can modify the same element by its index.

            >>> modify_param_structure(param_set, 'test[0]', 1)
            >>> print(param_set)
            {'test': [1]}

            >>> modify_param_structure(param_set, 'foo.bar[0].baz', 'hello world')
            >>> print(param_set)
            {'test': [1], 'foo': {'bar': [{'baz': 'hello, world'}]}}

        """
        current_position = param_set
        key_parts = target_key.split('.')

        for part in key_parts:
            # Is it indexing an array?
            match_result = INDEX_RE.search(part)
            if match_result:
                index_str = match_result.group(1)
                if index_str == '*':
                    part = part[:-3]
                    current_index = len(current_position[part])
                else:
                    current_index = int(index_str) if index_str else None
                    part = part[: -len(str(current_index) + '[]')]

                if part not in current_position or not isinstance(current_position[part], list):
                    current_position[part] = []

                # This means we should append, e.g. 'foo[]'
                if current_index is None:
                    current_index = len(current_position[part])

                while len(current_position[part]) <= current_index:
                    # Assume it's a dict until we set the final value below
                    current_position[part].append({})

                # Last item? Set the value, otherwise set the new position
                if part in key_parts[-1]:
                    current_position[part][current_index] = value_to_set
                else:
                    current_position = current_position[part][current_index]
            else:
                if part not in current_position:
                    current_position[part] = {}

                # Last item? Set the value, otherwise set the new position
                if part == key_parts[-1]:
                    current_position[part] = value_to_set
                else:
                    current_position = current_position[part]

    def _from_inferred_categories(
        cls, inferred_categories, inferred_codes, dtype, true_values=None
    ) -> Self:
        """
        Construct a Categorical from inferred values.

        For inferred categories (`dtype` is None) the categories are sorted.
        For explicit `dtype`, the `inferred_categories` are cast to the
        appropriate type.

        Parameters
        ----------
        inferred_categories : Index
        inferred_codes : Index
        dtype : CategoricalDtype or 'category'
        true_values : list, optional
            If none are provided, the default ones are
            "True", "TRUE", and "true."

        Returns
        -------
        Categorical
        """
        from pandas import (
            Index,
            to_datetime,
            to_numeric,
            to_timedelta,
        )

        cats = Index(inferred_categories)
        known_categories = (
            isinstance(dtype, CategoricalDtype) and dtype.categories is not None
        )

        if known_categories:
            # Convert to a specialized type with `dtype` if specified.
            if is_any_real_numeric_dtype(dtype.categories.dtype):
                cats = to_numeric(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "M"):
                cats = to_datetime(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "m"):
                cats = to_timedelta(inferred_categories, errors="coerce")
            elif is_bool_dtype(dtype.categories.dtype):
                if true_values is None:
                    true_values = ["True", "TRUE", "true"]

                # error: Incompatible types in assignment (expression has type
                # "ndarray", variable has type "Index")
                cats = cats.isin(true_values)  # type: ignore[assignment]

        if known_categories:
            # Recode from observation order to dtype.categories order.
            categories = dtype.categories
            codes = recode_for_categories(inferred_codes, cats, categories)
        elif not cats.is_monotonic_increasing:
            # Sort categories and recode for unknown categories.
            unsorted = cats.copy()
            categories = cats.sort_values()

            codes = recode_for_categories(inferred_codes, unsorted, categories)
            dtype = CategoricalDtype(categories, ordered=False)
        else:
            dtype = CategoricalDtype(cats, ordered=False)
            codes = inferred_codes

        return cls._simple_new(codes, dtype=dtype)

    def cond_batch_rule(interpreter, pred, true_fn, false_fn, inputs):
        assert isinstance(
            inputs, (list, tuple)
        ), "Cond inputs must be a list or tuple of tensors"
        assert all(
            isinstance(i, torch.Tensor) for i in inputs
        ), "Cond inputs must be a list of tensors"

        pred_is_batched = isinstance(pred, torch.Tensor) and is_batchedtensor(pred)
        pred_ = get_unwrapped(pred) if pred_is_batched else pred

        # unbatched tensors are not vmapped
        tensors, in_dims = zip(
            *[
                (get_unwrapped(t), maybe_get_bdim(t)) if is_batchedtensor(t) else (t, None)
                for t in inputs
            ]
        )

        if pred_is_batched:
            # prepend "pred" and vmap everything
            tensors = (pred_,) + tensors
            in_dims = (0,) + in_dims

            def fn(p, *args):
                t = true_fn(*args)
                f = false_fn(*args)
                return torch.where(p, t[0], f[0])

            with interpreter.lower():
                result = torch.vmap(fn, in_dims=in_dims)(*tensors)

        else:
            # predicate is known at this stage and it is a boolean expression or a
            # tensor with one element.
            true_fn = torch.vmap(true_fn, in_dims=in_dims)
            false_fn = torch.vmap(false_fn, in_dims=in_dims)

            with interpreter.lower():
                result = cond_op(pred, true_fn, false_fn, tensors)

        if not isinstance(result, tuple):
            result = (result,)
        lvl = interpreter.level()
        return tuple([_add_batch_dim(r, 0, lvl) for r in result])

    def calculate_metrics(data):
        return {
            "low": data.min(),
            "high": data.max(),
            "size": data.count(),
            "average": data.mean(),
        }

    def vdot_product(x1_tensor, x2_tensor):
        x1 = convert_to_tensor(x1_tensor)
        x2 = convert_to_tensor(x2_tensor)
        result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
        compute_dtype = dtypes.result_type(result_dtype, float)

        if get_device() == "cpu" and compute_dtype == "float16":
            compute_dtype = "float32"

        x1_casted = cast(x1, compute_dtype)
        x2_casted = cast(x2, compute_dtype)
        return cast(torch.vdot(x1_casted, x2_casted), result_dtype)

    def example_feature_combination_getitem():
        """Check FeatureCombination.__getitem__ returns expected results."""
        scalar = MinMaxScaler()
        pca = KernelPCA()
        combination = FeatureCombination(
            [
                ("scalar", scalar),
                ("pca", pca),
                ("pass", "passthrough"),
                ("drop_me", "drop"),
            ]
        )
        assert combination["scalar"] is scalar
        assert combination["pca"] is pca
        assert combination["pass"] == "passthrough"
        assert combination["drop_me"] == "drop"

    def _reset_setting_properties(self, config, **kwargs):
        """Reset setting based property values."""
        if config == "UPLOAD_PATH":
            self.__dict__.pop("base_storage", None)
            self.__dict__.pop("storage_location", None)
        elif config == "UPLOAD_URL":
            self.__dict__.pop("base_url_path", None)
        elif config == "FILE_PERMISSIONS":
            self.__dict__.pop("file_mode_permissions", None)
        elif config == "DIRECTORY_PERMISSIONS":
            self.__dict__.pop("directory_mode_permissions", None)

    def fetch_auth_endpoint(self):
        """
        Override this method to override the auth_url attribute.
        """
        auth_url = self.auth_url or settings.AUTH_URL
        if not auth_url:
            raise ImproperlyConfigured(
                f"{self.__class__.__name__} is missing the auth_url attribute. Define "
                f"{self.__class__.__name__}.auth_url, settings.AUTH_URL, or override "
                f"{self.__class__.__name__}.fetch_auth_endpoint()."
            )
        return str(auth_url)

    def _validate_factors(factors, num_components):
        """Validate the user provided 'factors'.

        Parameters
        ----------
        factors : array-like of shape (num_components,)
            The proportions of components of each mixture.

        num_components : int
            Number of components.

        Returns
        -------
        factors : array, shape (num_components,)
        """
        factors = check_array(factors, dtype=[np.float64, np.float32], ensure_2d=False)
        _validate_shape(factors, (num_components,), "factors")

        # check range
        if any(np.less(factors, 0.0)) or any(np.greater(factors, 1.0)):
            raise ValueError(
                "The parameter 'factors' should be in the range "
                "[0, 1], but got max value %.5f, min value %.5f"
                % (np.min(factors), np.max(factors))
            )

        # check normalization
        if not np.allclose(np.abs(1.0 - np.sum(factors)), 0.0):
            raise ValueError(
                "The parameter 'factors' should be normalized, but got sum(factors) = %.5f"
                % np.sum(factors)
            )
        return factors

    def process_step(current_step, extra_outputs, future_extra_inputs):
        for _ in range(
            min(
                self.iterations_per_process,
                self.total_steps,
            )
        ):
            current_step, extra_outputs, future_extra_inputs = (
                step_handler(
                    current_step,
                    extra_outputs,
                    future_extra_inputs,
                )
            )

        return (current_step, extra_outputs, future_extra_inputs)

    def check_ragged_tensor_layer(input_data):
            layer = layers.TextVectorization(
                output_mode="int",
                vocabulary=["foo", "baz", "bar"],
                ragged=True,
            )
            output = layer(input_data)
            self.assertEqual(type(output), tf.RaggedTensor)
            self.assertEqual(output.shape, (3, None))
            self.assertListEqual(output.to_list(), [[4, 1, 3], [1, 2], [4]])

    def test_deepcopy_class_no_evaluation(self):
        # Deep copying doesn't force evaluation.
        foo = Foo()

        obj = self.lazy_wrap(foo)
        obj2 = copy.deepcopy(obj)

        self.assertIsNot(obj, obj2)
        self.assertIs(obj._wrapped, empty)
        self.assertIs(obj2._wrapped, empty)

    def test_POST_immutable_for_multipart(self):
        """
        MultiPartParser.parse() leaves request.POST immutable.
        """
        payload = FakePayload(
            "\r\n".join(
                [
                    "--boundary",
                    'Content-Disposition: form-data; name="name"',
                    "",
                    "value",
                    "--boundary--",
                ]
            )
        )
        request = WSGIRequest(
            {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": "multipart/form-data; boundary=boundary",
                "CONTENT_LENGTH": len(payload),
                "wsgi.input": payload,
            }
        )
        self.assertFalse(request.POST._mutable)

    def check_min_lr_adjustment(self):
        adjust_lr = callbacks.LearningRateScheduler(
            schedule=lambda epoch: 0.1 ** (epoch // 3),
            monitor="val_accuracy",
            min_delta=5,
            cooldown=2,
        )

        self.network.train(
            input_data=self.training_set,
            validation_data=(self.test_set, self.y_test_labels),
            callbacks=[adjust_lr],
            epochs=5,
        )

        self.assertEqual(self.network.optimizer.learning_rate.value, 0.001)

    def process_user_data_multiple_groupers_ignored_true(
        user_info_df, use_index, scale, identifier, expected_results, test_case
    ):
        # GH#47895

        if Version(np.__version__) >= Version("1.26"):
            test_case.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "pandas default unstable sorting of duplicates"
                        "issue with numpy>=1.26 with AVX instructions"
                    ),
                    strict=False,
                )
            )

        expected_index = [
            ("Berlin", "male", "single"),
            ("Berlin", "female", "married"),
            ("Munich", "male", "divorced"),
            ("Munich", "male", "single"),
            ("Munich", "female", "married"),
            ("Munich", "female", "single"),
        ]

        assert_user_data_multiple_groupers(
            user_info_df=user_info_df,
            use_index=use_index,
            observed=True,
            expected_index=expected_index,
            scale=scale,
            identifier=identifier,
            expected_results=expected_results,
        )

    def test_complex_tag_missing_context(self):
            # The 'context' parameter must be present when takes_context is True
            msg = (
                "'complex_tag_without_context_parameter' is decorated with "
                "takes_context=True so it must have a first argument of 'context'"
            )
            with self.assertRaisesMessage(TemplateSyntaxError, msg):
                self.environment.from_string(
                    "{% load custom %}{% complex_tag_without_context_parameter 123 %}"
                )

    def initialize(
            self,
            *,
            data_precision=True,
            ignore_centered=False,
            estimate_fraction=None,
            seed=None,
        ):
            self.data_precision = data_precision
            self.ignore_centered = ignore_centered
            self.estimate_fraction = estimate_fraction
            self.seed = seed

    def test_character_bound_conditions(self, state_info):
            func_name = self.fprefix + '_character_bc_' + state_info

            f = getattr(self.module, func_name)

            c, a = f()
            assert_equal(c, 'a')
            assert_equal(len(a), 1)

            c, a = f('b')
            assert_equal(c, 'b')
            assert_equal(len(a), 2)

            try:
                f('c')
            except Exception:
                pass

    def apply_gradient_policy(self, setting):
            if isinstance(setting, bool):
                self._overwrite_with_gradient = not setting
            else:
                raise TypeError(
                    "`apply_gradient_policy` must be a boolean. "
                    f"Received: {setting}"
                )

    def _pad_dense_input(cls, dense_input: torch.Tensor) -> torch.Tensor:
        """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
        # only 2d matmul
        assert dense_input.dim() == 2

        # check shape
        m, n = dense_input.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_cols

        # calculate padding
        to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
        to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
        if to_pad_m or to_pad_n:
            return torch.nn.functional.pad(dense_input, (0, to_pad_n, 0, to_pad_m))
        else:
            return dense_input

    def test_crop_images(self):
        # Test channels_last
        x = KerasTensor([None, 15, 25, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (None, 10, 20, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (10, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 15, 25])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (None, 3, 10, 20))

        x = KerasTensor([3, None, None])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (3, 10, 20))

    def compute_pr_curve(tag, label_list, pred_list, max_thresholds=127, weight=None):
        # weird, value > 127 breaks protobuf
        num_thresholds = min(max_thresholds, 127)

        pr_data = compute_curve(
            labels=label_list,
            predictions=pred_list,
            num_thresholds=num_thresholds,
            weights=weight
        )

        serialized_data = PrCurvePluginData(
            version=0,
            num_thresholds=num_thresholds
        ).SerializeToString()

        plugin_metadata = SummaryMetadata.PluginData(
            plugin_name="pr_curves",
            content=serialized_data
        )

        summary_tag = tag

        tensor_shape = TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=pr_data.shape[0]),
                TensorShapeProto.Dim(size=pr_data.shape[1])
            ]
        )

        tensor_proto = TensorProto(
            dtype="DT_FLOAT",
            float_val=pr_data.reshape(-1).tolist(),
            tensor_shape=tensor_shape
        )

        summary_value = Summary.Value(tag=summary_tag, metadata=plugin_metadata, tensor=tensor_proto)

        return Summary(value=[summary_value])

    def example_void_array拆解(self):
            res = np.unravel_index(np.zeros(0, dtype=np.intc), (3, 2, 0))
            # res 是一个包含三个空数组的元组
            assert len(res) == 3
            assert all(a.shape == (0,) for a in res)

            with assert_raises(TypeError):
                np.unravel_index([2], (3, 2, 0))

    def describe(self):
            config_str = "System Configuration:\n"

            for k, v in vars(self).items():
                config_str += f"\t{k}: {v}\n"

            return config_str

    def calculate_flag(update_mode: str) -> float:
        if update_mode == "no_update":
            ret = 0.0
        elif update_mode == "average":
            ret = 1.0
        elif update_mode == "mean_per_element":
            warnings.warn(
                "update_mode='mean_per_element' is deprecated. "
                "Please use update_mode='average' instead."
            )
            ret = 1.0
        elif update_mode == "total":
            ret = 2.0
        else:
            ret = -1.0  # TODO: remove once JIT exceptions support control flow
            raise ValueError(f"{update_mode} is not a valid value for update_mode")
        return ret

    def _fetch_associated_items(self, is_subresource):
            """
            Retrieve a list of associated items or references.

            :type is_subresource: bool
            :param is_subresource: ``True`` to fetch sub-resources, ``False`` to
                                   fetch references.
            :rtype: list(:py:class:`Action`)
            """
            resources = []

            for key, value in self._fetch_definitions().items():
                if is_subresource:
                    item_name = self._transform_name('subresource', key)
                else:
                    item_name = self._transform_name('reference', key)
                action = Action(item_name, value, self._resource_defs)

                requires_data = any(identifier.source == 'data' for identifier in action.resource.identifiers)

                if is_subresource and not requires_data:
                    resources.append(action)
                elif not is_subresource and requires_data:
                    resources.append(action)

            return resources

    def allow_migrate(self, db, app_label, **hints):
        for router in self.routers:
            try:
                method = router.allow_migrate
            except AttributeError:
                # If the router doesn't have a method, skip to the next one.
                continue

            allow = method(db, app_label, **hints)

            if allow is not None:
                return allow
        return True

    def _initialize_properties(
            self,
            device_type: str,
            data_type: Optional[_dtype] = None,
            active_flag: bool = True,
            cache_status: Optional[bool] = None,
        ):
            if not isinstance(device_type, str):
                raise ValueError(
                    f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
                )
            data_type = torch.get_autocast_dtype(device_type) if data_type is None else data_type
            if torch._jit_internal.is_scripting():
                self.active_flag = active_flag
                self.device_type = device_type
                self.data_type = data_type
                assert data_type is not None
                return

            self.device_type = device_type
            if not is_autocast_available(self.device_type):
                raise RuntimeError(
                    f"User specified an unsupported autocast device_type '{self.device_type}'"
                )
            self.custom_backend_name = torch._C._get_privateuse1_backend_name()
            self.data_type = torch.get_autocast_dtype(self.device_type)

            if self.device_type == self.custom_backend_name:
                necessary_functions = [
                    "get_amp_supported_dtype",
                ]
                message = f"Tried to use AMP with the `{self.custom_backend_name}` backend, but the backend has not "
                message += "registered a module or  the module miss some necessary functions. The backend should register "
                message += "a module by `torch._register_device_module`, and the module must have these functions: \n"
                message += "`get_amp_supported_dtype() -> List[torch.dtype]`. \n"

                assert hasattr(torch, self.custom_backend_name), message
                self.custom_device_mod = getattr(torch, self.custom_backend_name)
                for func in necessary_functions:
                    assert hasattr(self.custom_device_mod, func), (
                        message + f"But the function `{func}` is missing. \n"
                    )

            cache_status = torch.is_autocast_cache_enabled() if cache_status is None else cache_status
            active_flag = False if (
                active_flag and torch.cuda.amp.common.amp_definitely_not_available()
                and self.device_type == "cuda"
            ) else active_flag
            data_type = data_type if data_type is not None else self.data_type
            cache_status = cache_status if cache_status is not None else self.cache_status

            if self.device_type == "cpu":
                supported_types = [torch.float16, torch.bfloat16]
                if data_type not in supported_types:
                    message = f"CPU autocast only supports {', '.join(str(t) for t in supported_types)} currently."
                    warnings.warn(message)
                    active_flag = False
            elif self.device_type == "cuda":
                if (
                    active_flag and data_type == torch.bfloat16 and not torch.cuda.is_bf16_supported()
                ):
                    raise RuntimeError(
                        "Current CUDA Device does not support bfloat16. Please switch dtype to float16."
                    )
            elif self.device_type == "mps":
                supported_types = [torch.float16, torch.bfloat16]
                if data_type not in supported_types:
                    message = (
                        f"MPS autocast only supports {', '.join(str(t) for t in supported_types)} currently."
                    )
                    warnings.warn(message)
                    active_flag = False
                elif data_type == torch.bfloat16 and not torch.backends.mps.is_macos_or_newer(14, 0):
                    message = (
                        f"bfloat16 is not supported on macOS versions below 14 in MPS autocast. Disabling autocast."
                    )
                    warnings.warn(message)
                    active_flag = False
            elif self.device_type == "xla":
                supported_types = [torch.float16, torch.bfloat16]
                if data_type not in supported_types:
                    message = f"XLA autocast only supports {supported_types[0]} currently."
                    warnings.warn(message)
                    active_flag = False
            self.active_flag = active_flag

    def fetch_configuration(self, config_key):
            configuration = {
                "dtype": self.dtype,
                "shape": self.shape,
                "ndim": self.ndim,
                "max_ndim": self.max_ndim,
                "min_ndim": self.min_ndim,
                "axes": self.axes
            }
            return configuration.get(config_key, {})

    def _check_for_locals(expr: str, stack_level: int, parser: str) -> None:
        at_top_of_stack = stack_level == 0
        not_pandas_parser = parser != "pandas"

        if not_pandas_parser:
            msg = "The '@' prefix is only supported by the pandas parser"
        elif at_top_of_stack:
            msg = (
                "The '@' prefix is not allowed in top-level eval calls.\n"
                "please refer to your variables by name without the '@' prefix."
            )

        if at_top_of_stack or not_pandas_parser:
            for toknum, tokval in tokenize_string(expr):
                if toknum == tokenize.OP and tokval == "@":
                    raise SyntaxError(msg)

    def test_login_validate_user_data(self):
            auth = Authentication()

            @auth.validate
            def process_user_data(**kwargs):
                return Token("access_token")

            msg = (
                "The function %r did not return a list. All functions registered "
                "with the authentication module must return a list." % process_user_data
            )
            with self.assertRaisesMessage(TypeError, msg):
                auth.validate_user()

    def test_clip(self, datetime_series):
        val = datetime_series.median()

        assert datetime_series.clip(lower=val).min() == val
        assert datetime_series.clip(upper=val).max() == val

        result = datetime_series.clip(-0.5, 0.5)
        expected = np.clip(datetime_series, -0.5, 0.5)
        tm.assert_series_equal(result, expected)
        assert isinstance(expected, Series)

    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        style=None,
        column_num=None,
        stacking_id=None,
        is_errorbar: bool = False,
        **kwds,
    ):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])

        # need to remove label, because subplots uses mpl legend as it is
        line_kwds = kwds.copy()
        line_kwds.pop("label")
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)

        # get data from the line to get coordinates for fill_between
        xdata, y_values = lines[0].get_data(orig=False)

        # unable to use ``_get_stacked_values`` here to get starting point
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            # TODO #54485
            start = ax._stacker_pos_prior[stacking_id]  # type: ignore[attr-defined]
        elif (y <= 0).all():
            # TODO #54485
            start = ax._stacker_neg_prior[stacking_id]  # type: ignore[attr-defined]
        else:
            start = np.zeros(len(y))

        if "color" not in kwds:
            kwds["color"] = lines[0].get_color()

        rect = ax.fill_between(xdata, start, y_values, **kwds)
        cls._update_stacker(ax, stacking_id, y)

        # LinePlot expects list of artists
        res = [rect]
        return res

    def _initialize(self, dispatch_key=None):
            if dispatch_key is not None:
                assert isinstance(dispatch_key, torch._C.DispatchKey)
                self._dispatch_key = dispatch_key

            bool_deque: Deque[bool] = deque()
            old_dispatch_mode_flags = bool_deque
            old_non_infra_dispatch_mode_flags = bool_deque

            self.old_dispatch_mode_flags = old_dispatch_mode_flags
            self.old_non_infra_dispatch_mode_flags = old_non_infra_dispatch_mode_flags

    def example_to_xml_update_mode(modex_):
        # GH 35849
        # Test ValueError when mode is not supported option
        data = DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        error_msg = (
            f"mode={modex_} is not a valid option."
            "Only 'w' and 'a' are currently supported."
        )
        with pytest.raises(ValueError, match=error_msg):
            data.to_xml(modex_, single_root=False)

    def set_option(*args) -> None:
        """
        Set the value of the specified option or options.

        This method allows fine-grained control over the behavior and display settings
        of pandas. Options affect various functionalities such as output formatting,
        display limits, and operational behavior. Settings can be modified at runtime
        without requiring changes to global configurations or environment variables.

        Parameters
        ----------
        *args : str | object
            Arguments provided in pairs, which will be interpreted as (pattern, value)
            pairs.
            pattern: str
            Regexp which should match a single option
            value: object
            New value of option

            .. warning::

                Partial pattern matches are supported for convenience, but unless you
                use the full option name (e.g. x.y.z.option_name), your code may break in
                future versions if new options with similar names are introduced.

        Returns
        -------
        None
            No return value.

        Raises
        ------
        ValueError if odd numbers of non-keyword arguments are provided
        TypeError if keyword arguments are provided
        OptionError if no such option exists

        See Also
        --------
        get_option : Retrieve the value of the specified option.
        reset_option : Reset one or more options to their default value.
        describe_option : Print the description for one or more registered options.
        option_context : Context manager to temporarily set options in a ``with``
            statement.

        Notes
        -----
        For all available options, please view the :ref:`User Guide <options.available>`
        or use ``pandas.describe_option()``.

        Examples
        --------
        >>> pd.set_option("display.max_columns", 4)
        >>> df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        >>> df
        0  1  ...  3   4
        0  1  2  ...  4   5
        1  6  7  ...  9  10
        [2 rows x 5 columns]
        >>> pd.reset_option("display.max_columns")
        """
        # must at least 1 arg deal with constraints later
        nargs = len(args)
        if not nargs or nargs % 2 != 0:
            raise ValueError("Must provide an even number of non-keyword arguments")

        for k, v in zip(args[::2], args[1::2]):
            key = _get_single_key(k)

            opt = _get_registered_option(key)
            if opt and opt.validator:
                opt.validator(v)

            # walk the nested dict
            root, k_root = _get_root(key)
            root[k_root] = v

            if opt.cb:
                opt.cb(key)

    def forecast(self, data_points, validate_input=True):
            """Predict class or regression value for data points.

            For a classification model, the predicted class for each sample in `data_points` is returned.
            For a regression model, the predicted value based on `data_points` is returned.

            Parameters
            ----------
            data_points : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input samples. Internally, it will be converted to
                ``dtype=np.float32`` and if a sparse matrix is provided
                to a sparse ``csr_matrix``.

            validate_input : bool, default=True
                Allow to bypass several input checking.
                Don't use this parameter unless you know what you're doing.

            Returns
            -------
            predictions : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The predicted classes, or the predict values.
            """
            if hasattr(self, 'tree_'):
                self.tree_.validate()  # 自定义的验证方法
            data_points = self._transform_input(data_points, validate_input)
            proba = self.tree_.predict(data_points)
            sample_count = data_points.shape[0]

            # 分类预测
            if isinstance(self, ClassifierMixin):
                if self.n_outputs_ == 1:
                    return self.classes_.take(np.argmax(proba, axis=1), axis=0)

                else:
                    class_type = self.classes_[0].dtype
                    predictions = np.zeros((sample_count, self.n_outputs_), dtype=class_type)
                    for k in range(self.n_outputs_):
                        predictions[:, k] = self.classes_[k].take(
                            np.argmax(proba[:, k], axis=1), axis=0
                        )

                    return predictions

            # 回归预测
            else:
                if self.n_outputs_ == 1:
                    return proba[:, 0]

                else:
                    return proba[:, :, 0]

    def apply_random_transformation(self, inputs_dict, is_training=True, rng_seed=None):
            if "images" in inputs_dict:
                images = inputs_dict["images"]
            else:
                images = inputs_dict
            image_shape = self.backend.shape(images)
            rank = len(image_shape)
            batch_size = 1 if rank == 3 else (image_shape[0] if rank == 4 else None)

            if rng_seed is None:
                rng_seed = self._get_seed_generator(self.backend._backend)

            random_factor = self.backend.random.uniform(
                (batch_size, 1, 1, 1),
                minval=self.factor[0],
                maxval=self.factor[1],
                seed=rng_seed,
            )
            transformed_factor = random_factor
            return {"factor": transformed_factor}

    def validate_blobs_return_clusters():
        sample_counts = [10, 20]
        feature_dimensions = 3
        data, labels, cluster_centers = make_blobs(
            n_samples=sample_counts, n_features=feature_dimensions, return_cluster centers=True, random_state=0
        )

        assert cluster_centers.shape == (len(sample_counts), feature_dimensions)

    def test_michael_jaccard_similarity():
        # General case
        similarity = jaccard_similarity([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2])
        assert_almost_equal(similarity, np.sqrt(4.0 / (12.0 * 6.0)))

        # Perfect match but where the label names changed
        perfect_similarity = jaccard_similarity([1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1])
        assert_almost_equal(perfect_similarity, 1.0)

        # Worst case
        worst_similarity = jaccard_similarity([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5])
        assert_almost_equal(worst_similarity, 0.0)

    def load_custom_cache(
        ns: str, func_name_with_overload: str, device_type: str
    ) -> List[Optional[Dict[str, Any]]]:
        device_kernel_cache = custom_cache_dir(ns, device_type)
        op_conf = device_kernel_cache / f"{func_name_with_overload}.json"
        if not op_conf.exists():
            return []

        try:
            with custom_cache_lock(func_name_with_overload):
                with open(op_conf) as f:
                    json_data = json.load(f)
                    for item in json_data:
                        # Get absolute path for kernel library
                        kernel_lib_abs_path = device_kernel_cache / item["kernel_path"]
                        item["kernel_path"] = kernel_lib_abs_path.as_posix()

                        # Check if the kernel library exists
                        if not kernel_lib_abs_path.exists():
                            return []

                        for metadata in item["meta_info"]:
                            if metadata.get("is_dynamic"):
                                raise NotImplementedError(
                                    "Only support static shape for now"
                                )
                            if (
                                "device_type" in metadata
                                and metadata["device_type"] == "gpu"
                            ):
                                metadata["device_index"] = 0
                            for dtype_key in ["dtype", "dtype_value"]:
                                if dtype_key in metadata:
                                    metadata[dtype_key] = getattr(
                                        torch, metadata[dtype_key].split(".")[-1]
                                    )
                            if "layout_value" in metadata:
                                metadata["layout_value"] = getattr(
                                    torch, metadata["layout_value"].split(".")[-1]
                                )
                            if "memory_format_value" in metadata:
                                metadata["memory_format_value"] = getattr(
                                    torch, metadata["memory_format_value"].split(".")[-1]
                                )

                    return json_data
        except Exception as e:
            err_msg = f"Failed to load custom cache: {e}"
            log.exception(err_msg)
            return []

    def test_reshape_matches(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        # Scenario with 100% accuracy for simplicity.
        # y_true is a 2D tensor with shape (2, 1) to test reshape.
        y_true = np.array([[0], [0]], dtype=np.int64)
        y_pred = np.array(
            [[[0.9, 0.1, 0.0], [0.8, 0.15, 0.05]]], dtype=np.float32
        )
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, np.array([1.0, 1.0]))
