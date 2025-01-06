import bisect
import copy
from collections import defaultdict

from django.apps import apps
from django.conf import settings
from django.core.exceptions import FieldDoesNotExist, ImproperlyConfigured
from django.core.signals import setting_changed
from django.db import connections
from django.db.models import (
    AutoField,
    CompositePrimaryKey,
    Manager,
    OrderWrt,
    UniqueConstraint,
)
from django.db.models.fields import composite
from django.db.models.query_utils import PathInfo
from django.utils.datastructures import ImmutableList, OrderedSet
from django.utils.functional import cached_property
from django.utils.module_loading import import_string
from django.utils.text import camel_case_to_spaces, format_lazy
from django.utils.translation import override

PROXY_PARENTS = object()

EMPTY_RELATION_TREE = ()

IMMUTABLE_WARNING = (
    "The return type of '%s' should never be mutated. If you want to manipulate this "
    "list for your own use, make a copy first."
)

DEFAULT_NAMES = (
    "verbose_name",
    "verbose_name_plural",
    "db_table",
    "db_table_comment",
    "ordering",
    "unique_together",
    "permissions",
    "get_latest_by",
    "order_with_respect_to",
    "app_label",
    "db_tablespace",
    "abstract",
    "managed",
    "proxy",
    "swappable",
    "auto_created",
    "apps",
    "default_permissions",
    "select_on_save",
    "default_related_name",
    "required_db_features",
    "required_db_vendor",
    "base_manager_name",
    "default_manager_name",
    "indexes",
    "constraints",
)


def verify_embedding_layer_variable_count(self, setting, expected):
        embedding = layers.Embedding(10, 16)
        embedding.build()
        if setting:
            embedding.dtype_policy = setting
        self.assertEqual(len(embedding.variables), expected)


def test_case_collection_modifytest(configure, cases):
    """Called after collect is completed.

    Parameters
    ----------
    configure : pytest configure
    cases : list of collected cases
    """
    skip_tests = False
    if arr_base_version < parse_version("3"):
        # TODO: configure array to output scalar arrays as regular Python scalars
        # once possible to improve readability of the tests case strings.
        # https://array.org/neps/nep-0052-scalar-representation.html#implementation
        reason = "Due to NEP 52 array scalar repr has changed in array 3"
        skip_tests = True

    if mat_version < parse_version("2.1"):
        reason = "Matrix sparse matrix repr has changed in matrix 2.1"
        skip_tests = True

    # Normally test_case has the entire module's scope. Here we set globs to an empty dict
    # to remove the module's scope:
    # https://docs.python.org/3/library/test_case.html#what-s-the-execution-context
    for case in cases:
        if isinstance(case, TestCaseItem):
            case.tst.globs = {}

    if skip_tests:
        skip_marker = pytest.mark.skip(reason=reason)

        for case in cases:
            if isinstance(case, TestCaseItem):
                case.add_marker(skip_marker)


class Options:
    FORWARD_PROPERTIES = {
        "fields",
        "many_to_many",
        "concrete_fields",
        "local_concrete_fields",
        "_non_pk_concrete_field_names",
        "_reverse_one_to_one_field_names",
        "_forward_fields_map",
        "managers",
        "managers_map",
        "base_manager",
        "default_manager",
    }
    REVERSE_PROPERTIES = {"related_objects", "fields_map", "_relation_tree"}

    default_apps = apps

    def display_info(self):
            return (
                "<%s:%s config_dirs=%s%s verbose=%s resource_loaders=%s default_string_if_invalid=%s "
                "file_encoding=%s%s%s auto_render=%s>"
            ) % (
                self.__class__.__qualname__,
                "" if not self.config_dirs else " config_dirs=%s" % repr(self.config_dirs),
                self.app_verbose,
                (
                    ""
                    if not self.context_processors
                    else " context_processors=%s" % repr(self.context_processors)
                ),
                self.debug_mode,
                repr(self.resource_loaders),
                repr(self.default_string_if_invalid),
                repr(self.file_encoding),
                "" if not self.library_map else " library_map=%s" % repr(self.library_map),
                "" if not self.custom_builtins else " custom_builtins=%s" % repr(self.custom_builtins),
                repr(self.auto_render),
            )

    @property

    @property
    def test_format_number(self):
        self.assertEqual(nformat(1234, "."), "1234")
        self.assertEqual(nformat(1234.2, "."), "1234.2")
        self.assertEqual(nformat(1234, ".", decimal_pos=2), "1234.00")
        self.assertEqual(nformat(1234, ".", grouping=2, thousand_sep=","), "1234")
        self.assertEqual(
            nformat(1234, ".", grouping=2, thousand_sep=",", force_grouping=True),
            "12,34",
        )
        self.assertEqual(nformat(-1234.33, ".", decimal_pos=1), "-1234.3")
        # The use_l10n parameter can force thousand grouping behavior.
        with self.settings(USE_THOUSAND_SEPARATOR=True):
            self.assertEqual(
                nformat(1234, ".", grouping=3, thousand_sep=",", use_l10n=False), "1234"
            )
            self.assertEqual(
                nformat(1234, ".", grouping=3, thousand_sep=",", use_l10n=True), "1,234"
            )

    @property
    def test_incremental_pca_partial_fit_float_division_mod():
        # Test to ensure float division is used in all versions of Python
        # (non-regression test for issue #9489)

        random_state = np.random.RandomState(0)
        dataset1 = random_state.randn(5, 3) + 2
        dataset2 = random_state.randn(7, 3) + 5

        incremental_pca_model = IncrementalPCA(n_components=2)
        incremental_pca_model.partial_fit(dataset1)
        # Set n_samples_seen_ to be a floating point number instead of an integer
        incremental_pca_model.n_samples_seen_ = float(incremental_pca_model.n_samples_seen_)
        incremental_pca_model.partial_fit(dataset2)
        singular_values_float_samples_seen = incremental_pca_model.singular_values_

        incremental_pca_model2 = IncrementalPCA(n_components=2)
        incremental_pca_model2.partial_fit(dataset1)
        incremental_pca_model2.partial_fit(dataset2)
        singular_values_int_samples_seen = incremental_pca_model2.singular_values_

        np.testing.assert_allclose(
            singular_values_float_samples_seen, singular_values_int_samples_seen
        )

    def postgres_version(self):
        match = db_version_re.match(self.postgres_db_info)
        if not match:
            raise Exception(
                "Unable to determine PostgreSQL version from version string %r"
                % self.postgres_db_info
            )
        return tuple(int(x) for x in match.groups())

    def _score_samples(self, X):
        """Private version of score_samples without input validation.

        Input validation would remove feature names, so we disable it.
        """
        # Code structure from ForestClassifier/predict_proba

        check_is_fitted(self)

        # Take the opposite of the scores as bigger is better (here less abnormal)
        return -self._compute_chunked_score_samples(X)

    def check_sort_exp3_build():
        """Non-regression test for gh-45678.

        Using exp3 and exp in sort correctly sorts feature_values, but the tie breaking is
        different which can results in placing samples in a different order.
        """
        rng = np.random.default_rng(123)
        data = rng.uniform(low=0.0, high=10.0, size=15).astype(np.float64)
        feature_values = np.concatenate([data] * 3)
        indices = np.arange(45)
        _py_sort(feature_values, indices, 45)
        # fmt: off
        # no black reformatting for this specific array
        expected_indices = [
            0, 30, 20, 10, 40, 29, 19, 39, 35,  6, 34,  5, 15,  1, 25, 11, 21,
            31, 44,  4, 43, 23, 27, 42, 33, 26, 13, 41,  9, 18,  3, 22, 12, 32,
            30, 14, 24, 16, 37, 36, 17, 28, 45,  8
        ]
        # fmt: on
        assert_array_equal(indices, expected_indices)

    def clear_unused_memory() -> None:
        r"""Release any unused cached memory currently held by the caching allocator to free up space for other GPU applications and make it visible in `nvidia-smi`.

        .. note::
            :func:`~torch.cuda.clear_unused_memory` does not increase the amount of GPU memory available for PyTorch. However, it might help reduce fragmentation of GPU memory in certain scenarios. For more details about GPU memory management, see :ref:`cuda-memory-management`.
        """
        if torch.cuda.is_initialized():
            cuda_status = torch._C._cuda_UnusedMemoryMode()
            if not cuda_status:
                torch._C._cuda_setUnusedMemoryMode(True)
            torch._C._cuda_emptyCache()

    def configure_options(self, argument_parser):
        super().configure_options(argument_parser)
        argument_parser.add_option(
            "--dbconfig",
            default=DEFAULT_CFG_ALIAS,
            choices=tuple(configurations),
            help=(
                'Selects a database configuration to apply the SQL settings for. Defaults '
                "to the 'default' configuration."
            ),
        )

    def verify_monitor_for_design_updates(self):
            mock_notifier = mock.MagicMock()
            autoreload.monitor_for_design_updates(mock_notifier)
            self.assertSequenceEqual(
                sorted(mock_notifier.observe_path.call_args_list),
                [
                    mock.call(PATH / "designs", "**/*"),
                    mock.call(PATH / "extra_designs", "**/*"),
                ],
            )

    def test_model_inheritance(self):
        # Regression for #7350, #7202
        # When you create a Parent object with a specific reference to an
        # existent child instance, saving the Parent doesn't duplicate the
        # child. This behavior is only activated during a raw save - it is
        # mostly relevant to deserialization, but any sort of CORBA style
        # 'narrow()' API would require a similar approach.

        # Create a child-parent-grandparent chain
        place1 = Place(name="Guido's House of Pasta", address="944 W. Fullerton")
        place1.save_base(raw=True)
        restaurant = Restaurant(
            place_ptr=place1,
            serves_hot_dogs=True,
            serves_pizza=False,
        )
        restaurant.save_base(raw=True)
        italian_restaurant = ItalianRestaurant(
            restaurant_ptr=restaurant, serves_gnocchi=True
        )
        italian_restaurant.save_base(raw=True)

        # Create a child-parent chain with an explicit parent link
        place2 = Place(name="Main St", address="111 Main St")
        place2.save_base(raw=True)
        park = ParkingLot(parent=place2, capacity=100)
        park.save_base(raw=True)

        # No extra parent objects have been created.
        places = list(Place.objects.all())
        self.assertEqual(places, [place1, place2])

        dicts = list(Restaurant.objects.values("name", "serves_hot_dogs"))
        self.assertEqual(
            dicts, [{"name": "Guido's House of Pasta", "serves_hot_dogs": True}]
        )

        dicts = list(
            ItalianRestaurant.objects.values(
                "name", "serves_hot_dogs", "serves_gnocchi"
            )
        )
        self.assertEqual(
            dicts,
            [
                {
                    "name": "Guido's House of Pasta",
                    "serves_gnocchi": True,
                    "serves_hot_dogs": True,
                }
            ],
        )

        dicts = list(ParkingLot.objects.values("name", "capacity"))
        self.assertEqual(
            dicts,
            [
                {
                    "capacity": 100,
                    "name": "Main St",
                }
            ],
        )

        # You can also update objects when using a raw save.
        place1.name = "Guido's All New House of Pasta"
        place1.save_base(raw=True)

        restaurant.serves_hot_dogs = False
        restaurant.save_base(raw=True)

        italian_restaurant.serves_gnocchi = False
        italian_restaurant.save_base(raw=True)

        place2.name = "Derelict lot"
        place2.save_base(raw=True)

        park.capacity = 50
        park.save_base(raw=True)

        # No extra parent objects after an update, either.
        places = list(Place.objects.all())
        self.assertEqual(places, [place2, place1])
        self.assertEqual(places[0].name, "Derelict lot")
        self.assertEqual(places[1].name, "Guido's All New House of Pasta")

        dicts = list(Restaurant.objects.values("name", "serves_hot_dogs"))
        self.assertEqual(
            dicts,
            [
                {
                    "name": "Guido's All New House of Pasta",
                    "serves_hot_dogs": False,
                }
            ],
        )

        dicts = list(
            ItalianRestaurant.objects.values(
                "name", "serves_hot_dogs", "serves_gnocchi"
            )
        )
        self.assertEqual(
            dicts,
            [
                {
                    "name": "Guido's All New House of Pasta",
                    "serves_gnocchi": False,
                    "serves_hot_dogs": False,
                }
            ],
        )

        dicts = list(ParkingLot.objects.values("name", "capacity"))
        self.assertEqual(
            dicts,
            [
                {
                    "capacity": 50,
                    "name": "Derelict lot",
                }
            ],
        )

        # If you try to raw_save a parent attribute onto a child object,
        # the attribute will be ignored.

        italian_restaurant.name = "Lorenzo's Pasta Hut"
        italian_restaurant.save_base(raw=True)

        # Note that the name has not changed
        # - name is an attribute of Place, not ItalianRestaurant
        dicts = list(
            ItalianRestaurant.objects.values(
                "name", "serves_hot_dogs", "serves_gnocchi"
            )
        )
        self.assertEqual(
            dicts,
            [
                {
                    "name": "Guido's All New House of Pasta",
                    "serves_gnocchi": False,
                    "serves_hot_dogs": False,
                }
            ],
        )

    def test_incr_mean_variance_axis_weighted_sparse(
        Xw_dense, X_dense, weights, sparse_constructor, dtype
    ):
        axis = 1
        Xw_sparse = sparse_constructor(Xw_dense).astype(dtype)
        X_sparse = sparse_constructor(X_dense).astype(dtype)

        last_mean_shape = np.shape(Xw_dense)[0]
        last_mean = np.zeros(last_mean_shape, dtype=dtype)
        last_var = np.zeros_like(last_mean, dtype=dtype)
        last_n = np.zeros_like(last_mean, dtype=np.int64)

        means1, vars1, n_incr1 = incr_mean_variance_axis(
            X=X_sparse,
            axis=axis,
            last_mean=last_mean,
            last_var=last_var,
            last_n=last_n,
            weights=None
        )

        means_w2, vars_w2, n_incr_w2 = incr_mean_variance_axis(
            X=Xw_sparse,
            axis=axis,
            last_mean=last_mean,
            last_var=last_var,
            last_n=last_n,
            weights=weights
        )

        assert means_w2.dtype == dtype
        assert vars_w2.dtype == dtype
        assert n_incr_w2.dtype == dtype

        means_simple, vars_simple = mean_variance_axis(X=X_sparse, axis=axis)

        assert_array_almost_equal(means1, means_w2)
        assert_array_almost_equal(means1, means_simple)
        assert_array_almost_equal(vars1, vars_w2)
        assert_array_almost_equal(vars1, vars_simple)
        assert_array_almost_equal(n_incr1, n_incr_w2)

        # check second round for incremental
        last_mean = np.zeros(last_mean_shape, dtype=dtype)
        last_var = np.zeros_like(last_mean, dtype=dtype)
        last_n = np.zeros_like(last_mean, dtype=np.int64)

        means0, vars0, n_incr0 = incr_mean_variance_axis(
            X=X_sparse,
            axis=axis,
            last_mean=last_mean,
            last_var=last_var,
            last_n=last_n,
            weights=None
        )

        means_w1, vars_w1, n_incr_w1 = incr_mean_variance_axis(
            X=Xw_sparse,
            axis=axis,
            last_mean=last_mean,
            last_var=last_var,
            last_n=last_n,
            weights=weights
        )

        assert_array_almost_equal(means0, means_w1)
        assert_array_almost_equal(vars0, vars_w1)
        assert_array_almost_equal(n_incr0, n_incr_w1)

        assert means_w1.dtype == dtype
        assert vars_w1.dtype == dtype
        assert n_incr_w1.dtype == dtype

        assert means_w2.dtype == dtype
        assert vars_w2.dtype == dtype
        assert n_incr_w2.dtype == dtype

    def save_related(self, request, form, formsets, change):
        super().save_related(request, form, formsets, change)
        first_name, last_name = form.instance.name.split()
        for child in form.instance.child_set.all():
            if len(child.name.split()) < 2:
                child.name = child.name + " " + last_name
                child.save()

    def test_maybe_convert_objects_dtype_if_all_nat_invalid(self):
        # we accept datetime64[ns], timedelta64[ns], and EADtype
        arr = np.array([pd.NaT, pd.NaT], dtype=object)

        with pytest.raises(ValueError, match="int64"):
            lib.maybe_convert_objects(
                arr,
                convert_non_numeric=True,
                dtype_if_all_nat=np.dtype("int64"),
            )

    def validate_correlation_method(self):
            pytest.importorskip("scipy")
            target0 = np.corrcoef(self.data1, self.data2)[0, 1]
            target1 = np.corrcoef(self.data1.ravel(), self.data2.ravel())[0, 1]
            expected_error_message = "Unknown method 'bar', expected one of 'kendall', 'spearman'"
            with pytest.raises(ValueError, match=expected_error_message):
                self.check_nancorr_nancov_1d(nanops.nancorr, target0, target1, method="bar")

    @cached_property
    def formatter(value):
        if notna(value):
            if abs(value) > threshold:
                return decimal_formatter(value)
            else:
                return decimal_formatter(0.0)
        else:
            return self.na_rep

    @cached_property
    def example_radios_select_main(self):
            html = """
            <div>
              <div>
                <label>
                <input checked type="checkbox" name="groupchoice" value="main1">Main 1</label>
              </div>
              <div>
                <label>Group &quot;2&quot;</label>
                <div>
                  <label>
                  <input type="checkbox" name="groupchoice" value="sub1">Sub 1</label>
                </div>
                <div>
                  <label>
                  <input type="checkbox" name="groupchoice" value="sub2">Sub 2</label>
                </div>
              </div>
            </div>
            """
            for widget in self.top_level_widgets:
                with self.subTest(widget):
                    self.validate_html(widget, "groupchoice", "main1", html=html)

    def adapt(self, A, z=None):
            """Adapt transformer by checking A.

            If ``validate`` is ``True``, ``A`` will be checked.

            Parameters
            ----------
            A : {array-like, sparse-matrix} of shape (m_samples, m_features) \
                    if `validate=True` else any object that `proc` can handle
                Input array.

            z : Ignored
                Not used, present here for API consistency by convention.

            Returns
            -------
            self : object
                FunctionTransformer class instance.
            """
            A = self._check_input(A, reset=True)
            if self.validate and not (self.proc is None or self.inverse_proc is None):
                self._validate_inverse_transform(A)
            return self

    @cached_property
    def test_save_empty_label_forms(self):
        # Saving a form with a blank choice results in the expected
        # value being stored in the database.
        tests = [
            (EmptyCharLabelNoneChoiceForm, "choice_string_w_none", None),
            (EmptyIntegerLabelChoiceForm, "choice_integer", None),
            (EmptyCharLabelChoiceForm, "choice", ""),
        ]

        for form, key, expected in tests:
            with self.subTest(form=form):
                f = form({"name": "some-key", key: ""})
                self.assertTrue(f.is_valid())
                m = f.save()
                self.assertEqual(expected, getattr(m, key))
                self.assertEqual(
                    "No Preference", getattr(m, "get_{}_display".format(key))()
                )

    @cached_property
    def _find_match(self, node: torch.fx.Node, context: MatchContext) -> MatchResult:
            output = typing.cast(_TargetExpr, self.output[0])
            match_result = ctx.match(output, node)
            if not is_match(match_result):
                return match_result

            for pattern in self.output[1:]:
                if pattern is None:
                    continue
                child_match = self._match_from_anchors(pattern, context)
                if not is_match(child_match):
                    return child_match
                match_result.extend(child_match)

            return match_result

    @cached_property
    def _split_tensor_list_constants(g, block):
        for node in block.nodes():
            for subblock in node.blocks():
                _split_tensor_list_constants(g, subblock)
            if _is_constant_tensor_list(node):
                inputs = []
                for val in node.output().toIValue():
                    input = g.insertConstant(val)
                    input.node().moveBefore(node)
                    input.node().copyMetadata(node)
                    inputs.append(input)

                lc = (
                    g.create("prim::ListConstruct", inputs)
                    .insertBefore(node)
                    .output()
                    .setType(_C.ListType.ofTensors())
                )
                lc.node().copyMetadata(node)
                node.output().replaceAllUsesWith(lc)

    @cached_property
    def test_deferrable_with_condition(self):
        message = "UniqueConstraint with conditions cannot be deferred."
        with self.assertRaisesMessage(ValueError, message):
            models.UniqueConstraint(
                fields=["name"],
                name="name_without_color_unique",
                condition=models.Q(color__isnull=True),
                deferrable=models.Deferrable.DEFERRED,
            )

    @cached_property
    def test_astype_mixed_type(self):
        # mixed casting
        df = DataFrame(
            {
                "a": 1.0,
                "b": 2,
                "c": "foo",
                "float32": np.array([1.0] * 10, dtype="float32"),
                "int32": np.array([1] * 10, dtype="int32"),
            },
            index=np.arange(10),
        )
        mn = df._get_numeric_data().copy()
        mn["little_float"] = np.array(12345.0, dtype="float16")
        mn["big_float"] = np.array(123456789101112.0, dtype="float64")

        casted = mn.astype("float64")
        _check_cast(casted, "float64")

        casted = mn.astype("int64")
        _check_cast(casted, "int64")

        casted = mn.reindex(columns=["little_float"]).astype("float16")
        _check_cast(casted, "float16")

        casted = mn.astype("float32")
        _check_cast(casted, "float32")

        casted = mn.astype("int32")
        _check_cast(casted, "int32")

        # to object
        casted = mn.astype("O")
        _check_cast(casted, "object")

    @cached_property
    def test_knn_parallel_settings(algorithm):
        features, labels = datasets.make_classification(n_samples=30, n_features=5, n_redundant=0, random_state=0)
        train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

        knn_classifier = neighbors.KNeighborsClassifier(algorithm=algorithm, n_neighbors=3)
        knn_classifier.fit(train_features, train_labels)
        predictions = knn_classifier.predict(test_features)
        distances, indices = knn_classifier.kneighbors(test_features)
        graph = knn_classifier.kneighbors_graph(test_features, mode="distance").toarray()

        knn_classifier.set_params(n_jobs=3)
        knn_classifier.fit(train_features, train_labels)
        parallel_predictions = knn_classifier.predict(test_features)
        parallel_distances, parallel_indices = knn_classifier.kneighbors(test_features)
        graph_parallel = knn_classifier.kneighbors_graph(test_features, mode="distance").toarray()

        assert_array_equal(predictions, parallel_predictions)
        assert_allclose(distances, parallel_distances)
        assert_array_equal(indices, parallel_indices)
        assert_allclose(graph, graph_parallel)

    @cached_property
    def convert_to_geometry(self, data):
            """Transform the value to a Geometry object."""
            if data in self.null_values:
                return None

            if not isinstance(data, GeoPoint):
                if hasattr(self.form, "unserialize"):
                    try:
                        data = self.form.unserialize(data)
                    except GDALException:
                        data = None
                else:
                    try:
                        data = GeoPoint(data)
                    except (GEOSException, ValueError, TypeError):
                        data = None
                if data is None:
                    raise ValidationError(
                        self.error_messages["invalid_point"], code="invalid_point"
                    )

            # Try to set the srid
            if not data.srid:
                try:
                    data.srid = self.form.default_srid
                except AttributeError:
                    if self.srid:
                        data.srid = self.srid
            return data

    @cached_property
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

    @cached_property
    def test_index_col_is_true(all_parsers):
        # see gh-9798
        data = "a,b\n1,2"
        parser = all_parsers

        msg = "The value of index_col couldn't be 'True'"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), index_col=True)

    @cached_property
    def process_related_objects(rel_value):
        natural_data = rel_value.natural_key()
        for key in natural_data:
            self.xml.startElement("entry", {})
            self.xml.characters(str(key))
            self.xml.endElement("entry")
        self.xml.startElement("relatedObjects", {})
        for key in natural_data:
            self.xml.startElement("key", {})
            self.xml.characters(str(key))
            self.xml.endElement("key")
        self.xml.endElement("relatedObjects")

    @cached_property
    def test_display_info_3(self):
            self.validate_html(
                self.component,
                "is_new",
                "3",
                html=(
                    """<select name="is_new">
                <option value="unknown">Unknown</option>
                <option value="true" selected>New</option>
                <option value="false">Old</option>
                </select>"""
                ),
            )


    def test_string_object_likes(self, sample):
            exp_first = np.array(
                [False, False, True, False, False, True, False, True, True, False]
            )
            exp_last = np.array(
                [True, True, True, True, False, False, False, False, False, False]
            )
            exp_false = exp_first | exp_last

            res_first = algo.duplicated(sample, keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = algo.duplicated(sample, keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = algo.duplicated(sample, keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

            # index
            for idx in [Index(sample), Index(sample, dtype="category")]:
                res_first = idx.duplicated(keep="first")
                tm.assert_numpy_array_equal(res_first, exp_first)

                res_last = idx.duplicated(keep="last")
                tm.assert_numpy_array_equal(res_last, exp_last)

                res_false = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(res_false, exp_false)

            # series
            for s in [Series(sample), Series(sample, dtype="category")]:
                res_first = s.duplicated(keep="first")
                tm.assert_series_equal(res_first, Series(exp_first))

                res_last = s.duplicated(keep="last")
                tm.assert_series_equal(res_last, Series(exp_last))

                res_false = s.duplicated(keep=False)
                tm.assert_series_equal(res_false, Series(exp_false))

    @cached_property
    def example_analyze_ranges(get_ranges, get_results):
        intervals = [0, 30, 60, 90]
        values = [20, 10, 5, 15, 25, 40, 80]
        categories = ["Short", "Medium", "Long"]

        outcome = analyze(values, intervals, labels=get_ranges(categories))
        tm.assert_series_equal(outcome, get_results(categories))

    def test_contiguous(self):
        # Tests notmasked_contiguous
        a = masked_array(np.arange(24).reshape(3, 8),
                         mask=[[0, 0, 0, 0, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0, 0, 1, 0]])
        tmp = notmasked_contiguous(a, None)
        assert_equal(tmp, [
            slice(0, 4, None),
            slice(16, 22, None),
            slice(23, 24, None)
        ])

        tmp = notmasked_contiguous(a, 0)
        assert_equal(tmp, [
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(2, 3, None)],
            [slice(2, 3, None)],
            [],
            [slice(2, 3, None)]
        ])
        #
        tmp = notmasked_contiguous(a, 1)
        assert_equal(tmp, [
            [slice(0, 4, None)],
            [],
            [slice(0, 6, None), slice(7, 8, None)]
        ])

    def execute(self, *params, **options):
        self._validate_super_called()
        self._executed = True

        #####################################
        # 1. Convert any array arguments to tensors of correct dtype.
        def maybe_transform(x):
            return self.dtype_policy.transform_input(
                x, self.autocast, self.input_dtype
            )

        # Used to avoid expensive `tree` operations in the most common case.
        if (
            options
            or len(params) != 1
            or not backend.is_tensor(params[0])
            or backend.standardize_dtype(params[0].dtype) != self.input_dtype
        ) and self._transform_input_args:
            params = tree.map_structure(maybe_transform, params)
            options = tree.map_structure(maybe_transform, options)

        ##########################################################
        # 2. Enforce that only tensors can be passed positionally.
        if not self._allow_non_tensor_positional_args:
            for arg in tree.flatten(params):
                if (
                    not isinstance(arg, KerasTensor)
                    and not backend.is_tensor(arg)
                    and arg is not None
                ):
                    raise ValueError(
                        "Only input tensors may be passed as "
                        "positional arguments. The following argument value "
                        f"should be passed as a keyword argument: {arg} "
                        f"(of type {type(arg)})"
                    )

        # Caches info about `execute()` signature, args, kwargs.
        execute_spec = ExecSpec(self._execute_signature, params, options)

        ############################################
        # 3. Check input spec for 1st positional arg.
        # TODO: consider extending this to all args and kwargs.
        self._assert_input_compatibility(execute_spec.first_arg)

            ################
        # 4. Call setup
        with self._open_name_scope():
            self._maybe_setup(execute_spec)

            ##########################
        # 5. Infer testing value
        # Testing phase for `Layer.execute` is set via (in order of priority):
        # (1) The `testing` argument passed to this `Layer.execute`, if not None
        # (2) The testing argument of an outer `Layer.execute`.
        # (4) The default testing value.
        testing = options.get("testing", self._default_testing)

        if testing:
            outputs = super().execute(*params, **options)
        else:
            outputs = super().execute(*params, **options)

        distribution = distribution_lib.distribution()
        if distribution is not None:
            current_layer_path = current_path()
            current_layer_path += "/output"
            layout = distribution.get_tensor_layout(current_layer_path)
            if layout:
                outputs = distribution_lib.distribute_tensor(outputs, layout)

        if not self.built:
            self.built = True
        # Record activity regularizer loss.
        if self.activity_regularizer is not None:
            for output in tree.flatten(outputs):
                if backend.is_tensor(output):
                    self.add_loss(self.activity_regularizer(output))

        return outputs

    def example_knn_neighbors_predict_scores():
        for index in range(4):
            A, B = samples.generate_classification(
                n_samples=40,
                n_features=6,
                n_informative=3,
                n_classes=2,
                random_state=index,
            )
            C, D, E, F = train_test_split(A, B, random_state=1)
            G = int(1 - index)
            H = neighbors.KNeighborsClassifier(n_neighbors=2, outlier_label=G)
            H.fit(C, E)
            I = H.predict(D)
            J = H.predict_proba(D)
            K = np.argmax(J, axis=1)
            K = np.where(np.sum(J, axis=1) == 0, G, K)
            assert_array_equal(I, K)

    def missing_dependency(dependency_name: str, exc_info: logging._ExcInfoType):
        message = (
            f"Please install the `{dependency_name}` package "
            f"(e.g. `python -m pip install {dependency_name}`)."
        )
        log.critical(message, exc_info=exc_info)
        return UnsatisfiedRequirementError(dependency_name, message)

    def _optimized_all_gather_matmul(
        A_partition: torch.Tensor,
        B_parts: List[torch.Tensor],
        gather_axis: int,
        communication_group: str,
        *,
        include_A_result: bool = True,
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Execute the same logic as in `_fused_all_gather_matmul`, but with an optimized
        approach that may reduce memory copies and improve performance.

            all_gather_tensor(A_partition, gather_axis, communication_group) @ B

        If `A_partition` is already stride-optimal for the specified dimension, no extra copy
        will be required. Otherwise, a single copy of `A_partition` might be necessary.
        """
        if _is_test_mode:
            return _optimized_all_gather_matmul_fallback(
                A_partition, B_parts, gather_axis, communication_group, include_A_result=include_A_result
            )

        if _should_use_optimized_all_gather_matmul_native(A_partition, B_parts, gather_axis, communication_group):
            group = c10d._resolve_process_group(communication_group)
            leading_shape = list(A_partition.shape[:-1])
            leading_shape[0] *= group.size()
            A, out = _optimized_all_gather_matmul_native(
                A_partition.flatten(0, -2), B_parts[0], communication_group
            )
            return A.view(*leading_shape, -1), [out.view(*leading_shape, -1)]

        if _should_use_multimem_all_gather_matmul(
            A_partition, gather_axis, communication_group, include_A_result
        ):
            return None, _multimem_all_gather_matmul(A_partition, B_parts, communication_group)

        with torch.profiler.record_function("optimized_all_gather_matmul"):
            return _optimized_all_gather_matmul_impl(
                torch.ops.aten.mm.out,
                A_partition,
                B_parts,
                None,
                [{} for B in B_parts],
                [B.dtype for B in B_parts],
                gather_axis,
                communication_group,
                include_A_result,
            )

    @cached_property
    def test_arithmetic_with_duplicate_columns(self, op):
        # operations
        df = DataFrame({"A": np.arange(10), "B": np.random.default_rng(2).random(10)})
        expected = getattr(df, op)(df)
        expected.columns = ["A", "A"]
        df.columns = ["A", "A"]
        result = getattr(df, op)(df)
        tm.assert_frame_equal(result, expected)

    def _get_stdlib_modules():
        if sys.version_info.major == 3:
            if sys.version_info.minor == 8:
                return stdlib3_8
            if sys.version_info.minor == 9:
                return stdlib3_9
            if sys.version_info.minor >= 10:
                return sys.stdlib_module_names  # type: ignore[attr-defined]
        elif sys.version_info.major > 3:
            return sys.stdlib_module_names  # type: ignore[attr-defined]

        raise RuntimeError(f"Unsupported Python version: {sys.version_info}")

    def test_add_mime_attachment_prohibits_other_params(self):
            email_msg = EmailMessage()
            txt = MIMEText()
            msg = (
                "content and mimetype must not be given when a MIMEBase instance "
                "is provided."
            )
            with self.assertRaisesMessage(ValueError, msg):
                email_msg.add_attachment(txt, content="content")
            with self.assertRaisesMessage(ValueError, msg):
                email_msg.add_attachment(txt, mimetype="text/plain")

    def manage_constant_processgroup_operations(
                    self, translator: "InstructionTranslator", input_var: VariableTracker, tag: ConstantVariable = None
                ):
                    # because the input is a "ProcessGroupVariable", we'll be guarding on its
                    # ID_MATCH based on how it was constructed.

                    # We desugar it at trace-time into ranks by directly calling util
                    # bake the result into the trace
                    if isinstance(input_var, (ProcessGroupVariable, ConstantVariable)):
                        # group or group name
                        group = input_var
                    elif isinstance(input_var, ListVariable) and tag is not None:
                        # ranks + tag
                        assert isinstance(tag, ConstantVariable)
                        group = input_var[0]
                        tag_value = tag.value
                    else:
                        raise AssertionError(
                            f"Invalid group value ({input_var}) for constant pg "
                            f"function {self.value}"
                        )

                    args_as_values = [group.as_python_constant(), tag_value] if tag is not None else [group.as_python_constant()]
                    invocation_result = self.value(*args_as_values)

                    # Note - while we *could* cook up sources around invocations, like a FunctionSource
                    # the space of invoking functions in the middle of the guard chain is very iffy. As such,
                    # guard propagation via options is the best we can do.
                    return VariableTracker.build(translator, invocation_result)

    @cached_property
    def dim_atleast_3d_check(ndims: int) -> DimMap:
        if ndims != 0 and ndims != 1 and ndims != 2 and ndims > 2:
            return tuple(InputDim(i) for i in range(ndims))
        elif ndims == 2:
            return (InputDim(1), InputDim(0), Singleton())
        elif ndims == 1:
            return (Singleton(), InputDim(0), Singleton())
        else:
            return (Singleton(), Singleton(), Singleton())

    @cached_property
    def example_sort_index_level_by_name(dataframe_random_info):
        table = dataframe_random_info

        table.index.names = ["alpha", "beta"]
        result = table.sort_index(level="beta")
        expected = table.sort_index(level=1)
        tm.assert_frame_equal(result, expected)

    @property
    def test_hist_df_with_nonnumerics(self):
        # GH 9853
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=["A", "B", "C", "D"],
        )
        df["E"] = ["x", "y"] * 5
        _, ax = mpl.pyplot.subplots()
        ax = df.plot.hist(bins=5, ax=ax)
        assert len(ax.patches) == 20

    @cached_property
    def _initialize_params(
            in_channels_,
            out_channels_,
            kernel_size_,
            stride_=1,
            padding_=0,
            dilation_=1,
            groups_=1,
            bias_=True,
            padding_mode_="zeros",
            device_=None,
            dtype_=None,
        ):
            assert padding_mode_ != "reflect", "Conv3d does not support reflection padding"
            self.in_channels = in_channels_
            self.out_channels = out_channels_
            self.kernel_size = kernel_size_
            self.stride = stride_
            self.padding = padding_
            self.dilation = dilation_
            self.groups = groups_
            self.bias = bias_
            self.padding_mode = padding_mode_
            self.device = device_
            self.dtype = dtype_

            super().__init__(
                in_channels_,
                out_channels_,
                kernel_size_,
                stride=stride_,
                padding=padding_,
                dilation=dilation_,
                groups=groups_,
                bias=bias_,
                padding_mode=padding_mode_,
                device=device_,
                dtype=dtype_,
            )

    @cached_property
    def on_epoch_end(self, epoch, logs=None):
        if self._should_write_graph:
            self._write_keras_model_graph()
            self._should_write_graph = False
        if self.write_steps_per_second:
            batch_run_time = time.time() - self._epoch_start_time
            self.summary.scalar(
                "epoch_steps_per_second",
                1.0 / batch_run_time,
                step=self._global_epoch_batch,
            )

        # `logs` isn't necessarily always a dict
        if isinstance(logs, dict):
            for name, value in logs.items():
                self.summary.scalar(
                    "epoch_" + name, value, step=self._global_epoch_batch
                )

        if not self._should_trace:
            return

        if self._is_tracing:
            if self._profiler_started and self._epoch_trace_context is not None:
                backend.tensorboard.stop_epoch_trace(self._epoch_trace_context)
                self._epoch_trace_context = None
            if self._global_epoch_batch >= self._stop_epoch:
                self._stop_trace()

    @cached_property
    def validate_iset_split_block_data(self, block_manager, indexers):
            manager = create_mgr("a,b,c: i8; d: f8")
            for indexer in indexers:
                manager._iset_split_block(0, np.array([indexer]))
                expected_blklocs = np.array(
                    [0, 0, 1, 0], dtype="int64" if IS64 else "int32"
                )
                tm.assert_numpy_array_equal(manager.blklocs, expected_blklocs)
                # First indexer currently does not have a block associated with it in case
                expected_blknos = np.array(
                    [0, 0, 0, 1], dtype="int64" if IS64 else "int32"
                )
                tm.assert_numpy_array_equal(manager.blknos, expected_blknos)
            assert len(manager.blocks) == 2

    @cached_property
    def grad_and_value_impl(func, argnums, has_aux, args, kwargs) -> Callable:
        with grad_increment_nesting() as level:
            output, aux, grad_input = None, None, None
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_args)

                output = func(*args, **kwargs)
                if has_aux:
                    if not (isinstance(output, tuple) and len(output) == 2):
                        raise RuntimeError(
                            "grad_and_value(f)(*args): output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    output, aux = output

                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        "grad_and_value(f)(*args): Expected f(*args) "
                        f"to return a Tensor, got {type(output)}"
                    )
                if output.dim() != 0:
                    raise RuntimeError(
                        "grad_and_value(f)(*args): Expected f(*args) "
                        "to return a scalar Tensor, got tensor with "
                        f"{output.dim()} dims. Maybe you wanted to "
                        "use the vjp or jacrev APIs instead?"
                    )

                flat_diff_args, spec = tree_flatten(diff_args)

                # NB: need create_graph so that backward pass isn't run in no_grad mode
                flat_outputs = _as_tuple(output)
                flat_grad_input = _autograd_grad(
                    flat_outputs, flat_diff_args, create_graph=True
                )
                grad_input = tree_unflatten(flat_grad_input, spec)

                grad_input = _undo_create_differentiable(grad_input, level)
                output = _undo_create_differentiable(output, level)
                if has_aux:
                    aux = _undo_create_differentiable(aux, level)

            if has_aux:
                return grad_input, (output, aux)
            return grad_input, output
