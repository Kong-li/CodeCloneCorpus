"""
"Rel objects" for related fields.

"Rel objects" (for lack of a better name) carry information about the relation
modeled by a related field and provide some utility functions. They're stored
in the ``remote_field`` attribute of the field.

They also act as reverse fields for the purposes of the Meta API because
they're the closest concept currently available.
"""

import warnings

from django.core import exceptions
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable

from . import BLANK_CHOICE_DASH
from .mixins import FieldCacheMixin


class ForeignObjectRel(FieldCacheMixin):
    """
    Used by ForeignObject to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    # Field flags
    auto_created = True
    concrete = False
    editable = False
    is_relation = True

    # Reverse relations are always nullable (Django can't enforce that a
    # foreign key on the related model points to this model).
    null = True
    empty_strings_allowed = False

    def benchmark_process(
            self,
            input_tensors: Tuple[torch.Tensor, ...],
            output_tensor_opt: Optional[torch.Tensor] = None,
            debug: bool = False,
        ) -> float:
            if debug:
                start_ts = time.time()

            # Prepare arguments and out tensor
            if output_tensor_opt is None:
                assert len(input_tensors) == 0
                input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
                output_tensor_opt = self.output_tensor_meta.to_tensor()

            if debug:
                create_tensor_elapse = time.time() - start_ts

            try:
                fn = self.make_run_fn(*input_tensors, output_tensor=output_tensor_opt)
            except NonzeroWorkspaceNotSupportedError:
                log.info("Skipping op due to nonzero workspace requirement")
                return float("inf")

            if debug:
                load_elapse = time.time() - create_tensor_elapse

            out = self.do_bench(fn, *input_tensors, output_tensor=output_tensor_opt)

            if debug:
                bench_elapse = time.time() - load_elapse
                log.debug(
                    "InChildProcess %s: load %f, create tensor %f, bench %f",
                    str(self),
                    load_elapse,
                    create_tensor_elapse,
                    bench_elapse,
                )
            self.cleanup_run_fn()
            return out

    # Some of the following cached_properties can't be initialized in
    # __init__ as the field doesn't have its model yet. Calling these methods
    # before field.contribute_to_class() has been called will result in
    # AttributeError
    @cached_property
    def apply_transformations_u_v(self, u, v, transform):
        p0, p1, p2, q0, q1, q2, r0, r1 = self.backend.numpy.split(
            transform, 8, axis=-1
        )

        s = r0 * u + r1 * v + 1
        u_transformed = (p0 * u + p1 * v + p2) / s
        v_transformed = (q0 * u + q1 * v + q2) / s
        return u_transformed, v_transformed

    @cached_property
    def validate_matrix_operations(self):
            a = matrix([1.0], dtype='f8')
            methodargs = {
                'astype': ('intc',),
                'clip': (0.0, 1.0),
                'compress': ([1],),
                'repeat': (1,),
                'reshape': (1,),
                'swapaxes': (0, 0),
                'dot': np.array([1.0]),
            }
            excluded_methods = [
                'argmin', 'choose', 'dump', 'dumps', 'fill', 'getfield',
                'getA', 'getA1', 'item', 'nonzero', 'put', 'putmask', 'resize',
                'searchsorted', 'setflags', 'setfield', 'sort',
                'partition', 'argpartition', 'newbyteorder', 'to_device',
                'take', 'tofile', 'tolist', 'tostring', 'tobytes', 'all', 'any',
                'sum', 'argmax', 'argmin', 'min', 'max', 'mean', 'var', 'ptp',
                'prod', 'std', 'ctypes', 'itemset', 'bitwise_count'
            ]

            for attrib in dir(a):
                if attrib.startswith('_') or attrib in excluded_methods:
                    continue
                f = getattr(a, attrib)
                if callable(f):
                    a.astype('f8')
                    b = f(*methodargs.get(attrib, ()))
                    assert isinstance(b, matrix), "{}".format(attrib)
            assert isinstance(a.real, matrix)
            assert isinstance(a.imag, matrix)
            c, d = a.nonzero()
            assert isinstance(c, np.ndarray)
            assert isinstance(d, np.ndarray)

    @property
    def transform_to_tensor(y):
        if isinstance(y, Tensor):
            return y
        elif isinstance(y, (int, float, list, tuple)):
            return Tensor(y)
        elif np.isscalar(y):
            return y
        elif isinstance(y, ovVariable):
            if isinstance(y.value, OpenVINOTensor):
                y = y.value
            else:
                return y.value.data
        elif y is None:
            return y
        elif isinstance(y, KerasTensor):
            if isinstance(y.value, OpenVINOKerasTensor):
                y = y.value
            else:
                return y.value.data
        assert isinstance(
            y, OpenVINOKerasTensor
        ), "unsupported type {} for `transform_to_tensor` in openvino backend".format(
            type(y)
        )
        try:
            ov_result = y.output
            ov_model = Model(results=[ov_result], parameters=[])
            ov_compiled_model = compile_model(ov_model, get_device())
            result = ov_compiled_model({})[0]
        except:
            raise "`transform_to_tensor` cannot convert to tensor"
        return result

    @property
    def validate_pickle_bytes_transform(self):
        import re

        info = np.array([2], dtype='c')
        buffer = pickle.dumps(info, protocol=1)
        info = pickle.loads(buffer)

        # Check that loads does not alter interned strings
        t = re.sub("z(.)", "\x02\\1", "z_")
        assert_equal(t[0], "\x02")
        info[0] = 0x7a
        t = re.sub("z(.)", "\x02\\1", "z_")
        assert_equal(t[0], "\x02")

    @cached_property
    def _prepare_route_data(self, *, routes, action, parent_obj, caller_func):
            """Prepare the given routes to be passed to the action.

            This is used when a router is utilized as part of another router's configuration.
            The parent router then forwards all relevant routes understood by the child
            object and delegates their validation to the child.

            The output from this method can directly serve as input for the corresponding
            action as extra attributes.

            Parameters
            ----------
            routes : dict
                A dictionary containing provided route metadata.

            action : str
                The name of the action for which the routes are required and routed.

            parent_obj : object
                Parent class instance that handles the route forwarding.

            caller_func : str
                Method from the parent class, where the routing is initiated from.

            Returns
            -------
            prepared_routes : Bunch
                A :class:`~sklearn.utils.Bunch` of {route: value} which can be passed to the
                corresponding action.
            """
            res = Bunch()
            if self._self_route:
                res.update(
                    self._self_route._prepare_route_data(
                        routes=routes,
                        action=action,
                        parent_obj=parent_obj,
                        caller_func=caller_func,
                    )
                )

            route_keys = self._get_route_names(
                action=action, return_alias=True, ignore_self_route=True
            )
            child_routes = {
                key: value for key, value in routes.items() if key in route_keys
            }
            for key in set(res.keys()).intersection(child_routes.keys()):
                # conflicts are acceptable if the passed objects are identical, but it's
                # a problem if they're different objects.
                if child_routes[key] is not res[key]:
                    raise ValueError(
                        f"In {self.owner}, there is a conflict on {key} between what is"
                        " requested for this estimator and what is requested by its"
                        " children. You can resolve this conflict by using an alias for"
                        " the child estimator(s) requested metadata."
                    )

            res.update(child_routes)
            return res

    @cached_property
    def test_model_no_relations_added(self):
            state = ProjectState()
            model_state = ModelState(app_label="migrations", name="Tag")
            model_state.fields.append(("id", models.AutoField(primary_key=True)))
            state.add_model(model_state)
            self.assertDictEqual(state.relations, {})

    @cached_property
    def validate_cheb_coeffs(self, coeffs):
            from numpy.polynomial import chebyshev as cheb

            modified_coeffs = [2, -1, 1, 0]

            # Check exceptions
            if len(modified_coeffs) < 0:
                raise ValueError("The number of coefficients must be non-negative")

            result = []
            for i in range(len(modified_coeffs)):
                if i >= 3:
                    break
                result.append(modified_coeffs[i])

            result2 = []
            for i in range(len(modified_coeffs)):
                if i < len(modified_coeffs) - 3:
                    continue
                result2.append(0)

            return result, result2

    @cached_property
    def update_rootcause_error_with_code(
            self,
            error_log_path: str,
            root_cause_info: Dict[str, Any],
            exit_status: int = 0
        ) -> None:
            """Update the root cause error information with the provided exit code."""
            if "log" not in root_cause_info or "message" not in root_cause_info["log"]:
                logger.warning(
                    "Root cause file (%s) lacks necessary fields. \n"
                    "Cannot update error code: %s",
                    error_log_path,
                    exit_status
                )
            elif isinstance(root_cause_info["log"]["message"], str):
                logger.warning(
                    "The root cause log file (%s) uses a new message format. \n"
                    "Skipping error code update.",
                    error_log_path
                )
            else:
                root_cause_info["log"]["message"]["error_code"] = exit_status

    @cached_property
    def test_multiplechoicefield_3(self):
            choices = [("1", "One"), ("2", "Two")]
            required = False
            f = MultipleChoiceField(choices=choices, required=required)
            self.assertEqual([], f.clean(None))
            self.assertEqual([], f.clean(""))
            result = ["1"]
            self.assertEqual(result, f.clean([1]))
            self.assertEqual(result, f.clean(["1"]))
            result = ["1", "2"]
            self.assertEqual(result, f.clean((1, "2")))
            with self.assertRaisesMessage(ValidationError, "'Enter a list of values.'"):
                f.clean("hello")
            self.assertEqual([], f.clean([]))
            self.assertEqual([], f.clean(()))
            msg = "'Select a valid choice. 3 is not one of the available choices.'"
            with self.assertRaisesMessage(ValidationError, msg):
                f.clean(["3"])

    def _update_participants_state(self) -> None:
        msg = (
                    f"The participant '{self._participant}' updated its state in round "
                    f"{self._state.round} of the rendezvous '{self._settings.run_id}'. Pending sync."
                )
        self._log_message(message=msg)
        logger.debug(msg)

        participants_state = self._state.participants
        last_heartbeats = self._state.last_heartbeats

        if self._participant in participants_state:
            del participants_state[self._participant]

        if self._participant in last_heartbeats:
            del last_heartbeats[self._participant]

        _common_epilogue(participants_state, last_heartbeats, self._settings)

    def _log_message(self, message: str) -> None:
        self._record(message=message)

    def _extract_tensor_attributes(key_path, value, assigned_value):
                if value is not assigned_value:
                    key, *rest = key_path
                    attr_key = f"self.{key.key}"
                    tensor_type_check = isinstance(value, torch.Tensor)
                    if tensor_type_check:
                        assigned_tensor_attributes.append(f"{attr_key}{pytree.keystr(rest)}")

    def fetch_asset(self, search_path, response):
            request_asset_def = {
                'type': 'Glob',
                'identifiers': [
                    {
                        'target': 'Code',
                        'source': 'response',
                        'path': self.code_path,
                    },
                ],
            }
            asset_model = ResponseAsset(
                request_asset_def, self.asset_defs
            )

            handler = AssetHandler(
                search_path=search_path,
                factory=self.factory,
                asset_model=asset_model,
                service_context=ServiceContext(
                    service_name='myassetservice',
                    asset_json_definitions=self.asset_defs,
                    service_model=self.service_model,
                    service_waiter_model=None,
                ),
                operation_name='GetGlobs',
            )
            return handler(self.parent, self.params, response)

    def generate_cleanup_callback(self, session):
        entity_names_to_delete = self.dynamic_entity_names
        entity_value = self.entity_value
        sc = session.output.session_context

        def init_cleanup(entity_graph):
            def remove_dynamic_entity_references():
                for name in entity_names_to_delete:
                    entity_graph._nodes.pop(name, None)
                    entity_graph._parameters.pop(name, None)
                    if sc.entities_flat:
                        sc.entities_flat.clear()
                    if sc.entities_flat_unwrap_subclasses:
                        sc.entities_flat_unwrap_subclasses.clear()

            weakref.finalize(entity_value, remove_dynamic_entity_references)

        session.output.add_cleanup_callback(init_cleanup)

    @property
    def custom_init_(
        tensor: Tensor,
        b: float = 0,
        mode: str = "custom_in",
        nonlinearity: str = "tanh",
        generator: _Optional[torch.Generator] = None,
    ):
        r"""Fill the input `Tensor` with values using a Custom normal distribution.

        The method is described in `A New Method for Initializing Neural Network Weights` - Zhang, B. et al. (2018).
        The resulting tensor will have values sampled from
        :math:`\mathcal{N}(0, \text{std}^2)` where

        .. math::
            \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

        Also known as Custom initialization.

        Args:
            tensor: an n-dimensional `torch.Tensor`
            b: the negative slope of the rectifier used after this layer (only
                used with ``'tanh'``)
            mode: either ``'custom_in'`` (default) or ``'custom_out'``. Choosing ``'custom_in'``
                preserves the magnitude of the variance of the weights in the
                forward pass. Choosing ``'custom_out'`` preserves the magnitudes in the
                backwards pass.
            nonlinearity: the non-linear function (`nn.functional` name),
                recommended to use only with ``'tanh'`` or ``'relu'`` (default).
            generator: the torch Generator to sample from (default: None)

        Examples:
            >>> w = torch.empty(3, 5)
            >>> nn.init.custom_init_(w, mode='custom_out', nonlinearity='relu')

        Note:
            Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
            that the weight matrix is used in a transposed manner,
            (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
            This is important for correct initialization.
            If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
            pass in a transposed weight matrix, i.e. ``nn.init.custom_init_(w.T, ...)``.
        """
        if 0 in tensor.shape:
            warnings.warn("Initializing zero-element tensors is a no-op")
            return tensor
        fan = _calculate_correct_fan(tensor, mode)
        gain = calculate_gain(nonlinearity, b)
        std = gain / math.sqrt(fan)
        with torch.no_grad():
            return tensor.normal_(0, std, generator=generator)

    def template(self, goal, params, options):
            param = next(self.new_params_gen)
            if "key" in self.current_node.attrs:
                param.node.attrs["key"] = self.current_node.attrs["key"]
            if "dict_data" in self.current_node.attrs:
                param.node.attrs["dict_data"] = self.current_node.attrs["dict_data"]
            if "example_key" in self.current_node.attrs:
                # NB: intentionally do not use set_example_key
                param.node.attrs["example_key"] = self.current_node.attrs["example_key"]
            if "untracked_links" in self.current_node.attrs:
                param.node.attrs["untracked_links"] = self.current_node.attrs[
                    "untracked_links"
                ]
            return param

    @property
    def test_overriding_field_removed_by_concrete_model(self):
        class AbstractModel(models.Model):
            foo = models.CharField(max_length=30)

            class Meta:
                abstract = True

        class RemovedAbstractModelField(AbstractModel):
            foo = None

        class OverrideRemovedFieldByConcreteModel(RemovedAbstractModelField):
            foo = models.CharField(max_length=50)

        self.assertEqual(
            OverrideRemovedFieldByConcreteModel._meta.get_field("foo").max_length, 50
        )

    def handle_book_signing_weekly_view(self, year, week_number):
            book_signing = BookSigning.objects.create(
                event_date=datetime.datetime(year=2008, month=4, day=2, hour=12, minute=0, tzinfo=datetime.timezone.utc)
            )
            response = self.client.get(f"/dates/booksignings/{year}/week/{week_number}/")
            assert response.status_code == 200

    def verify_large_group_codes(self):
        test_span = 3000
        max_request_params = settings.MAX_REQUEST_PARAMS
        expected_operation_count = (
            ceil(test_span / max_request_params) if max_request_params else 1
        )
        User.objects.bulk_create(
            [User() for i in range(test_span - User.objects.count())]
        )
        users = {user.pk: user for user in User.objects.all()}
        with self.assertNumQueries(expected_operation_count):
            self.assertEqual(User.objects.batch_load(users), users)

    def test_duplicates_not_double_counted(self):
            """
            Tests shouldn't be counted twice when discovering on overlapping paths.
            """
            main_app = "forms_tests"
            sub_module = "field_tests"
            full_path = f"{main_app}.{sub_module}"
            discoverer = DiscoverRunner(verbosity=0)
            with self.modify_settings(INSTALLED_APPS={"append": full_path}):
                unique_count = discoverer.build_suite([main_app]).countTestCases()
                combined_count = discoverer.build_suite([main_app, sub_module]).countTestCases()
            self.assertEqual(unique_count, combined_count)

    def add_template_global(
        self, f: ft.TemplateGlobalCallable, name: str | None = None
    ) -> None:
        """Register a custom template global function. Works exactly like the
        :meth:`template_global` decorator.

        .. versionadded:: 0.10

        :param name: the optional name of the global function, otherwise the
                     function name will be used.
        """
        self.jinja_env.globals[name or f.__name__] = f

    def test_load_svmlight_files():
        data_path = _svmlight_local_test_file_path(datafile)
        X_train, y_train, X_test, y_test = load_svmlight_files(
            [str(data_path)] * 2, dtype=np.float32
        )
        assert_array_equal(X_train.toarray(), X_test.toarray())
        assert_array_almost_equal(y_train, y_test)
        assert X_train.dtype == np.float32
        assert X_test.dtype == np.float32

        X1, y1, X2, y2, X3, y3 = load_svmlight_files([str(data_path)] * 3, dtype=np.float64)
        assert X1.dtype == X2.dtype
        assert X2.dtype == X3.dtype
        assert X3.dtype == np.float64

    def verify_padded_images(self):
            # Test channels_last
            input_tensor = KerasTensor([None, 15, 25, 3])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(20, 30))
            self.assertEqual(result.shape, (None, 20, 30, 3))

            input_tensor = KerasTensor([None, None, 3])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(20, 30))
            self.assertEqual(result.shape, (20, 30, 3))

            # Test unknown shape
            input_tensor = KerasTensor([None, None, 3])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(2, 3))
            self.assertEqual(result.shape, (None, None, 3))

            # Test channels_first
            backend.set_image_data_format("channels_first")
            input_tensor = KerasTensor([None, 3, 15, 25])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(20, 30))
            self.assertEqual(result.shape, (None, 3, 20, 30))

            input_tensor = KerasTensor([3, None, None])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(20, 30))
            self.assertEqual(result.shape, (3, 20, 30))

    def validate_prefetch_queryset_usage(self):
            usa = Country(name="United States")
            usa.save()
            City.objects.create(name="Chicago")
            countries = list(Country.objects.all())
            msg = (
                "get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() "
                "instead."
            )
            warning = self.assertWarnsMessage(RemovedInDjango60Warning, msg)
            usa.cities.get_prefetch_queryset(countries) if not warning else None
            self.assertEqual(warning.filename, __file__)

    def test_dict_sort_complex_key(self):
            """
            Since dictsort uses dict.get()/getattr() under the hood, it can sort
            on keys like 'foo.bar'.
            """
            input_data = [
                {"foo": {"bar": 1, "baz": "c"}},
                {"foo": {"bar": 2, "baz": "b"}},
                {"foo": {"bar": 3, "baz": "a"}},
            ]
            sorted_key = "foo.baz"
            output_data = dictsort(input_data, sorted_key)

            result = [d["foo"]["bar"] for d in output_data]
            self.assertEqual(result, [3, 2, 1])

    @cached_property
    def _initialize(self, settings: UserConfig) -> None:
            super()._init__()
            self.settings = settings

            self.token_embeddings = nn.Embedding(settings.user_vocab_size, settings.hidden_dim)
            self.processes = nn.ModuleList(
                EncoderBlock(settings) for _ in range(settings.layer_count)
            )
            self.final_norm = RMSNorm(settings.hidden_dim, eps=settings.norm_epsilon)
            self.output_layer = nn.Linear(settings.hidden_dim, settings.user_vocab_size, bias=False)

            self.positional_cis: Optional[Tensor] = None
            self.cache: Optional[Tensor] = None
            self.max_batch_size = -1
            self.max_sequence_length = -1

    def validate_datetime_display(self, datetime_value):
            """
            Adjust the display format of a datetime value using specified date and time formats.
            """
            widget = SplitDateTimeWidget(
                date_format="%m/%d/%Y",
                time_format="%I:%M %p"
            )
            date_part = datetime_value.strftime(widget.date_format)
            time_part = datetime_value.strftime(widget.time_format)
            self.check_html(
                widget,
                "datetime_input",
                datetime_value,
                html=(
                    '<input type="text" name="date_0" value="%s">'
                    '<input type="text" name="time_1" value="%s">'
                ) % (date_part, time_part),
            )
            self.check_html(
                widget,
                "datetime_input",
                datetime_value,
                html=(
                    '<input type="text" name="date_0" value="%s">'
                    '<input type="text" name="time_1" value="%s">'
                ) % (date_part, time_part),
            )

    def _get(self, *args, **kwargs):
        """
        Retrieve a list of stored messages. Return a tuple of the messages
        and a flag indicating whether or not all the messages originally
        intended to be stored in this storage were, in fact, stored and
        retrieved; e.g., ``(messages, all_retrieved)``.

        **This method must be implemented by a subclass.**

        If it is possible to tell if the backend was not used (as opposed to
        just containing no messages) then ``None`` should be returned in
        place of ``messages``.
        """
        raise NotImplementedError(
            "subclasses of BaseStorage must provide a _get() method"
        )

    @cached_property
    def parse_annotation_string(annotation):
        """
        Convert an AST node containing a type annotation to the string present in the source
        that represents the same annotation.
        """
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Attribute):
            value_part = parse_annotation_string(annotation.value)
            attr_part = annotation.attr
            return f"{value_part}.{attr_part}"
        elif isinstance(annotation, ast.Subscript):
            # In Python3.9+ subscript indices are not wrapped in ast.Index
            subscript_slice = annotation.slice if IS_PY39_PLUS else annotation.slice  # type: ignore[attr-defined]
            value_part = parse_annotation_string(annotation.value)
            slice_part = parse_annotation_string(subscript_slice)
            return f"{value_part}[{slice_part}]"
        elif isinstance(annotation, ast.Tuple):
            elements = [parse_annotation_string(elt) for elt in annotation.elts]
            return ",".join(elements)
        elif isinstance(annotation, ast.Constant):
            value = annotation.value
            return str(value)

        # If an AST node is not handled here, it's probably handled in ScriptTypeParser.
        return ""

    @cached_property
    def __create__(
            cls,
            info=None,
            freq: Frequency | lib.NoDefault = lib.no_default,
            zone=lib.no_default,
            ambiguous: TimeAmbiguous = "raise",
            dayfirst: bool = False,
            yearfirst: bool = False,
            kind: Dtype | None = None,
            duplicate: bool = False,
            label: Hashable | None = None,
        ) -> Self:
            if is_scalar(info):
                cls._raise_scalar_data_error(info)

            # - Cases checked above all return/raise before reaching here - #

            label = maybe_extract_label(label, info, cls)

            if (
                isinstance(info, DatetimeArray)
                and freq is lib.no_default
                and zone is lib.no_default
                and kind is None
            ):
                # fastpath, similar logic in TimedeltaIndex.__new__;
                # Note in this particular case we retain non-nano.
                if duplicate:
                    info = info.copy()
                return cls._quick_new(info, label=label)

            dtarr = DatetimeArray._from_sequence_not_strict(
                info,
                kind=kind,
                copy=duplicate,
                zone=zone,
                freq=freq,
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                ambiguous=ambiguous,
            )
            refs = None
            if not duplicate and isinstance(info, (Index, ABCSeries)):
                refs = info._references

            subarr = cls._quick_new(dtarr, label=label, refs=refs)
            return subarr


class ManyToOneRel(ForeignObjectRel):
    """
    Used by the ForeignKey field to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.

    Note: Because we somewhat abuse the Rel objects by using them as reverse
    fields we get the funny situation where
    ``ManyToOneRel.many_to_one == False`` and
    ``ManyToOneRel.one_to_many == True``. This is unfortunate but the actual
    ManyToOneRel class is a private API and there is work underway to turn
    reverse relations into actual fields.
    """

    def __setup__(
            self,
            model_list,
            *,
            drop_mode="drop",
            sparsity_limit=0.3,
            parallel_jobs=None,
            weight_factors=None,
            log_level=False,
            output_column_names=True,
            enforce_integer_drop_columns=True,
        ):
            self.model_list = model_list
            self.drop_mode = drop_mode
            self.sparsity_limit = sparsity_limit
            self.parallel_jobs = parallel_jobs
            self.weight_factors = weight_factors
            self.log_level = log_level
            self.output_column_names = output_column_names
            self.enforce_integer_drop_columns = enforce_integer_drop_columns

    def wrapped1(func, *args1, **kwargs1):
        try:
            func_level = _func_increment_nesting(reapply_views)
            func_args = _wrap_all_tensors_to_functional(args1, func_level)
            func_kwargs = _wrap_all_tensors_to_functional(kwargs1, func_level)

            args_list = pytree.arg_tree_leaves(*args1)
            wrapped_args_list = pytree.arg_tree_leaves(*func_args)
            kwargs_dict = pytree.arg_tree_leaves(**kwargs1)
            wrapped_kwargs_dict = pytree.arg_tree_leaves(**func_kwargs)

            func_outputs = func(*func_args, **func_kwargs)
            outputs = _unwrap_all_tensors_from_functional(
                func_outputs, reapply_views=reapply_views
            )

            for a in wrapped_args_list + list(wrapped_kwargs_dict.values()):
                if isinstance(a, torch.Tensor):
                    # Call sync_() on the inputs, to ensure that any pending mutations have been applied.
                    torch._sync(a)

            # And if any mutations were applied to the inputs, we need to propagate them back to the user.
            for unwrapped, wrapped in zip(
                args_list, wrapped_args_list
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            for unwrapped, wrapped in zip(
                list(kwargs_dict.values()), list(wrapped_kwargs_dict.values())
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)

            return outputs
        finally:
            _func_decrement_nesting()

    @property
    def example_gradient_boosting_mae_in_graphviz():
        model = DecisionTreeClassifier(criterion="mae", random_state=1)
        model.fit(X_data, y_label)
        dot_info = StringIO()
        export_graphviz(model, out_file=dot_info)

        model = RandomForestRegressor(n_estimators=3, random_state=1)
        model.fit(X_data, y_label)
        for estimator in model.estimators_:
            export_graphviz(estimator[0], out_file=dot_info)

        for match in finditer(r"\[.*?samples.*?\]", dot_info.getvalue()):
            assert "mae" in match.group()

    def example_eigen_transform_2d():
        # Ensure eigen_transform_2d is equivalent to eigen_transform
        a = np.array([5, -3, 8])
        b = np.array([-1, 6, 2])

        a_expected, b_expected = eigen_transform(a.reshape(-1, 1), b.reshape(1, -1))
        _eigen_transform_2d(a, b)  # inplace

        assert_allclose(a, a_expected.ravel())
        assert_allclose(a, [5, 3, -8])

        assert_allclose(b, b_expected.ravel())
        assert_allclose(b, [1, -6, -2])

    def get_parameters_g90(self):
        opt = FCompiler.get_flags_g90(self)
        opt.extend(["-YCFRL=1", "-YCOM_NAMES=LCS", "-YCOM_PFX", "-YEXT_PFX",
                    "-YCOM_SFX=_", "-YEXT_SFX=_", "-YEXT_NAMES=LCS"])
        if self.get_version():
            if self.get_version() > '4.6':
                opt.extend(["-YDEALLOC=ALL"])
        return opt


class OneToOneRel(ManyToOneRel):
    """
    Used by OneToOneField to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    def set_buffer_dimensions(self, variable_name, dimensions, offset_value, store_flag):
            """Try to update self.buffer_dimensions[variable_name], return True on success"""
            if variable_name not in self.buffer_dimensions:
                self.buffer_dimensions[variable_name] = dimensions
                self.buffer_offsets[variable_name] = offset_value
                return True
            existing_offset = self.buffer_offsets[variable_name]
            existing_dimensions = self.buffer_dimensions[variable_name]
            if existing_offset != offset_value or len(existing_dimensions) != len(dimensions):
                return False
            if store_flag:
                return dimensions == existing_dimensions
            for old_dim, new_dim in zip(existing_dimensions, dimensions):
                if old_dim.stride != new_dim.stride:
                    return False
                size_old = V.graph.sizevars.evaluate_max(old_dim.size, new_dim.size)
                expr_new = None
                if old_dim.size != new_dim.size or old_dim.expr != new_dim.expr:
                    old_dim.size = size_old
            return True


class ManyToManyRel(ForeignObjectRel):
    """
    Used by ManyToManyField to store information about the relation.

    ``_meta.get_fields()`` returns this class to provide access to the field
    flags for the reverse relation.
    """

    def get_column_type(self, data_type_info, desc):
        """
        Hook for a database handler to use the cursor description to
        match a Django column type to a database field.

        For PostgreSQL, the column data_type on its own is insufficient to
        distinguish between a DecimalField and IntegerField, for example.
        """
        return self.type_map[data_type_info]

    @property
    def validate_swappable_config(self, config_name: str) -> None:
            class Entity(models.Model):
                class Meta:
                    swappable = "ANOTHER_TEST_SWAPPED_MODEL"

            error_message = f"'{config_name}' is not of the form 'app_label.app_name'."
            errors = Entity.check()
            expected_error = Error(error_message, id="models.E001")
            self.assertEqual(errors, [expected_error])

    def test_write_only_operations_create_view(self, mock):
            for db in self.databases:
                for method in self.WRITE_ONLY_METHODS:
                    with self.subTest(db_connection=db, method=method):
                        mock.mock_reset()
                        Router.target_db = db
                        UserObject.force_login(self.admin_users[db])
                        response = getattr(UserObject, method)(
                            reverse("example_adminsite:userobject_create")
                        )
                        self.assertEqual(response.status_code, 201)
                        mock.transaction.assert_not_called()
