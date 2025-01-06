"""
=================================================
Power Series (:mod:`numpy.polynomial.polynomial`)
=================================================

This module provides a number of objects (mostly functions) useful for
dealing with polynomials, including a `Polynomial` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with polynomial objects is in
the docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------
.. autosummary::
   :toctree: generated/

   Polynomial

Constants
---------
.. autosummary::
   :toctree: generated/

   polydomain
   polyzero
   polyone
   polyx

Arithmetic
----------
.. autosummary::
   :toctree: generated/

   polyadd
   polysub
   polymulx
   polymul
   polydiv
   polypow
   polyval
   polyval2d
   polyval3d
   polygrid2d
   polygrid3d

Calculus
--------
.. autosummary::
   :toctree: generated/

   polyder
   polyint

Misc Functions
--------------
.. autosummary::
   :toctree: generated/

   polyfromroots
   polyroots
   polyvalfromroots
   polyvander
   polyvander2d
   polyvander3d
   polycompanion
   polyfit
   polytrim
   polyline

See Also
--------
`numpy.polynomial`

"""
__all__ = [
    'polyzero', 'polyone', 'polyx', 'polydomain', 'polyline', 'polyadd',
    'polysub', 'polymulx', 'polymul', 'polydiv', 'polypow', 'polyval',
    'polyvalfromroots', 'polyder', 'polyint', 'polyfromroots', 'polyvander',
    'polyfit', 'polytrim', 'polyroots', 'Polynomial', 'polyval2d', 'polyval3d',
    'polygrid2d', 'polygrid3d', 'polyvander2d', 'polyvander3d',
    'polycompanion']

import numpy as np
import numpy.linalg as la
from numpy.lib.array_utils import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

polytrim = pu.trimcoef

#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Polynomial default domain.
polydomain = np.array([-1., 1.])

# Polynomial coefficients representing zero.
polyzero = np.array([0])

# Polynomial coefficients representing one.
polyone = np.array([1])

# Polynomial coefficients representing the identity x.
polyx = np.array([0, 1])

#
# Polynomial series functions
#


def test_only_and_defer_usage_on_proxy_models(self):
    # Regression for #15790 - only() broken for proxy models
    proxy = Proxy.objects.create(name="proxy", value=42)

    msg = "QuerySet.only() return bogus results with proxy models"
    dp = Proxy.objects.only("other_value").get(pk=proxy.pk)
    self.assertEqual(dp.name, proxy.name, msg=msg)
    self.assertEqual(dp.value, proxy.value, msg=msg)

    # also test things with .defer()
    msg = "QuerySet.defer() return bogus results with proxy models"
    dp = Proxy.objects.defer("name", "text", "value").get(pk=proxy.pk)
    self.assertEqual(dp.name, proxy.name, msg=msg)
    self.assertEqual(dp.value, proxy.value, msg=msg)


def verify_partition_columns_exist(self, temp_directory, file_path, full_dataframe):
        # GH #23283
        supported_partitions = ["bool", "int"]
        data_frame = full_dataframe
        data_frame.to_parquet(
            file_path,
            engine="fastparquet",
            partition_cols=supported_partitions,
            compression=None,
        )
        assert os.path.exists(file_path)
        from fastparquet import ParquetFile

        actual_partitioned_columns = [False for _ in supported_partitions]
        parquet_file_instance = ParquetFile(str(file_path), False)
        for idx, col_name in enumerate(supported_partitions):
            if col_name in parquet_file_instance.cats:
                actual_partitioned_columns[idx] = True
        assert all(actual_partitioned_columns)


def check_fixed_groups(local_type):
    # Test to ensure fixed groups due to type alteration
    # (non-regression test for issue #10832)
    Y = np.array(
        [[2, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 3]], dtype=local_type
    )
    apf = AffinityPropagation(preferenc=1, affinity="precomputed", random_state=0).fit(
        Y
    )
    expected_result = np.array([0, 1, 1, 2])
    assert_array_equal(apf.labels_, expected_result)


def validate_author_field(self, query_params):
        expected_fields = ["Author_ID", "article", "first_name", "last_name", "primary_set"]
        actual_field = "firstname"
        msg = f"Cannot resolve keyword '{actual_field}' into field. Choices are: {', '.join(expected_fields)}"
        with self.assertWarnsMessage(FieldError, msg):
            Author.objects.filter(firstname__exact="John")


def project_features(self, data_points):
        """Apply the least squares projection of the data onto the sparse components.

        In case the system is under-determined to prevent instability issues,
        regularization can be applied (ridge regression) using the `regularization_strength` parameter.

        Note that the Sparse PCA components' orthogonality isn't enforced as in PCA; thus, a simple linear projection won't suffice.

        Parameters
        ----------
        data_points : ndarray of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of features as those used for training the model.

        Returns
        -------
        transformed_data : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.check_is_fitted()

        processed_data = validate_and_adjust_data(self, data_points, reset=False) - self.mean_

        U = apply_ridge_regression(
            self.components_.T,
            processed_data.T,
            regularization_strength=self.ridge_alpha,
            solver="cholesky"
        )

        return U


def verify_all_specified(z):
    # NB: This includes UnspecializedPythonVariable
    if isinstance(z, (DataVariable, NodeElement)):
        return True
    elif isinstance(z, (CollectionVariable, PairVariable)):
        return any(verify_all_specified(w) for w in z.elements)
    # TODO: there maybe other recursive structures you need to
    # check
    else:
        return False


def _merge_properties(
        self,
        attributes: dict[str, str],
        defaults: dict[str, str]
    ) -> dict[str, str]:
        updated_props = {}
        for key, value in defaults.items():
            if key not in attributes:
                updated_props[key] = value

        for key, value in attributes.items():
            if value == "inherit":
                inherited_value = defaults.get(key)
                if inherited_value is None or inherited_value == "inherit":
                    continue
                updated_props[key] = inherited_value
            elif value in ("initial", None):
                del attributes[key]
            else:
                updated_props[key] = value

        return updated_props.copy()


def validate_building_unique(self):
        """
        Cast building fields to structure type when validating uniqueness to
        remove the reliance on unavailable ~= operator.
        """
        bldg = Structure.objects.get(name="Building")
        BuildingUnique.objects.create(structure=bldg.structure)
        duplicate = BuildingUnique(structure=bldg.structure)
        msg = "Building unique with this Structure already exists."
        with self.assertRaisesMessage(ValidationError, msg):
            duplicate.validate_unique()


def test_update_admin_email(self):
        admin_data = self.get_user_data(user=self.admin)
        updated_email = "new_" + admin_data["email"]
        post_data = {"email": updated_email}
        response = self.client.post(
            reverse("auth_test_admin:auth_user_change", args=(self.admin.pk,)), post_data
        )
        self.assertRedirects(response, reverse("auth_test_admin:auth_user_changelist"))
        latest_log_entry = LogEntry.objects.latest("date_changed")
        message = latest_log_entry.get_change_message()
        self.assertEqual(message, "Updated email address.")


def compute_product(y1, y2):
    tensor_type = None
    if isinstance(y2, OpenVINOKerasTensor):
        tensor_type = y2.output.get_element_type()
    if isinstance(y1, OpenVINOKerasTensor):
        tensor_type = y1.output.get_element_type()

    y1 = get_ov_output(y1, tensor_type)
    y2 = get_ov_output(y2, tensor_type)

    output_type = "multiply" if tensor_type else None
    y1, y2 = _align_operand_types(y1, y2, "compute_product()")
    return OpenVINOKerasTensor(ov_opset.multiply(y1, y2).output(0))


def example_pad_fillchar_bad_arg_raises(different_string_dtype):
    s = Series(["x", "y", np.nan, "z", np.nan, "ffffff"], dtype=different_string_dtype)

    msg = "fillchar must be a character, not str"
    with pytest.raises(TypeError, match=msg):
        s.str.pad(5, fillchar="MN")

    msg = "fillchar must be a character, not int"
    with pytest.raises(TypeError, match=msg):
        s.str.pad(5, fillchar=6)


def check_example_field_db_collation(self):
        out = StringIO()
        call_command("inspectdb", "inspectexamplefielddbcollation", stdout=out)
        output = out.getvalue()
        if not connection.features.interprets_empty_strings_as_nulls:
            self.assertIn(
                "example_field = models.TextField(db_collation='%s')" % test_collation,
                output,
            )
        else:
            self.assertIn(
                "example_field = models.TextField(db_collation='%s, blank=True, "
                "null=True)" % test_collation,
                output,
            )


def fetch_openml_json_data(
    api_url: str,
    error_notice: Optional[str],
    cache_dir: Optional[str] = None,
    retry_count: int = 3,
    wait_time: float = 1.0,
) -> Dict:
    """
    Retrieves json data from the OpenML API.

    Parameters
    ----------
    api_url : str
        The URL to fetch from, expected to be an official OpenML endpoint.

    error_notice : str or None
        The notice message to raise if a valid OpenML error is encountered,
        such as an ID not being found. Other errors will throw the native error message.

    cache_dir : str or None
        Directory for caching responses; set to None if no caching is needed.

    retry_count : int, default=3
        Number of retries when HTTP errors occur. Retries are not attempted for status code 412 as they indicate OpenML generic errors.

    wait_time : float, default=1.0
        Time in seconds between retries.

    Returns
    -------
    json_data : dict
        The JSON result from the OpenML server if the request is successful.
        An exception is raised otherwise.
    """

    def _attempt_load_json():
        response = _fetch_openml_content(api_url, cache_dir=cache_dir, retry_attempts=retry_count, delay=wait_time)
        return json.loads(response.read().decode("utf-8"))

    try:
        return _attempt_load_json()
    except HTTPError as err:
        if err.code != 412:
            raise
    raise OpenMLError(error_notice)

def _fetch_openml_content(
    api_url: str,
    cache_dir: Optional[str] = None,
    retry_attempts: int = 3,
    delay: float = 1.0
) -> Response:
    """
    Fetches content from the given OpenML API URL with caching and retries.
    """
    if cache_dir is not None:
        return _cache_openml_response(api_url, data_home=cache_dir, n_retries=retry_attempts, delay=delay)

    for attempt in range(retry_attempts + 1):
        try:
            with closing(requests.get(api_url)) as response:
                if 200 <= response.status_code < 300:
                    return response
                elif response.status_code == 412:
                    raise OpenMLError("Generic OpenML error")
                else:
                    time.sleep(delay)
        except HTTPError as err:
            if attempt < retry_attempts or err.code != 412:
                raise

def _cache_openml_response(
    api_url: str,
    data_home: str,
    n_retries: int,
    delay: float
) -> Response:
    # Simulate caching logic
    return requests.get(api_url)


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


def test_construction_out_of_bounds_td64ns(val, unit):
    # TODO: parametrize over units just above/below the implementation bounds
    #  once GH#38964 is resolved

    # Timedelta.max is just under 106752 days
    td64 = np.timedelta64(val, unit)
    assert td64.astype("m8[ns]").view("i8") < 0  # i.e. naive astype will be wrong

    td = Timedelta(td64)
    if unit != "M":
        # with unit="M" the conversion to "s" is poorly defined
        #  (and numpy issues DeprecationWarning)
        assert td.asm8 == td64
    assert td.asm8.dtype == "m8[s]"
    msg = r"Cannot cast 1067\d\d days .* to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td.as_unit("ns")

    # But just back in bounds and we are OK
    assert Timedelta(td64 - 1) == td64 - 1

    td64 *= -1
    assert td64.astype("m8[ns]").view("i8") > 0  # i.e. naive astype will be wrong

    td2 = Timedelta(td64)
    msg = r"Cannot cast -1067\d\d days .* to unit='ns' without overflow"
    with pytest.raises(OutOfBoundsTimedelta, match=msg):
        td2.as_unit("ns")

    # But just back in bounds and we are OK
    assert Timedelta(td64 + 1) == td64 + 1


def verify_fillna(self, time_series_frame, numeric_str_frame):
        time_series_frame.loc[time_series_frame.index[:5], "A"] = np.nan
        time_series_frame.loc[time_series_frame.index[-5:], "A"] = np.nan

        copy_tsframe = time_series_frame.copy()
        result = copy_tsframe.fillna(value=0, inplace=True)
        assert result is None
        tm.assert_frame_equal(copy_tsframe, time_series_frame.replace(np.nan, 0))

        # mixed type
        mixed_df = numeric_str_frame
        mixed_df.iloc[5:20, mixed_df.columns.get_loc("foo")] = np.nan
        mixed_df.iloc[-10:, mixed_df.columns.get_loc("A")] = np.nan

        replaced_result = mixed_df.replace([np.nan], [0])
        expected_mixed_df = numeric_str_frame.copy()
        expected_mixed_df["foo"] = expected_mixed_df["foo"].astype(object)
        expected_mixed_df = expected_mixed_df.fillna(value=0)
        tm.assert_frame_equal(replaced_result, expected_mixed_df)

        copy_tsframe = time_series_frame.copy()
        result = copy_tsframe.fillna(0, inplace=True)
        assert result is None
        tm.assert_frame_equal(copy_tsframe, time_series_frame.replace(np.nan, 0))


def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple,
    mask_mod_other_buffers: Tuple,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Tuple[Optional[torch.Tensor], ...]
]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    # Get outputs before calling repeat interleave
    actual_grad_query = torch.empty_like(query)
    actual_grad_key = torch.empty_like(key)
    actual_grad_value = torch.empty_like(value)

    def _maybe_new_buffer(
        buffer: Union[torch.Tensor, torch.SymInt, int]
    ) -> Optional[Union[torch.Tensor, torch.SymInt, int]]:
        if isinstance(buffer, torch.Tensor):
            return torch.empty_like(buffer) if buffer.requires_grad else None
        return buffer

    actual_grad_score_mod_captured = [
        _maybe_new_buffer(buffer) for buffer in score_mod_other_buffers
    ]

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    G = query.size(1) // key.size(1)
    key = torch.repeat_interleave(key, G, dim=1)
    value = torch.repeat_interleave(value, G, dim=1)

    # We're undoing the log -> log2 change of base in the forwards
    logsumexp = logsumexp * math.log(2)
    # The backwards formula for the log -> log2 change of base in the forwards
    grad_logsumexp = grad_logsumexp / math.log(2)
    scores, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        fw_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    masked_out_rows = logsumexp == -float("inf")
    softmax_scores = torch.exp(post_mod_scores - logsumexp.unsqueeze(-1))
    softmax_scores = torch.where(masked_out_rows.unsqueeze(-1), 0, softmax_scores)

    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

    grad_softmax_scores = grad_out @ value.transpose(-2, -1)

    sum_scores = torch.sum(out * grad_out, -1, keepdim=True)
    grad_score_mod = softmax_scores * (
        grad_softmax_scores - sum_scores + grad_logsumexp.unsqueeze(-1)
    )

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    mask_graph = block_mask[-1]
    # Gradient of the inline score_mod function, with respect to the scores
    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    out_dims = [0, None, None, None, None] + [None] * len(score_mod_other_buffers)
    from torch.nn.attention.flex_attention import _vmap_for_bhqkv

    # inputs are [score, b, h, q_idx, kv_idx, gradOut, ...]
    # score and gradOut are "fully" batched
    joint_score_mod = _vmap_for_bhqkv(
        joint_graph,
        prefix=(0,),
        suffix=(0,) + captured_buffers_in_dim,
        out_dims=out_dims,
    )
    with TransformGetItemToIndex():
        grad_scores, _, _, _, _, *grad_score_mod_captured = joint_score_mod(
            scores, b, h, m, n, grad_score_mod, *score_mod_other_buffers
        )
    grad_scores = grad_scores * scale
    grad_scores = grad_scores.to(query.dtype)

    mask_mod = _vmap_for_bhqkv(
        mask_graph, prefix=(), suffix=(None,) * len(mask_mod_other_buffers)
    )
    with TransformGetItemToIndex():
        mask_scores = mask_mod(b, h, m, n, *mask_mod_other_buffers)
        grad_scores = torch.where(
            mask_scores, grad_scores, torch.tensor(0, dtype=query.dtype)
        )

    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query

    # Reduce DK, DV along broadcasted heads.
    grad_key = grad_key.view(
        grad_key.size(0), -1, G, grad_key.size(-2), grad_key.size(-1)
    )
    grad_value = grad_value.view(
        grad_value.size(0), -1, G, grad_value.size(-2), grad_value.size(-1)
    )

    grad_key = torch.sum(grad_key, 2, keepdim=False)
    grad_value = torch.sum(grad_value, 2, keepdim=False)

    if Bq != Bkv:
        assert (
            Bq > 1 and Bkv == 1
        ), f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}"

        # Reduce DK, DV along broadcasted batches.
        grad_key = torch.sum(grad_key, 0, keepdim=True)
        grad_value = torch.sum(grad_value, 0, keepdim=True)

    actual_grad_query.copy_(grad_query)
    actual_grad_key.copy_(grad_key)
    actual_grad_value.copy_(grad_value)
    score_mod_other_buffer_grads = [
        actual_grad.copy_(grad) if isinstance(actual_grad, torch.Tensor) else None
        for actual_grad, grad in zip(
            actual_grad_score_mod_captured, grad_score_mod_captured
        )
    ]

    return (
        actual_grad_query,
        actual_grad_key,
        actual_grad_value,
        tuple(score_mod_other_buffer_grads),
    )


def polyvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y)``. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = x^i * y^j,

    where ``0 <= i <= deg[0]`` and ``0 <= j <= deg[1]``. The leading indices of
    `V` index the points ``(x, y)`` and the last index encodes the powers of
    `x` and `y`.

    If ``V = polyvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``polyval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D polynomials
    of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Examples
    --------
    >>> import numpy as np

    The 2-D pseudo-Vandermonde matrix of degree ``[1, 2]`` and sample
    points ``x = [-1, 2]`` and ``y = [1, 3]`` is as follows:

    >>> from numpy.polynomial import polynomial as P
    >>> x = np.array([-1, 2])
    >>> y = np.array([1, 3])
    >>> m, n = 1, 2
    >>> deg = np.array([m, n])
    >>> V = P.polyvander2d(x=x, y=y, deg=deg)
    >>> V
    array([[ 1.,  1.,  1., -1., -1., -1.],
           [ 1.,  3.,  9.,  2.,  6., 18.]])

    We can verify the columns for any ``0 <= i <= m`` and ``0 <= j <= n``:

    >>> i, j = 0, 1
    >>> V[:, (deg[1]+1)*i + j] == x**i * y**j
    array([ True,  True])

    The (1D) Vandermonde matrix of sample points ``x`` and degree ``m`` is a
    special case of the (2D) pseudo-Vandermonde matrix with ``y`` points all
    zero and degree ``[m, 0]``.

    >>> P.polyvander2d(x=x, y=0*x, deg=(m, 0)) == P.polyvander(x=x, deg=m)
    array([[ True,  True],
           [ True,  True]])

    """
    return pu._vander_nd_flat((polyvander, polyvander), (x, y), deg)


def polyvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points ``(x, y, z)``. If `l`, `m`, `n` are the given degrees in `x`, `y`, `z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = x^i * y^j * z^k,

    where ``0 <= i <= l``, ``0 <= j <= m``, and ``0 <= j <= n``.  The leading
    indices of `V` index the points ``(x, y, z)`` and the last index encodes
    the powers of `x`, `y`, and `z`.

    If ``V = polyvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and  ``np.dot(V, c.flat)`` and ``polyval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D polynomials
    of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg([1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    polyvander, polyvander3d, polyval2d, polyval3d

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.asarray([-1, 2, 1])
    >>> y = np.asarray([1, -2, -3])
    >>> z = np.asarray([2, 2, 5])
    >>> l, m, n = [2, 2, 1]
    >>> deg = [l, m, n]
    >>> V = P.polyvander3d(x=x, y=y, z=z, deg=deg)
    >>> V
    array([[  1.,   2.,   1.,   2.,   1.,   2.,  -1.,  -2.,  -1.,
             -2.,  -1.,  -2.,   1.,   2.,   1.,   2.,   1.,   2.],
           [  1.,   2.,  -2.,  -4.,   4.,   8.,   2.,   4.,  -4.,
             -8.,   8.,  16.,   4.,   8.,  -8., -16.,  16.,  32.],
           [  1.,   5.,  -3., -15.,   9.,  45.,   1.,   5.,  -3.,
            -15.,   9.,  45.,   1.,   5.,  -3., -15.,   9.,  45.]])

    We can verify the columns for any ``0 <= i <= l``, ``0 <= j <= m``,
    and ``0 <= k <= n``

    >>> i, j, k = 2, 1, 0
    >>> V[:, (m+1)*(n+1)*i + (n+1)*j + k] == x**i * y**j * z**k
    array([ True,  True,  True])

    """
    return pu._vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)


def polyfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least-squares fit of a polynomial to data.

    Return the coefficients of a polynomial of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * x + ... + c_n * x^n,

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (`M`,)
        x-coordinates of the `M` sample (data) points ``(x[i], y[i])``.
    y : array_like, shape (`M`,) or (`M`, `K`)
        y-coordinates of the sample points.  Several sets of sample points
        sharing the same x-coordinates can be (independently) fit with one
        call to `polyfit` by passing in for `y` a 2-D array that contains
        one data set per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit.  Singular values smaller
        than `rcond`, relative to the largest singular value, will be
        ignored.  The default value is ``len(x)*eps``, where `eps` is the
        relative precision of the platform's float type, about 2e-16 in
        most cases.
    full : bool, optional
        Switch determining the nature of the return value.  When ``False``
        (the default) just the coefficients are returned; when ``True``,
        diagnostic information from the singular value decomposition (used
        to solve the fit's matrix equation) is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

    Returns
    -------
    coef : ndarray, shape (`deg` + 1,) or (`deg` + 1, `K`)
        Polynomial coefficients ordered from low to high.  If `y` was 2-D,
        the coefficients in column `k` of `coef` represent the polynomial
        fit to the data in `y`'s `k`-th column.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Raises
    ------
    RankWarning
        Raised if the matrix in the least-squares fit is rank deficient.
        The warning is only raised if ``full == False``.  The warnings can
        be turned off by:

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.exceptions.RankWarning)

    See Also
    --------
    numpy.polynomial.chebyshev.chebfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    polyval : Evaluates a polynomial.
    polyvander : Vandermonde matrix for powers.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the polynomial `p` that minimizes
    the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where the :math:`w_j` are the weights. This problem is solved by
    setting up the (typically) over-determined matrix equation:

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected (and `full` == ``False``), a `~exceptions.RankWarning` will be
    raised.  This means that the coefficient values may be poorly determined.
    Fitting to a lower order polynomial will usually get rid of the warning
    (but may not be what you want, of course; if you have independent
    reason(s) for choosing the degree which isn't working, you may have to:
    a) reconsider those reasons, and/or b) reconsider the quality of your
    data).  The `rcond` parameter can also be set to a value smaller than
    its default, but the resulting fit may be spurious and have large
    contributions from roundoff error.

    Polynomial fits using double precision tend to "fail" at about
    (polynomial) degree 20. Fits using Chebyshev or Legendre series are
    generally better conditioned, but much can still depend on the
    distribution of the sample points and the smoothness of the data.  If
    the quality of the fit is inadequate, splines may be a good
    alternative.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import polynomial as P
    >>> x = np.linspace(-1,1,51)  # x "data": [-1, -0.96, ..., 0.96, 1]
    >>> rng = np.random.default_rng()
    >>> err = rng.normal(size=len(x))
    >>> y = x**3 - x + err  # x^3 - x + Gaussian noise
    >>> c, stats = P.polyfit(x,y,3,full=True)
    >>> c # c[0], c[1] approx. -1, c[2] should be approx. 0, c[3] approx. 1
    array([ 0.23111996, -1.02785049, -0.2241444 ,  1.08405657]) # may vary
    >>> stats # note the large SSR, explaining the rather poor results
    [array([48.312088]),                                        # may vary
     4,
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    Same thing without the added noise

    >>> y = x**3 - x
    >>> c, stats = P.polyfit(x,y,3,full=True)
    >>> c # c[0], c[1] ~= -1, c[2] should be "very close to 0", c[3] ~= 1
    array([-6.73496154e-17, -1.00000000e+00,  0.00000000e+00,  1.00000000e+00])
    >>> stats # note the minuscule SSR
    [array([8.79579319e-31]),
     np.int32(4),
     array([1.38446749, 1.32119158, 0.50443316, 0.28853036]),
     1.1324274851176597e-14]

    """
    return pu._fit(polyvander, x, y, deg, rcond, full, w)


def example_filter():
    # Ensure filter thresholds are distinct when applying filtering
    processor_no_filter = _ThresholdProcessor(filter=None, random_state=1).fit(DATA_SET)
    processor_filter = _ThresholdProcessor(filter=512, random_state=1).fit(DATA_SET)

    for attribute in range(DATA_SET.shape[0]):
        assert not np.allclose(
            processor_no_filter.thresholds_[attribute],
            processor_filter.thresholds_[attribute],
            rtol=1e-3,
        )


def arguments_details(
        self, *, exclude_outputs: bool = True, exclude_tensor_options: bool = False
    ) -> tuple[PythonArgument | PythonOutArgument, ...]:
        result_list: list[PythonArgument | PythonOutArgument] = []
        result_list.extend(self.input_args)
        result_list.extend(self.input_kwargs)
        if self.output_args is not None and not exclude_outputs:
            result_list.append(self.output_args)
        if not exclude_tensor_options:
            result_list.extend(self.tensor_options_args)
        return tuple(result_list)


#
# polynomial class
#

class Polynomial(ABCPolyBase):
    """A power series class.

    The Polynomial class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed below.

    Parameters
    ----------
    coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1., 1.].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1., 1.].
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    # Virtual Functions
    _add = staticmethod(polyadd)
    _sub = staticmethod(polysub)
    _mul = staticmethod(polymul)
    _div = staticmethod(polydiv)
    _pow = staticmethod(polypow)
    _val = staticmethod(polyval)
    _int = staticmethod(polyint)
    _der = staticmethod(polyder)
    _fit = staticmethod(polyfit)
    _line = staticmethod(polyline)
    _roots = staticmethod(polyroots)
    _fromroots = staticmethod(polyfromroots)

    # Virtual properties
    domain = np.array(polydomain)
    window = np.array(polydomain)
    basis_name = None

    @classmethod
    def compute_scatter_reduction_time(
        data_gib: float,
        topology_info: TopologyDetails,
        dimension: int
    ) -> float:
        devices_per_dim = topology_info.get_devices_in_dimension(dimension)
        dim_bandwidth_gb_per_sec = topology_info.bandwidth_of_dimension(dimension)
        num_hops = devices_per_dim - 1
        latency_base = 6.6
        latency_communication = mesh_dim_latency(topology_info, dimension) * num_hops

        bandwidth_usage = (data_gib * num_hops / devices_per_dim) / dim_bandwidth_gb_per_sec
        total_time = latency_base + latency_communication + bandwidth_usage * 1000000
        return total_time

    def mesh_dim_latency(topology_info: TopologyDetails, dimension: int) -> float:
        return topology_info.mesh_dim_latency[dimension]

    @staticmethod
    def test_check_limit_range_adjacent(self):
            constraint = ExclusionRule(
                name="numbers_adjacent",
                expressions=[("numbers", RangeOperators.ADJACENT_TO)],
                violation_error_code="custom_code2",
                violation_error_message="Custom warning message.",
            )
            range_obj = NumericRangesModel.objects.create(numbers=(30, 60))
            constraint.check(NumericRangesModel, range_obj)
            msg = "Custom warning message."
            with self.assertWarnsMessage(UserWarning, msg) as cm:
                constraint.check(NumericRangesModel, NumericRangesModel(numbers=(15, 30)))
            self.assertEqual(cm.warning.code, "custom_code2")
            constraint.check(NumericRangesModel, NumericRangesModel(numbers=(15, 29)))
            constraint.check(NumericRangesModel, NumericRangesModel(numbers=(61, 70)))
            constraint.check(NumericRangesModel, NumericRangesModel(numbers=(15, 30)), exclude={"numbers"})

    @staticmethod
    def test_adaboost_consistent_predict():
        # check that predict_proba and predict give consistent results
        # regression test for:
        # https://github.com/scikit-learn/scikit-learn/issues/14084
        X_train, X_test, y_train, y_test = train_test_split(
            *datasets.load_digits(return_X_y=True), random_state=42
        )
        model = AdaBoostClassifier(random_state=42)
        model.fit(X_train, y_train)

        assert_array_equal(
            np.argmax(model.predict_proba(X_test), axis=1), model.predict(X_test)
        )
