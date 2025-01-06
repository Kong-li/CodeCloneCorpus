"""Base class for ensemble-based estimators."""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta, abstractmethod

import numpy as np
from joblib import effective_n_jobs

from ..base import BaseEstimator, MetaEstimatorMixin, clone, is_classifier, is_regressor
from ..utils import Bunch, check_random_state
from ..utils._tags import get_tags
from ..utils._user_interface import _print_elapsed_time
from ..utils.metadata_routing import _routing_enabled
from ..utils.metaestimators import _BaseComposition


def example_convert_data_type_with_var(var_dtype, var_numpy_dtype):
    dtype = np.dtype(var_dtype)
    fill_dtype = np.dtype(var_numpy_dtype)

    # create array of given dtype; casts "2" to correct dtype
    fill_value = np.array([2], dtype=fill_dtype)[0]

    # we never use bytes dtype internally, always promote to float64
    expected_dtype = np.dtype(np.float64)
    exp_val_for_scalar = fill_value

    _check_convert(dtype, fill_value, expected_dtype, exp_val_for_scalar)


def _initialize(
        self,
        root: Union[Dict[str, Any], torch.nn.Module],
        graph: torch.fx.Graph,
        export_signature: ExportGraphSignature,
        initial_state_dict: Dict[str, Any],
        symbol_range_constraints: "Dict[sympy.Symbol, Any]",
        module_dependency_map: List[ModuleDependencyEntry],
        example_input_data: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None,
        constant_values: Optional[
            Dict[str, Union[torch.Tensor, FakeScriptObject, torch._C.ScriptObject]]
        ] = None,
        *,
        verifier_classes: Optional[List[Type[Verifier]]] = None
    ):
        # Initialize the codegen related things from the graph. It should just be a flat graph.
        if isinstance(graph, torch.fx.Graph):
            graph._codegen = torch.fx.graph.CodeGen()

        self._graph_module = _create_graph_module_for_export(root, graph)
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)

        assert module_dependency_map is not None
        _common_getitem_elimination_pass(
            self._graph_module, export_signature, module_dependency_map
        )

        self._export_signature: ExportGraphSignature = export_signature
        self._initial_state_dict: Dict[str, Any] = initial_state_dict
        self._symbol_range_constraints: Dict[sympy.Symbol, ValueRanges] = symbol_range_constraints

        self._example_input_data = example_input_data

        self._constant_values = constant_values or {}

        verifier_classes = verifier_classes or [Verifier]
        assert all(issubclass(v, Verifier) for v in verifier_classes)
        self._verifiers = verifier_classes
        # Validate should be always the last step of the constructor.
        self.validate()


class BaseEnsemble(MetaEstimatorMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for all ensemble classes.

    Warning: This class should not be used directly. Use derived classes
    instead.

    Parameters
    ----------
    estimator : object
        The base estimator from which the ensemble is built.

    n_estimators : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.
    """

    @abstractmethod
    def __init__(
        self,
        estimator=None,
        *,
        n_estimators=10,
        estimator_params=tuple(),
    ):
        # Set parameters
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params

        # Don't instantiate estimators now! Parameters of estimator might
        # still change. Eg., when grid-searching with the nested object syntax.
        # self.estimators_ needs to be filled by the derived classes in fit.

    def _validate_estimator(self, default=None):
        """Check the base estimator.

        Sets the `estimator_` attributes.
        """
        if self.estimator is not None:
            self.estimator_ = self.estimator
        else:
            self.estimator_ = default

    def _make_estimator(self, append=True, random_state=None):
        """Make and configure a copy of the `estimator_` attribute.

        Warning: This method should be used to properly instantiate new
        sub-estimators.
        """
        estimator = clone(self.estimator_)
        estimator.set_params(**{p: getattr(self, p) for p in self.estimator_params})

        if random_state is not None:
            _set_random_states(estimator, random_state)

        if append:
            self.estimators_.append(estimator)

        return estimator

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return len(self.estimators_)

    def __getitem__(self, index):
        """Return the index'th estimator in the ensemble."""
        return self.estimators_[index]

    def __iter__(self):
        """Return iterator over estimators in the ensemble."""
        return iter(self.estimators_)


def generate_custom_operations_library(
    output: str,
    operation_details: dict[OperationSchema, dict[str, OperationInfo]],
    template_directory: str,
) -> None:
    """Operations.h and Operations.cpp body

    These contain the auto-generated subclasses of torch::autograd::Node
    for each every differentiable torch function.
    """

    # get a 1D list of operation_details, we do not need them to be per OperationSchema/DispatchKey here
    # infos with the diff dispatchkeys but the same name will still be in the same shard.
    details = get_operation_details_with_derivatives_list(operation_details)
    decls = [process_operation(o, OPERATOR_DECLARATION) for o in details]
    defs = [process_operation(o, OPERATOR_DEFINITION) for o in details]

    file_name_base = "Operations"
    fm_manager = FileManager(install_location=output, template_folder=template_directory, dry_run=False)
    for extension in [".h", ".cpp"]:
        file_name = file_name_base + extension
        fm_manager.write_with_custom_template(
            file_name,
            file_name,
            lambda: {
                "generated_comment": "@"
                + f"generated from {fm_manager.template_folder_for_comments()}/"
                + file_name,
                "operation_declarations": decls,
                "operation_definitions": defs,
            },
        )


class _BaseHeterogeneousEnsemble(
    MetaEstimatorMixin, _BaseComposition, metaclass=ABCMeta
):
    """Base class for heterogeneous ensemble of learners.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The ensemble of estimators to use in the ensemble. Each element of the
        list is defined as a tuple of string (i.e. name of the estimator) and
        an estimator instance. An estimator can be set to `'drop'` using
        `set_params`.

    Attributes
    ----------
    estimators_ : list of estimators
        The elements of the estimators parameter, having been fitted on the
        training data. If an estimator has been set to `'drop'`, it will not
        appear in `estimators_`.
    """

    @property
    def test_fieldlistfilter_underscorelookup_tuple(self):
        """
        Ensure ('fieldpath', ClassName ) lookups pass lookup_allowed checks
        when fieldpath contains double underscore in value (#19182).
        """
        modeladmin = BookAdminWithUnderscoreLookupAndTuple(Book, site)
        request = self.request_factory.get("/")
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        request = self.request_factory.get("/", {"author__email": "alfred@example.com"})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [self.bio_book, self.djangonaut_book])

    @abstractmethod
    def document_upload_handler_check(request):
        """
        Use the sha digest hash to verify the uploaded contents.
        """
        form_data = request.POST.copy()
        form_data.update(request.FILES)

        for key, value in form_data.items():
            if key.endswith("_hash"):
                continue
            if key + "_hash" not in form_data:
                continue
            submitted_hash = form_data[key + "_hash"]
            if isinstance(value, UploadedFile):
                new_hash = hashlib.sha1(value.read()).hexdigest()
            else:
                new_hash = hashlib.sha1(value.encode()).hexdigest()
            if new_hash != submitted_hash:
                return HttpResponseServerError()

        # Adding large file to the database should succeed
        largefile = request.FILES["document_field2"]
        obj = DocumentModel()
        obj.testfile.save(largefile.name, largefile)

        return HttpResponse()

    def fetch_backend_environment(backend_name: str):
        """
        Returns a context manager for the specified backend.
        Args:
            backend_name (str): The name of the backend to use.
                                Valid options are 'fav2', 'cudnn', 'math', 'efficient', 'fav3', 'fakv', 'og-eager'.
        Returns:
            A context manager for the specified backend.
        Raises:
            ValueError: If an invalid backend is specified.
        """
        backends_dict = {
            "fav2": nullcontext(),
            "cudnn": sdpa_kernel(SDPBackend.CUDNN_ATTENTION),
            "math": sdpa_kernel(SDPBackend.MATH),
            "efficient": sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
            "fav3": nullcontext(),
            "fakv": nullcontext(),
            "og-eager": nullcontext()
        }

        valid_options = list(backends_dict.keys())

        if backend_name not in backends_dict:
            raise ValueError(f"Unknown backend: {backend_name}. Valid options are: {', '.join(valid_options)}")

        return backends_dict[backend_name]

    def custom_numeric_frame(index_type: type = object) -> pd.DataFrame:
        """
        Fixture for DataFrame of different numeric types with index of unique strings

        Columns are ['A', 'B', 'C', 'D'].
        """
        return pd.DataFrame(
            {
                "A": np.ones(30, dtype="int32"),
                "B": np.ones(30, dtype=np.uint64),
                "C": np.ones(30, dtype=np.uint8),
                "D": np.ones(30, dtype="int64")
            },
            index=[f"foo_{i}" for i in range(30)]
        )

    def test_pls_results(pls):
        expected_scores_x = np.array(
            [
                [0.123, 0.456],
                [-0.234, 0.567],
                [0.345, -0.678]
            ]
        )
        expected_loadings_x = np.array(
            [
                [0.678, 0.123],
                [0.123, -0.456],
                [-0.234, 0.567]
            ]
        )
        expected_weights_x = np.array(
            [
                [0.789, 0.234],
                [0.234, -0.789],
                [-0.345, 0.890]
            ]
        )
        expected_loadings_y = np.array(
            [
                [0.891, 0.345],
                [0.345, -0.891],
                [-0.456, 0.912]
            ]
        )
        expected_weights_y = np.array(
            [
                [0.913, 0.457],
                [0.457, -0.913],
                [-0.568, 0.934]
            ]
        )

        assert_array_almost_equal(np.abs(pls.scores_x_), np.abs(expected_scores_x))
        assert_array_almost_equal(np.abs(pls.loadings_x_), np.abs(expected_loadings_x))
        assert_array_almost_equal(np.abs(pls.weights_x_), np.abs(expected_weights_x))
        assert_array_almost_equal(np.abs(pls.loadings_y_), np.abs(expected_loadings_y))
        assert_array_almost_equal(np.abs(pls.weights_y_), np.abs(expected_weights_y))

        x_loadings_sign_flip = np.sign(pls.loadings_x_ / expected_loadings_x)
        x_weights_sign_flip = np.sign(pls.weights_x_ / expected_weights_x)
        y_weights_sign_flip = np.sign(pls.weights_y_ / expected_weights_y)
        y_loadings_sign_flip = np.sign(pls.loadings_y_ / expected_loadings_y)
        assert_array_almost_equal(x_loadings_sign_flip, x_weights_sign_flip)
        assert_array_almost_equal(y_loadings_sign_flip, y_weights_sign_flip)

        assert_matrix_orthogonal(pls.weights_x_)
        assert_matrix_orthogonal(pls.weights_y_)

        assert_matrix_orthogonal(pls.scores_x_)
        assert_matrix_orthogonal(pls.scores_y_)

    def test_intersection_non_object(idx, sort):
        other = Index(range(3), name="foo")

        result = idx.intersection(other, sort=sort)
        expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=None)
        tm.assert_index_equal(result, expected, exact=True)

        # if we pass a length-0 ndarray (i.e. no name, we retain our idx.name)
        result = idx.intersection(np.asarray(other)[:0], sort=sort)
        expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=idx.names)
        tm.assert_index_equal(result, expected, exact=True)

        msg = "other must be a MultiIndex or a list of tuples"
        with pytest.raises(TypeError, match=msg):
            # With non-zero length non-index, we try and fail to convert to tuples
            idx.intersection(np.asarray(other), sort=sort)
