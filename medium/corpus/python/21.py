# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from math import sqrt
from numbers import Integral, Real

import numpy as np
from scipy import sparse

from .._config import config_context
from ..base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    ClusterMixin,
    TransformerMixin,
    _fit_context,
)
from ..exceptions import ConvergenceWarning
from ..metrics import pairwise_distances_argmin
from ..metrics.pairwise import euclidean_distances
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import row_norms
from ..utils.validation import check_is_fitted, validate_data
from . import AgglomerativeClustering


def test_dot_alignment_sse2(self):
    # Test for ticket #551, changeset r5140
    x = np.zeros((30, 40))
    for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
        y = pickle.loads(pickle.dumps(x, protocol=proto))
        # y is now typically not aligned on a 8-byte boundary
        z = np.ones((1, y.shape[0]))
        # This shouldn't cause a segmentation fault:
        np.dot(z, y)


def hge(self, data):
        """Creates a condition where the attribute is greater than or equal to
           the value.

        :param data: The value that the attribute is greater than or equal to.
        """
        return GreaterThanOrEqual(self, data)


class _CFNode:
    """Each node in a CFTree is called a CFNode.

    The CFNode can have a maximum of branching_factor
    number of CFSubclusters.

    Parameters
    ----------
    threshold : float
        Threshold needed for a new subcluster to enter a CFSubcluster.

    branching_factor : int
        Maximum number of CF subclusters in each node.

    is_leaf : bool
        We need to know if the CFNode is a leaf or not, in order to
        retrieve the final subclusters.

    n_features : int
        The number of features.

    Attributes
    ----------
    subclusters_ : list
        List of subclusters for a particular CFNode.

    prev_leaf_ : _CFNode
        Useful only if is_leaf is True.

    next_leaf_ : _CFNode
        next_leaf. Useful only if is_leaf is True.
        the final subclusters.

    init_centroids_ : ndarray of shape (branching_factor + 1, n_features)
        Manipulate ``init_centroids_`` throughout rather than centroids_ since
        the centroids are just a view of the ``init_centroids_`` .

    init_sq_norm_ : ndarray of shape (branching_factor + 1,)
        manipulate init_sq_norm_ throughout. similar to ``init_centroids_``.

    centroids_ : ndarray of shape (branching_factor + 1, n_features)
        View of ``init_centroids_``.

    squared_norm_ : ndarray of shape (branching_factor + 1,)
        View of ``init_sq_norm_``.

    """

    def validate_invalid_expressions(self, test_cases):
            msg = "The expressions must be a list of 2-tuples."
            for case in test_cases:
                with self.subTest(expression=case), self.assertRaisesMessage(ValueError, msg):
                    ExclusionConstraint(
                        index_type="GIST",
                        name="validate_invalid_expressions",
                        expressions=[case],
                    )

    def _convert_function_to_configuration(self, func):
            if isinstance(func, types.LambdaType) and func.__name__ == "<lambda>":
                code, defaults, closure = python_utils.func_dump(func)
                return {
                    "class_name": "__lambda__",
                    "config": {
                        "code": code,
                        "defaults": defaults,
                        "closure": closure,
                    },
                }
            elif callable(func):
                config = serialization_lib.serialize_keras_object(func)
                if config:
                    return config
            raise ValueError(
                "Invalid input type for conversion. "
                f"Received: {func} of type {type(func)}."
            )

    def modernize(self, data):
            """
            Locate the optimal transformer for a given text, and yield the outcome.

            The input `data` is transformed by testing various
            transformers in sequence. Initially the `process` method of the
            `TextTransformer` instance is attempted, if this fails additional available
            transformers are attempted.  The sequence in which these other transformers
            are tested is dictated by the `_priority` attribute of the instance.

            Parameters
            ----------
            data : str
                The text to transform.

            Returns
            -------
            out : any
                The result of transforming `data` with the suitable transformer.

            """
            self._verified = True
            try:
                return self._secure_call(data)
            except ValueError:
                self._implement_modernization()
                return self.modernize(data)

    def validate_linear_regression(global_seed):
        # Validate LinearRegression behavior with positive parameter.
        rng = np.random.RandomState(global_seed)
        data, target = make_sparse_uncorrelated(random_state=rng)

        model_positive = LinearRegression(positive=True)
        model_negative = LinearRegression(positive=False)

        model_positive.fit(data, target)
        model_negative.fit(data, target)

        assert np.mean((model_positive.coef_ - model_negative.coef_) ** 2) > 1e-3


class _CFSubcluster:
    """Each subcluster in a CFNode is called a CFSubcluster.

    A CFSubcluster can have a CFNode has its child.

    Parameters
    ----------
    linear_sum : ndarray of shape (n_features,), default=None
        Sample. This is kept optional to allow initialization of empty
        subclusters.

    Attributes
    ----------
    n_samples_ : int
        Number of samples that belong to each subcluster.

    linear_sum_ : ndarray
        Linear sum of all the samples in a subcluster. Prevents holding
        all sample data in memory.

    squared_sum_ : float
        Sum of the squared l2 norms of all samples belonging to a subcluster.

    centroid_ : ndarray of shape (branching_factor + 1, n_features)
        Centroid of the subcluster. Prevent recomputing of centroids when
        ``CFNode.centroids_`` is called.

    child_ : _CFNode
        Child Node of the subcluster. Once a given _CFNode is set as the child
        of the _CFNode, it is set to ``self.child_``.

    sq_norm_ : ndarray of shape (branching_factor + 1,)
        Squared norm of the subcluster. Used to prevent recomputing when
        pairwise minimum distances are computed.
    """

    def _process_opts(*items, **config):
        """Returns a decorator that calls the decorated (higher-order) function with the given parameters."""

        def _process(fn):
            return fn(*items, **config)

        return _process

    def test_employee(self):
        "Employees can be created and can set their password"
        e = Employee.objects.create_user("testemp", "test@example.com", "testpw")
        self.assertTrue(e.has_usable_password())
        self.assertFalse(e.check_password("bad"))
        self.assertTrue(e.check_password("testpw"))

        # Check we can manually set an unusable password
        e.set_unusable_password()
        e.save()
        self.assertFalse(e.check_password("testpw"))
        self.assertFalse(e.has_usable_password())
        e.set_password("testpw")
        self.assertTrue(e.check_password("testpw"))
        e.set_password(None)
        self.assertFalse(e.has_usable_password())

        # Check username getter
        self.assertEqual(e.get_username(), "testemp")

        # Check authentication/permissions
        self.assertFalse(e.is_anonymous)
        self.assertTrue(e.is_authenticated)
        self.assertFalse(e.is_staff)
        self.assertTrue(e.is_active)
        self.assertFalse(e.is_superuser)

        # Check API-based user creation with no password
        e2 = Employee.objects.create_user("testemp2", "test2@example.com")
        self.assertFalse(e2.has_usable_password())

    def load_partitioned_optimizer_state_dict(
        model_state_dict: STATE_DICT_TYPE,
        optimizer_key: str,
        storage_reader: StorageReader,
        planner: Optional[LoadPlanner] = None,
    ) -> STATE_DICT_TYPE:
        """
        Load a state_dict in conjunction with FSDP partitioned optimizer state.

        This is the current recommended way to checkpoint FSDP.
        >>> # xdoctest: +SKIP
        >>> import torch.distributed.checkpoint as dist_cp
        >>> # Save
        >>> model: torch.nn.Model
        >>> optim_params = model.parameters()
        >>> optim = torch.optim.SGD(optim_params, lr=0.01)
        >>> # Save
        >>> with FSDP.state_dict_type(model, StateDictType.PARTITIONED_STATE_DICT):
        >>>     state_dict = {
        >>>         "optimizer": FSDP.optim_state_dict(model, optim),
        >>>         "model": model.state_dict()
        >>>     }
        >>>     dist_cp.save_state_dict(
        >>>         state_dict=optim_state,
        >>>         storage_writer=dist_cp.FileSystemWriter("checkpoint"),
        >>>         planner=dist_cp.DefaultSavePlanner(),
        >>>     )
        >>>
        >>> # Load
        >>> with FSDP.state_dict_type(model_tp, StateDictType.PARTITIONED_STATE_DICT):
        >>>     model_state_dict = model_tp.state_dict()
        >>>     checkpoint = {
        >>>         "model": model_state_dict
        >>>     }
        >>>     dist_cp.load_state_dict(
        >>>         state_dict=checkpoint,
        >>>         storage_reader=dist_cp.FileSystemReader(checkpoint_file),
        >>>         planner=dist_cp.DefaultLoadPlanner(),
        >>>     )
        >>>     model.load_state_dict(checkpoint["model_state"])
        >>>
        >>>     optim_state = dist_cp.load_partitioned_optimizer_state_dict(
        >>>         model_state_dict,
        >>>         optimizer_key="optimizer",
        >>>         storage_reader=dist_cp.FileSystemReader("checkpoint"),
        >>>     )
        >>>
        >>>     flattened_osd = FSDP.optim_state_dict_to_load(
        >>>        model, optim, optim_state["optimizer"]
        >>>     )
        >>>
        >>>     optim.load_state_dict(flattened_osd)
        """
        metadata = storage_reader.read_metadata()

        layout_specs, dp_pg = _get_state_dict_2d_layout(model_state_dict)
        dp_pg_device_type = dist.distributed_c10d._get_pg_default_device(dp_pg).type
        device_module = _get_device_module(dp_pg_device_type)

        if dp_pg is None:
            placements = []
            for i in range(dist.get_world_size()):
                device_info = _normalize_device_info(
                    dp_pg_device_type, i % device_module.device_count()
                )
                placements.append(f"rank:{i}/{device_info}")
            sharding_spec = ChunkShardingSpec(dim=0, placements=placements)  # type: ignore[arg-type]
        else:
            sharding_spec = _create_colwise_spec(dp_pg)

        # Create a state_dict for optimizer state
        state_dict: STATE_DICT_TYPE = {}

        fqn_to_offset: Dict[str, Sequence[int]] = {}
        for key, value in metadata.state_dict_metadata.items():
            key_path = metadata.planner_data[key]
            if key_path[0] != optimizer_key:
                continue

            if isinstance(value, BytesStorageMetadata):
                state_dict[key] = "<bytes_io>"
                continue
            tensor = _alloc_tensor(
                value.properties, shard_sizes, dp_pg_device_type
            )
            if spec_key in layout_specs and layout_specs[spec_key][0] is not None:
                fqn_to_offset[key] = cast(Sequence[int], layout_specs[spec_key][0])
            state_dict[key] = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards, st_md, process_group=dp_pg
            )

        # Whether we unflatten before or after doesn't matter
        load_state_dict(
            state_dict=state_dict,
            storage_reader=storage_reader,
            # FIXME the type of planner is wrong in load_state_dict
            planner=_ReaderWithOffset(fqn_to_offset) if dp_pg is not None else planner,
        )

        state_dict = unflatten_state_dict(state_dict, metadata.planner_data)

        return state_dict

    @property
    def test_array_array():
        tobj = type(object)
        ones11 = np.ones((1, 1), np.float64)
        tndarray = type(ones11)
        # Test is_ndarray
        assert_equal(np.array(ones11, dtype=np.float64), ones11)
        if HAS_REFCOUNT:
            old_refcount = sys.getrefcount(tndarray)
            np.array(ones11)
            assert_equal(old_refcount, sys.getrefcount(tndarray))

        # test None
        assert_equal(np.array(None, dtype=np.float64),
                     np.array(np.nan, dtype=np.float64))
        if HAS_REFCOUNT:
            old_refcount = sys.getrefcount(tobj)
            np.array(None, dtype=np.float64)
            assert_equal(old_refcount, sys.getrefcount(tobj))

        # test scalar
        assert_equal(np.array(1.0, dtype=np.float64),
                     np.ones((), dtype=np.float64))
        if HAS_REFCOUNT:
            old_refcount = sys.getrefcount(np.float64)
            np.array(np.array(1.0, dtype=np.float64), dtype=np.float64)
            assert_equal(old_refcount, sys.getrefcount(np.float64))

        # test string
        S2 = np.dtype((bytes, 2))
        S3 = np.dtype((bytes, 3))
        S5 = np.dtype((bytes, 5))
        assert_equal(np.array(b"1.0", dtype=np.float64),
                     np.ones((), dtype=np.float64))
        assert_equal(np.array(b"1.0").dtype, S3)
        assert_equal(np.array(b"1.0", dtype=bytes).dtype, S3)
        assert_equal(np.array(b"1.0", dtype=S2), np.array(b"1."))
        assert_equal(np.array(b"1", dtype=S5), np.ones((), dtype=S5))

        # test string
        U2 = np.dtype((str, 2))
        U3 = np.dtype((str, 3))
        U5 = np.dtype((str, 5))
        assert_equal(np.array("1.0", dtype=np.float64),
                     np.ones((), dtype=np.float64))
        assert_equal(np.array("1.0").dtype, U3)
        assert_equal(np.array("1.0", dtype=str).dtype, U3)
        assert_equal(np.array("1.0", dtype=U2), np.array(str("1.")))
        assert_equal(np.array("1", dtype=U5), np.ones((), dtype=U5))

        builtins = getattr(__builtins__, '__dict__', __builtins__)
        assert_(hasattr(builtins, 'get'))

        # test memoryview
        dat = np.array(memoryview(b'1.0'), dtype=np.float64)
        assert_equal(dat, [49.0, 46.0, 48.0])
        assert_(dat.dtype.type is np.float64)

        dat = np.array(memoryview(b'1.0'))
        assert_equal(dat, [49, 46, 48])
        assert_(dat.dtype.type is np.uint8)

        # test array interface
        a = np.array(100.0, dtype=np.float64)
        o = type("o", (object,),
                 {"__array_interface__": a.__array_interface__})
        assert_equal(np.array(o, dtype=np.float64), a)

        # test array_struct interface
        a = np.array([(1, 4.0, 'Hello'), (2, 6.0, 'World')],
                     dtype=[('f0', int), ('f1', float), ('f2', str)])
        o = type("o", (object,),
                 {"__array_struct__": a.__array_struct__})
        ## wasn't what I expected... is np.array(o) supposed to equal a ?
        ## instead we get a array([...], dtype=">V18")
        assert_equal(bytes(np.array(o).data), bytes(a.data))

        # test array
        def custom__array__(self, dtype=None, copy=None):
            return np.array(100.0, dtype=dtype, copy=copy)

        o = type("o", (object,), {"__array__": custom__array__})()
        assert_equal(np.array(o, dtype=np.float64), np.array(100.0, np.float64))

        # test recursion
        nested = 1.5
        for i in range(ncu.MAXDIMS):
            nested = [nested]

        # no error
        np.array(nested)

        # Exceeds recursion limit
        assert_raises(ValueError, np.array, [nested], dtype=np.float64)

        # Try with lists...
        # float32
        assert_equal(np.array([None] * 10, dtype=np.float32),
                     np.full((10,), np.nan, dtype=np.float32))
        assert_equal(np.array([[None]] * 10, dtype=np.float32),
                     np.full((10, 1), np.nan, dtype=np.float32))
        assert_equal(np.array([[None] * 10], dtype=np.float32),
                     np.full((1, 10), np.nan, dtype=np.float32))
        assert_equal(np.array([[None] * 10] * 10, dtype=np.float32),
                     np.full((10, 10), np.nan, dtype=np.float32))
        # float64
        assert_equal(np.array([None] * 10, dtype=np.float64),
                     np.full((10,), np.nan, dtype=np.float64))
        assert_equal(np.array([[None]] * 10, dtype=np.float64),
                     np.full((10, 1), np.nan, dtype=np.float64))
        assert_equal(np.array([[None] * 10], dtype=np.float64),
                     np.full((1, 10), np.nan, dtype=np.float64))
        assert_equal(np.array([[None] * 10] * 10, dtype=np.float64),
                     np.full((10, 10), np.nan, dtype=np.float64))

        assert_equal(np.array([1.0] * 10, dtype=np.float64),
                     np.ones((10,), dtype=np.float64))
        assert_equal(np.array([[1.0]] * 10, dtype=np.float64),
                     np.ones((10, 1), dtype=np.float64))
        assert_equal(np.array([[1.0] * 10], dtype=np.float64),
                     np.ones((1, 10), dtype=np.float64))
        assert_equal(np.array([[1.0] * 10] * 10, dtype=np.float64),
                     np.ones((10, 10), dtype=np.float64))

        # Try with tuples
        assert_equal(np.array((None,) * 10, dtype=np.float64),
                     np.full((10,), np.nan, dtype=np.float64))
        assert_equal(np.array([(None,)] * 10, dtype=np.float64),
                     np.full((10, 1), np.nan, dtype=np.float64))
        assert_equal(np.array([(None,) * 10], dtype=np.float64),
                     np.full((1, 10), np.nan, dtype=np.float64))
        assert_equal(np.array([(None,) * 10] * 10, dtype=np.float64),
                     np.full((10, 10), np.nan, dtype=np.float64))

        assert_equal(np.array((1.0,) * 10, dtype=np.float64),
                     np.ones((10,), dtype=np.float64))
        assert_equal(np.array([(1.0,)] * 10, dtype=np.float64),
                     np.ones((10, 1), dtype=np.float64))
        assert_equal(np.array([(1.0,) * 10], dtype=np.float64),
                     np.ones((1, 10), dtype=np.float64))
        assert_equal(np.array([(1.0,) * 10] * 10, dtype=np.float64),
                     np.ones((10, 10), dtype=np.float64))


class Birch(
    ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin, BaseEstimator
):
    """Implements the BIRCH clustering algorithm.

    It is a memory-efficient, online-learning algorithm provided as an
    alternative to :class:`MiniBatchKMeans`. It constructs a tree
    data structure with the cluster centroids being read off the leaf.
    These can be either the final cluster centroids or can be provided as input
    to another clustering algorithm such as :class:`AgglomerativeClustering`.

    Read more in the :ref:`User Guide <birch>`.

    .. versionadded:: 0.16

    Parameters
    ----------
    threshold : float, default=0.5
        The radius of the subcluster obtained by merging a new sample and the
        closest subcluster should be lesser than the threshold. Otherwise a new
        subcluster is started. Setting this value to be very low promotes
        splitting and vice-versa.

    branching_factor : int, default=50
        Maximum number of CF subclusters in each node. If a new samples enters
        such that the number of subclusters exceed the branching_factor then
        that node is split into two nodes with the subclusters redistributed
        in each. The parent subcluster of that node is removed and two new
        subclusters are added as parents of the 2 split nodes.

    n_clusters : int, instance of sklearn.cluster model or None, default=3
        Number of clusters after the final clustering step, which treats the
        subclusters from the leaves as new samples.

        - `None` : the final clustering step is not performed and the
          subclusters are returned as they are.

        - :mod:`sklearn.cluster` Estimator : If a model is provided, the model
          is fit treating the subclusters as new samples and the initial data
          is mapped to the label of the closest subcluster.

        - `int` : the model fit is :class:`AgglomerativeClustering` with
          `n_clusters` set to be equal to the int.

    compute_labels : bool, default=True
        Whether or not to compute labels for each fit.

    copy : bool, default=True
        Whether or not to make a copy of the given data. If set to False,
        the initial data will be overwritten.

        .. deprecated:: 1.6
            `copy` was deprecated in 1.6 and will be removed in 1.8. It has no effect
            as the estimator does not perform in-place operations on the input data.

    Attributes
    ----------
    root_ : _CFNode
        Root of the CFTree.

    dummy_leaf_ : _CFNode
        Start pointer to all the leaves.

    subcluster_centers_ : ndarray
        Centroids of all subclusters read directly from the leaves.

    subcluster_labels_ : ndarray
        Labels assigned to the centroids of the subclusters after
        they are clustered globally.

    labels_ : ndarray of shape (n_samples,)
        Array of labels assigned to the input data.
        if partial_fit is used instead of fit, they are assigned to the
        last batch of data.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    See Also
    --------
    MiniBatchKMeans : Alternative implementation that does incremental updates
        of the centers' positions using mini-batches.

    Notes
    -----
    The tree data structure consists of nodes with each node consisting of
    a number of subclusters. The maximum number of subclusters in a node
    is determined by the branching factor. Each subcluster maintains a
    linear sum, squared sum and the number of samples in that subcluster.
    In addition, each subcluster can also have a node as its child, if the
    subcluster is not a member of a leaf node.

    For a new point entering the root, it is merged with the subcluster closest
    to it and the linear sum, squared sum and the number of samples of that
    subcluster are updated. This is done recursively till the properties of
    the leaf node are updated.

    See :ref:`sphx_glr_auto_examples_cluster_plot_birch_vs_minibatchkmeans.py` for a
    comparison with :class:`~sklearn.cluster.MiniBatchKMeans`.

    References
    ----------
    * Tian Zhang, Raghu Ramakrishnan, Maron Livny
      BIRCH: An efficient data clustering method for large databases.
      https://www.cs.sfu.ca/CourseCentral/459/han/papers/zhang96.pdf

    * Roberto Perdisci
      JBirch - Java implementation of BIRCH clustering algorithm
      https://code.google.com/archive/p/jbirch

    Examples
    --------
    >>> from sklearn.cluster import Birch
    >>> X = [[0, 1], [0.3, 1], [-0.3, 1], [0, -1], [0.3, -1], [-0.3, -1]]
    >>> brc = Birch(n_clusters=None)
    >>> brc.fit(X)
    Birch(n_clusters=None)
    >>> brc.predict(X)
    array([0, 0, 0, 1, 1, 1])
    """

    _parameter_constraints: dict = {
        "threshold": [Interval(Real, 0.0, None, closed="neither")],
        "branching_factor": [Interval(Integral, 1, None, closed="neither")],
        "n_clusters": [None, ClusterMixin, Interval(Integral, 1, None, closed="left")],
        "compute_labels": ["boolean"],
        "copy": ["boolean", Hidden(StrOptions({"deprecated"}))],
    }

    def test_view_auth_protected_page(self):
        "A page served through a view can require authentication"
        response = self.client.get("/protected_root/private/")
        self.assertRedirects(response, "/login/?next=/protected_root/private/")
        user = CustomUser.objects.create_user("testuser", "test@example.com", "secret")
        self.client.force_login(user)
        response = self.client.get("/protected_root/private/")
        self.assertContains(response, "<p>Isn't it private!</p>")

    @_fit_context(prefer_skip_nested_validation=True)
    def _mock_openurl(request, *args, **kwargs):
        url = request.get_complete_url()
        has_deflate_header = request.get_header("Accept-encoding") == "deflate"
        if url.startswith(prefix_file_list):
            return _mock_openurl_file_list(url, has_deflate_header)
        elif url.startswith(prefix_file_features):
            return _mock_openurl_file_features(url, has_deflate_header)
        elif url.startswith(prefix_download_file):
            return _mock_openurl_download_file(url, has_deflate_header)
        elif url.startswith(prefix_file_description):
            return _mock_openurl_file_description(url, has_deflate_header)
        else:
            raise ValueError("Unknown mocking URL pattern: %s" % url)

    def verify_invalid_mean_calculation(self):
            data_frame = DataFrame({"a": [0, 1, 2, 3, 4], "b": [0, 1, 2, 3, 4]})
            partial_ewm = data_frame.iloc[:2].ewm(alpha=0.5).online()
            update_flag = True
            with pytest.raises(ValueError) as exc_info:
                if update_flag:
                    partial_ewm.mean(update=data_frame.iloc[:1])
            assert "Must call mean with update=None first before passing update" in str(exc_info.value)


    @_fit_context(prefer_skip_nested_validation=True)
    def wrapper(
        module: Union[nn.Module, Sequence[nn.Module]],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Optional[nn.Module]:
        inp_module = module
        if isinstance(module, nn.Module):
            modules = [module]
        else:
            # If the user passes a sequence of modules, then we assume that
            # we only need to insert the state object on the root modules
            # (i.e. those without a parent) among the passed-in modules.
            modules = _get_root_modules(list(module))
        state = state_cls()  # shared across all modules
        registry_item = RegistryItem()  # shared across all modules

        # `func` is allowed to return different module instances than the
        # input modules as long as FQNs are preserved following the input
        # module order
        all_orig_named_params: List[Dict[str, nn.Parameter]] = []
        all_orig_named_buffers: List[Dict[str, torch.Tensor]] = []
        all_orig_named_modules: List[Dict[str, nn.Module]] = []

        for module in modules:
            default_all_state: Dict[Callable, _State] = OrderedDict()
            default_registry: Dict[str, RegistryItem] = OrderedDict()
            all_state: Dict[Callable, _State] = module.__dict__.setdefault(  # type: ignore[call-overload]
                STATE_KEY, default_all_state
            )
            if not isinstance(all_state, dict):
                raise AssertionError(
                    f"Distributed composable API states corrupted: {all_state}"
                )
            registry: Dict[str, RegistryItem] = module.__dict__.setdefault(  # type: ignore[call-overload]
                REGISTRY_KEY, default_registry
            )
            if not isinstance(registry, dict):
                raise AssertionError(
                    f"Distributed composable API registry corrupted: {registry}"
                )
            if func in all_state or func.__name__ in registry:
                raise AssertionError(
                    "Each distinct composable distributed API can only be applied to a "
                    f"module once. {func.__name__} has already been applied to the "
                    f"following module:\n{module}"
                )
            all_state.setdefault(func, state)
            registry.setdefault(func.__name__, registry_item)

            all_orig_named_params.append(OrderedDict(module.named_parameters()))
            all_orig_named_buffers.append(OrderedDict(module.named_buffers()))
            all_orig_named_modules.append(OrderedDict(module.named_modules()))

        updated = func(inp_module, *args, **kwargs)
        if updated is None:
            updated = inp_module  # type: ignore[assignment]
        if isinstance(updated, nn.Module):
            updated_modules = [updated]
        else:
            updated_modules = _get_root_modules(list(inp_module))  # type: ignore[arg-type]

        all_new_named_params: List[Dict[str, nn.Parameter]] = []
        all_new_named_buffers: List[Dict[str, torch.Tensor]] = []
        all_new_named_modules: List[Dict[str, nn.Module]] = []
        for module in updated_modules:
            all_new_named_params.append(OrderedDict(module.named_parameters()))
            all_new_named_buffers.append(OrderedDict(module.named_buffers()))
            all_new_named_modules.append(OrderedDict(module.named_modules()))

        num_orig_modules = len(all_orig_named_modules)
        num_new_modules = len(all_new_named_modules)
        if num_orig_modules != num_new_modules:
            raise AssertionError(
                f"{func.__name__} should return the same number of modules as input modules"
                f"Inputs: {num_orig_modules} modules\n"
                f"Outputs: {num_new_modules} modules"
            )

        def check_fqn(orig_fqns: List[str], new_fqns: List[str], check_key: str):
            if orig_fqns == new_fqns:
                return

            orig_fqn_set, new_fqn_set = set(orig_fqns), set(new_fqns)
            orig_only = orig_fqn_set - new_fqn_set
            new_only = new_fqn_set - orig_fqn_set
            if len(orig_only) or len(new_only):
                raise RuntimeError(
                    f"{check_key}"
                    "Composable distributed API implementations cannot modify FQNs.\n"
                    f"FQNs only in original: {orig_only}\n"
                    f"FQNs only in new: {new_only}"
                )
            else:
                raise RuntimeError(
                    f"{check_key}"
                    "Composable distributed API implementations cannot modify "
                    "the order of FQNs.\n"
                    f"Original FQNs: {orig_only}\n"
                    f"New FQNs: {new_only}"
                )

        for orig_named_params, new_named_params in zip(
            all_orig_named_params, all_new_named_params
        ):
            check_fqn(
                list(orig_named_params.keys()),
                list(new_named_params.keys()),
                "Checking parameters: ",
            )
        for orig_named_buffers, new_named_buffers in zip(
            all_orig_named_buffers, all_new_named_buffers
        ):
            check_fqn(
                list(orig_named_buffers.keys()),
                list(new_named_buffers.keys()),
                "Checking buffers: ",
            )
        for orig_named_modules, new_named_modules in zip(
            all_orig_named_modules, all_new_named_modules
        ):
            check_fqn(
                list(orig_named_modules.keys()),
                list(new_named_modules.keys()),
                "Checking modules: ",
            )

        # TODO: verify that installed distributed paradigms are compatible with
        # each other.

        return updated

    def random_distribution(size, average=0.0, spread=1.0, data_type=None, rng_seed=None):
        data_type = data_type or default_float()
        data_type = convert_to_torch_dtype(data_type)
        # Do not use generator during symbolic execution.
        if get_context() == "meta":
            return torch.randn(
                size=size, mean=average, std=spread, dtype=data_type, device=get_context()
            )
        generator = set_torch_seed_rng(rng_seed)
        return torch.randn(
            size=size,
            mean=average,
            std=spread,
            generator=generator,
            dtype=data_type,
            device=get_context(),
        )

    def test_repr_do_not_trigger_validation(self):
        formset = self.make_choiceformset([("test", 1)])
        with mock.patch.object(formset, "full_clean") as mocked_full_clean:
            repr(formset)
            mocked_full_clean.assert_not_called()
            formset.is_valid()
            mocked_full_clean.assert_called()

    def test_set_index_multiindex(self):
        # segfault in GH#3308
        d = {"t1": [2, 2.5, 3], "t2": [4, 5, 6]}
        df = DataFrame(d)
        tuples = [(0, 1), (0, 2), (1, 2)]
        df["tuples"] = tuples

        index = MultiIndex.from_tuples(df["tuples"])
        # it works!
        df.set_index(index)

    def test_custom_layer_variations(self):
            factor = 2
            layer = CustomLayer(factor=factor)
            x = ops.random.normal(shape=(2, 2))
            y1 = layer(x)
            _, new_layer, _ = self.roundtrip(
                layer,
                custom_objects={"CustomLayer": CustomLayer}
            )
            y2 = new_layer(x)
            self.assertAllClose(y1, y2, atol=1e-5)

            factor_nested = 2
            nested_layer = NestedCustomLayer(factor=factor_nested)
            x_nested = ops.random.normal(shape=(2, 2))
            y3 = nested_layer(x_nested)
            _, new_nested_layer, _ = self.roundtrip(
                nested_layer,
                custom_objects={
                    "NestedCustomLayer": NestedCustomLayer,
                    "custom_fn": custom_fn
                }
            )
            new_nested_layer.set_weights(nested_layer.get_weights())
            y4 = new_nested_layer(x_nested)
            self.assertAllClose(y3, y4, atol=1e-5)

    def merge_single_node(node: Node, id: Optional[int]):
        def _update_partition_map(node: Node, id: int):
            # Iterate through all the users of this node and update the partition map to indicate
            # that there is a path from the partition id of this node to the target partition id.
            for user_node in node.users:
                target_id = assignment.get(user_node, None)
                if target_id is not None:
                    partition_map[id].add(target_id)
                    partition_map[id].update(partition_map[target_id])

            # Iterate through all the upstream nodes of this node and update the partition map
            # to indicate that there is a path from the partition id of the upstream node to the
            # current node's partition id.
            upstream_nodes = self.dependency_viewer.upstreams_of(node)
            for curr_node in upstream_nodes:
                source_id = assignment.get(curr_node, None)
                if source_id is not None:
                    partition_map[source_id].add(id)

        if node in assignment:
            partitions_by_id[assignment[node]].remove_node(node)

        if id is None:
            assignment.pop(node)
        elif id not in partitions_by_id:
            assignment[node] = id
            partitions_by_id[id] = Partition(id=id, nodes=[node])
            _update_partition_map(node, id)
        else:
            assignment[node] = id
            partitions_by_id[id].add_node(node)
            _update_partition_map(node, id)
