def compute_silhouette(
    features, cluster_labels, *, distance_metric="euclidean", sample_subset=None, seed=None, **kwargs
):
    """Compute the average Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels
    is ``2 <= n_labels <= n_samples - 1``.

    This function returns the average Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`compute_silhouette_samples`.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Read more in the :ref:`User Guide <silhouette_coefficient>`.

    Parameters
    ----------
    features : {array-like, sparse matrix} of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    cluster_labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    distance_metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`~sklearn.metrics.pairwise_distances`. If ``features`` is
        the distance array itself, use ``distance_metric="precomputed"``.

    sample_subset : int, default=None
        The size of the subset to use when computing the Silhouette Coefficient
        on a random selection of data.
        If ``sample_subset is None``, no sampling is used.

    seed : int, RandomState instance or None, default=None
        Determines random number generation for selecting a subset of samples.
        Used when ``sample_subset is not None``.
        Pass an integer for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    **kwargs : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    silhouette_score : float
        Average Silhouette Coefficient for all samples.

    References
    ----------

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.metrics import compute_silhouette
    >>> X, y = make_blobs(random_state=42)
    >>> kmeans = KMeans(n_clusters=2, random_state=42)
    >>> silhouette_score = compute_silhouette(X, kmeans.fit_predict(X))
    0.49...
    """
    if sample_subset is not None:
        features, cluster_labels = check_X_y(features, cluster_labels, accept_sparse=["csc", "csr"])
        seed = check_random_state(seed)
        indices = seed.permutation(features.shape[0])[:sample_subset]
        if distance_metric == "precomputed":
            features, cluster_labels = features[indices].T[indices].T, cluster_labels[indices]
        else:
            features, cluster_labels = features[indices], cluster_labels[indices]
    return np.mean(compute_silhouette_samples(features, cluster_labels, metric=distance_metric, **kwargs))

def aot_eager_decomp_partition(gm, fake_tensor_inputs, **kwargs):
    if kwargs:
        log.warning(
            "aot_eager_decomp_partition backend ignoring extra kwargs %s", kwargs
        )

    from torch._inductor.compiler_bisector import CompilerBisector

    config_patches = {"unlift_effect_tokens": True}
    if bisect_changes := CompilerBisector.get_config_change(
        "aot_eager_decomp_partition"
    ):
        config_patches.update(bisect_changes)

    with functorch_config.patch(config_patches):
        return aot_autograd(
            # these are taken from memory_efficient_fusion()
            fw_compiler=get_nop_func(),
            bw_compiler=get_nop_func(),
            # NB: lambda here is to delay import of inductor
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        )(gm, fake_tensor_inputs)

def validate_invalid_cookies(self):
        """
        Cookie strings that violate RFC 6265 but are sent by browsers via document.cookie.
        """
        # Chunks without an equals sign appear as unnamed values per the bug report at https://bugzilla.mozilla.org/show_bug.cgi?id=169091
        self.assertIn("django_language", parse_cookie(cookie_str="abc=def; unnamed; django_language=en"))
        # Even a double quote may be an unnamed value.
        self.assertEqual(parse_cookie('a=b; "; c=d'), {"a": "b", "unnamed": '"', "c": "d"})
        # Spaces in names and values, and an equals sign in values.
        parsed_cookies = parse_cookie("a b c=d e = f; gh=i")
        self.assertEqual(parsed_cookies["a b c"], "d e = f")
        self.assertEqual(parsed_cookies["gh"], "i")
        # More characters the spec forbids.
        self.assertEqual(
            parse_cookie('a   b,c<>@:/[]?{}=d  "  =e,f g'),
            {"a   b,c<>@:/[]?{}": 'd  "  =e,f g'}
        )
        # Unicode characters. The spec only allows ASCII.
        self.assertEqual(
            parse_cookie("saint=André Bessette"),
            {"saint": "André Bessette"}
        )
        # Browsers don't send extra whitespace or semicolons in Cookie headers, but parse_cookie() should handle it the same way document.cookie does.
        parsed_cookies = parse_cookie("  =  b  ;  ;  =  ;   c  =  ;  ")
        self.assertEqual(parsed_cookies["unnamed"], "b")
        self.assertEqual(parsed_cookies["c"], "")

def test_refit():
    # Regression test for bug in refitting
    # Simulates re-fitting a broken estimator; this used to break with
    # sparse SVMs.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = GridSearchCV(
        BrokenClassifier(), [{"parameter": [0, 1]}], scoring="precision", refit=True
    )
    clf.fit(X, y)

