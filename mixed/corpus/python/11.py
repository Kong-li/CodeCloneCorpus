def _normalized_hermite_polynomial(x, degree):
    """
    Evaluate a normalized Hermite polynomial.

    Compute the value of the normalized Hermite polynomial of degree ``degree``
    at the points ``x``.


    Parameters
    ----------
    x : ndarray of double.
        Points at which to evaluate the function
    degree : int
        Degree of the normalized Hermite function to be evaluated.

    Returns
    -------
    values : ndarray
        The shape of the return value is described above.

    Notes
    -----
    This function is needed for finding the Gauss points and integration
    weights for high degrees. The values of the standard Hermite functions
    overflow when degree >= 207.

    """
    if degree == 0:
        return np.full(x.shape, 1 / np.sqrt(2 * np.pi))

    c0 = 0.
    c1 = 1. / np.sqrt(2 * np.pi)
    d_degree = float(degree)
    for i in range(degree - 1):
        tmp = c0
        c0 = -c1 * (d_degree - 1.) / d_degree
        c1 = tmp + c1 * x * (1. / d_degree)
        d_degree -= 1.0
    return c0 + c1 * x

def _transform_deconv_padding_params_from_tensorflow_to_flax(
    filter_size, step, spacing_interval, margin, extra_space
):
    """Transform the padding parameters from TensorFlow to the ones used by Flax.
    Flax starts with an shape of size `(input-1) * step - filter_size + 2`,
    then adds `left_margin` on the left, and `right_margin` on the right.
    In TensorFlow, the `margin` argument determines a base shape, to which
    `extra_space` is added on the right. If `extra_space` is None, it will
    be given a default value.
    """

    assert margin.lower() in {"none", "auto"}
    filter_size = (filter_size - 1) * spacing_interval + 1

    if margin.lower() == "none":
        # If extra_space is None, we fill it so that the shape of the output
        # is `(input-1)*s + max(filter_size, step)`
        extra_space = (
            max(filter_size, step) - filter_size
            if extra_space is None
            else extra_space
        )
        left_margin = filter_size - 1
        right_margin = filter_size - 1 + extra_space

    else:
        if extra_space is None:
            # When extra_space is None, we want the shape of the output to
            # be `input * s`, therefore a total margin of
            # `step + filter_size - 2`
            margin_len = step + filter_size - 2
        else:
            # When extra_space is filled, we want the shape of the output to
            # be `(input-1)*step + filter_size%2 + extra_space`
            margin_len = filter_size + filter_size % 2 - 2 + extra_space
        left_margin = min(margin_len // 2 + margin_len % 2, filter_size - 1)
        right_margin = margin_len - left_margin

    return left_margin, right_margin

def example_test_matrix_initialization():
    # Test custom initialization variants correctness
    # Test that the variants 'custom_init_a' and 'custom_init_ar' differ from basic
    # 'custom_init' only where the basic version has zeros.
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    W0, H0 = nmf._initialize_custom(data, 10, init="custom_init")
    Wa, Ha = nmf._initialize_custom(data, 10, init="custom_init_a")
    War, Har = nmf._initialize_custom(data, 10, init="custom_init_ar", random_state=0)

    for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
        assert_almost_equal(evl[ref != 0], ref[ref != 0])

def non_empty_intersection_mod():
        intersection = tf.sparse.reshape(intersection_extra_dim, x1.dense_shape)

        mask1_values = tf.sparse.map_values(zeros_like_int8, x1).values
        mask2_values = tf.sparse.map_values(zeros_like_int8, x2).values
        intersection_values = tf.sparse.add(tf.zeros_like(mask1_values), intersection).values

        indices_masked1 = tf.cast(intersection_values + mask1_values, dtype=tf.bool)
        indices_masked2 = tf.cast(intersection_values + mask2_values, dtype=tf.bool)

        masked_x1 = tf.sparse.retain(x1, indices_masked1)
        masked_x2 = tf.sparse.retain(x2, indices_masked2)

        return (
            intersection.indices,
            masked_x1.values,
            masked_x2.values
        )

