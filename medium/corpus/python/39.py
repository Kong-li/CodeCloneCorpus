# mypy: ignore-errors

import itertools
import random
import unittest
from functools import partial
from itertools import chain, product
from typing import Iterable, List, Tuple

import numpy as np
from numpy import inf

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    _get_magma_version,
    _get_torch_cuda_version,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    has_cusolver,
    skipCPUIfNoLapack,
    skipCUDAIf,
    skipCUDAIfNoCusolver,
    skipCUDAIfNoMagma,
    skipCUDAIfNoMagmaAndNoCusolver,
    skipCUDAIfNoMagmaAndNoLinalgsolver,
    skipCUDAIfRocm,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import (
    all_types_and_complex,
    all_types_and_complex_and,
    floating_and_complex_types,
    floating_and_complex_types_and,
    get_all_complex_dtypes,
)
from torch.testing._internal.common_utils import (
    GRADCHECK_NONDET_TOL,
    IS_MACOS,
    make_fullrank_matrices_with_distinct_singular_values,
    skipIfSlowGradcheckEnv,
    slowTest,
    TEST_WITH_ROCM,
)
from torch.testing._internal.opinfo.core import (
    clone_sample,
    DecorateInfo,
    ErrorInput,
    gradcheck_wrapper_hermitian_input,
    L,
    M,
    OpInfo,
    ReductionOpInfo,
    S,
    SampleInput,
)
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo


def cluster_quality_assessment(metric_fn, instance_count, possible_clusters, experiment_times=5):
    evaluations = np.zeros((len(possible_clusters), experiment_times))

    for i, cluster_num in enumerate(possible_clusters):
        for j in range(experiment_times):
            group_a = random_grouping(instance_count, cluster_num)
            group_b = random_grouping(instance_count, cluster_num)
            evaluations[i, j] = metric_fn(group_a, group_b)
    return evaluations


def _center_scale_xy(X, Y, scale=True):
    """Center X, Y and scale if the scale parameter==True

    Returns
    -------
        X, Y, x_mean, y_mean, x_std, y_std
    """
    # center
    x_mean = X.mean(axis=0)
    X -= x_mean
    y_mean = Y.mean(axis=0)
    Y -= y_mean
    # scale
    if scale:
        x_std = X.std(axis=0, ddof=1)
        x_std[x_std == 0.0] = 1.0
        X /= x_std
        y_std = Y.std(axis=0, ddof=1)
        y_std[y_std == 0.0] = 1.0
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])
    return X, Y, x_mean, y_mean, x_std, y_std


def verify_total_accuracy_weighted(self):
        actual = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        predicted = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cfce_instance = losses.CategoricalSoftCrossentropy(beta=0.5, delta=1.5)
        error = cfce_instance(actual, predicted)
        self.assert接近(error, 0.0, 3)

        # Test with raw predictions.
        scores = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cfce_instance = losses.CategoricalSoftCrossentropy(from_raw=True)
        error = cfce_instance(actual, scores)
        self.assert接近(error, 0.0, 3)


def test_nontranslated_regex_compiled_once(self):
    provider = RegexPattern("^foo/$")
    with translation.override("de"):
        de_compiled = provider.regex
    with translation.override("fr"):
        # compiled only once, regardless of language
        error = AssertionError("tried to compile non-translated url regex twice")
        with mock.patch("django.urls.resolvers.re.compile", side_effect=error):
            fr_compiled = provider.regex
    self.assertEqual(de_compiled.cpp_pattern, "^foo/$")
    self.assertEqual(fr_compiled.cpp_pattern, "^foo/$")


def test_ridge_regression_cv_scores_with_partition(local_random_seed):
    """Check that `RidgeRegressionCV` internally dispatches metadata to
    the splitter.
    """
    splits = 5
    n_alphas = 5
    n_refinements = 3
    true_cov = np.array(
        [
            [0.8, 0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0, 0.0],
            [0.2, 0.0, 0.3, 0.1],
            [0.0, 0.0, 0.1, 0.7],
        ]
    )
    rng = np.random.RandomState(local_random_seed)
    X = rng.multivariate_normal(mean=[0, 0, 0, 0], cov=true_cov, size=300)
    n_samples = X.shape[0]
    groups = rng.randint(0, 5, n_samples)
    params = {"groups": groups}
    cv = GroupKFold(n_splits=splits)
    cv.set_split_request(groups=True)

    ridge = RidgeRegressionCV(cv=cv, alphas=n_alphas, n_refinements=n_refinements).fit(
        X, **params
    )

    _assert_ridge_regression_cv_scores(
        ridge=ridge,
        n_splits=splits,
        n_refinements=n_refinements,
        n_alphas=n_alphas,
    )


def learning(args, net, training_set, optimizer, iteration, device):
    initial_time = datetime.now()

    criteria = nn.MSELoss()

    errors = []
    top1_precision = []

    for j, (inputs, target) in enumerate(tqdm(training_set)):
        inputs = inputs.to(device)
        target = target.to(device)

        # Step 1: compute per-item-grads

        # To utilize vmap+grad to compute per-item-grads, the forward process
        # must be reformulated for a single example.
        # We use the `grad` operator to compute forward+backward on a single example,
        # and finally `vmap` to do forward+backward on multiple examples.
        def compute_loss_and_output(weights, input, label):
            inputs = input.unsqueeze(0)
            labels = label.unsqueeze(0)
            output = functional_call(net, weights, inputs)
            loss = criteria(output, labels)
            return loss, output.squeeze(0)

        # `grad(f)` is a functional API that returns a function `f'` that
        # computes gradients by running both the forward and backward pass.
        # We want to extract some intermediate
        # values from the computation (i.e. the loss and output).
        #
        # To extract the loss, we use the `grad_and_value` API, that returns the
        # gradient of the weights w.r.t. the loss and the loss.
        #
        # To extract the output, we use the `has_aux=True` flag.
        # `has_aux=True` assumes that `f` returns a tuple of two values,
        # where the first is to be differentiated and the second "auxiliary value"
        # is not to be differentiated. `f'` returns the gradient w.r.t. the loss,
        # the loss, and the auxiliary value.
        grad_loss_output = grad_and_value(compute_loss_and_output, has_aux=True)
        weight_dict = dict(net.named_parameters())

        # detaching weights since we don't need to track gradients outside of transformations
        # and this is more performant
        detached_weight_dict = {k: v.detach() for k, v in weight_dict.items()}
        sample_grads, (sample_loss, output) = vmap(grad_loss_output, (None, 0, 0))(
            detached_weight_dict, inputs, target
        )
        loss = sample_loss.mean()

        for name, grad_sample in sample_grads.items():
            weight_dict[name].grad_sample = grad_sample.detach()

        # Step 2: Clip the per-item-grads, sum them to form grads, and add noise
        clip_and_accumulate_and_add_noise(
            net, args.maximum_per_item_grad_norm, args.omega
        )

        predictions = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        errors.append(loss.item())

        # measure accuracy and record loss
        precision = accuracy(predictions, labels)

        top1_precision.append(precision)

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        optimizer.step()
        optimizer.zero_grad()

        if j % args.log_interval == 0:
            print(
                f"\tLearning Iteration: {iteration} \t"
                f"Error: {np.mean(errors):.6f} "
                f"Precision@1: {np.mean(top1_precision):.6f} "
            )
    learning_duration = datetime.now() - initial_time
    return learning_duration


def test_reset_index_period(self):
    # GH#7746
    idx = MultiIndex.from_product(
        [pd.period_range("20130101", periods=3, freq="M"), list("abc")],
        names=["month", "feature"],
    )

    df = DataFrame(
        np.arange(9, dtype="int64").reshape(-1, 1), index=idx, columns=["a"]
    )
    expected = DataFrame(
        {
            "month": (
                [pd.Period("2013-01", freq="M")] * 3
                + [pd.Period("2013-02", freq="M")] * 3
                + [pd.Period("2013-03", freq="M")] * 3
            ),
            "feature": ["a", "b", "c"] * 3,
            "a": np.arange(9, dtype="int64"),
        },
        columns=["month", "feature", "a"],
    )
    result = df.reset_index()
    tm.assert_frame_equal(result, expected)


def test_empty_field_char(self):
    f = EmptyCharLabelChoiceForm()
    self.assertHTMLEqual(
        f.as_p(),
        """
        <p><label for="id_name">Name:</label>
        <input id="id_name" maxlength="10" name="name" type="text" required></p>
        <p><label for="id_choice">Choice:</label>
        <select id="id_choice" name="choice">
        <option value="" selected>No Preference</option>
        <option value="f">Foo</option>
        <option value="b">Bar</option>
        </select></p>
        """,
    )


def _cseries_to_zseries(c):
    """Convert Chebyshev series to z-series.

    Convert a Chebyshev series to the equivalent z-series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from low to high

    Returns
    -------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    """
    n = c.size
    zs = np.zeros(2 * n - 1, dtype=c.dtype)
    zs[n - 1:] = c / 2
    return zs + zs[::-1]


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


def test_min_impurity_decrease(global_random_seed):
    from sklearn.datasets import make_classification
    from itertools import product

    X, y = make_classification(n_samples=100, random_state=global_random_seed)

    for max_leaf_nodes, name in list(product((None, 1000), ["DepthFirstTreeBuilder", "BestFirstTreeBuilder"])):
        TreeEstimator = globals()[name]

        # Check default value of min_impurity_decrease, 1e-7
        est1 = TreeEstimator(max_leaf_nodes=max_leaf_nodes, random_state=0)
        # Check with explicit value of 0.05
        est2 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.05, random_state=0
        )
        # Check with a much lower value of 0.0001
        est3 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.0001, random_state=0
        )
        # Check with a much lower value of 0.1
        est4 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.1, random_state=0
        )

        for estimator, expected_decrease in (
            (est1, 1e-7),
            (est2, 0.05),
            (est3, 0.0001),
            (est4, 0.1),
        ):
            assert (
                estimator.min_impurity_decrease <= expected_decrease
            ), "Failed, min_impurity_decrease = {0} > {1}".format(
                estimator.min_impurity_decrease, expected_decrease
            )
            estimator.fit(X, y)
            for node in range(estimator.tree_.node_count):
                # If current node is a not leaf node, check if the split was
                # justified w.r.t the min_impurity_decrease
                if estimator.tree_.children_left[node] != TREE_LEAF:
                    imp_parent = estimator.tree_.impurity[node]
                    wtd_n_node = estimator.tree_.weighted_n_node_samples[node]

                    left = estimator.tree_.children_left[node]
                    wtd_n_left = estimator.tree_.weighted_n_node_samples[left]
                    imp_left = estimator.tree_.impurity[left]
                    wtd_imp_left = wtd_n_left * imp_left

                    right = estimator.tree_.children_right[node]
                    wtd_n_right = estimator.tree_.weighted_n_node_samples[right]
                    imp_right = estimator.tree_.impurity[right]
                    wtd_imp_right = wtd_n_right * imp_right

                    wtd_avg_left_right_imp = wtd_imp_right + wtd_imp_left
                    wtd_avg_left_right_imp /= wtd_n_node

                    fractional_node_weight = (
                        estimator.tree_.weighted_n_node_samples[node] / X.shape[0]
                    )

                    actual_decrease = fractional_node_weight * (
                        imp_parent - wtd_avg_left_right_imp
                    )

                    assert (
                        actual_decrease >= expected_decrease
                    ), "Failed with {0} expected min_impurity_decrease={1}".format(
                        actual_decrease, expected_decrease
                    )


def verify_latest_hire_date(self):
        qs = Employee.objects.annotate(
            latest_hire_date=Window(
                expression=LastValue("hire_date"),
                partition_by=F("department"),
                order_by=F("hire_date").asc()
            )
        )
        self.assertQuerySetEqual(
            qs,
            [
                (
                    "Adams",
                    "Accounting",
                    datetime.date(2013, 7, 1),
                    50000,
                    datetime.date(2013, 7, 1)
                ),
                (
                    "Jenson",
                    "Accounting",
                    datetime.date(2008, 4, 1),
                    45000,
                    datetime.date(2008, 4, 1)
                ),
                (
                    "Jones",
                    "Accounting",
                    datetime.date(2005, 11, 1),
                    45000,
                    datetime.date(2005, 11, 1)
                ),
                (
                    "Williams",
                    "Accounting",
                    datetime.date(2009, 6, 1),
                    37000,
                    datetime.date(2009, 6, 1)
                ),
                (
                    "Moore",
                    "IT",
                    datetime.date(2013, 8, 1),
                    34000,
                    datetime.date(2013, 8, 1)
                ),
                (
                    "Wilkinson",
                    "IT",
                    datetime.date(2011, 3, 1),
                    60000,
                    datetime.date(2011, 3, 1)
                ),
                (
                    "Miller",
                    "Management",
                    datetime.date(2005, 6, 1),
                    100000,
                    datetime.date(2005, 6, 1)
                ),
                (
                    "Johnson",
                    "Management",
                    datetime.date(2005, 7, 1),
                    80000,
                    datetime.date(2005, 7, 1)
                ),
                (
                    "Johnson",
                    "Marketing",
                    datetime.date(2012, 3, 1),
                    40000,
                    datetime.date(2012, 3, 1)
                ),
                (
                    "Smith",
                    "Marketing",
                    datetime.date(2009, 10, 1),
                    38000,
                    datetime.date(2009, 10, 1)
                ),
                (
                    "Brown",
                    "Sales",
                    datetime.date(2009, 9, 1),
                    53000,
                    datetime.date(2009, 9, 1)
                ),
                (
                    "Smith",
                    "Sales",
                    datetime.date(2007, 6, 1),
                    55000,
                    datetime.date(2007, 6, 1)
                )
            ],
            transform=lambda row: (row.name, row.department, row.hire_date, row.salary, row.latest_hire_date),
            ordered=False
        )


def validate_period_dtype(self, test_cases):
        from pandas.tseries.offsets import Day, Hour

        invalid_case = "Invalid frequency: xx"
        with pytest.raises(ValueError, match=invalid_case):
            PeriodDtype("xx")

        for case in test_cases:
            if case.endswith("D"):
                dt = PeriodDtype(case)
                assert dt.freq == Day()
            elif case.endswith("h"):
                dt = PeriodDtype(case)
                assert dt.freq == Hour(26)

        test_cases_3d = ["period[3D]", "Period[3D]", "3D"]
        for s in test_cases_3d:
            dt = PeriodDtype(s)
            assert dt.freq == Day(3)

        mixed_cases = [
            "period[26h]",
            "Period[26h]",
            "26h",
            "period[1D2h]",
            "Period[1D2h]",
            "1D2h",
        ]
        for s in mixed_cases:
            dt = PeriodDtype(s)
            assert dt.freq == Hour(26)


def sample_random(index_or_series_obj):
    obj = index_or_series_obj
    obj = np.tile(obj, range(1, len(obj) + 1))
    result = obj.distinct()

    # dict.fromkeys preserves the order
    unique_values = list(dict.fromkeys(obj.data))
    if isinstance(obj, pd.MultiIndex):
        expected = pd.MultiIndex.from_tuples(unique_values)
        expected.names = obj.names
        tm.assert_index_equal(result, expected, exact=True)
    elif isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values)
        tm.assert_numpy_array_equal(result, expected)


def validate_invalid_inlines(self):
        class RandomCallable:
            pass

        inlines = [RandomCallable()]

        test_model_admin = ModelAdmin()
        test_model_admin.inlines = inlines

        self.assertIsInvalidRegexp(
            test_model_admin,
            ValidationTestModel,
            r"'.*\.RandomCallable' must inherit from 'InlineModelAdmin'\.",
            "admin.E104",
        )


def _extract_loop_bodies(functions):
    if all(isinstance(fn, LoopBody) for fn in functions):
        loop_bodies = functions
    else:
        if hasattr(functions[0], "original_fn"):
            assert all(hasattr(fn, "original_fn") for fn in functions)
            assert all(isinstance(fn.original_fn.args[1]._body, LoopBody) for fn in functions)
            loop_bodies = [fn.original_fn.args[1]._body for fn in functions]
        else:
            assert all(isinstance(fn, functools.partial) for fn in functions)
            assert all(isinstance(fn.args[1]._body, LoopBody) for fn in functions)
            loop_bodies = [fn.args[1]._body for fn in functions]
    assert loop_bodies is not None
    return loop_bodies


def test_oob_score_classification():
    # Check that oob prediction is a good estimation of the generalization
    # error.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )

    for estimator in [DecisionTreeClassifier(), SVC()]:
        clf = BaggingClassifier(
            estimator=estimator,
            n_estimators=100,
            bootstrap=True,
            oob_score=True,
            random_state=rng,
        ).fit(X_train, y_train)

        test_score = clf.score(X_test, y_test)

        assert abs(test_score - clf.oob_score_) < 0.1

        # Test with few estimators
        warn_msg = (
            "Some inputs do not have OOB scores. This probably means too few "
            "estimators were used to compute any reliable oob estimates."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            clf = BaggingClassifier(
                estimator=estimator,
                n_estimators=1,
                bootstrap=True,
                oob_score=True,
                random_state=rng,
            )
            clf.fit(X_train, y_train)


def _test_aot_autograd_forwards_backwards_helper_mod(
    helper_f, compiled_helper_f, args_tuple, assert_raises_regex_fn, assert_equals_fn,
    try_check_data_specialization, skip_correctness_check=False):

    def call_forwards_backwards(f, args):
        flat_args = pytree.arg_tree_leaves(*args)
        diff_args = [arg for arg in flat_args if isinstance(arg, torch.Tensor) and
                     arg.requires_grad]
        out = wrapper_set_seed(f, args_tuple)
        flat_out = pytree.tree_leaves(out)

        sm = 0
        for i in flat_out:
            if isinstance(i, torch.Tensor):
                # We need to call .abs() because it is possible that the output of the
                # operator is a complex Tensor and autograd will yell at autograd.grad
                # on a complex Tensor unless we manually provide the grad_output flag.
                sm += i.sum().abs()
        assert isinstance(sm, torch.Tensor)
        return out, torch.autograd.grad(sm, diff_args, allow_unused=True)

    def check(args, ignore_failure=False):
        try:
            orig_out, orig_grad = call_forwards_backwards(helper_f, args_tuple)
        except Exception:
            if ignore_failure:
                return
            raise

        # See https://github.com/pytorch/pytorch/pull/98960#issuecomment-1505962215
        tensor_args = [x for x in pytree.tree_flatten(args_tuple)[0] if isinstance(x, torch.Tensor)]
        any_non_leaves = any(x.grad_fn is not None for x in tensor_args)
        if all(x is None for x in orig_grad) and any_non_leaves:
            with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
                call_forwards_backwards(compiled_helper_f, args_tuple)
            return

        msg = (
            "Gradients of the operator are different in eager-mode PyTorch vs "
            "AOTAutograd. This means the operator will have incorrect gradients "
            "underneath torch.compile. This could be because the operator's "
            "backward is incorrectly registered or not traceable or that there "
            "is a bug in AOTAutograd."
        )

        compiled_out, compiled_grad = call_forwards_backwards(compiled_helper_f, args_tuple)
        if not skip_correctness_check:
            assert_equals_fn(compiled_out, orig_out, msg=outputs_msg)
            assert_equals_fn(compiled_grad, orig_grad, msg=msg)

    check(args_tuple, ignore_failure=False)

    # Randomize the data and run the traced graph with it, to catch bugs
    # where we may have baked in Tensor data into the trace.
    # This is not guaranteed to succeed, because `f` might have preconditions
    # on the values of the inputs, so we just ignore if this test fails.
    if try_check_data_specialization:
        args_tuple = randomize(args_tuple)
        check(args_tuple, ignore_failure=True)


def test_feed_last_modified_time_naive_date(self):
    """
    Tests the Last-Modified header with naive publication dates.
    """
    response = self.client.get("/syndication/naive-dates/")
    self.assertEqual(
        response.headers["Last-Modified"], "Tue, 26 Mar 2013 01:00:00 GMT"
    )


def test_remove_replaced_nodes(self):
    """
    Replaced nodes are properly removed and dependencies remapped.
    """
    # Add some dummy nodes to be replaced.
    graph = MigrationGraph()
    graph.add_dummy_node(
        key=("app_a", "0001"), origin="app_a.0002", error_message="BAD!"
    )
    graph.add_dummy_node(
        key=("app_a", "0002"), origin="app_b.0001", error_message="BAD!"
    )
    graph.add_dependency(
        "app_a.0002", ("app_a", "0002"), ("app_a", "0001"), skip_validation=True
    )
    # Add some normal parent and child nodes to test dependency remapping.
    graph.add_node(("app_c", "0001"), None)
    graph.add_node(("app_b", "0001"), None)
    graph.add_dependency(
        "app_a.0001", ("app_a", "0001"), ("app_c", "0001"), skip_validation=True
    )
    graph.add_dependency(
        "app_b.0001", ("app_b", "0001"), ("app_a", "0002"), skip_validation=True
    )
    # Try replacing before replacement node exists.
    msg = (
        "Unable to find replacement node ('app_a', '0001_squashed_0002'). It was "
        "either never added to the migration graph, or has been removed."
    )
    with self.assertRaisesMessage(NodeNotFoundError, msg):
        graph.remove_replaced_nodes(
            replacement=("app_a", "0001_squashed_0002"),
            replaced=[("app_a", "0001"), ("app_a", "0002")],
        )
    graph.add_node(("app_a", "0001_squashed_0002"), None)
    # Ensure `validate_consistency()` still raises an error at this stage.
    with self.assertRaisesMessage(NodeNotFoundError, "BAD!"):
        graph.validate_consistency()
    # Remove the dummy nodes.
    graph.remove_replaced_nodes(
        replacement=("app_a", "0001_squashed_0002"),
        replaced=[("app_a", "0001"), ("app_a", "0002")],
    )
    # Ensure graph is now consistent and dependencies have been remapped
    graph.validate_consistency()
    parent_node = graph.node_map[("app_c", "0001")]
    replacement_node = graph.node_map[("app_a", "0001_squashed_0002")]
    child_node = graph.node_map[("app_b", "0001")]
    self.assertIn(parent_node, replacement_node.parents)
    self.assertIn(replacement_node, parent_node.children)
    self.assertIn(child_node, replacement_node.children)
    self.assertIn(replacement_node, child_node.parents)


def test_unique_inheritance_filter_test(self):
    """
    Regression test for #14003: When using a ManyToMany in list_filter,
    results shouldn't appear more than once. Model managed in the
    admin inherits from the one that defines the relationship.
    """
    artist = Painter.objects.create(name="Pablo")
    group = ArtGroup.objects.create(name="The Masters")
    Membership.objects.create(association=group, artist=artist, role="lead painter")
    Membership.objects.create(association=group, artist=artist, role="sculptor")

    a = ArtGroupAdmin(ArtGroup, custom_site)
    request = self.factory.get("/art_group/", data={"creators": artist.pk})
    request.user = self.superuser

    cl = a.get_changelist_instance(request)
    cl.get_results(request)

    # There's only one ArtGroup instance
    self.assertEqual(cl.result_count, 1)
    # Queryset must be deletable.
    cl.queryset.delete()
    self.assertEqual(cl.queryset.count(), 0)


def verify_invalid_codes_input(self, codes_input, categories_or_dtype):
        if not isinstance(categories_or_dtype, CategoricalDtype):
            categories = categories_or_dtype.categories
        else:
            categories = categories_or_dtype
        msg = "codes need to be between "
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes=codes_input, categories=categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes=codes_input, dtype=categories_or_dtype)

    verify_invalid_codes_input([1, 2], CategoricalDtype(categories=[1, 2]))


def parse_dims_from_args(
    params: Union[List[Any], Tuple[List[Any], ...]]
) -> List[Any]:
    if params and isinstance(params[0], list):
        assert len(params) == 1
        params = cast(Tuple[List[Any]], params)
        return cast(List[Any], params[0])
    else:
        return cast(List[Any], params)


def modified_blackman(
    num_points: int,
    *,
    is_symmetric: bool = True,
    data_type: Optional[torch.dtype] = None,
    storage_layout: torch.layout = torch.strided,
    compute_device: Optional[torch.device] = None,
    requires_grad_flag: bool = False
) -> Tensor:
    if data_type is None:
        data_type = torch.get_default_dtype()

    modified_a_values = [0.42, 0.5, 0.08]
    _window_function_checks("blackman", num_points, data_type, storage_layout)

    return general_cosine(
        num_points,
        a=modified_a_values,
        sym=is_symmetric,
        dtype=data_type,
        layout=storage_layout,
        device=compute_device,
        requires_grad=requires_grad_flag
    )


def decide_clipboard():
    """
    Decide the OS/platform and set the copy() and paste() functions
    accordingly.
    """
    global Foundation, AppKit, qtpylib, PyQt4lib, PyQt5lib

    # Setup for the CYGWIN platform:
    if (
        "cygwin" in platform.system().lower()
    ):  # Cygwin has a variety of values returned by platform.system(),
        # such as 'CYGWIN_NT-6.1'
        # FIXME(pyperclip#55): pyperclip currently does not support Cygwin,
        # see https://github.com/asweigart/pyperclip/issues/55
        if os.path.exists("/dev/clipboard"):
            warnings.warn(
                "Pyperclip's support for Cygwin is not perfect, "
                "see https://github.com/asweigart/pyperclip/issues/55",
                stacklevel=find_stack_level(),
            )
            return init_dev_clipboard_clipboard()

    # Setup for the WINDOWS platform:
    elif os.name == "nt" or platform.system() == "Windows":
        return init_windows_clipboard()

    if platform.system() == "Linux":
        if _executable_exists("wslconfig.exe"):
            return init_wsl_clipboard()

    # Setup for the macOS platform:
    if os.name == "mac" or platform.system() == "Darwin":
        try:
            import AppKitlib
            import Foundationlib  # check if pyobjc is installed
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()

    # Setup for the LINUX platform:
    if HAS_DISPLAY:
        if os.environ.get("WAYLAND_DISPLAY") and _executable_exists("wl-copy"):
            return init_wl_clipboard()
        if _executable_exists("xsel"):
            return init_xsel_clipboard()
        if _executable_exists("xclip"):
            return init_xclip_clipboard()
        if _executable_exists("klipperlib") and _executable_exists("qdbus"):
            return init_klipper_clipboard()

        try:
            # qtpy is a small abstraction layer that lets you write applications
            # using a single api call to either PyQt or PySide.
            # https://pypi.python.org/project/QtPy
            import qtpylib  # check if qtpy is installed
        except ImportError:
            # If qtpy isn't installed, fall back on importing PyQt4.
            try:
                import PyQt5lib  # check if PyQt5 is installed
            except ImportError:
                try:
                    import PyQt4lib  # check if PyQt4 is installed
                except ImportError:
                    pass  # We want to fail fast for all non-ImportError exceptions.
                else:
                    return init_qt_clipboard()
            else:
                return init_qt_clipboard()
        else:
            return init_qt_clipboard()

    return init_no_clipboard()


def test_lasso_lars_fit_copyX_behaviour(copy_X):
    """
    Test that user input to .fit for copy_X overrides default __init__ value

    """
    lasso_lars = LassoLarsIC(precompute=False)
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 5))
    X_copy = X.copy()
    y = X[:, 2]
    lasso_lars.fit(X, y, copy_X=copy_X)
    assert copy_X == np.array_equal(X, X_copy)


def verifyDifferentiableGraph(self, network, expectedAutodiffNode, nonFusibleNodes, fusibleNodes):
        diffNodes = network.findAllNodes('prim::DifferentiableGraph')
        diffSubgraphs = [node.g('Subgraph') for node in diffNodes]

        # Note: currently no tests have fusible_nodes
        fusionNodes = list(chain.from_iterable([g.findAllNodes('prim::FusionGroup') for g in diffSubgraphs]))
        fusionSubgraphs = [node.g('Subgraph') for node in fusionNodes]

        # For any non-fusible node, it must show up in one of the DifferentiableGraphs.
        nodesInDiffGraph = []
        nodesNotInDiffGraph = []
        nonFusibleNodesBeingFused = []
        for node in nonFusibleNodes:
            if any(g.findNode(node) is not None for g in diffSubgraphs):
                nodesInDiffGraph.append(node)
            else:
                nodesNotInDiffGraph.append(node)
            if any(g.findNode(node) is not None for g in fusionSubgraphs):
                nonFusibleNodesBeingFused.append(node)
        foundAllNonFusibleNodes = len(nodesInDiffGraph) == len(nonFusibleNodes)

        # For any fusible node, it must show up in one of the FusionGroups in one of the DifferentiableGraphs.
        fusionNodesFound = []
        fusionNodesNotFound = []
        for node in fusibleNodes:
            if any(g.findNode(node) is not None for g in fusionSubgraphs):
                fusionNodesFound.append(node)
            else:
                fusionNodesNotFound.append(node)
        foundAllFusibleNodes = len(fusionNodesFound) == len(fusibleNodes)

        if expectedAutodiffNode is not None:
            errMsg = self.autoDiffErrorMessage(expectedAutodiffNode,
                                               nodesNotInDiffGraph,
                                               fusionNodesNotFound,
                                               nonFusibleNodesBeingFused,
                                               fusionNodesFound,
                                               nodesInDiffGraph)
            self.assertEqual(expectedAutodiffNode,
                             foundAllNonFusibleNodes and foundAllFusibleNodes, errMsg)


def forest_check_only(
    elem_type_or_types: Type[U],
    /,
    condition: Fn[U, bool],
    collection: Forest,
    is_root: Optional[Callable[[Forest], bool]] = None,
) -> bool:
    ...


def validate_rolling_window(series, expected_results, window_size, min_valid_points):
    # GH 11704
    expected_series = [Series(vals, index=index) for (vals, index) in expected_results]

    for exp, act in zip(expected_series, series.rolling(window=window_size, min_periods=min_valid_points)):
        assert_series_equal(act, exp)


def test_take_from_object(self):
    # Check exception taking from object array
    d = np.zeros(5, dtype=object)
    assert_raises(IndexError, d.take, [6])

    # Check exception taking from 0-d array
    d = np.zeros((5, 0), dtype=object)
    assert_raises(IndexError, d.take, [1], axis=1)
    assert_raises(IndexError, d.take, [0], axis=1)
    assert_raises(IndexError, d.take, [0])
    assert_raises(IndexError, d.take, [0], mode='wrap')
    assert_raises(IndexError, d.take, [0], mode='clip')


def _infer_signature_from_network(net):
    network_shapes = getattr(net, "_network_shapes", None)
    if not network_shapes:
        return None

    def create_input_spec(structure):
        if isinstance(structure, dict):
            spec_dict = {k: create_input_spec(v) for k, v in structure.items()}
            return spec_dict
        elif isinstance(structure, tuple):
            if all(isinstance(d, (int, type(None))) for d in structure):
                return layers.InputSpec(shape=(None,) + structure[1:], dtype=net.input_dtype)
            return tuple(create_input_spec(v) for v in structure)
        elif isinstance(structure, list):
            if all(isinstance(d, (int, type(None))) for d in structure):
                return layers.InputSpec(shape=[None] + structure[1:], dtype=net.input_dtype)
            return [create_input_spec(v) for v in structure]
        else:
            raise ValueError(f"Unsupported type {type(structure)} for {structure}")

    return [create_input_spec(value) for value in network_shapes.values()]


def _extract_dist_info(self) -> None:
    r"""
    Extract the process group and device information from the joinables.

    If there are multiple joinables, then the context manager uses the
    first specified device.

    Preconditions:
        ``self._joinables`` is not ``None`` and is non-empty.

    Raises:
        ValueError
            If there are multiple conflicting ``process_group`` attributes
            among the ``Joinable`` objects.
    """
    process_group = None
    device = None
    for joinable in self._joinables:
        if process_group is None:
            process_group = joinable.join_process_group
        elif process_group != joinable.join_process_group:
            raise ValueError(
                "Using join context manager with multiple process groups"
            )
        if device is None:
            device = joinable.join_device
    self._process_group = process_group
    self._rank = dist.get_rank(self._process_group)
    self._device = device


def verify_place_unset_for_underground_bar(self):
        """
        Regression for #13839 and #17439.

        The target of a one-to-one relation is always cached.
        """
        underground_bar = UndergroundBar(place=self.location, serves_cocktails=True)
        underground_bar.save()
        self.assertNumQueries(0, lambda: setattr(self.location, 'undergroundbar', underground_bar))
        underground_bar.place = None
        underground_bar.save()
        with self.assertNumQueries(0):
            self.assertIsNone(getattr(self.location, 'undergroundbar', None))




from typing import Optional, List

def extract_reactivated_issues(text: Optional[str]) -> List[int]:
    if text is None:
        return []

    reactivated_keywords = ["close", "fix", "resolve"]
    issue_ids = []

    for keyword in reactivated_keywords:
        matches = [match[5:] for match in text.split() if match.startswith(keyword + " #")]
        issue_ids.extend(matches)

    return list(set(issue_ids))


def test_weighted_percentile():
    y = np.empty(102, dtype=np.float64)
    y[:50] = 0
    y[-51:] = 2
    y[-1] = 100000
    y[50] = 1
    sw = np.ones(102, dtype=np.float64)
    sw[-1] = 0.0
    score = _weighted_percentile(y, sw, 50)
    assert approx(score) == 1




def test_getitem_ix_mixed_integer2(self):
    # 11320
    df = DataFrame(
        {
            "rna": (1.5, 2.2, 3.2, 4.5),
            -1000: [11, 21, 36, 40],
            0: [10, 22, 43, 34],
            1000: [0, 10, 20, 30],
        },
        columns=["rna", -1000, 0, 1000],
    )
    result = df[[1000]]
    expected = df.iloc[:, [3]]
    tm.assert_frame_equal(result, expected)
    result = df[[-1000]]
    expected = df.iloc[:, [1]]
    tm.assert_frame_equal(result, expected)


def decompose_customized_functionalized(network):
    """Decomposes customized_functionalized nodes into clones and the underlying
    mutation node.

    We assume that the reinplacing pass runs before this; the reinplacing pass
    tells us (via rewriting the arguments or .meta to those nodes) which
    Tensors we should clone and which Tensors are safe to reinplace.
    """
    network_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.revised_order.customized_functionalized),
        pass_dict=network_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._revised_order_ops.customized_functionalize import customized_functionalized_dense

        only_clone_these_tensors = tuple(
            match.nodes[0].meta.get("only_clone_these_tensors", [])
        )

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            assert len(args) == 1
            mode = args[0]
            return customized_functionalized_dense(mode, only_clone_these_tensors, **kwargs)

        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.revised_order.customized_functionalized_v2),
        pass_dict=network_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._revised_order_ops.customized_functionalize import (
            customized_functionalized_v2_dense,
        )

        only_clone_these_bases = tuple(
            match.nodes[0].meta.get("only_clone_these_tensors", [])
        )

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            assert len(args) == 1
            mutable_op = args[0]
            return customized_functionalized_v2_dense(
                mutable_op, only_clone_these_bases, **kwargs
            )

        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    network_pass.apply(network)

    for _ in network.find_nodes(
        op="call_function", target=torch.ops.revised_order.customized_functionalized
    ):
        raise AssertionError("customized_functionalized was not removed")

    for _ in network.find_nodes(
        op="call_function",
        target=torch.ops.revised_order.customized_functionalized_v2,
    ):
        raise AssertionError("customized_functionalized_v2 was not removed")


def _get_conv_bn_pattern_nodes(r: ReplacedPatterns) -> Dict[str, Tuple[Node, Node]]:
    """
    Helper function to extract the nodes in the conv-bn fusion pattern after
    subgraph rewriting, in the form of a map:

        {name: (original_node, replacement_node)}

    The following names must exist in the map:

        "conv", "conv_weight", "conv_input", "bn", "getitem"

    The following names may exist in the map:

        "conv_weight_q", "conv_weight_dq", "conv_bias",
        "conv_bias_q", "conv_bias_dq"
    """

    def _get_nodes(nodes: List[Node]) -> Tuple[Node, Node, Optional[Node]]:
        """
        Return a 3-tuple of (conv_node, bn_node, getitem_node).
        This asserts that the match contains exactly one of each node.
        """
        conv_node, bn_node, getitem_node = None, None, None
        for n in nodes:
            if n.op != "call_function":
                continue
            if _is_conv_or_conv_transpose_node(n):
                assert conv_node is None
                conv_node = n
            if _is_bn_node(n):
                assert bn_node is None
                bn_node = n
            if n.target == operator.getitem:
                assert getitem_node is None
                getitem_node = n
        assert conv_node is not None
        assert bn_node is not None
        return (conv_node, bn_node, getitem_node)

    def _get_q_dq_nodes(n: Node) -> Tuple[Node, Node, Node]:
        """
        Return a 3-tuple of (orig_node, q_node, dq_node).
        """
        assert _is_dequantize(n)
        q_node = n.args[0]
        assert isinstance(q_node, Node)
        assert _is_quantize(q_node)
        orig_node = q_node.args[0]
        assert isinstance(orig_node, Node)
        return (orig_node, q_node, n)

    original_nodes = list(_filter_nodes_map(r.nodes_map).values())
    o_conv, o_bn, o_getitem = _get_nodes(original_nodes)
    r_conv, r_bn, r_getitem = _get_nodes(r.replacements)

    # Create the mapping from original node to replacement node
    assert o_getitem is None
    assert r_getitem is None
    mapping = {
        "conv": (o_conv, r_conv),
        "bn": (o_bn, r_bn),
    }

    # Extract conv input and weight
    # Note: here we extract the original nodes indirectly through the pattern nodes
    # because the args of the original nodes are no longer available after replacement
    (p_conv, _, _) = _get_nodes(list(r.nodes_map.keys()))
    (p_conv_input, p_conv_weight, *_) = p_conv.args
    (r_conv_input, r_conv_weight, *_) = r_conv.args
    assert isinstance(p_conv_input, Node)
    assert isinstance(p_conv_weight, Node)
    assert isinstance(r_conv_input, Node)
    assert isinstance(r_conv_weight, Node)
    o_conv_input = r.nodes_map[p_conv_input]
    o_conv_weight = r.nodes_map[p_conv_weight]

    # If conv weight is quantized, extract the q - dq nodes
    if _is_dequantize(p_conv_weight):
        p_conv_weight, p_conv_weight_q, p_conv_weight_dq = _get_q_dq_nodes(
            p_conv_weight
        )
        r_conv_weight, r_conv_weight_q, r_conv_weight_dq = _get_q_dq_nodes(
            r_conv_weight
        )
        o_conv_weight = r.nodes_map[p_conv_weight]
        o_conv_weight_q = r.nodes_map[p_conv_weight_q]
        o_conv_weight_dq = r.nodes_map[p_conv_weight_dq]
        mapping["conv_weight_q"] = (o_conv_weight_q, r_conv_weight_q)
        mapping["conv_weight_dq"] = (o_conv_weight_dq, r_conv_weight_dq)
    mapping["conv_input"] = (o_conv_input, r_conv_input)
    mapping["conv_weight"] = (o_conv_weight, r_conv_weight)

    # Extract conv bias
    if len(p_conv.args) > 2 and len(r_conv.args) > 2:
        p_conv_bias = p_conv.args[2]
        r_conv_bias = r_conv.args[2]
        assert isinstance(p_conv_bias, Node)
        assert isinstance(r_conv_bias, Node)
        o_conv_bias = r.nodes_map[p_conv_bias]

        # If conv bias is quantized, extract the q - dq nodes
        if _is_dequantize(p_conv_bias):
            p_conv_bias, p_conv_bias_q, p_conv_bias_dq = _get_q_dq_nodes(p_conv_bias)
            r_conv_bias, r_conv_bias_q, r_conv_bias_dq = _get_q_dq_nodes(r_conv_bias)
            o_conv_bias = r.nodes_map[p_conv_bias]
            o_conv_bias_q = r.nodes_map[p_conv_bias_q]
            o_conv_bias_dq = r.nodes_map[p_conv_bias_dq]
            mapping["conv_bias_q"] = (o_conv_bias_q, r_conv_bias_q)
            mapping["conv_bias_dq"] = (o_conv_bias_dq, r_conv_bias_dq)
        mapping["conv_bias"] = (o_conv_bias, r_conv_bias)
    return mapping


op_db: List[OpInfo] = [
    OpInfo(
        "linalg.cross",
        ref=lambda x, y, dim=-1: np.cross(x, y, axis=dim),
        op=torch.linalg.cross,
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        aten_name="linalg_cross",
        sample_inputs_func=sample_inputs_cross,
        error_inputs_func=error_inputs_cross,
        supports_out=True,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    OpInfo(
        "linalg.det",
        aten_name="linalg_det",
        op=torch.linalg.det,
        aliases=("det",),
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        check_batched_gradgrad=False,
    ),
    OpInfo(
        "linalg.det",
        aten_name="linalg_det",
        op=torch.linalg.det,
        variant_test_name="singular",
        aliases=("det",),
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_det_singular,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        skips=(
            DecorateInfo(
                unittest.skip("The backward may give different results"),
                "TestCommon",
                "test_noncontiguous_samples",
            ),
            DecorateInfo(
                unittest.skip("Gradients are incorrect on macos"),
                "TestBwdGradients",
                "test_fn_grad",
                device_type="cpu",
                dtypes=(torch.float64,),
                active_if=IS_MACOS,
            ),
            DecorateInfo(
                unittest.skip("Gradients are incorrect on macos"),
                "TestFwdGradients",
                "test_forward_mode_AD",
                device_type="cpu",
                dtypes=(torch.float64,),
                active_if=IS_MACOS,
            ),
            # Both Hessians are incorrect on complex inputs??
            DecorateInfo(
                unittest.expectedFailure,
                "TestBwdGradients",
                "test_fn_gradgrad",
                dtypes=(torch.complex128,),
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                dtypes=(torch.complex128,),
            ),
            DecorateInfo(
                unittest.skip("Skipped, see https://github.com//issues/84192"),
                "TestBwdGradients",
                "test_fn_gradgrad",
                device_type="cuda",
            ),
            DecorateInfo(
                unittest.skip("Skipped, see https://github.com//issues/84192"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
            ),
            DecorateInfo(
                unittest.skip(
                    "Flaky on ROCm https://github.com/pytorch/pytorch/issues/93044"
                ),
                "TestBwdGradients",
                "test_fn_grad",
                device_type="cuda",
                dtypes=get_all_complex_dtypes(),
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip(
                    "Flaky on ROCm https://github.com/pytorch/pytorch/issues/93045"
                ),
                "TestFwdGradients",
                "test_forward_mode_AD",
                device_type="cuda",
                dtypes=get_all_complex_dtypes(),
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.diagonal",
        aten_name="linalg_diagonal",
        aten_backward_name="diagonal_backward",
        dtypes=all_types_and_complex_and(
            torch.bool, torch.bfloat16, torch.float16, torch.chalf
        ),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_diagonal_diag_embed,
        error_inputs_func=error_inputs_diagonal_diag_embed,
    ),
    OpInfo(
        "linalg.cholesky",
        aten_name="linalg_cholesky",
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_cholesky,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.cholesky_ex",
        aten_name="linalg_cholesky_ex",
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_cholesky,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.vecdot",
        aten_name="linalg_vecdot",
        ref=lambda x, y, *, dim=-1: (x.conj() * y).sum(dim),
        dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_linalg_vecdot,
        check_batched_forward_grad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
            DecorateInfo(
                toleranceOverride({torch.half: tol(atol=1.2e-2, rtol=1.7e-2)}),
                "TestInductorOpInfo",
                "test_comprehensive",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.cond",
        aten_name="linalg_cond",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_cond,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.eig",
        aten_name="linalg_eig",
        op=torch.linalg.eig,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eig,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # AssertionError: Scalars are not equal!
            DecorateInfo(
                unittest.expectedFailure, "TestCommon", "test_out", device_type="cpu"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, with_tf32_off],
    ),
    OpInfo(
        "linalg.eigvals",
        aten_name="linalg_eigvals",
        op=torch.linalg.eigvals,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.eigh",
        aten_name="linalg_eigh",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eigh,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.eigvalsh",
        aten_name="linalg_eigvalsh",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eigh,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # Pre-existing condition; Needs to be fixed
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.householder_product",
        aten_name="linalg_householder_product",
        op=torch.linalg.householder_product,
        aliases=("orgqr",),
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        # TODO: backward uses in-place operations that vmap doesn't like
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_householder_product,
        decorators=[
            skipCUDAIfNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.complex64: tol(atol=1e-3, rtol=1e-3)})
            ),
            DecorateInfo(
                unittest.skip("Skipped! Flaky"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cpu",
                dtypes=(torch.complex128,),
            ),
        ],
    ),
    OpInfo(
        "linalg.ldl_factor",
        aten_name="linalg_ldl_factor",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_factor,
        decorators=[skipCUDAIfNoMagmaAndNoLinalgsolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.ldl_factor_ex",
        aten_name="linalg_ldl_factor_ex",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_factor,
        decorators=[skipCUDAIfNoMagmaAndNoLinalgsolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.ldl_solve",
        aten_name="linalg_ldl_solve",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_solve,
        decorators=[
            skipCUDAIf(
                _get_torch_cuda_version() < (11, 4), "not available before CUDA 11.3.1"
            ),
            skipCUDAIfNoCusolver,
            skipCUDAIfRocm,
            skipCPUIfNoLapack,
        ],
    ),
    OpInfo(
        "linalg.lstsq",
        aten_name="linalg_lstsq",
        dtypes=floating_and_complex_types(),
        supports_out=True,
        sample_inputs_func=sample_inputs_linalg_lstsq,
        error_inputs_func=error_inputs_lstsq,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # we skip gradient checks for this suite as they are tested in
            # variant_test_name='grad_oriented'
            DecorateInfo(unittest.skip("Skipped!"), "TestFwdGradients"),
            DecorateInfo(unittest.skip("Skipped!"), "TestBwdGradients"),
            # The values for attribute 'shape' do not match
            DecorateInfo(unittest.skip("Skipped!"), "TestCommon", "test_out"),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.lstsq",
        aten_name="linalg_lstsq",
        variant_test_name="grad_oriented",
        # gradchecks for forward AD fails with multi-Tensor outputs
        op=lambda a, b, driver: torch.linalg.lstsq(a, b, driver=driver)[0],
        supports_out=False,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_lstsq,
        error_inputs_func=error_inputs_lstsq_grad_oriented,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_autograd=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # tests do not work with passing lambda for op
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
        ),
    ),
    OpInfo(
        "linalg.matrix_power",
        aliases=("matrix_power",),
        aten_name="linalg_matrix_power",
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_inplace_autograd=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_matrix_power,
    ),
    OpInfo(
        "linalg.multi_dot",
        # Need this lambda because gradcheck does not work with TensorList inputs
        aten_name="linalg_multi_dot",
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
        supports_inplace_autograd=False,
        # Batched grad checks fail for empty input tensors (see https://github.com/pytorch/pytorch/issues/53407)
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # https://github.com/pytorch/pytorch/issues/66357
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_multi_dot,
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        skips=(
            # https://github.com/pytorch/pytorch/issues/67470
            DecorateInfo(
                unittest.skip("67470!"), "TestCommon", "test_noncontiguous_samples"
            ),
            # Fails on XLA.
            # AssertionError: False is not true : Tensors failed to compare as equal!
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestOpInfo",
                device_type="xla",
                dtypes=(torch.long,),
            ),
            # https://github.com/pytorch/pytorch/issues/71774
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestNNCOpInfo",
                "test_nnc_correctness",
                device_type="cpu",
                dtypes=(torch.long,),
            ),
        ),
    ),
    # NB: linalg.norm has two variants so that different skips can be used for different sample inputs
    OpInfo(
        "linalg.norm",
        aten_name="linalg_norm",
        op=torch.linalg.norm,
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_norm,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_gradgrad"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.norm",
        op=torch.linalg.norm,
        variant_test_name="subgradients_at_zero",
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=partial(
            sample_inputs_linalg_norm, variant="subgradient_at_zero"
        ),
        aten_name="linalg_norm",
        supports_forward_ad=True,
        # torch.autograd.gradcheck.GradcheckError: While computing batched gradients, got:
        # Could not allocate memory to change Tensor SizesAndStrides!
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        skips=(
            # [NEW] Skips specifically for sample inputs at zero
            # norm's vjp/jvp are not well-conditioned near zero
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_gradgrad"
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestFwdGradients", "test_fn_fwgrad_bwgrad"
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestFwdGradients", "test_forward_mode_AD"
            ),
            DecorateInfo(unittest.expectedFailure, "TestBwdGradients", "test_fn_grad"),
        ),
    ),
    OpInfo(
        "linalg.matrix_norm",
        aten_name="linalg_matrix_norm",
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        check_batched_gradgrad=False,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_matrix_norm,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.qr",
        aten_name="linalg_qr",
        op=torch.linalg.qr,
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # In-place ops
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_qr_geqrf,
        decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.slogdet",
        aten_name="linalg_slogdet",
        op=torch.linalg.slogdet,
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.vander",
        aten_name="linalg_vander",
        ref=np_vander_batched,
        op=torch.linalg.vander,
        dtypes=all_types_and_complex(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
        sample_inputs_func=sample_inputs_linalg_vander,
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    ReductionOpInfo(
        "linalg.vector_norm",
        op=torch.linalg.vector_norm,
        identity=0,
        nan_policy="propagate",
        supports_multiple_dims=True,
        complex_to_real=True,
        supports_forward_ad=True,
        # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
        # got: Could not allocate memory to change Tensor SizesAndStrides!
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        generate_args_kwargs=sample_kwargs_vector_norm,
        aten_name="linalg_vector_norm",
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
        ),
    ),
    OpInfo(
        "linalg.lu_factor",
        aten_name="linalg_lu_factor",
        op=torch.linalg.lu_factor,
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu_factor_ex",
        aten_name="linalg_lu_factor_ex",
        op=torch.linalg.lu_factor_ex,
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu",
        aten_name="linalg_lu",
        op=torch.linalg.lu,
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        # Runs very slowly on slow-gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu_solve",
        op=torch.linalg.lu_solve,
        aten_name="linalg_lu_solve",
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_lu_solve,
        skips=(
            DecorateInfo(
                unittest.skip("Tests different backward paths"),
                "TestCommon",
                "test_floating_inputs_are_differentiable",
            ),
        ),
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
    ),
    OpInfo(
        "linalg.inv",
        aten_name="linalg_inv",
        op=torch.linalg.inv,
        aliases=("inverse",),
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.inv_ex",
        aten_name="linalg_inv_ex",
        op=torch.linalg.inv_ex,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve",
        aten_name="linalg_solve",
        op=torch.linalg.solve,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[
            skipCUDAIfNoMagmaAndNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1.3e-05, rtol=6e-04)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cpu",
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve_ex",
        aten_name="linalg_solve_ex",
        op=torch.linalg.solve_ex,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[
            skipCUDAIfNoMagmaAndNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1.3e-05, rtol=6e-04)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cpu",
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve_triangular",
        aten_name="linalg_solve_triangular",
        op=torch.linalg.solve_triangular,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve_triangular,
        supports_fwgrad_bwgrad=True,
        skips=(skipCPUIfNoLapack,),
        # linalg.solve_triangular cannot be batched over because of a call to out.copy_(result);
        supports_forward_ad=True,
    ),
    OpInfo(
        "linalg.matrix_rank",
        aten_name="linalg_matrix_rank",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_matrix_rank,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            # jit doesn't accept tensor inputs for matrix rank
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=[torch.complex64, torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.matrix_rank",
        aten_name="linalg_matrix_rank",
        variant_test_name="hermitian",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        op=torch.linalg.pinv,
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_pinv,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # errors with "leaked XXXX bytes CUDA memory on device 0"
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        variant_test_name="singular",
        # pinv is Frechet-differentiable in a rank-preserving neighborhood,
        # so we feed inputs that are the products of two full-rank factors,
        # to avoid any rank changes caused by the perturbations in the gradcheck
        op=lambda a, b: torch.linalg.pinv(a @ b.mT),
        dtypes=floating_and_complex_types(),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_pinv_singular,
        # Only large tensors show issues with implicit backward used prior to
        # explicit backward implementation.
        decorators=[slowTest, skipCUDAIfNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # CUDA runs out of memory
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
                dtypes=[torch.cdouble],
            ),
            # This test takes almost 2 hours to run!
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestBwdGradients",
                "test_fn_gradgrad",
                device_type="cuda",
                dtypes=[torch.cdouble],
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        variant_test_name="hermitian",
        dtypes=floating_and_complex_types(),
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-5, rtol=1e-5)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cuda",
            ),
            # This test is flaky under slow gradcheck, likely due to rounding issues
            DecorateInfo(
                skipIfSlowGradcheckEnv,
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.svd",
        op=torch.linalg.svd,
        aten_name="linalg_svd",
        decomp_aten_name="_linalg_svd",
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow-gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        # We're using at::allclose, which does not have a batching rule
        check_batched_grad=False,
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_svd,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.svdvals",
        op=torch.linalg.svdvals,
        aten_name="linalg_svdvals",
        decomp_aten_name="_linalg_svd",
        dtypes=floating_and_complex_types(),
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        # We're using at::allclose, which does not have a batching rule
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_svdvals,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.tensorinv",
        ref=np.linalg.tensorinv,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_tensorinv,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    OpInfo(
        "linalg.tensorsolve",
        ref=lambda a, b, dims=None: np.linalg.tensorsolve(a, b, axes=dims),
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_tensorsolve,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[
            skipCUDAIfNoMagmaAndNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-03, rtol=1e-03)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cuda",
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=8e-04, rtol=7e-06)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cpu",
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
]

python_ref_db: List[OpInfo] = [
    #
    # torch.linalg
    #
    PythonRefInfo(
        "_refs.linalg.cross",
        torch_opinfo_name="linalg.cross",
        supports_out=True,
        op_db=op_db,
        skips=(
            # TODO: is this really needed?
            DecorateInfo(
                unittest.expectedFailure, "TestCommon", "test_python_ref_errors"
            ),
        ),
    ),
    PythonRefInfo(
        "_refs.linalg.diagonal",
        torch_opinfo_name="linalg.diagonal",
        supports_out=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.vecdot",
        torch_opinfo_name="linalg.vecdot",
        op_db=op_db,
    ),
    ReductionPythonRefInfo(
        "_refs.linalg.vector_norm",
        torch_opinfo_name="linalg.vector_norm",
        supports_out=True,
        op_db=op_db,
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
        ),
    ),
    PythonRefInfo(
        "_refs.linalg.matrix_norm",
        torch_opinfo_name="linalg.matrix_norm",
        supports_out=True,
        # Uses vector_norm inside and vector_norm is affected by
        # https://github.com/pytorch/pytorch/issues/77216
        validate_view_consistency=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.norm",
        torch_opinfo_name="linalg.norm",
        supports_out=True,
        # Uses vector_norm inside and vector_norm is affected by
        # https://github.com/pytorch/pytorch/issues/77216
        validate_view_consistency=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.svd",
        torch_opinfo_name="linalg.svd",
        supports_out=True,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.svdvals",
        torch_opinfo_name="linalg.svdvals",
        supports_out=True,
        op_db=op_db,
    ),
]
