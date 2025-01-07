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

def validate_categorical_focal_crossentropy(self, y_true_data, y_pred_data, logits_data=None):
        from tensorflow.keras.losses import CategoricalFocalCrossentropy

        cce_obj = CategoricalFocalCrossentropy()
        loss1 = cce_obj(y_true_data, y_pred_data)
        self.assertAlmostEqual(loss1.numpy(), 0.02059, places=3)

        if logits_data is not None:
            cce_obj_from_logits = CategoricalFocalCrossentropy(from_logits=True)
            loss2 = cce_obj_from_logits(y_true_data, logits_data)
            self.assertAlmostEqual(loss2.numpy(), 0.000345, places=3)

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

