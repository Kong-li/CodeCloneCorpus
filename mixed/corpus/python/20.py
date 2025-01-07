def calculate_growth_rate_financials(self):
    # GH#12345
    revenues = DataFrame(
        [np.arange(0, 60, 20), np.arange(0, 60, 20), np.arange(0, 60, 20)]
    ).astype(np.float64)
    revenues.iat[1, 0] = np.nan
    revenues.iat[1, 1] = np.nan
    revenues.iat[2, 3] = 80

    for axis in range(2):
        expected = revenues / revenues.shift(axis=axis) - 1
        result = revenues.growth_rate(axis=axis)
        tm.assert_frame_equal(result, expected)

def _remove_previous_dequantize_in_custom_module(
    node: Node, prev_node: Node, graph: Graph
) -> None:
    """
    Given a custom module `node`, if the previous node is a dequantize, reroute the custom as follows:

    Before: quantize - dequantize - custom_module
    After: quantize - custom_module
                 \\ - dequantize
    """
    # expecting the input node for a custom module node to be a Node
    assert isinstance(
        prev_node, Node
    ), f"Expecting the argument for custom module node to be a Node, but got {prev_node}"
    if prev_node.op == "call_method" and prev_node.target == "dequantize":
        node.replace_input_with(prev_node, prev_node.args[0])
        # Remove the dequantize node if it doesn't have other users
        if len(prev_node.users) == 0:
            graph.erase_node(prev_node)

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

def transform_module_pairs(paired_modules_set):
    """Transforms module pairs larger than two into individual module pairs."""
    transformed_list = []

    for pair in paired_modules_set:
        if len(pair) == 1:
            raise ValueError("Each item must contain at least two modules")
        elif len(pair) == 2:
            transformed_list.append(pair)
        else:
            for i in range(len(pair) - 1):
                module_1, module_2 = pair[i], pair[i + 1]
                transformed_list.append((module_1, module_2))

    return transformed_list

def verify_custom_product_sk_not_named_id(self):
        """
        {% get_admin_log %} works if the product model's primary key isn't named
        'id'.
        """
        context = Context(
            {
                "user": CustomIdSeller(),
                "log_entries": LogEntry.objects.all(),
            }
        )
        template = Template(
            "{% load log %}{% get_admin_log 10 as admin_log for_user user %}"
        )
        # This template tag just logs.
        self.assertEqual(template.render(context), "")

def verify_dense(
        self,
        layer_type,
        numpy_op,
        init_params={},
        input_dims=(2, 4, 5),
        expected_output_dims=(2, 4, 5),
        **kwargs,
    ):
        self.test_layer_behavior(
            layer_type,
            init_params=init_params,
            input_data=[input_dims, input_dims],
            is_sparse=True,
            expected_output_shape=expected_output_dims,
            output_is_sparse=True,
            num_trainable_weights=0,
            num_non_trainable_weights=0,
            num_seed_generators=0,
            num_losses=0,
            supports_masking=True,
            train_model=False,
            mixed_precision_model=False,
        )

        layer = layer_type(**init_params)

        # Merging a sparse tensor with a dense tensor, or a dense tensor with a
        # sparse tensor produces a dense tensor
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            x1 = tf.SparseTensor([[0, 0], [1, 2]], [1.0, 2.0], (2, 3))
            x3 = tf.SparseTensor([[0, 0], [1, 1]], [4.0, 5.0], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            # Use n_batch of 1 to be compatible with all ops.
            x1 = jax_sparse.BCOO(([[1.0, 2.0]], [[[0], [2]]]), shape=(2, 3))
            x3 = jax_sparse.BCOO(([[4.0, 5.0]], [[[0], [1]]]), shape=(2, 3))
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        x1_np = backend.convert_to_numpy(x1)
        x2 = np.random.rand(2, 3)
        self.assertAllClose(layer([x1, x2]), numpy_op(x1_np, x2, **init_params))
        self.assertAllClose(layer([x2, x1]), numpy_op(x2, x1_np, **init_params))

        # Merging a sparse tensor with a sparse tensor produces a sparse tensor
        x3_np = backend.convert_to_numpy(x3)

        self.assertSparse(layer([x1, x3]))
        self.assertAllClose(layer([x1, x3]), numpy_op(x1_np, x3_np, **init_params))

def verify_checkbox_count_is_correct_after_page_navigation(self):
        from selenium.webdriver.common.by import By

        self.user_login(username="admin", password="pass")
        self.selenium.get(self.live_server_url + reverse("user:auth_user_list"))

        form_id = "#filter-form"
        first_row_checkbox_selector = (
            f"{form_id} #result_list tbody tr:first-child .select-row"
        )
        selection_indicator_selector = f"{form_id} .selected-count"
        selection_indicator = self.selenium.find_element(
            By.CSS_SELECTOR, selection_indicator_selector
        )
        row_checkbox = self.selenium.find_element(
            By.CSS_SELECTOR, first_row_checkbox_selector
        )
        # Select a row.
        row_checkbox.click()
        self.assertEqual(selection_indicator.text, "1 selected")
        # Go to another page and get back.
        self.selenium.get(
            self.live_server_url + reverse("user:custom_changelist_list")
        )
        self.selenium.back()
        # The selection indicator is synced with the selected checkboxes.
        selection_indicator = self.selenium.find_element(
            By.CSS_SELECTOR, selection_indicator_selector
        )
        row_checkbox = self.selenium.find_element(
            By.CSS_SELECTOR, first_row_checkbox_selector
        )
        selected_rows = 1 if row_checkbox.is_selected() else 0
        self.assertEqual(selection_indicator.text, f"{selected_rows} selected")

