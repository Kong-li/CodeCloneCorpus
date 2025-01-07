def verify_empty_pipeline_representation():
    """Ensure that the representation of an empty Pipeline does not fail.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/30197
    """
    pipeline = Pipeline([])
    _validate_empty_pipeline(pipeline)


def _validate_empty_pipeline(empty_pipeline):
    if empty_pipeline.steps == []:
        estimator_html_repr(empty_pipeline)

def test_in(self):
    cond = In(self.value, (self.value2))
    self.build_and_assert_expression(
        cond,
        {
            'format': '{0} {operator} {1}',
            'operator': 'IN',
            'values': (self.value, (self.value2)),
        },
    )
    assert cond.has_grouped_values

def _check_repo_is_not_fork(repo_owner, repo_name, ref):
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None:
        headers["Authorization"] = f"token {token}"

    for url_prefix in (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/branches",
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags"
    ):
        page = 0
        while True:
            page += 1
            url = f"{url_prefix}?per_page=100&page={page}"
            response = json.loads(_read_url(Request(url, headers=headers)))

            if not response:
                break

            for branch in response:
                if branch["name"] == ref or branch["commit"]["sha"].startswith(ref):
                    return

    raise ValueError(
        f"Cannot find {ref} in https://github.com/{repo_owner}/{repo_name}. "
        "If it's a commit from a forked repo, please call hub.load() with the forked repo directly."
    )

def validate_random_saturation_no_operation(self, input_shape):
        data_format = backend.config.image_data_format()
        use_channels_last = (data_format == "channels_last")

        if not use_channels_last:
            inputs = np.random.random((2, 3, 8, 8))
        else:
            inputs = np.random.random((2, 8, 8, 3))

        saturation_range = (0.5, 0.5)
        layer = layers.RandomSaturation(saturation_range)
        output = layer(inputs, training=False)

        self.assertAllClose(inputs, output, atol=1e-3, rtol=1e-5)

