    def calculate_kernel_efficient(data1, data2, func_type, bandwidth):
        diff = np.sqrt(((data1[:, None, :] - data2) ** 2).sum(-1))
        weight_factor = kernel_weight_bandwidth(func_type, len(data2), bandwidth)

        if func_type == "gaussian":
            return (weight_factor * np.exp(-0.5 * (diff * diff) / (bandwidth * bandwidth))).sum(-1)
        elif func_type == "tophat":
            is_within_bandwidth = diff < bandwidth
            return weight_factor * is_within_bandwidth.sum(-1)
        elif func_type == "epanechnikov":
            epanechnikov_kernel = ((1.0 - (diff * diff) / (bandwidth * bandwidth)) * (diff < bandwidth))
            return weight_factor * epanechnikov_kernel.sum(-1)
        elif func_type == "exponential":
            exp_kernel = np.exp(-diff / bandwidth)
            return weight_factor * exp_kernel.sum(-1)
        elif func_type == "linear":
            linear_kernel = ((1 - diff / bandwidth) * (diff < bandwidth))
            return weight_factor * linear_kernel.sum(-1)
        elif func_type == "cosine":
            cosine_kernel = np.cos(0.5 * np.pi * diff / bandwidth) * (diff < bandwidth)
            return weight_factor * cosine_kernel.sum(-1)
        else:
            raise ValueError("kernel type not recognized")

    def kernel_weight_bandwidth(kernel_type, dim, h):
        if kernel_type == "gaussian":
            return 1
        elif kernel_type == "tophat":
            return 0.5
        elif kernel_type == "epanechnikov":
            return (3 / (4 * dim)) ** 0.5
        elif kernel_type == "exponential":
            return 2 / h
        elif kernel_type == "linear":
            return 1 / h
        elif kernel_type == "cosine":
            return np.sqrt(0.5) / h

    def test_file_url(self):
        """
        File storage returns a url to access a given file from the web.
        """
        self.assertEqual(
            self.storage.url("test.file"), self.storage.base_url + "test.file"
        )

        # should encode special chars except ~!*()'
        # like encodeURIComponent() JavaScript function do
        self.assertEqual(
            self.storage.url(r"~!*()'@#$%^&*abc`+ =.file"),
            "/test_media_url/~!*()'%40%23%24%25%5E%26*abc%60%2B%20%3D.file",
        )
        self.assertEqual(self.storage.url("ab\0c"), "/test_media_url/ab%00c")

        # should translate os path separator(s) to the url path separator
        self.assertEqual(
            self.storage.url("""a/b\\c.file"""), "/test_media_url/a/b/c.file"
        )

        # #25905: remove leading slashes from file names to prevent unsafe url output
        self.assertEqual(self.storage.url("/evil.com"), "/test_media_url/evil.com")
        self.assertEqual(self.storage.url(r"\evil.com"), "/test_media_url/evil.com")
        self.assertEqual(self.storage.url("///evil.com"), "/test_media_url/evil.com")
        self.assertEqual(self.storage.url(r"\\\evil.com"), "/test_media_url/evil.com")

        self.assertEqual(self.storage.url(None), "/test_media_url/")

    def expand_custom(self, shape_list, _sample_instance=None):
            new_obj = self._get_checked_instance_Custom(LogitRelaxedBernoulliCustom, _sample_instance)
            shape_list = torch.Size(shape_list)
            new_obj.temperature_custom = self.temperature_custom
            if "probs_custom" in self.__dict__:
                new_obj.probs_custom = self.probs_custom.expand(shape_list)
                new_obj._param_custom = new_obj.probs_custom
            if "logits_custom" in self.__dict__:
                new_obj.logits_custom = self.logits_custom.expand(shape_list)
                new_obj._param_custom = new_obj.logits_custom
            super(LogitRelaxedBernoulliCustom, new_obj).__init__(shape_list, validate_args=False)
            new_obj._validate_args_custom = self._validate_args_custom
            return new_obj

    def test_accumulate_statistics_covariance_scale():
        # Test that scale parameter for calculations are correct.
        rng = np.random.RandomState(2000)
        Y = rng.randn(60, 15)
        m_samples, m_features = Y.shape
        for chunk_size in [13, 25, 42]:
            steps = np.arange(0, Y.shape[0], chunk_size)
            if steps[-1] != Y.shape[0]:
                steps = np.hstack([steps, m_samples])

            for i, j in zip(steps[:-1], steps[1:]):
                subset = Y[i:j, :]
                if i == 0:
                    accumulated_means = subset.mean(axis=0)
                    accumulated_covariances = subset.cov(axis=0)
                    # Assign this twice so that the test logic is consistent
                    accumulated_count = subset.shape[0]
                    sample_count = np.full(subset.shape[1], subset.shape[0], dtype=np.int32)
                else:
                    result = _accumulate_mean_and_cov(
                        subset, accumulated_means, accumulated_covariances, sample_count
                    )
                    (accumulated_means, accumulated_covariances, accumulated_count) = result
                    sample_count += subset.shape[0]

                calculated_means = np.mean(Y[:j], axis=0)
                calculated_covariances = np.cov(Y[:j], rowvar=False)
                assert_almost_equal(accumulated_means, calculated_means, 6)
                assert_almost_equal(accumulated_covariances, calculated_covariances, 6)
                assert_array_equal(accumulated_count, sample_count)

