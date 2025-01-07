    def example_feature_combination_getitem():
        """Check FeatureCombination.__getitem__ returns expected results."""
        scalar = MinMaxScaler()
        pca = KernelPCA()
        combination = FeatureCombination(
            [
                ("scalar", scalar),
                ("pca", pca),
                ("pass", "passthrough"),
                ("drop_me", "drop"),
            ]
        )
        assert combination["scalar"] is scalar
        assert combination["pca"] is pca
        assert combination["pass"] == "passthrough"
        assert combination["drop_me"] == "drop"

    def _reset_setting_properties(self, config, **kwargs):
        """Reset setting based property values."""
        if config == "UPLOAD_PATH":
            self.__dict__.pop("base_storage", None)
            self.__dict__.pop("storage_location", None)
        elif config == "UPLOAD_URL":
            self.__dict__.pop("base_url_path", None)
        elif config == "FILE_PERMISSIONS":
            self.__dict__.pop("file_mode_permissions", None)
        elif config == "DIRECTORY_PERMISSIONS":
            self.__dict__.pop("directory_mode_permissions", None)

    def _validate_factors(factors, num_components):
        """Validate the user provided 'factors'.

        Parameters
        ----------
        factors : array-like of shape (num_components,)
            The proportions of components of each mixture.

        num_components : int
            Number of components.

        Returns
        -------
        factors : array, shape (num_components,)
        """
        factors = check_array(factors, dtype=[np.float64, np.float32], ensure_2d=False)
        _validate_shape(factors, (num_components,), "factors")

        # check range
        if any(np.less(factors, 0.0)) or any(np.greater(factors, 1.0)):
            raise ValueError(
                "The parameter 'factors' should be in the range "
                "[0, 1], but got max value %.5f, min value %.5f"
                % (np.min(factors), np.max(factors))
            )

        # check normalization
        if not np.allclose(np.abs(1.0 - np.sum(factors)), 0.0):
            raise ValueError(
                "The parameter 'factors' should be normalized, but got sum(factors) = %.5f"
                % np.sum(factors)
            )
        return factors

    def fetch_auth_endpoint(self):
        """
        Override this method to override the auth_url attribute.
        """
        auth_url = self.auth_url or settings.AUTH_URL
        if not auth_url:
            raise ImproperlyConfigured(
                f"{self.__class__.__name__} is missing the auth_url attribute. Define "
                f"{self.__class__.__name__}.auth_url, settings.AUTH_URL, or override "
                f"{self.__class__.__name__}.fetch_auth_endpoint()."
            )
        return str(auth_url)

