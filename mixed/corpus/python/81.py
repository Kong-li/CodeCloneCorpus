    def validate_namespace(ns: str) -> None:
        if "." in ns:
            raise ValueError(
                f'custom_op(..., ns="{ns}"): expected ns to not contain any . (and be a '
                f"valid variable name)"
            )
        if ns in RESERVED_NS:
            raise ValueError(
                f"custom_op(..., ns='{ns}'): '{ns}' is a reserved namespace, "
                f"please choose something else. "
            )

    def example_check(self, system, call):
            default_result = call(["users"]).output
            user_output = call(["users", "-s", "user"]).output
            assert default_result == user_output
            self.verify_sequence(
                ["login", "register", "profile"],
                call(["users", "-s", "status"]).output,
            )
            self.verify_sequence(
                ["profile", "login", "register"],
                call(["users", "-s", "priority"]).output,
            )
            match_sequence = [r.user for r in system.user_list.iter_users()]
            self.verify_sequence(match_sequence, call(["users", "-s", "match"]).output)

    def parse_config() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Upload configuration file to Bucket")
        parser.add_argument(
            "--project", type=str, required=True, help="Path to the project"
        )
        parser.add_argument(
            "--storage", type=str, required=True, help="S3 storage to upload config file to"
        )
        parser.add_argument(
            "--file-key",
            type=str,
            required=True,
            help="S3 key to upload config file to",
        )
        parser.add_argument("--preview", action="store_true", help="Preview run")
        args = parser.parse_args()
        # Sanitize the input a bit by removing s3:// prefix + trailing/leading
        # slashes
        if args.storage.startswith("s3://"):
            args.storage = args.storage[5:]
        args.storage = args.storage.strip("/")
        args.file_key = args.file_key.strip("/")
        return args

    def fit_transform(self, X, y=None, **params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

            .. versionchanged:: 1.5
                `**params` can now be routed via metadata routing API.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit_transform", **params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            for name, obj in self.transformer_list:
                if hasattr(obj, "fit_transform"):
                    routed_params[name] = Bunch(fit_transform={})
                    routed_params[name].fit_transform = params
                else:
                    routed_params[name] = Bunch(fit={})
                    routed_params[name] = Bunch(transform={})
                    routed_params[name].fit = params

        results = self._parallel_func(X, y, _fit_transform_one, routed_params)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        return self._hstack(Xs)

