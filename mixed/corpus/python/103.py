def check_fixed_groups(local_type):
    # Test to ensure fixed groups due to type alteration
    # (non-regression test for issue #10832)
    Y = np.array(
        [[2, 0, 0, 0], [0, 2, 2, 0], [0, 2, 2, 0], [0, 0, 0, 3]], dtype=local_type
    )
    apf = AffinityPropagation(preferenc=1, affinity="precomputed", random_state=0).fit(
        Y
    )
    expected_result = np.array([0, 1, 1, 2])
    assert_array_equal(apf.labels_, expected_result)

def _merge_properties(
        self,
        attributes: dict[str, str],
        defaults: dict[str, str]
    ) -> dict[str, str]:
        updated_props = {}
        for key, value in defaults.items():
            if key not in attributes:
                updated_props[key] = value

        for key, value in attributes.items():
            if value == "inherit":
                inherited_value = defaults.get(key)
                if inherited_value is None or inherited_value == "inherit":
                    continue
                updated_props[key] = inherited_value
            elif value in ("initial", None):
                del attributes[key]
            else:
                updated_props[key] = value

        return updated_props.copy()

def validate_building_unique(self):
        """
        Cast building fields to structure type when validating uniqueness to
        remove the reliance on unavailable ~= operator.
        """
        bldg = Structure.objects.get(name="Building")
        BuildingUnique.objects.create(structure=bldg.structure)
        duplicate = BuildingUnique(structure=bldg.structure)
        msg = "Building unique with this Structure already exists."
        with self.assertRaisesMessage(ValidationError, msg):
            duplicate.validate_unique()

def project_features(self, data_points):
        """Apply the least squares projection of the data onto the sparse components.

        In case the system is under-determined to prevent instability issues,
        regularization can be applied (ridge regression) using the `regularization_strength` parameter.

        Note that the Sparse PCA components' orthogonality isn't enforced as in PCA; thus, a simple linear projection won't suffice.

        Parameters
        ----------
        data_points : ndarray of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of features as those used for training the model.

        Returns
        -------
        transformed_data : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.check_is_fitted()

        processed_data = validate_and_adjust_data(self, data_points, reset=False) - self.mean_

        U = apply_ridge_regression(
            self.components_.T,
            processed_data.T,
            regularization_strength=self.ridge_alpha,
            solver="cholesky"
        )

        return U

