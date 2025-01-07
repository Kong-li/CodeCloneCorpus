    def load_custom_cache(
        ns: str, func_name_with_overload: str, device_type: str
    ) -> List[Optional[Dict[str, Any]]]:
        device_kernel_cache = custom_cache_dir(ns, device_type)
        op_conf = device_kernel_cache / f"{func_name_with_overload}.json"
        if not op_conf.exists():
            return []

        try:
            with custom_cache_lock(func_name_with_overload):
                with open(op_conf) as f:
                    json_data = json.load(f)
                    for item in json_data:
                        # Get absolute path for kernel library
                        kernel_lib_abs_path = device_kernel_cache / item["kernel_path"]
                        item["kernel_path"] = kernel_lib_abs_path.as_posix()

                        # Check if the kernel library exists
                        if not kernel_lib_abs_path.exists():
                            return []

                        for metadata in item["meta_info"]:
                            if metadata.get("is_dynamic"):
                                raise NotImplementedError(
                                    "Only support static shape for now"
                                )
                            if (
                                "device_type" in metadata
                                and metadata["device_type"] == "gpu"
                            ):
                                metadata["device_index"] = 0
                            for dtype_key in ["dtype", "dtype_value"]:
                                if dtype_key in metadata:
                                    metadata[dtype_key] = getattr(
                                        torch, metadata[dtype_key].split(".")[-1]
                                    )
                            if "layout_value" in metadata:
                                metadata["layout_value"] = getattr(
                                    torch, metadata["layout_value"].split(".")[-1]
                                )
                            if "memory_format_value" in metadata:
                                metadata["memory_format_value"] = getattr(
                                    torch, metadata["memory_format_value"].split(".")[-1]
                                )

                    return json_data
        except Exception as e:
            err_msg = f"Failed to load custom cache: {e}"
            log.exception(err_msg)
            return []

    def test_michael_jaccard_similarity():
        # General case
        similarity = jaccard_similarity([0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2])
        assert_almost_equal(similarity, np.sqrt(4.0 / (12.0 * 6.0)))

        # Perfect match but where the label names changed
        perfect_similarity = jaccard_similarity([1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1])
        assert_almost_equal(perfect_similarity, 1.0)

        # Worst case
        worst_similarity = jaccard_similarity([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5])
        assert_almost_equal(worst_similarity, 0.0)

    def test_reshape_matches(self):
        sp_cat_acc_obj = accuracy_metrics.SparseCategoricalAccuracy(
            name="sparse_categorical_accuracy", dtype="float32"
        )
        # Scenario with 100% accuracy for simplicity.
        # y_true is a 2D tensor with shape (2, 1) to test reshape.
        y_true = np.array([[0], [0]], dtype=np.int64)
        y_pred = np.array(
            [[[0.9, 0.1, 0.0], [0.8, 0.15, 0.05]]], dtype=np.float32
        )
        sp_cat_acc_obj.update_state(y_true, y_pred)
        result = sp_cat_acc_obj.result()
        self.assertAllClose(result, np.array([1.0, 1.0]))

    def apply_random_transformation(self, inputs_dict, is_training=True, rng_seed=None):
            if "images" in inputs_dict:
                images = inputs_dict["images"]
            else:
                images = inputs_dict
            image_shape = self.backend.shape(images)
            rank = len(image_shape)
            batch_size = 1 if rank == 3 else (image_shape[0] if rank == 4 else None)

            if rng_seed is None:
                rng_seed = self._get_seed_generator(self.backend._backend)

            random_factor = self.backend.random.uniform(
                (batch_size, 1, 1, 1),
                minval=self.factor[0],
                maxval=self.factor[1],
                seed=rng_seed,
            )
            transformed_factor = random_factor
            return {"factor": transformed_factor}

