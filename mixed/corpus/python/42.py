    def check_adjust_lr_on_training(self):
            adjust_lr = callbacks.LearningRateScheduler(
                schedule=lambda x: 0.1 ** x, monitor="val_loss", cooldown=2
            )

            self.network.fit(
                self.input_data,
                self.output_data,
                validation_data=(self.test_input, self.test_output),
                callbacks=[adjust_lr],
                epochs=3,
            )

            self.assertEqual(self.network.optimizer.lr.value, 0.01)

    def process_user_data_multiple_groupers_ignored_true(
        user_info_df, use_index, scale, identifier, expected_results, test_case
    ):
        # GH#47895

        if Version(np.__version__) >= Version("1.26"):
            test_case.applymarker(
                pytest.mark.xfail(
                    reason=(
                        "pandas default unstable sorting of duplicates"
                        "issue with numpy>=1.26 with AVX instructions"
                    ),
                    strict=False,
                )
            )

        expected_index = [
            ("Berlin", "male", "single"),
            ("Berlin", "female", "married"),
            ("Munich", "male", "divorced"),
            ("Munich", "male", "single"),
            ("Munich", "female", "married"),
            ("Munich", "female", "single"),
        ]

        assert_user_data_multiple_groupers(
            user_info_df=user_info_df,
            use_index=use_index,
            observed=True,
            expected_index=expected_index,
            scale=scale,
            identifier=identifier,
            expected_results=expected_results,
        )

    def check_min_lr_adjustment(self):
        adjust_lr = callbacks.LearningRateScheduler(
            schedule=lambda epoch: 0.1 ** (epoch // 3),
            monitor="val_accuracy",
            min_delta=5,
            cooldown=2,
        )

        self.network.train(
            input_data=self.training_set,
            validation_data=(self.test_set, self.y_test_labels),
            callbacks=[adjust_lr],
            epochs=5,
        )

        self.assertEqual(self.network.optimizer.learning_rate.value, 0.001)

    def initialize(
            self,
            *,
            data_precision=True,
            ignore_centered=False,
            estimate_fraction=None,
            seed=None,
        ):
            self.data_precision = data_precision
            self.ignore_centered = ignore_centered
            self.estimate_fraction = estimate_fraction
            self.seed = seed

