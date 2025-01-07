    def calculateDifferences(self, alternative):
            """
            Generates a delta against another ModuleContextCheckpointState.

            Returns None if no delta is found, otherwise, return a set() of mismatched
            module key names.
            """
            r = set(self.nnModules.keys()).difference(set(alternative.nnModules.keys()))
            if len(r) == 0:
                return None
            return r

    def generate_overall_schedule(
            self, total_plans: list[SavePlan]
        ) -> Tuple[list[SavePlan], Metadata]:
            total_plans = remove_duplicate_saves(total_plans, not self.deduplicate_higher_rank)

            overall_plan, metadata = setup_initial_global_save_plan(total_plans)

            if not self.merge_state_dict:
                planner_data_list = [p.planner_data for p in overall_plan]
                merged_mappings = dict(ChainMap(*planner_data_list))
                metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

            if _check_schedule_validity(overall_plan, metadata):
                raise ValueError("Global schedule validation failed")

            self.overall_plan = overall_plan
            self.metadata = metadata

            return self.overall_plan, self.metadata

def _verify_shard_positional_overlap(metadata_a: ChunkStorageMetadata, metadata_b: ChunkStorageMetadata) -> bool:
    """Check if two shards overlap. Tuples are (offsets, sizes)."""
    ndims = len(metadata_a.offsets)
    for dim_index in range(ndims):
        shard_a_offset = metadata_a.offsets[dim_index]
        shard_b_offset = metadata_b.offsets[dim_index]
        shard_a_size = metadata_a.sizes[dim_index]
        shard_b_size = metadata_b.sizes[dim_index]

        if shard_a_offset >= shard_b_offset + shard_b_size:
            return False
        elif shard_b_offset >= shard_a_offset + shard_a_size:
            return False

    return True

    def validate_aggregate_test(self):
            AggregateTestModel.objects.all().delete()
            tests = [
                (ArrayAgg("char_field", default=Value(["empty"], StringField())), ["empty"]),
                (ArrayAgg("integer_field", default=[1]), [1]),
                (ArrayAgg("boolean_field", default=[True]), [True]),
                (BitAnd("integer_field", default=0), 0),
                (BitOr("integer_field", default=0), 0),
                (BoolAnd("boolean_field", default=True), True),
                (BoolOr("boolean_field", default=True), True),
                (JSONBAgg("integer_field", default=Value(["empty"], JSONField())), ["empty"]),
                (
                    JSONBAgg("integer_field", default=Value(["empty"], JSONField())),
                    ["empty"],
                ),
                (StringAgg("char_field", delimiter=";", default="<empty>"), "<empty>"),
                (
                    StringAgg("char_field", delimiter=";", default=Value("<empty>", CharField())),
                    "<empty>",
                ),
                (BitXor("integer_field", default=0), 0),
            ]
            for test, expected in tests:
                with self.subTest(test=test):
                    # Empty result with non-execution optimization.
                    with self.assertNumQueries(1 if test.default == Value(["empty"], StringField()) else 0):
                        values = AggregateTestModel.objects.none().aggregate(
                            aggregation=test,
                        )
                        self.assertEqual(values, {"aggregation": expected})
                    # Empty result when query must be executed.
                    with transaction.atomic(), self.subTest(test=test), self.assertNumQueries(1 if test.default == Value(["empty"], StringField()) else 2):
                        values = AggregateTestModel.objects.aggregate(
                            aggregation=test,
                        )
                        self.assertEqual(values, {"aggregation": expected})

    def log_processing_steps():
        pc = ProfilingContext.try_get()
        if pc is None:
            yield None
            return
        old_log_messages = pc.log_messages
        pc.log_messages = []
        try:
            yield pc.log_messages
        finally:
            pc.log_messages = old_log_messages

    def process_step(current_step, extra_outputs, future_extra_inputs):
        for _ in range(
            min(
                self.iterations_per_process,
                self.total_steps,
            )
        ):
            current_step, extra_outputs, future_extra_inputs = (
                step_handler(
                    current_step,
                    extra_outputs,
                    future_extra_inputs,
                )
            )

        return (current_step, extra_outputs, future_extra_inputs)

