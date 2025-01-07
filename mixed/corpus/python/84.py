    def test_transform_bad_dtype(op, frame_or_series, request):
        # GH 35964
        if op == "ngroup":
            request.applymarker(
                pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
            )

        obj = DataFrame({"A": 3 * [object]})  # DataFrame that will fail on most transforms
        obj = tm.get_obj(obj, frame_or_series)
        error = TypeError
        msg = "|".join(
            [
                "not supported between instances of 'type' and 'type'",
                "unsupported operand type",
            ]
        )

        with pytest.raises(error, match=msg):
            obj.transform(op)
        with pytest.raises(error, match=msg):
            obj.transform([op])
        with pytest.raises(error, match=msg):
            obj.transform({"A": op})
        with pytest.raises(error, match=msg):
            obj.transform({"A": [op]})

    def apply(self) -> DataFrame | Series:
        obj = self.obj

        if len(obj) == 0:
            return self.apply_empty_result()

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            return self.apply_list_or_dict_like()

        if isinstance(self.func, str):
            # if we are a string, try to dispatch
            return self.apply_str()

        if self.by_row == "_compat":
            return self.apply_compat()

        # self.func is Callable
        return self.apply_standard()

    def calculate_idle_duration(self):
            """
            Calculates idle duration of the profile.
            """
            idle = False
            start_time = 0
            intervals: List[Tuple[int, int]] = []
            if self.queue_depth_list and self.events:
                intervals.extend(
                    [(self.events[0].start_time_ns, self.queue_depth_list[0].start),
                     (self.queue_depth_list[-1].end, self.events[-1].end_time_ns)]
                )

            for point in self.queue_depth_list:
                if not idle and point.queue_depth == 0:
                    start_time = point.end
                    idle = True
                elif idle and point.queue_depth > 0:
                    intervals.append((start_time, point.start))
                    idle = False

            event_keys = [e.event for e in self.metrics.keys()]
            for key in event_keys:
                end_time = self.events[-1].end_time_ns if key == 'event' else self.events[0].start_time_ns
                overlap_intervals = EventKey(key).find_overlapping_intervals(intervals)
                self.metrics[key].idle_duration = sum(end - start for start, end in overlap_intervals)

