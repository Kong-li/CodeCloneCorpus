    def info(self):
        """
        A pointer to the memory area of the array as a Python integer.
        This memory area may contain data that is not aligned, or not in
        correct byte-order. The memory area may not even be writeable.
        The array flags and data-type of this array should be respected
        when passing this attribute to arbitrary C-code to avoid trouble
        that can include Python crashing. User Beware! The value of this
        attribute is exactly the same as:
        ``self._array_interface_['data'][0]``.

        Note that unlike ``data_as``, a reference won't be kept to the array:
        code like ``ctypes.c_void_p((a + b).ctypes.info)`` will result in a
        pointer to a deallocated array, and should be spelt
        ``(a + b).ctypes.info_as(ctypes.c_void_p)``
        """
        return self._info.value

    def dense_flying(
        elements: List[Vector],
        updates: List[Vector],
        avg_weights: List[Vector],
        avg_squares: List[Vector],
        step_counts: List[int],
        *,
        epsilon: float,
        beta1: float,
        beta2: float,
        learning_rate: float,
        reduce_flag: bool,
    ):
        r"""Functional API that performs Dense Flying algorithm computation.

        See :class:`~torch.optim.DenseFlying` for details.
        """
        for i, element in enumerate(elements):
            update = updates[i]
            update = update if not reduce_flag else -update
            update = update.coalesce()  # the update is non-linear so indices must be unique
            update_indices = update._indices()
            update_values = update._values()
            if update_values.numel() == 0:
                # Skip update for empty grad
                continue
            size = update.size()

            avg_weight = avg_weights[i]
            avg_squares = avg_squares[i]
            step = step_counts[i]

            def make_dense(values):
                constructor = update.new
                if update_indices.dim() == 0 or values.dim() == 0:
                    return constructor().resize_as_(update)
                return constructor(update_indices, values, size)

            # Decay the first and second moment running average coefficient
            #      old <- b * old + (1 - b) * new
            # <==> old += (1 - b) * (new - old)
            old_avg_weight_values = avg_weight.dense_mask(update)._values()
            avg_weight_update_values = update_values.sub(old_avg_weight_values).mul_(1 - beta1)
            avg_weight.add_(make_dense(avg_weight_update_values))
            old_avg_square_values = avg_squares.sparse_mask(update)._values()
            avg_square_update_values = (
                update_values.pow(2).sub_(old_avg_square_values).mul_(1 - beta2)
            )
            avg_squares.add_(make_dense(avg_square_update_values))

            # Dense addition again is intended, avoiding another sparse_mask
            numer = avg_weight_update_values.add_(old_avg_weight_values)
            avg_square_update_values.add_(old_avg_square_values)
            denom = avg_square_update_values.sqrt_().add_(epsilon)
            del avg_weight_update_values, avg_square_update_values

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            step_size = learning_rate * math.sqrt(bias_correction2) / bias_correction1

            element.add_(make_dense(-step_size * numer.div_(denom)))

    def verify_keep_alive_connection_reset_request_data(self):
            server_host = LiveServerViews.server_thread.host
            server_port = LiveServerViews.server_thread.port
            conn = HTTPConnection(server_host, server_port)
            try:
                headers = {"Connection": "keep-alive"}
                conn.request("POST", "/method_view/", b"{}", headers=headers)
                response = conn.getresponse()
                self.assertTrue(not response.will_close)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b"POST")

                conn.request("POST", "/method_view/", b"{}", headers={"Connection": "close"})
                response = conn.getresponse()
                self.assertFalse(response.will_close)
                self.assertEqual(response.status, 200)
                self.assertEqual(response.read(), b"POST")
            finally:
                conn.close()

    def test_datetimes_has_lazy_iterator(self):
        pub_dates = [
            datetime.datetime(2005, 7, 28, 12, 15),
            datetime.datetime(2005, 7, 29, 2, 15),
            datetime.datetime(2005, 7, 30, 5, 15),
            datetime.datetime(2005, 7, 31, 19, 15),
        ]
        for i, pub_date in enumerate(pub_dates):
            Article(pub_date=pub_date, title="title #{}".format(i)).save()
        # Use iterator() with datetimes() to return a generator that lazily
        # requests each result one at a time, to save memory.
        dates = []
        with self.assertNumQueries(0):
            article_datetimes_iterator = Article.objects.datetimes(
                "pub_date", "day", order="DESC"
            ).iterator()

        with self.assertNumQueries(1):
            for article in article_datetimes_iterator:
                dates.append(article)
        self.assertEqual(
            dates,
            [
                datetime.datetime(2005, 7, 31, 0, 0),
                datetime.datetime(2005, 7, 30, 0, 0),
                datetime.datetime(2005, 7, 29, 0, 0),
                datetime.datetime(2005, 7, 28, 0, 0),
            ],
        )

