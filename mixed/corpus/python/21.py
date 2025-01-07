def check_POST_multipart_data(self):
        payload = FakePayload(
            "\r\n".join(
                [
                    f"--{BOUNDARY}",
                    'Content-Disposition: form-data; name="username"',
                    "",
                    "user1",
                    f"--{BOUNDARY}",
                    *self._data_payload,
                    f"--{BOUNDARY}--",
                ]
            )
        )
        request = WSGIRequest(
            {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": MULTIPART_CONTENT,
                "CONTENT_LENGTH": len(payload),
                "wsgi.input": payload,
            }
        )
        self.assertEqual(
            request.POST,
            {
                "username": ["user1"],
                "INFO": [
                    '{"id": 1, "type": "store.product", "attributes": {"name": "Laptop", "price": 999.99}}'
                ],
            },
        )

    def _sync_parameters_across_ranks(self, source_rank: int):
            r"""
            Synchronize the shard of parameters from a specified rank across all ranks asynchronously.

            Arguments:
                source_rank (int): the originating rank for parameter synchronization.

            Returns:
                A :class:`list` of async work handles for the ``broadcast()`` operations
                executed to synchronize the parameters.
            """
            assert not self._overlap_with_ddp, (
                "`_sync_parameters_across_ranks()` should not be invoked if "
                "`overlap_with_ddp=True`; parameter synchronization should occur in the DDP communication hook"
            )
            handles = []
            if self.parameters_as_bucket_view:
                for dev_i_buckets in self._buckets:
                    bucket = dev_i_buckets[source_rank]
                    global_rank = dist.distributed_c10d.get_global_rank(
                        process_group=self.process_group, rank=source_rank
                    )
                    handle = dist.broadcast(
                        tensor=bucket,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                    handles.append(handle)
            else:
                param_groups = self._partition_parameters()[source_rank]
                global_rank = dist.distributed_c10d.get_global_rank(
                    process_group=self.process_group, rank=source_rank
                )
                for param_group in param_groups:
                    for param in param_group["params"]:
                        handle = dist.broadcast(
                            tensor=param.data,
                            src=global_rank,
                            group=self.process_group,
                            async_op=True,
                        )
                        handles.append(handle)
            return handles

def polyfromroots(roots):
    """
    Generate a monic polynomial with given roots.

    Return the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    where the :math:`r_n` are the roots specified in `roots`.  If a zero has
    multiplicity n, then it must appear in `roots` n times. For instance,
    if 2 is a root of multiplicity three and 3 is a root of multiplicity 2,
    then `roots` looks something like [2, 2, 2, 3, 3]. The roots can appear
    in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * x + ... +  x^n

    The coefficient of the last term is 1 for monic polynomials in this
    form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of the polynomial's coefficients If all the roots are
        real, then `out` is also real, otherwise it is complex.  (see
        Examples below).

    See Also
    --------
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Notes
    -----
    The coefficients are determined by multiplying together linear factors
    of the form ``(x - r_i)``, i.e.

    .. math:: p(x) = (x - r_0) (x - r_1) ... (x - r_n)

    where ``n == len(roots) - 1``; note that this implies that ``1`` is always
    returned for :math:`a_n`.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyfromroots((-1,0,1))  # x(x - 1)(x + 1) = x^3 - x
    array([ 0., -1.,  0.,  1.])
    >>> j = complex(0,1)
    >>> P.polyfromroots((-j,j))  # complex returned, though values are real
    array([1.+0.j,  0.+0.j,  1.+0.j])

    """
    return pu._fromroots(polyline, polymul, roots)

    def test_POST_immutable_for_multipart(self):
        """
        MultiPartParser.parse() leaves request.POST immutable.
        """
        payload = FakePayload(
            "\r\n".join(
                [
                    "--boundary",
                    'Content-Disposition: form-data; name="name"',
                    "",
                    "value",
                    "--boundary--",
                ]
            )
        )
        request = WSGIRequest(
            {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": "multipart/form-data; boundary=boundary",
                "CONTENT_LENGTH": len(payload),
                "wsgi.input": payload,
            }
        )
        self.assertFalse(request.POST._mutable)

    def verify_dataframe_aggregation():
        # GH 12363

        dtf = DataFrame(
            {
                "X": ["alpha", "beta", "alpha", "beta", "alpha", "beta", "alpha", "beta"],
                "Y": ["one", "one", "two", "two", "two", "two", "one", "two"],
                "Z": np.random.default_rng(2).standard_normal(8) + 1.0,
                "W": np.arange(8),
            }
        )

        expected = dtf.groupby(["X"]).Y.count()
        result = dtf.Y.groupby(dtf.X).count()
        tm.assert_series_equal(result, expected)

def _partition_param_group(
    self, param_group: Dict[str, Any], params_per_rank: List[List[torch.Tensor]]
) -> None:
    r"""
    Partition the parameter group ``param_group`` according to ``params_per_rank``.

    The partition will modify the ``self._partition_parameters_cache``. This method should
    only be used as a subroutine for :meth:`_partition_parameters`.

    Arguments:
        param_group (dict[str, Any]): a parameter group as normally defined
            in an optimizer state.
        params_per_rank (list[list[torch.Tensor]]): a :class:`list` of
            length world size containing :class:`list` s of parameters to
            assign to each rank.
    """
    for rank, params in enumerate(params_per_rank):
        rank_param_group = copy.copy(param_group)
        rank_param_group["params"] = params
        self._partition_parameters_cache[rank].append(rank_param_group)

def test_log_deletions(self):
    ma = ModelAdmin(Band, self.site)
    mock_request = MockRequest()
    mock_request.user = User.objects.create(username="akash")
    content_type = get_content_type_for_model(self.band)
    Band.objects.create(
        name="The Beatles",
        bio="A legendary rock band from Liverpool.",
        sign_date=date(1962, 1, 1),
    )
    Band.objects.create(
        name="Mohiner Ghoraguli",
        bio="A progressive rock band from Calcutta.",
        sign_date=date(1975, 1, 1),
    )
    queryset = Band.objects.all().order_by("-id")[:3]
    self.assertEqual(len(queryset), 3)
    with self.assertNumQueries(1):
        ma.log_deletions(mock_request, queryset)
    logs = (
        LogEntry.objects.filter(action_flag=DELETION)
        .order_by("id")
        .values_list(
            "user_id",
            "content_type",
            "object_id",
            "object_repr",
            "action_flag",
            "change_message",
        )
    )
    expected_log_values = [
        (
            mock_request.user.id,
            content_type.id,
            str(obj.pk),
            str(obj),
            DELETION,
            "",
        )
        for obj in queryset
    ]
    self.assertSequenceEqual(logs, expected_log_values)

    def test_POST_multipart_json(self):
        payload = FakePayload(
            "\r\n".join(
                [
                    f"--{BOUNDARY}",
                    'Content-Disposition: form-data; name="name"',
                    "",
                    "value",
                    f"--{BOUNDARY}",
                    *self._json_payload,
                    f"--{BOUNDARY}--",
                ]
            )
        )
        request = WSGIRequest(
            {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": MULTIPART_CONTENT,
                "CONTENT_LENGTH": len(payload),
                "wsgi.input": payload,
            }
        )
        self.assertEqual(
            request.POST,
            {
                "name": ["value"],
                "JSON": [
                    '{"pk": 1, "model": "store.book", "fields": {"name": "Mostly '
                    'Harmless", "author": ["Douglas", Adams"]}}'
                ],
            },
        )

