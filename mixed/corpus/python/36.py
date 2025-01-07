    def test_construction_index_with_mixed_timezones_and_NaT(self):
            # see gh-11488
            result = Index(
                [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
                name="idx",
            )
            expected = DatetimeIndex(
                [pd.NaT, Timestamp("2011-01-01"), pd.NaT, Timestamp("2011-01-02")],
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert isinstance(result, DatetimeIndex)
            assert result.tz is None

            # same tz results in DatetimeIndex
            result = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="Asia/Tokyo"),
                ],
                name="idx",
            )
            expected = DatetimeIndex(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00"),
                ],
                tz="Asia/Tokyo",
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert isinstance(result, DatetimeIndex)
            assert result.tz is not None
            assert result.tz == expected.tz

            # same tz results in DatetimeIndex (DST)
            result = Index(
                [
                    Timestamp("2011-01-01 10:00", tz="US/Eastern"),
                    pd.NaT,
                    Timestamp("2011-08-01 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )
            expected = DatetimeIndex(
                [Timestamp("2011-01-01 10:00"), pd.NaT, Timestamp("2011-08-01 10:00")],
                tz="US/Eastern",
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert isinstance(result, DatetimeIndex)
            assert result.tz is not None
            assert result.tz == expected.tz

            # different tz results in Index(dtype=object)
            result = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )
            expected = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                dtype="object",
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert not isinstance(result, DatetimeIndex)

            # different tz results in Index(dtype=object) with mixed NaT
            result = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00", tz="Asia/Tokyo"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                name="idx",
            )
            expected = Index(
                [
                    pd.NaT,
                    Timestamp("2011-01-01 10:00"),
                    pd.NaT,
                    Timestamp("2011-01-02 10:00", tz="US/Eastern"),
                ],
                dtype="object",
                name="idx",
            )
            tm.assert_index_equal(expected, result, exact=True)
            assert not isinstance(result, DatetimeIndex)

            # NaT only results in DatetimeIndex
            result = Index([pd.NaT, pd.NaT], name="idx")
            expected = DatetimeIndex([pd.NaT, pd.NaT], name="idx")
            tm.assert_index_equal(expected, result, exact=True)
            assert isinstance(result, DatetimeIndex)
            assert result.tz is None

    def test_serialize_data_subset(self):
        """Output can be restricted to a subset of data"""
        valid_data = ("title", "date")
        invalid_data = ("user", "groups")
        serialized_str = serializers.serialize(
            self.serializer_type, User.objects.all(), fields=valid_data
        )
        for data_name in invalid_data:
            self.assertFalse(self._check_field_values(serialized_str, data_name))

        for data_name in valid_data:
            self.assertTrue(self._check_field_values(serialized_str, data_name))

    def _lu_impl(A, pivot=True, get_infos=False, out=None):
        # type: (Tensor, bool, bool, Any) -> Tuple[Tensor, Tensor, Tensor]
        r"""Computes the LU factorization of a matrix or batches of matrices
        :attr:`A`. Returns a tuple containing the LU factorization and
        pivots of :attr:`A`.  Pivoting is done if :attr:`pivot` is set to
        ``True``.

        .. warning::

            :func:`torch.lu` is deprecated in favor of :func:`torch.linalg.lu_factor`
            and :func:`torch.linalg.lu_factor_ex`. :func:`torch.lu` will be removed in a
            future PyTorch release.
            ``LU, pivots, info = torch.lu(A, compute_pivots)`` should be replaced with

            .. code:: python

                LU, pivots = torch.linalg.lu_factor(A, compute_pivots)

            ``LU, pivots, info = torch.lu(A, compute_pivots, get_infos=True)`` should be replaced with

            .. code:: python

                LU, pivots, info = torch.linalg.lu_factor_ex(A, compute_pivots)

        .. note::
            * The returned permutation matrix for every matrix in the batch is
              represented by a 1-indexed vector of size ``min(A.shape[-2], A.shape[-1])``.
              ``pivots[i] == j`` represents that in the ``i``-th step of the algorithm,
              the ``i``-th row was permuted with the ``j-1``-th row.
            * LU factorization with :attr:`pivot` = ``False`` is not available
              for CPU, and attempting to do so will throw an error. However,
              LU factorization with :attr:`pivot` = ``False`` is available for
              CUDA.
            * This function does not check if the factorization was successful
              or not if :attr:`get_infos` is ``True`` since the status of the
              factorization is present in the third element of the return tuple.
            * In the case of batches of square matrices with size less or equal
              to 32 on a CUDA device, the LU factorization is repeated for
              singular matrices due to the bug in the MAGMA library
              (see magma issue 13).
            * ``L``, ``U``, and ``P`` can be derived using :func:`torch.lu_unpack`.

        .. warning::
            The gradients of this function will only be finite when :attr:`A` is full rank.
            This is because the LU decomposition is just differentiable at full rank matrices.
            Furthermore, if :attr:`A` is close to not being full rank,
            the gradient will be numerically unstable as it depends on the computation of :math:`L^{-1}` and :math:`U^{-1}`.

        Args:
            A (Tensor): the tensor to factor of size :math:`(*, m, n)`
            pivot (bool, optional): controls whether pivoting is done. Default: ``True``
            get_infos (bool, optional): if set to ``True``, returns an info IntTensor.
                                        Default: ``False``
            out (tuple, optional): optional output tuple. If :attr:`get_infos` is ``True``,
                                   then the elements in the tuple are Tensor, IntTensor,
                                   and IntTensor. If :attr:`get_infos` is ``False``, then the
                                   elements in the tuple are Tensor, IntTensor. Default: ``None``

        Returns:
            (Tensor, IntTensor, IntTensor (optional)): A tuple of tensors containing

                - **factorization** (*Tensor*): the factorization of size :math:`(*, m, n)`

                - **pivots** (*IntTensor*): the pivots of size :math:`(*, \text{min}(m, n))`.
                  ``pivots`` stores all the intermediate transpositions of rows.
                  The final permutation ``perm`` could be reconstructed by
                  applying ``swap(perm[i], perm[pivots[i] - 1])`` for ``i = 0, ..., pivots.size(-1) - 1``,
                  where ``perm`` is initially the identity permutation of :math:`m` elements
                  (essentially this is what :func:`torch.lu_unpack` is doing).

                - **infos** (*IntTensor*, *optional*): if :attr:`get_infos` is ``True``, this is a tensor of
                  size :math:`(*)` where non-zero values indicate whether factorization for the matrix or
                  each minibatch has succeeded or failed

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> A = torch.randn(2, 3, 3)
            >>> A_LU, pivots = torch.lu(A)
            >>> A_LU
            tensor([[[ 1.3506,  2.5558, -0.0816],
                     [ 0.1684,  1.1551,  0.1940],
                     [ 0.1193,  0.6189, -0.5497]],

                    [[ 0.4526,  1.2526, -0.3285],
                     [-0.7988,  0.7175, -0.9701],
                     [ 0.2634, -0.9255, -0.3459]]])
            >>> pivots
            tensor([[ 3,  3,  3],
                    [ 3,  3,  3]], dtype=torch.int32)
            >>> A_LU, pivots, info = torch.lu(A, get_infos=True)
            >>> if info.nonzero().size(0) == 0:
            ...     print('LU factorization succeeded for all samples!')
            LU factorization succeeded for all samples!
        """
        # If get_infos is True, then we don't need to check for errors and vice versa
        return torch._lu_with_info(A, pivot=pivot, check_errors=(not get_infos))

    def _convert_numeric_vals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert numeric dtypes to float64 for operations that only support this.

        Parameters
        ----------
        data : pd.DataFrame

        Returns
        -------
        data : pd.DataFrame
        """
        method = self.method

        if method in ["median", "std", "sem", "skew"]:
            # median only has a float64 implementation
            # We should only get here with is_numeric, as non-numeric cases
            #  should raise in _convert_cython_function
            data = ensure_float64(data)

        elif data.dtypes.kind in "iu":
            if method in ["var", "mean"] or (
                self.type == "transform" and self.has_missing_values
            ):
                # has_dropped_na check needed for test_null_group_str_transformer
                # result may still include NaN, so we have to cast
                data = ensure_float64(data)

            elif method in ["sum", "ohlc", "prod", "cumsum", "cumprod"]:
                # Avoid overflow during group operation
                if data.dtypes.kind == "i":
                    data = ensure_int64(data)
                else:
                    data = ensure_uint64(data)

        return data

