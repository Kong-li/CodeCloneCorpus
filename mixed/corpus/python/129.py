def get_stable_regression(device: torch.device) -> GetterReturnType:
    M = 15
    L = 8

    # X.shape: (M, L + 1), Y.shape: (M, 1)
    X = torch.rand(M, L + 1, device=device)
    Y = torch.rand(M, 1, device=device)

    # Predefined mu_alpha and mu_beta, mu_alpha.shape: (1, 1), mu_beta.shape: (1, 1)
    mu_alpha = torch.rand(1, 1, device=device)
    mu_beta = torch.rand(1, 1, device=device)
    mu = dist.Gamma(mu_alpha, mu_beta)

    # Predefined tau_rate: tau_rate.shape: (M, 1)
    tau_rate = torch.rand(M, 1, device=device)
    tau = dist.Exponential(tau_rate)

    # Predefined alpha_mean and alpha_sigma: alpha_mean.shape: (L + 1, 1), alpha_sigma.shape: (L + 1, 1)
    alpha_mean = torch.rand(L + 1, 1, device=device)
    alpha_sigma = torch.rand(L + 1, 1, device=device)
    alpha = dist.Normal(alpha_mean, alpha_sigma)

    mu_value = mu.sample()
    mu_value.requires_grad_(True)

    tau_value = tau.sample()
    tau_unconstrained_value = tau_value.log()
    tau_unconstrained_value.requires_grad_(True)

    alpha_value = alpha.sample()
    alpha_value.requires_grad_(True)

    def forward(
        mu_value: Tensor, tau_unconstrained_value: Tensor, alpha_value: Tensor
    ) -> Tensor:
        tau_constrained_value = tau_unconstrained_value.exp()
        beta = X.mm(alpha_value)

        # For this model, we need to compute the following three scores:
        # We need to compute the first and second gradient of this score with respect
        # to mu_value.
        mu_score = dist.StudentT(mu_value, beta, tau_constrained_value).log_prob(
            Y
        ).sum() + mu.log_prob(mu_value)

        # We need to compute the first and second gradient of this score with respect
        # to tau_unconstrained_value.
        tau_score = (
            dist.StudentT(mu_value, beta, tau_constrained_value).log_prob(Y).sum()
            + tau.log_prob(tau_constrained_value)
            + tau_unconstrained_value
        )

        # We need to compute the first and second gradient of this score with respect
        # to alpha_value.
        alpha_score = dist.StudentT(mu_value, beta, tau_constrained_value).log_prob(
            Y
        ).sum() + alpha.log_prob(alpha_value)

        return mu_score.sum() + tau_score.sum() + alpha_score.sum()

    return forward, (
        mu_value.to(device),
        tau_unconstrained_value.to(device),
        alpha_value.to(device),
    )

def _bsr_softmax_kernel_mod(
    crow_indices_ptr,
    crow_indices_batch_stride,
    crow_indices_stride,
    values_ptr,
    values_batch_stride,
    values_row_block_stride,
    values_nnz_col_block_stride,
    row_block,
    col_block,
    max_row_nnz: tl.constexpr,
    tile: tl.constexpr
):
    batch_pid = tl.program_id(2)
    row_block_offset_pid = tl.program_id(1)
    row_block_pid = tl.program_id(0)

    crow_indices_offset_ptr = (
        crow_indices_ptr
        + crow_indices_batch_stride * batch_pid
        + crow_indices_stride * row_block_pid
    )
    nnz_offset = tl.load(crow_indices_offset_ptr)
    nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

    # Compute nnz for the row with number row_block_pid.
    # If it is zero, skip the row.
    row_nnz = nnz_offset_next - nnz_offset
    if row_nnz == 0:
        return

    row_arange = tl.arange(0, tile)
    mask = row_arange < row_nnz * col_block

    curr_row_values_ptrs = (
        values_ptr
        + values_batch_stride * batch_pid
        + values_row_block_stride * row_block_offset_pid
        + nnz_offset * col_block
    )

    # find max in the row
    row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")).to(tl.float32)
    max_row_value = tl.max(row_tile, axis=0)
    for offset in range(1, max_row_nnz // tile):
        row_arange += tile
        mask = row_arange < row_nnz * col_block
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")).to(tl.float32)
        curr_max_row_value = tl.max(row_tile, axis=0)
        max_row_value = tl.where(max_row_value > curr_max_row_value, max_row_value, curr_max_row_value)

    # find denominator for stable softmax
    num = tl.exp(row_tile - max_row_value)
    denom = tl.sum(num, axis=0)
    for offset in range(1, max_row_nnz // tile):
        row_arange -= tile
        mask = row_arange < row_nnz * col_block
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")).to(tl.float32)
        num = tl.exp(row_tile - max_row_value)
        denom += tl.sum(num, axis=0)

    # populate output
    for i in range(row_nnz * col_block):
        if i < row_nnz * col_block:
            mask_i = i < row_nnz * col_block
            row_arange_i = tl.arange(0, tile)
            curr_row_values_ptrs_i = (
                values_ptr
                + values_batch_stride * batch_pid
                + values_row_block_stride * row_block_offset_pid
                + (nnz_offset + i // col_block) * col_block
            )
            row_tile_i = tl.load(curr_row_values_ptrs_i + row_arange_i, mask=mask_i, other=-float("inf")).to(tl.float32)
            num_i = tl.exp(row_tile_i - max_row_value)
            denom_i = tl.sum(num_i, axis=0)
            tl.store(
                curr_row_values_ptrs_i + row_arange_i,
                (num_i / denom_i).to(values_ptr.dtype.element_ty),
                mask=mask_i
            )

def check_inlineformset_factory_nulls_default_pks_child_editable_pk(self):
        """
        #24958 - Variant of test_inlineformset_factory_nulls_default_pks for
        the case of a parent object with a UUID primary key and a child
        object with an editable natural key for a primary key.
        """
        FormSet = inlineformset_factory(
            UUIDPKParentClass, ChildWithEditablePKClass, fields="__all__"
        )
        formset = FormSet()
        self.assertIsNone(formset.forms[0].fields["parent_field"].initial)

def _check_not_almost_equal_inverted(x, y, **kwargs):
    """
    Verify that two objects are not approximately equal.

    This verification is carried out in a non-commutative manner.

    Parameters
    ----------
    x : object
        The first object to compare.
    y : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    if not _compare_objects(x, y, **kwargs):
        return
    if not _compare_objects(y, x, **kwargs):
        return

def _compare_objects(a, b, **kwargs):
    """
    Helper function to compare two objects for approximate equality.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    return not _assert_almost_equal(a, b, **kwargs)

def expect_target_flags(self, targets, groups={}, **kwargs):
    match_dict = self.arg_regex(**kwargs)
    if match_dict is None:
        return
    assert(isinstance(match_dict, dict))
    _, tar_flags = self.get_targets(targets=targets, groups=groups)

    for match_tar, match_flags in match_dict.items():
        if match_tar not in tar_flags:
            raise AssertionError(
                'expected to find target "%s"' % match_tar
            )
        flags = tar_flags[match_tar]
        if not match_flags:
            if len(flags) != 0:
                raise AssertionError(
                    'expected to find empty flags in target "%s"' % match_tar
                )
        if not re.match(match_flags, flags):
            raise AssertionError(
                '"%s" flags "%s" not match "%s"' % (match_tar, flags, match_flags)
            )

