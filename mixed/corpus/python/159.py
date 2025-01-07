def validate_join_parameters(left_set, right_set):
    # GH 46622
    # Check invalid arguments for merge validation
    valid_options = ["1:1", "1:m", "m:1", "m:m", "one_to_one", "one_to_many", "many_to_one", "many_to_many"]
    error_message = (
        f'"{validate}" is not a valid argument. Valid arguments are:\n'
        + '\n'.join(f'- "{opt}"' for opt in valid_options)
    )

    if validate not in valid_options:
        with pytest.raises(ValueError, match=error_message):
            left_set.merge(right_set, on="a", validate=validate)

def sample(self, sample_shape=torch.Size()):
    """
    Generates a sample_shape shaped sample or sample_shape shaped batch of
    samples if the distribution parameters are batched. Samples first from
    base distribution and applies `transform()` for every transform in the
    list.
    """
    with torch.no_grad():
        x = self.base_dist.sample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

def modified_sample_inputs_linalg_cond(op_info, device_type, tensor_dtype, need_grad=False, **kwargs):
    make_arg = partial(
        make_tensor, dtype=tensor_dtype, device=device_type, requires_grad=need_grad
    )

    shapes_list = [
        (S, S),
        (2, S, S),
        (2, 1, S, S)
    ]

    for shape in reversed(shapes_list):
        yield SampleInput(make_arg(shape), kwargs)

