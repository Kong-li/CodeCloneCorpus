def _clone_cookie_store(store):
    if store is None:
        return None

    if hasattr(store, "clone"):
        # We're dealing with an instance of RequestsCookieStore
        return store.clone()
    # We're dealing with a generic CookieStore instance
    new_store = copy.copy(store)
    new_store.clear()
    for item in store:
        new_store.add_item(copy.copy(item))
    return new_store

def _merge(delimiter, elements):
    """
    Return a string which is the concatenation of the strings in the
    sequence `elements`.

    Calls :meth:`str.join` element-wise.

    Parameters
    ----------
    delimiter : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    elements : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    Returns
    -------
    result : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.join

    Examples
    --------
    >>> import numpy as np
    >>> np.strings.merge('_', 'abc')  # doctest: +SKIP
    array('a-b-c', dtype='<U4')  # doctest: +SKIP

    >>> np.strings.merge(['-', '.'], ['ghc', 'osd'])  # doctest: +SKIP
    array(['g-h-c', 'o.s.d'], dtype='<U5')  # doctest: +SKIP

    """
    return _to_bytes_or_str_array(
        _vec_string(delimiter, np.object_, 'join', (elements,)), elements)

def update_value(input_data):
        result = input_data
        iter_count = n_iter if n_iter is not None and n_iter > 0 else 1

        for _ in range(iter_count):
            result = base_pass(result)

        predicate_func = predicate if predicate is not None else lambda x: True

        while predicate_func(result):
            result = base_pass(result)

        if iter_count == 0 and predicate_func(result):
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given "
                f"{n_iter}) xor predicate (given {predicate})"
            )

        return result

def analyze_dimensions(cls, input_chars: List[str], output_char: str) -> "DimensionAnalysis":
    """
    Analyze the dimensions and extract the contracting, batch, and free dimensions
    for the left and right hand sides.
    """
    dimension_set: Set[str] = set()
    for input_char in input_chars:
        dimension_set.update(input_char)

    # get a deterministic order of all dim chars
    all_dimension_chars = sorted(dimension_set)

    # parse input and output dimensions
    lhs_out_only_chars, rhs_out_only_chars = [], []
    batch_chars, contracting_chars = [], []

    for dimension_char in all_dimension_chars:
        if dimension_char not in output_char:
            contracting_chars.append(dimension_char)
        else:
            is_batch_char = True
            for input_char in input_chars:
                is_batch_char = is_batch_char and dimension_char in input_char

            if is_batch_char:
                batch_chars.append(dimension_char)
            else:
                assert (
                    len(input_chars) == 2
                ), "free dimension only supported for two inputs!"
                left, right = input_chars
                if dimension_char in left:
                    lhs_out_only_chars.append(dimension_char)
                elif dimension_char in right:
                    rhs_out_only_chars.append(dimension_char)
                else:
                    raise RuntimeError("Invalid character")

    return cls(
        contracting_chars=contracting_chars,
        batch_chars=batch_chars,
        lhs_out_only_chars=lhs_out_only_chars,
        rhs_out_only_chars=rhs_out_only_chars,
    )

    def _partition(b, deli=None, maxdiv=None):
        """
        For each element in `b`, return a list of the words in the
        string, using `deli` as the delimiter string.

        Calls :meth:`str.split` element-wise.

        Parameters
        ----------
        b : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

        deli : str or unicode, optional
           If `deli` is not specified or None, any whitespace string is a
           separator.

        maxdiv : int, optional
            If `maxdiv` is given, at most `maxdiv` splits are done.

        Returns
        -------
        out : ndarray
            Array of list objects

        Examples
        --------
        >>> import numpy as np
        >>> y = np.array("Numpy is nice!")
        >>> np.strings.partition(y, " ")  # doctest: +SKIP
        array(list(['Numpy', 'is', 'nice!']), dtype=object)  # doctest: +SKIP

        >>> np.strings.partition(y, " ", 1)  # doctest: +SKIP
        array(list(['Numpy', 'is nice!']), dtype=object)  # doctest: +SKIP

        See Also
        --------
        str.split, rsplit

        """
        # This will return an array of lists of different sizes, so we
        # leave it as an object array
        return _vec_string(
            b, np.object_, 'split', [deli] + _clean_args(maxdiv))

