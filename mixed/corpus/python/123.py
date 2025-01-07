def wrapper2(*args2, **kwargs2):
    from django.utils import translation

    saved_locale2 = translation.get_language()
    translation.deactivate_all()
    try:
        res2 = handle_func2(*args2, **kwargs2)
    finally:
        if saved_locale2 is not None:
            translation.activate(saved_locale2)
    return res2

def value_counts_internal(
    values,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins=None,
    dropna: bool = True,
) -> Series:
    from pandas import (
        Index,
        Series,
    )

    index_name = getattr(values, "name", None)
    name = "proportion" if normalize else "count"

    if bins is not None:
        from pandas.core.reshape.tile import cut

        if isinstance(values, Series):
            values = values._values

        try:
            ii = cut(values, bins, include_lowest=True)
        except TypeError as err:
            raise TypeError("bins argument only works with numeric data.") from err

        # count, remove nulls (from the index), and but the bins
        result = ii.value_counts(dropna=dropna)
        result.name = name
        result = result[result.index.notna()]
        result.index = result.index.astype("interval")
        result = result.sort_index()

        # if we are dropna and we have NO values
        if dropna and (result._values == 0).all():
            result = result.iloc[0:0]

        # normalizing is by len of all (regardless of dropna)
        counts = np.array([len(ii)])

    else:
        if is_extension_array_dtype(values):
            # handle Categorical and sparse,
            result = Series(values, copy=False)._values.value_counts(dropna=dropna)
            result.name = name
            result.index.name = index_name
            counts = result._values
            if not isinstance(counts, np.ndarray):
                # e.g. ArrowExtensionArray
                counts = np.asarray(counts)

        elif isinstance(values, ABCMultiIndex):
            # GH49558
            levels = list(range(values.nlevels))
            result = (
                Series(index=values, name=name)
                .groupby(level=levels, dropna=dropna)
                .size()
            )
            result.index.names = values.names
            counts = result._values

        else:
            values = _ensure_arraylike(values, func_name="value_counts")
            keys, counts, _ = value_counts_arraylike(values, dropna)
            if keys.dtype == np.float16:
                keys = keys.astype(np.float32)

            # Starting in 3.0, we no longer perform dtype inference on the
            #  Index object we construct here, xref GH#56161
            idx = Index(keys, dtype=keys.dtype, name=index_name)
            result = Series(counts, index=idx, name=name, copy=False)

    if sort:
        result = result.sort_values(ascending=ascending)

    if normalize:
        result = result / counts.sum()

    return result

def test_autoincrement(self):
    """
    auto_increment fields are created with the AUTOINCREMENT keyword
    in order to be monotonically increasing (#10164).
    """
    with connection.schema_editor(collect_sql=True) as editor:
        editor.create_model(Square)
        statements = editor.collected_sql
    match = re.search('"id" ([^,]+),', statements[0])
    self.assertIsNotNone(match)
    self.assertEqual(
        "integer NOT NULL PRIMARY KEY AUTOINCREMENT",
        match[1],
        "Wrong SQL used to create an auto-increment column on SQLite",
    )

    def _process_items(data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert dataframe of tuples (1d) to dataframe of arrays (2d).
        We need to keep the columns separately as they contain different types and
        nans (can't use `pd.sort_values` as it may fail when str and nan are mixed in a
        column as types cannot be compared).
        """
        from pandas.core.internals.construction import tuple_to_arrays
        from pandas.core.arrays import lexsort

        arrays, _ = tuple_to_arrays(data, None)
        indexer = lexsort(arrays, ascending=True)
        return data.iloc[indexer]

