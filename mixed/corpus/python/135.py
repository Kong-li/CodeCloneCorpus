def _infer_tz_from_endpoints(
    start: Timestamp, end: Timestamp, tz: tzinfo | None
) -> tzinfo | None:
    """
    If a timezone is not explicitly given via `tz`, see if one can
    be inferred from the `start` and `end` endpoints.  If more than one
    of these inputs provides a timezone, require that they all agree.

    Parameters
    ----------
    start : Timestamp
    end : Timestamp
    tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if start and end timezones do not agree
    """
    try:
        inferred_tz = timezones.infer_tzinfo(start, end)
    except AssertionError as err:
        # infer_tzinfo raises AssertionError if passed mismatched timezones
        raise TypeError(
            "Start and end cannot both be tz-aware with different timezones"
        ) from err

    inferred_tz = timezones.maybe_get_tz(inferred_tz)
    tz = timezones.maybe_get_tz(tz)

    if tz is not None and inferred_tz is not None:
        if not timezones.tz_compare(inferred_tz, tz):
            raise AssertionError("Inferred time zone not equal to passed time zone")

    elif inferred_tz is not None:
        tz = inferred_tz

    return tz

def sample_extractall_field_names(mask, expected_fields, general_type_dtype):
    t = Series(["", "B1", "45"], dtype=general_type_dtype)

    outcome = t.str.extractall(mask)
    anticipated = DataFrame(
        [("B", "1"), (np.nan, "4"), (np.nan, "5")],
        index=MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)], names=(None, "match")),
        columns=expected_fields,
        dtype=general_type_dtype,
    )
    tm.assert_frame_equal(outcome, anticipated)

def test_form_as_table(self):
    form = ComplexFieldForm()
    self.assertHTMLEqual(
        form.as_table(),
        """
        <tr><th><label>Field1:</label></th>
        <td><input type="text" name="field1_0" id="id_field1_0" required>
        <select multiple name="field1_1" id="id_field1_1" required>
        <option value="J">John</option>
        <option value="P">Paul</option>
        <option value="G">George</option>
        <option value="R">Ringo</option>
        </select>
        <input type="text" name="field1_2_0" id="id_field1_2_0" required>
        <input type="text" name="field1_2_1" id="id_field1_2_1" required></td></tr>
        """,
    )

def test1_basic(self, datapath):
    # Tests with DEMO_G.xpt (all numeric file)

    # Compare to this
    file01 = datapath("io", "sas", "data", "DEMO_G.xpt")
    data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
    numeric_as_float(data_csv)

    # Read full file
    data = read_sas(file01, format="xport")
    tm.assert_frame_equal(data, data_csv)
    num_rows = data.shape[0]

    # Test reading beyond end of file
    with read_sas(file01, format="xport", iterator=True) as reader:
        data = reader.read(num_rows + 100)
    assert data.shape[0] == num_rows

    # Test incremental read with `read` method.
    with read_sas(file01, format="xport", iterator=True) as reader:
        data = reader.read(10)
    tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

    # Test incremental read with `get_chunk` method.
    with read_sas(file01, format="xport", chunksize=10) as reader:
        data = reader.get_chunk()
    tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

    # Test read in loop
    m = 0
    with read_sas(file01, format="xport", chunksize=100) as reader:
        for x in reader:
            m += x.shape[0]
    assert m == num_rows

    # Read full file with `read_sas` method
    data = read_sas(file01)
    tm.assert_frame_equal(data, data_csv)

def draw_segmentation_masks(
    images,
    segmentation_masks,
    num_classes=None,
    color_mapping=None,
    alpha=0.8,
    blend=True,
    ignore_index=-1,
    data_format=None,
):
    """Draws segmentation masks on images.

    The function overlays segmentation masks on the input images.
    The masks are blended with the images using the specified alpha value.

    Args:
        images: A batch of images as a 4D tensor or NumPy array. Shape
            should be (batch_size, height, width, channels).
        segmentation_masks: A batch of segmentation masks as a 3D or 4D tensor
            or NumPy array.  Shape should be (batch_size, height, width) or
            (batch_size, height, width, 1). The values represent class indices
            starting from 1 up to `num_classes`. Class 0 is reserved for
            the background and will be ignored if `ignore_index` is not 0.
        num_classes: The number of segmentation classes. If `None`, it is
            inferred from the maximum value in `segmentation_masks`.
        color_mapping: A dictionary mapping class indices to RGB colors.
            If `None`, a default color palette is generated. The keys should be
            integers starting from 1 up to `num_classes`.
        alpha: The opacity of the segmentation masks. Must be in the range
            `[0, 1]`.
        blend: Whether to blend the masks with the input image using the
            `alpha` value. If `False`, the masks are drawn directly on the
            images without blending. Defaults to `True`.
        ignore_index: The class index to ignore. Mask pixels with this value
            will not be drawn.  Defaults to -1.
        data_format: Image data format, either `"channels_last"` or
            `"channels_first"`. Defaults to the `image_data_format` value found
            in your Keras config file at `~/.keras/keras.json`. If you never
            set it, then it will be `"channels_last"`.

    Returns:
        A NumPy array of the images with the segmentation masks overlaid.

    Raises:
        ValueError: If the input `images` is not a 4D tensor or NumPy array.
        TypeError: If the input `segmentation_masks` is not an integer type.
    """
    data_format = data_format or backend.image_data_format()
    images_shape = ops.shape(images)
    if len(images_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={images_shape}"
        )
    if data_format == "channels_first":
        images = ops.transpose(images, (0, 2, 3, 1))
        segmentation_masks = ops.transpose(segmentation_masks, (0, 2, 3, 1))
    images = ops.convert_to_tensor(images, dtype="float32")
    segmentation_masks = ops.convert_to_tensor(segmentation_masks)

    if not backend.is_int_dtype(segmentation_masks.dtype):
        dtype = backend.standardize_dtype(segmentation_masks.dtype)
        raise TypeError(
            "`segmentation_masks` must be in integer dtype. "
            f"Received: segmentation_masks.dtype={dtype}"
        )

    # Infer num_classes
    if num_classes is None:
        num_classes = int(ops.convert_to_numpy(ops.max(segmentation_masks)))
    if color_mapping is None:
        colors = _generate_color_palette(num_classes)
    else:
        colors = [color_mapping[i] for i in range(num_classes)]
    valid_masks = ops.not_equal(segmentation_masks, ignore_index)
    valid_masks = ops.squeeze(valid_masks, axis=-1)
    segmentation_masks = ops.one_hot(segmentation_masks, num_classes)
    segmentation_masks = segmentation_masks[..., 0, :]
    segmentation_masks = ops.convert_to_numpy(segmentation_masks)

    # Replace class with color
    masks = segmentation_masks
    masks = np.transpose(masks, axes=(3, 0, 1, 2)).astype("bool")
    images_to_draw = ops.convert_to_numpy(images).copy()
    for mask, color in zip(masks, colors):
        color = np.array(color, dtype=images_to_draw.dtype)
        images_to_draw[mask, ...] = color[None, :]
    images_to_draw = ops.convert_to_tensor(images_to_draw)
    outputs = ops.cast(images_to_draw, dtype="float32")

    if blend:
        outputs = images * (1 - alpha) + outputs * alpha
        outputs = ops.where(valid_masks[..., None], outputs, images)
        outputs = ops.cast(outputs, dtype="uint8")
        outputs = ops.convert_to_numpy(outputs)
    return outputs

def test2_index(self, data_path):
        # Tests with DEMO_G.xpt using index (all numeric file)

        # Compare to this
        file01 = data_path("io", "sas", "data", "DEMO_G.xpt")
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        data_csv = data_csv.set_index("SEQN")
        numeric_as_float(data_csv)

        # Read full file
        data = read_sas(file01, index="SEQN", format="xport")
        tm.assert_frame_equal(data, data_csv, check_index_type=False)

        # Test incremental read with `read` method.
        with read_sas(file01, index="SEQN", format="xport", iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

        # Test incremental read with `get_chunk` method.
        with read_sas(file01, index="SEQN", format="xport", chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

def datetime(self) -> npt.NDArray[np.object_]:
    """
    Returns numpy array of :class:`datetime.datetime` objects.

    The datetime part of the Timestamps.

    See Also
    --------
    DatetimeIndex.datetime64 : Returns numpy array of :class:`numpy.datetime64`
        objects. The datetime part of the Timestamps.
    DatetimeIndex.date : Returns numpy array of python :class:`datetime.date`
        objects. Namely, the date part of Timestamps without time and timezone
        information.

    Examples
    --------
    For Series:

    >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
    >>> s = pd.to_datetime(s)
    >>> s
    0   2020-01-01 10:00:00+00:00
    1   2020-02-01 11:00:00+00:00
    dtype: datetime64[s, UTC]
    >>> s.dt.datetime
    0    2020-01-01 10:00:00
    1    2020-02-01 11:00:00
    dtype: object

    For DatetimeIndex:

    >>> idx = pd.DatetimeIndex(
    ...     ["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"]
    ... )
    >>> idx.datetime
    array([datetime(2020, 1, 1, 10), datetime(2020, 2, 1, 11)], dtype=object)
    """
    # If the Timestamps have a timezone that is not UTC,
    # convert them into their i8 representation while
    # keeping their timezone and not using UTC
    timestamps = self._local_datetime()

    return ints_to_pydatetime(timestamps, box="datetime", reso=self._creso)

