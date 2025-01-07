    def test_models_not_loaded(self):
        """
        apps.get_models() raises an exception if apps.models_ready isn't True.
        """
        apps.models_ready = False
        try:
            # The cache must be cleared to trigger the exception.
            apps.get_models.cache_clear()
            with self.assertRaisesMessage(
                AppRegistryNotReady, "Models aren't loaded yet."
            ):
                apps.get_models()
        finally:
            apps.models_ready = True

    def reverse_transform(self, Y):
        """Convert the data back to the original representation.

        Inverts the `transform` operation performed on an array.
        This operation can only be performed after :class:`SimpleFiller` is
        instantiated with `add_indicator=True`.

        Note that `reverse_transform` can only invert the transform in
        features that have binary indicators for missing values. If a feature
        has no missing values at `fit` time, the feature won't have a binary
        indicator, and the filling done at `transform` time won't be reversed.

        .. versionadded:: 0.24

        Parameters
        ----------
        Y : array-like of shape \
                (n_samples, n_features + n_features_missing_indicator)
            The filled data to be reverted to original data. It has to be
            an augmented array of filled data and the missing indicator mask.

        Returns
        -------
        Y_original : ndarray of shape (n_samples, n_features)
            The original `Y` with missing values as it was prior
            to filling.
        """
        check_is_fitted(self)

        if not self.add_indicator:
            raise ValueError(
                "'reverse_transform' works only when "
                "'SimpleFiller' is instantiated with "
                "'add_indicator=True'. "
                f"Got 'add_indicator={self.add_indicator}' "
                "instead."
            )

        n_features_missing = len(self.indicator_.features_)
        non_empty_feature_count = Y.shape[1] - n_features_missing
        array_filled = Y[:, :non_empty_feature_count].copy()
        missing_mask = Y[:, non_empty_feature_count:].astype(bool)

        n_features_original = len(self.statistics_)
        shape_original = (Y.shape[0], n_features_original)
        Y_original = np.zeros(shape_original)
        Y_original[:, self.indicator_.features_] = missing_mask
        full_mask = Y_original.astype(bool)

        filled_idx, original_idx = 0, 0
        while filled_idx < len(array_filled.T):
            if not np.all(Y_original[:, original_idx]):
                Y_original[:, original_idx] = array_filled.T[filled_idx]
                filled_idx += 1
                original_idx += 1
            else:
                original_idx += 1

        Y_original[full_mask] = self.missing_values
        return Y_original

    def test_byteswapping_and_unaligned(dtype, value, swap):
        # Try to create "interesting" values within the valid unicode range:
        dtype = np.dtype(dtype)
        data = [f"x,{value}\n"]  # repr as PyPy `str` truncates some
        if swap:
            dtype = dtype.newbyteorder()
        full_dt = np.dtype([("a", "S1"), ("b", dtype)], align=False)
        # The above ensures that the interesting "b" field is unaligned:
        assert full_dt.fields["b"][1] == 1
        res = np.loadtxt(data, dtype=full_dt, delimiter=",",
                         max_rows=1)  # max-rows prevents over-allocation
        assert res["b"] == dtype.type(value)

    def duplicate(self, new_name=None):
        """Return a duplicate of this GDALRaster."""
        if not new_name:
            if self.driver.name != "MEM":
                new_name = f"{self.name}_dup.{self.driver.name}"
            else:
                new_name = os.path.join(VSI_MEM_FILESYSTEM_BASE_PATH, str(uuid.uuid4()))

        return GDALRaster(
            capi.copy_ds(
                self.driver._ptr,
                force_bytes(new_name),
                self._ptr,
                c_int(),
                c_char_p(),
                c_void_p(),
                c_void_p()
            ),
            write=self._write
        )

def process_whitespace_delimited(ws):
    from io import StringIO
    import numpy as np

    txt = StringIO(
        f"1 2{ws}30\n\n{ws}\n"
        f"4 5 60{ws}\n  {ws}  \n"
        f"7 8 {ws} 90\n  # comment\n"
        f"3 2 1"
    )

    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])

    delimiter_none_result = np.loadtxt(txt, dtype=int, delimiter=None, comments="#")
    assert_equal(delimiter_none_result, expected)

def transform(self, gs_input, resample="Bilinear", tolerance=0.0):
    """
    Return a transformed GDALRaster with the given input characteristics.

    The input is expected to be a dictionary containing the parameters
    of the target raster. Allowed values are width, height, SRID, origin,
    scale, skew, datatype, driver, and name (filename).

    By default, the transform function keeps all parameters equal to the values
    of the original source raster. For the name of the target raster, the
    name of the source raster will be used and appended with
    _copy. + source_driver_name.

    In addition, the resampling algorithm can be specified with the "resample"
    input parameter. The default is Bilinear. For a list of all options
    consult the GDAL_RESAMPLE_ALGORITHMS constant.
    """
    # Get the parameters defining the geotransform, srid, and size of the raster
    gs_input.setdefault("width", self.width)
    gs_input.setdefault("height", self.height)
    gs_input.setdefault("srid", self.srs.srid)
    gs_input.setdefault("origin", self.origin)
    gs_input.setdefault("scale", self.scale)
    gs_input.setdefault("skew", self.skew)
    # Get the driver, name, and datatype of the target raster
    gs_input.setdefault("driver", self.driver.name)

    if "name" not in gs_input:
        gs_input["name"] = self.name + "_copy." + self.driver.name

    if "datatype" not in gs_input:
        gs_input["datatype"] = self.bands[0].datatype()

    # Instantiate raster bands filled with nodata values.
    gs_input["bands"] = [{"nodata_value": bnd.nodata_value} for bnd in self.bands]

    # Create target raster
    target = GDALRaster(gs_input, write=True)

    # Select resampling algorithm
    algorithm = GDAL_RESAMPLE_ALGORITHMS[resample]

    # Reproject image
    capi.reproject_image(
        self._ptr,
        self.srs.wkt.encode(),
        target._ptr,
        target.srs.wkt.encode(),
        algorithm,
        0.0,
        tolerance,
        c_void_p(),
        c_void_p(),
        c_void_p(),
    )

    # Make sure all data is written to file
    target._flush()

    return target

