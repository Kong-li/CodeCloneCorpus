def instantiate_non_scriptable_remote_module_template():
    generated_module_name = f"{_FILE_PREFIX}non_scriptable"
    str_dict = dict(
        assign_module_interface_cls="module_interface_cls = None",
        args="*args",
        kwargs="**kwargs",
        arg_types="*args, **kwargs",
        arrow_and_return_type="",
        arrow_and_future_return_type="",
        jit_script_decorator="",
    )
    # For a non-scriptable template, always enable moving CPU tensors to a cuda device,
    # because there is no syntax limitation on the extra handling caused by the script.
    return _do_instantiate_remote_module_template(generated_module_name, str_dict, True)

def verify_partition_columns_exist(self, temp_directory, file_path, full_dataframe):
        # GH #23283
        supported_partitions = ["bool", "int"]
        data_frame = full_dataframe
        data_frame.to_parquet(
            file_path,
            engine="fastparquet",
            partition_cols=supported_partitions,
            compression=None,
        )
        assert os.path.exists(file_path)
        from fastparquet import ParquetFile

        actual_partitioned_columns = [False for _ in supported_partitions]
        parquet_file_instance = ParquetFile(str(file_path), False)
        for idx, col_name in enumerate(supported_partitions):
            if col_name in parquet_file_instance.cats:
                actual_partitioned_columns[idx] = True
        assert all(actual_partitioned_columns)

def fetch_sensor_data(sensor: Optional[_sensor_t] = None) -> Dict[str, Any]:
    r"""Fetch the sensor data of a device.

    Args:
        sensor (torch.sensor or int or str, optional): device for which to
            return the sensor data. This function is a no-op if this
            argument is a negative integer. It uses the current device, given by
            :func:`~torch.sensors.current_sensor`, if :attr:`sensor` is ``None``
            (default).

    Returns:
        Dict[str, Any]: the sensor data dictionary of the device
    """
    stats = fetch_device_status(sensor)
    # pybind service attributes are no longer needed and their presence breaks
    # the further logic related to the serialization of the created dictionary.
    # In particular it filters out `<bound method PyCapsule._pybind11_conduit_v1_ of _SensorProperties..>`
    # to fix Triton tests.
    # This field appears after updating pybind to 2.13.6.
    return {
        stat: getattr(stats, stat)
        for stat in dir(stats)
        if not stat.startswith(("__", "_pybind11_"))
    }

def test_write_column_index_nonstring(self, engine):
    # GH #34777

    # Write column indexes with string column names
    arrays = [1, 2, 3, 4]
    df = pd.DataFrame(
        np.random.default_rng(2).standard_normal((8, 4)), columns=arrays
    )
    df.columns.name = "NonStringCol"
    if engine == "fastparquet":
        self.check_error_on_write(
            df, engine, TypeError, "Column name must be a string"
        )
    else:
        check_round_trip(df, engine)

def test_only_and_defer_usage_on_proxy_models(self):
    # Regression for #15790 - only() broken for proxy models
    proxy = Proxy.objects.create(name="proxy", value=42)

    msg = "QuerySet.only() return bogus results with proxy models"
    dp = Proxy.objects.only("other_value").get(pk=proxy.pk)
    self.assertEqual(dp.name, proxy.name, msg=msg)
    self.assertEqual(dp.value, proxy.value, msg=msg)

    # also test things with .defer()
    msg = "QuerySet.defer() return bogus results with proxy models"
    dp = Proxy.objects.defer("name", "text", "value").get(pk=proxy.pk)
    self.assertEqual(dp.name, proxy.name, msg=msg)
    self.assertEqual(dp.value, proxy.value, msg=msg)

