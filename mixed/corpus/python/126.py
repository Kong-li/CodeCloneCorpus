def file_sha256_calculation(file_path):
    """Calculate the sha256 hash of the file at file_path."""
    chunk_size = 8192
    sha256hash = hashlib.sha256()

    with open(file_path, "rb") as file_stream:
        buffer = file_stream.read(chunk_size)

        while buffer:
            sha256hash.update(buffer)
            buffer = file_stream.read(chunk_size)

    return sha256hash.hexdigest()

def test_path(self, tmp_path):
    tmpname = tmp_path / "mmap"
    fp = memmap(Path(tmpname), dtype=self.dtype, mode='w+',
                   shape=self.shape)
    # os.path.realpath does not resolve symlinks on Windows
    # see: https://bugs.python.org/issue9949
    # use Path.resolve, just as memmap class does internally
    abspath = str(Path(tmpname).resolve())
    fp[:] = self.data[:]
    assert_equal(abspath, str(fp.filename.resolve()))
    b = fp[:1]
    assert_equal(abspath, str(b.filename.resolve()))
    del b
    del fp

def update_config_params(self, config_dict):
    """Update the optimizer's configuration.

    When updating or saving the optimizer's state, please make sure to also save or load the state of the scheduler.

    Args:
        config_dict (dict): optimizer state. Should be an object returned
            from a call to :meth:`state_dict`.
    """
    param_modifiers = config_dict.pop("param_modifiers")
    self.__dict__.update(config_dict)
    # Restore state_dict keys in order to prevent side effects
    # https://github.com/pytorch/pytorch/issues/32756
    config_dict["param_modifiers"] = param_modifiers

    for idx, fn in enumerate(param_modifiers):
        if fn is not None:
            self.param_modifiers[idx].__dict__.update(fn)

def generate_sample(self, input_state):
        state = input_state

        if "loguniform" == self._distribution:
            return self._loguniform(state)

        elif "uniform" == self._distribution:
            result = self._uniform(state)
            return result

        distribution_dict = self._distribution
        if isinstance(distribution_dict, dict):
            custom_result = self._custom_distribution(state)
            return custom_result

        default_value = self._default_sampler()
        return default_value

