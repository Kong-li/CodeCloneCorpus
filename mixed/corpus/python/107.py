    def test_employee(self):
        "Employees can be created and can set their password"
        e = Employee.objects.create_user("testemp", "test@example.com", "testpw")
        self.assertTrue(e.has_usable_password())
        self.assertFalse(e.check_password("bad"))
        self.assertTrue(e.check_password("testpw"))

        # Check we can manually set an unusable password
        e.set_unusable_password()
        e.save()
        self.assertFalse(e.check_password("testpw"))
        self.assertFalse(e.has_usable_password())
        e.set_password("testpw")
        self.assertTrue(e.check_password("testpw"))
        e.set_password(None)
        self.assertFalse(e.has_usable_password())

        # Check username getter
        self.assertEqual(e.get_username(), "testemp")

        # Check authentication/permissions
        self.assertFalse(e.is_anonymous)
        self.assertTrue(e.is_authenticated)
        self.assertFalse(e.is_staff)
        self.assertTrue(e.is_active)
        self.assertFalse(e.is_superuser)

        # Check API-based user creation with no password
        e2 = Employee.objects.create_user("testemp2", "test2@example.com")
        self.assertFalse(e2.has_usable_password())

    def load_partitioned_optimizer_state_dict(
        model_state_dict: STATE_DICT_TYPE,
        optimizer_key: str,
        storage_reader: StorageReader,
        planner: Optional[LoadPlanner] = None,
    ) -> STATE_DICT_TYPE:
        """
        Load a state_dict in conjunction with FSDP partitioned optimizer state.

        This is the current recommended way to checkpoint FSDP.
        >>> # xdoctest: +SKIP
        >>> import torch.distributed.checkpoint as dist_cp
        >>> # Save
        >>> model: torch.nn.Model
        >>> optim_params = model.parameters()
        >>> optim = torch.optim.SGD(optim_params, lr=0.01)
        >>> # Save
        >>> with FSDP.state_dict_type(model, StateDictType.PARTITIONED_STATE_DICT):
        >>>     state_dict = {
        >>>         "optimizer": FSDP.optim_state_dict(model, optim),
        >>>         "model": model.state_dict()
        >>>     }
        >>>     dist_cp.save_state_dict(
        >>>         state_dict=optim_state,
        >>>         storage_writer=dist_cp.FileSystemWriter("checkpoint"),
        >>>         planner=dist_cp.DefaultSavePlanner(),
        >>>     )
        >>>
        >>> # Load
        >>> with FSDP.state_dict_type(model_tp, StateDictType.PARTITIONED_STATE_DICT):
        >>>     model_state_dict = model_tp.state_dict()
        >>>     checkpoint = {
        >>>         "model": model_state_dict
        >>>     }
        >>>     dist_cp.load_state_dict(
        >>>         state_dict=checkpoint,
        >>>         storage_reader=dist_cp.FileSystemReader(checkpoint_file),
        >>>         planner=dist_cp.DefaultLoadPlanner(),
        >>>     )
        >>>     model.load_state_dict(checkpoint["model_state"])
        >>>
        >>>     optim_state = dist_cp.load_partitioned_optimizer_state_dict(
        >>>         model_state_dict,
        >>>         optimizer_key="optimizer",
        >>>         storage_reader=dist_cp.FileSystemReader("checkpoint"),
        >>>     )
        >>>
        >>>     flattened_osd = FSDP.optim_state_dict_to_load(
        >>>        model, optim, optim_state["optimizer"]
        >>>     )
        >>>
        >>>     optim.load_state_dict(flattened_osd)
        """
        metadata = storage_reader.read_metadata()

        layout_specs, dp_pg = _get_state_dict_2d_layout(model_state_dict)
        dp_pg_device_type = dist.distributed_c10d._get_pg_default_device(dp_pg).type
        device_module = _get_device_module(dp_pg_device_type)

        if dp_pg is None:
            placements = []
            for i in range(dist.get_world_size()):
                device_info = _normalize_device_info(
                    dp_pg_device_type, i % device_module.device_count()
                )
                placements.append(f"rank:{i}/{device_info}")
            sharding_spec = ChunkShardingSpec(dim=0, placements=placements)  # type: ignore[arg-type]
        else:
            sharding_spec = _create_colwise_spec(dp_pg)

        # Create a state_dict for optimizer state
        state_dict: STATE_DICT_TYPE = {}

        fqn_to_offset: Dict[str, Sequence[int]] = {}
        for key, value in metadata.state_dict_metadata.items():
            key_path = metadata.planner_data[key]
            if key_path[0] != optimizer_key:
                continue

            if isinstance(value, BytesStorageMetadata):
                state_dict[key] = "<bytes_io>"
                continue
            tensor = _alloc_tensor(
                value.properties, shard_sizes, dp_pg_device_type
            )
            if spec_key in layout_specs and layout_specs[spec_key][0] is not None:
                fqn_to_offset[key] = cast(Sequence[int], layout_specs[spec_key][0])
            state_dict[key] = ShardedTensor._init_from_local_shards_and_global_metadata(
                local_shards, st_md, process_group=dp_pg
            )

        # Whether we unflatten before or after doesn't matter
        load_state_dict(
            state_dict=state_dict,
            storage_reader=storage_reader,
            # FIXME the type of planner is wrong in load_state_dict
            planner=_ReaderWithOffset(fqn_to_offset) if dp_pg is not None else planner,
        )

        state_dict = unflatten_state_dict(state_dict, metadata.planner_data)

        return state_dict

    def test_array_array():
        tobj = type(object)
        ones11 = np.ones((1, 1), np.float64)
        tndarray = type(ones11)
        # Test is_ndarray
        assert_equal(np.array(ones11, dtype=np.float64), ones11)
        if HAS_REFCOUNT:
            old_refcount = sys.getrefcount(tndarray)
            np.array(ones11)
            assert_equal(old_refcount, sys.getrefcount(tndarray))

        # test None
        assert_equal(np.array(None, dtype=np.float64),
                     np.array(np.nan, dtype=np.float64))
        if HAS_REFCOUNT:
            old_refcount = sys.getrefcount(tobj)
            np.array(None, dtype=np.float64)
            assert_equal(old_refcount, sys.getrefcount(tobj))

        # test scalar
        assert_equal(np.array(1.0, dtype=np.float64),
                     np.ones((), dtype=np.float64))
        if HAS_REFCOUNT:
            old_refcount = sys.getrefcount(np.float64)
            np.array(np.array(1.0, dtype=np.float64), dtype=np.float64)
            assert_equal(old_refcount, sys.getrefcount(np.float64))

        # test string
        S2 = np.dtype((bytes, 2))
        S3 = np.dtype((bytes, 3))
        S5 = np.dtype((bytes, 5))
        assert_equal(np.array(b"1.0", dtype=np.float64),
                     np.ones((), dtype=np.float64))
        assert_equal(np.array(b"1.0").dtype, S3)
        assert_equal(np.array(b"1.0", dtype=bytes).dtype, S3)
        assert_equal(np.array(b"1.0", dtype=S2), np.array(b"1."))
        assert_equal(np.array(b"1", dtype=S5), np.ones((), dtype=S5))

        # test string
        U2 = np.dtype((str, 2))
        U3 = np.dtype((str, 3))
        U5 = np.dtype((str, 5))
        assert_equal(np.array("1.0", dtype=np.float64),
                     np.ones((), dtype=np.float64))
        assert_equal(np.array("1.0").dtype, U3)
        assert_equal(np.array("1.0", dtype=str).dtype, U3)
        assert_equal(np.array("1.0", dtype=U2), np.array(str("1.")))
        assert_equal(np.array("1", dtype=U5), np.ones((), dtype=U5))

        builtins = getattr(__builtins__, '__dict__', __builtins__)
        assert_(hasattr(builtins, 'get'))

        # test memoryview
        dat = np.array(memoryview(b'1.0'), dtype=np.float64)
        assert_equal(dat, [49.0, 46.0, 48.0])
        assert_(dat.dtype.type is np.float64)

        dat = np.array(memoryview(b'1.0'))
        assert_equal(dat, [49, 46, 48])
        assert_(dat.dtype.type is np.uint8)

        # test array interface
        a = np.array(100.0, dtype=np.float64)
        o = type("o", (object,),
                 {"__array_interface__": a.__array_interface__})
        assert_equal(np.array(o, dtype=np.float64), a)

        # test array_struct interface
        a = np.array([(1, 4.0, 'Hello'), (2, 6.0, 'World')],
                     dtype=[('f0', int), ('f1', float), ('f2', str)])
        o = type("o", (object,),
                 {"__array_struct__": a.__array_struct__})
        ## wasn't what I expected... is np.array(o) supposed to equal a ?
        ## instead we get a array([...], dtype=">V18")
        assert_equal(bytes(np.array(o).data), bytes(a.data))

        # test array
        def custom__array__(self, dtype=None, copy=None):
            return np.array(100.0, dtype=dtype, copy=copy)

        o = type("o", (object,), {"__array__": custom__array__})()
        assert_equal(np.array(o, dtype=np.float64), np.array(100.0, np.float64))

        # test recursion
        nested = 1.5
        for i in range(ncu.MAXDIMS):
            nested = [nested]

        # no error
        np.array(nested)

        # Exceeds recursion limit
        assert_raises(ValueError, np.array, [nested], dtype=np.float64)

        # Try with lists...
        # float32
        assert_equal(np.array([None] * 10, dtype=np.float32),
                     np.full((10,), np.nan, dtype=np.float32))
        assert_equal(np.array([[None]] * 10, dtype=np.float32),
                     np.full((10, 1), np.nan, dtype=np.float32))
        assert_equal(np.array([[None] * 10], dtype=np.float32),
                     np.full((1, 10), np.nan, dtype=np.float32))
        assert_equal(np.array([[None] * 10] * 10, dtype=np.float32),
                     np.full((10, 10), np.nan, dtype=np.float32))
        # float64
        assert_equal(np.array([None] * 10, dtype=np.float64),
                     np.full((10,), np.nan, dtype=np.float64))
        assert_equal(np.array([[None]] * 10, dtype=np.float64),
                     np.full((10, 1), np.nan, dtype=np.float64))
        assert_equal(np.array([[None] * 10], dtype=np.float64),
                     np.full((1, 10), np.nan, dtype=np.float64))
        assert_equal(np.array([[None] * 10] * 10, dtype=np.float64),
                     np.full((10, 10), np.nan, dtype=np.float64))

        assert_equal(np.array([1.0] * 10, dtype=np.float64),
                     np.ones((10,), dtype=np.float64))
        assert_equal(np.array([[1.0]] * 10, dtype=np.float64),
                     np.ones((10, 1), dtype=np.float64))
        assert_equal(np.array([[1.0] * 10], dtype=np.float64),
                     np.ones((1, 10), dtype=np.float64))
        assert_equal(np.array([[1.0] * 10] * 10, dtype=np.float64),
                     np.ones((10, 10), dtype=np.float64))

        # Try with tuples
        assert_equal(np.array((None,) * 10, dtype=np.float64),
                     np.full((10,), np.nan, dtype=np.float64))
        assert_equal(np.array([(None,)] * 10, dtype=np.float64),
                     np.full((10, 1), np.nan, dtype=np.float64))
        assert_equal(np.array([(None,) * 10], dtype=np.float64),
                     np.full((1, 10), np.nan, dtype=np.float64))
        assert_equal(np.array([(None,) * 10] * 10, dtype=np.float64),
                     np.full((10, 10), np.nan, dtype=np.float64))

        assert_equal(np.array((1.0,) * 10, dtype=np.float64),
                     np.ones((10,), dtype=np.float64))
        assert_equal(np.array([(1.0,)] * 10, dtype=np.float64),
                     np.ones((10, 1), dtype=np.float64))
        assert_equal(np.array([(1.0,) * 10], dtype=np.float64),
                     np.ones((1, 10), dtype=np.float64))
        assert_equal(np.array([(1.0,) * 10] * 10, dtype=np.float64),
                     np.ones((10, 10), dtype=np.float64))

