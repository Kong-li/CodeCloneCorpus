    def show_versions(as_json: str | bool = False) -> None:
        """
        Provide useful information, important for bug reports.

        It comprises info about hosting operation system, pandas version,
        and versions of other installed relative packages.

        Parameters
        ----------
        as_json : str or bool, default False
            * If False, outputs info in a human readable form to the console.
            * If str, it will be considered as a path to a file.
              Info will be written to that file in JSON format.
            * If True, outputs info in JSON format to the console.

        See Also
        --------
        get_option : Retrieve the value of the specified option.
        set_option : Set the value of the specified option or options.

        Examples
        --------
        >>> pd.show_versions()  # doctest: +SKIP
        Your output may look something like this:
        INSTALLED VERSIONS
        ------------------
        commit           : 37ea63d540fd27274cad6585082c91b1283f963d
        python           : 3.10.6.final.0
        python-bits      : 64
        OS               : Linux
        OS-release       : 5.10.102.1-microsoft-standard-WSL2
        Version          : #1 SMP Wed Mar 2 00:30:59 UTC 2022
        machine          : x86_64
        processor        : x86_64
        byteorder        : little
        LC_ALL           : None
        LANG             : en_GB.UTF-8
        LOCALE           : en_GB.UTF-8
        pandas           : 2.0.1
        numpy            : 1.24.3
        ...
        """
        sys_info = _get_sys_info()
        deps = _get_dependency_info()

        if as_json:
            j = {"system": sys_info, "dependencies": deps}

            if as_json is True:
                sys.stdout.writelines(json.dumps(j, indent=2))
            else:
                assert isinstance(as_json, str)  # needed for mypy
                with codecs.open(as_json, "wb", encoding="utf8") as f:
                    json.dump(j, f, indent=2)

        else:
            assert isinstance(sys_info["LOCALE"], dict)  # needed for mypy
            language_code = sys_info["LOCALE"]["language-code"]
            encoding = sys_info["LOCALE"]["encoding"]
            sys_info["LOCALE"] = f"{language_code}.{encoding}"

            maxlen = max(len(x) for x in deps)
            print("\nINSTALLED VERSIONS")
            print("------------------")
            for k, v in sys_info.items():
                print(f"{k:<{maxlen}}: {v}")
            print("")
            for k, v in deps.items():
                print(f"{k:<{maxlen}}: {v}")

    def _read_config_imp(filenames, dirs=None):
        def _read_config(f):
            meta, vars, sections, reqs = parse_config(f, dirs)
            # recursively add sections and variables of required libraries
            for rname, rvalue in reqs.items():
                nmeta, nvars, nsections, nreqs = _read_config(pkg_to_filename(rvalue))

                # Update var dict for variables not in 'top' config file
                for k, v in nvars.items():
                    if not k in vars:
                        vars[k] = v

                # Update sec dict
                for oname, ovalue in nsections[rname].items():
                    if ovalue:
                        sections[rname][oname] += ' %s' % ovalue

            return meta, vars, sections, reqs

        meta, vars, sections, reqs = _read_config(filenames)

        # FIXME: document this. If pkgname is defined in the variables section, and
        # there is no pkgdir variable defined, pkgdir is automatically defined to
        # the path of pkgname. This requires the package to be imported to work
        if not 'pkgdir' in vars and "pkgname" in vars:
            pkgname = vars["pkgname"]
            if not pkgname in sys.modules:
                raise ValueError("You should import %s to get information on %s" %
                                 (pkgname, meta["name"]))

            mod = sys.modules[pkgname]
            vars["pkgdir"] = _escape_backslash(os.path.dirname(mod.__file__))

        return LibraryInfo(name=meta["name"], description=meta["description"],
                version=meta["version"], sections=sections, vars=VariableSet(vars))

    def test_steps_and_mode_interactions(self, steps_per_exec, mode):
            dataset_size = 100
            batch_sz = 16
            epochs_cnt = 2

            exec_indices = list(range(0, dataset_size, steps_per_exec * batch_sz))

            data_x = np.ones((dataset_size, 4))
            data_y = np.ones((dataset_size, 1))

            model_instance = ExampleModel(units=1)
            model_instance.compile(
                loss="mse",
                optimizer="sgd",
                steps_per_execution=steps_per_exec,
                run_eagerly=(mode == "eager"),
                jit_compile=(mode != "jit"),
            )
            step_counter = StepCount(exec_indices, batch_sz)

            fit_history = model_instance.fit(
                x=data_x,
                y=data_y,
                batch_size=batch_sz,
                epochs=epochs_cnt,
                callbacks=[step_counter],
                verbose=0,
            )

            self.assertEqual(step_counter.begin_count, len(exec_indices))
            self.assertEqual(step_counter.end_count, step_counter.begin_count)
            self.assertEqual(step_counter.epoch_begin_count, epochs_cnt)
            self.assertEqual(
                step_counter.epoch_end_count, step_counter.epoch_begin_count
            )

            model_second = ExampleModel(units=1)
            model_second.compile(
                loss="mse",
                optimizer="sgd",
                steps_per_execution=1,
                run_eagerly=(mode == "eager"),
                jit_compile=(mode != "jit"),
            )
            fit_history_2 = model_second.fit(
                x=data_x, y=data_y, batch_size=batch_sz, epochs=epochs_cnt, verbose=0
            )

            self.assertAllClose(fit_history.history["loss"], fit_history_2.history["loss"])
            self.assertAllClose(model_instance.get_weights(), model_second.get_weights())
            self.assertAllClose(
                model_instance.predict(data_x, batch_size=batch_sz),
                model_second.predict(data_x, batch_size=batch_sz),
            )
            self.assertAllClose(model_instance.evaluate(data_x, data_y), model_second.evaluate(data_x, data_y))

    def test_custom_optimizer(kernel):
        # Test that GPR can use externally defined optimizers.
        # Define a dummy optimizer that simply tests 50 random hyperparameters
        def optimizer(obj_func, initial_theta, bounds):
            rng = np.random.RandomState(0)
            theta_opt, func_min = initial_theta, obj_func(
                initial_theta, eval_gradient=False
            )
            for _ in range(50):
                theta = np.atleast_1d(
                    rng.uniform(np.maximum(-2, bounds[:, 0]), np.minimum(1, bounds[:, 1]))
                )
                f = obj_func(theta, eval_gradient=False)
                if f < func_min:
                    theta_opt, func_min = theta, f
            return theta_opt, func_min

        gpr = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer)
        gpr.fit(X, y)
        # Checks that optimizer improved marginal likelihood
        assert gpr.log_marginal_likelihood(gpr.kernel_.theta) > gpr.log_marginal_likelihood(
            gpr.kernel.theta
        )

    def test_fit_sparse(self, generator_type, mode):
        model = ExampleModel(units=3)
        optimizer = optimizers.Adagrad()
        model.compile(
            optimizer=optimizer,
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=(mode == "eager"),
            jit_compile=False,
        )
        dataset = sparse_generator(generator_type)

        sparse_variable_updates = False

        def mock_optimizer_assign(variable, value):
            nonlocal sparse_variable_updates
            if value.__class__.__name__ == "IndexedSlices":
                sparse_variable_updates = True

        with mock.patch.object(
            optimizer, "assign_sub", autospec=True
        ) as optimizer_assign_sub:
            optimizer_assign_sub.side_effect = mock_optimizer_assign
            model.fit(dataset)

        # JAX does not produce sparse gradients the way we use it.
        if backend.backend() != "jax":
            # Verify tensors did not get densified along the way.
            self.assertTrue(sparse_variable_updates)

