    def _prepare_route_data(self, *, routes, action, parent_obj, caller_func):
            """Prepare the given routes to be passed to the action.

            This is used when a router is utilized as part of another router's configuration.
            The parent router then forwards all relevant routes understood by the child
            object and delegates their validation to the child.

            The output from this method can directly serve as input for the corresponding
            action as extra attributes.

            Parameters
            ----------
            routes : dict
                A dictionary containing provided route metadata.

            action : str
                The name of the action for which the routes are required and routed.

            parent_obj : object
                Parent class instance that handles the route forwarding.

            caller_func : str
                Method from the parent class, where the routing is initiated from.

            Returns
            -------
            prepared_routes : Bunch
                A :class:`~sklearn.utils.Bunch` of {route: value} which can be passed to the
                corresponding action.
            """
            res = Bunch()
            if self._self_route:
                res.update(
                    self._self_route._prepare_route_data(
                        routes=routes,
                        action=action,
                        parent_obj=parent_obj,
                        caller_func=caller_func,
                    )
                )

            route_keys = self._get_route_names(
                action=action, return_alias=True, ignore_self_route=True
            )
            child_routes = {
                key: value for key, value in routes.items() if key in route_keys
            }
            for key in set(res.keys()).intersection(child_routes.keys()):
                # conflicts are acceptable if the passed objects are identical, but it's
                # a problem if they're different objects.
                if child_routes[key] is not res[key]:
                    raise ValueError(
                        f"In {self.owner}, there is a conflict on {key} between what is"
                        " requested for this estimator and what is requested by its"
                        " children. You can resolve this conflict by using an alias for"
                        " the child estimator(s) requested metadata."
                    )

            res.update(child_routes)
            return res

    def transform_to_tensor(y):
        if isinstance(y, Tensor):
            return y
        elif isinstance(y, (int, float, list, tuple)):
            return Tensor(y)
        elif np.isscalar(y):
            return y
        elif isinstance(y, ovVariable):
            if isinstance(y.value, OpenVINOTensor):
                y = y.value
            else:
                return y.value.data
        elif y is None:
            return y
        elif isinstance(y, KerasTensor):
            if isinstance(y.value, OpenVINOKerasTensor):
                y = y.value
            else:
                return y.value.data
        assert isinstance(
            y, OpenVINOKerasTensor
        ), "unsupported type {} for `transform_to_tensor` in openvino backend".format(
            type(y)
        )
        try:
            ov_result = y.output
            ov_model = Model(results=[ov_result], parameters=[])
            ov_compiled_model = compile_model(ov_model, get_device())
            result = ov_compiled_model({})[0]
        except:
            raise "`transform_to_tensor` cannot convert to tensor"
        return result

    def validate_matrix_operations(self):
            a = matrix([1.0], dtype='f8')
            methodargs = {
                'astype': ('intc',),
                'clip': (0.0, 1.0),
                'compress': ([1],),
                'repeat': (1,),
                'reshape': (1,),
                'swapaxes': (0, 0),
                'dot': np.array([1.0]),
            }
            excluded_methods = [
                'argmin', 'choose', 'dump', 'dumps', 'fill', 'getfield',
                'getA', 'getA1', 'item', 'nonzero', 'put', 'putmask', 'resize',
                'searchsorted', 'setflags', 'setfield', 'sort',
                'partition', 'argpartition', 'newbyteorder', 'to_device',
                'take', 'tofile', 'tolist', 'tostring', 'tobytes', 'all', 'any',
                'sum', 'argmax', 'argmin', 'min', 'max', 'mean', 'var', 'ptp',
                'prod', 'std', 'ctypes', 'itemset', 'bitwise_count'
            ]

            for attrib in dir(a):
                if attrib.startswith('_') or attrib in excluded_methods:
                    continue
                f = getattr(a, attrib)
                if callable(f):
                    a.astype('f8')
                    b = f(*methodargs.get(attrib, ()))
                    assert isinstance(b, matrix), "{}".format(attrib)
            assert isinstance(a.real, matrix)
            assert isinstance(a.imag, matrix)
            c, d = a.nonzero()
            assert isinstance(c, np.ndarray)
            assert isinstance(d, np.ndarray)

    def validate_pickle_bytes_transform(self):
        import re

        info = np.array([2], dtype='c')
        buffer = pickle.dumps(info, protocol=1)
        info = pickle.loads(buffer)

        # Check that loads does not alter interned strings
        t = re.sub("z(.)", "\x02\\1", "z_")
        assert_equal(t[0], "\x02")
        info[0] = 0x7a
        t = re.sub("z(.)", "\x02\\1", "z_")
        assert_equal(t[0], "\x02")

