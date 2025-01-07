    def _update_list(self, size, elements):
            mem_ptr = self._create_point(size, elements)
            if mem_ptr:
                srid_value = self.srid
                capi.destroy_geom(self.ptr)
                self._ptr = mem_ptr
                if srid_value is not None:
                    self.srid = srid_value
                self._post_init()
            else:
                # can this happen?
                raise GEOSException("Geometry resulting from slice deletion was invalid.")

    def decorator(klass):
        def __new__(cls, *args, **kwargs):
            # We capture the arguments to make returning them trivial
            obj = super(klass, cls).__new__(cls)
            obj._constructor_args = (args, kwargs)
            return obj

        def deconstruct(obj):
            """
            Return a 3-tuple of class import path, positional arguments,
            and keyword arguments.
            """
            # Fallback version
            if path and type(obj) is klass:
                module_name, _, name = path.rpartition(".")
            else:
                module_name = obj.__module__
                name = obj.__class__.__name__
            # Make sure it's actually there and not an inner class
            module = import_module(module_name)
            if not hasattr(module, name):
                raise ValueError(
                    "Could not find object %s in %s.\n"
                    "Please note that you cannot serialize things like inner "
                    "classes. Please move the object into the main module "
                    "body to use migrations.\n"
                    "For more information, see "
                    "https://docs.djangoproject.com/en/%s/topics/migrations/"
                    "#serializing-values" % (name, module_name, get_docs_version())
                )
            return (
                (
                    path
                    if path and type(obj) is klass
                    else f"{obj.__class__.__module__}.{name}"
                ),
                obj._constructor_args[0],
                obj._constructor_args[1],
            )

        klass.__new__ = staticmethod(__new__)
        klass.deconstruct = deconstruct

        return klass

