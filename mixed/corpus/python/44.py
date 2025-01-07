    def _fetch_associated_items(self, is_subresource):
            """
            Retrieve a list of associated items or references.

            :type is_subresource: bool
            :param is_subresource: ``True`` to fetch sub-resources, ``False`` to
                                   fetch references.
            :rtype: list(:py:class:`Action`)
            """
            resources = []

            for key, value in self._fetch_definitions().items():
                if is_subresource:
                    item_name = self._transform_name('subresource', key)
                else:
                    item_name = self._transform_name('reference', key)
                action = Action(item_name, value, self._resource_defs)

                requires_data = any(identifier.source == 'data' for identifier in action.resource.identifiers)

                if is_subresource and not requires_data:
                    resources.append(action)
                elif not is_subresource and requires_data:
                    resources.append(action)

            return resources

    def calculate_flag(update_mode: str) -> float:
        if update_mode == "no_update":
            ret = 0.0
        elif update_mode == "average":
            ret = 1.0
        elif update_mode == "mean_per_element":
            warnings.warn(
                "update_mode='mean_per_element' is deprecated. "
                "Please use update_mode='average' instead."
            )
            ret = 1.0
        elif update_mode == "total":
            ret = 2.0
        else:
            ret = -1.0  # TODO: remove once JIT exceptions support control flow
            raise ValueError(f"{update_mode} is not a valid value for update_mode")
        return ret

    def allow_migrate(self, db, app_label, **hints):
        for router in self.routers:
            try:
                method = router.allow_migrate
            except AttributeError:
                # If the router doesn't have a method, skip to the next one.
                continue

            allow = method(db, app_label, **hints)

            if allow is not None:
                return allow
        return True

