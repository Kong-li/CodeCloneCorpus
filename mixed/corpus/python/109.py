    def merge_single_node(node: Node, id: Optional[int]):
        def _update_partition_map(node: Node, id: int):
            # Iterate through all the users of this node and update the partition map to indicate
            # that there is a path from the partition id of this node to the target partition id.
            for user_node in node.users:
                target_id = assignment.get(user_node, None)
                if target_id is not None:
                    partition_map[id].add(target_id)
                    partition_map[id].update(partition_map[target_id])

            # Iterate through all the upstream nodes of this node and update the partition map
            # to indicate that there is a path from the partition id of the upstream node to the
            # current node's partition id.
            upstream_nodes = self.dependency_viewer.upstreams_of(node)
            for curr_node in upstream_nodes:
                source_id = assignment.get(curr_node, None)
                if source_id is not None:
                    partition_map[source_id].add(id)

        if node in assignment:
            partitions_by_id[assignment[node]].remove_node(node)

        if id is None:
            assignment.pop(node)
        elif id not in partitions_by_id:
            assignment[node] = id
            partitions_by_id[id] = Partition(id=id, nodes=[node])
            _update_partition_map(node, id)
        else:
            assignment[node] = id
            partitions_by_id[id].add_node(node)
            _update_partition_map(node, id)

def record_error(
    self,
    err_info: (tuple[type, BaseException, TracebackType] | tuple[None, None, None]),
) -> None:
    """Records an error.  This is called by :meth:`process_exception`
    if debugging is disabled and right before the handler is called.
    The default implementation logs the error as critical on the
    :attr:`logger`.

    .. versionadded:: 0.9
    """
    self.logger.critical(
        f"Error on {request.url} [{request.method}]", exc_info=err_info
    )

def get_data_files(data):
    if is_string(data):
        return [data]
    sources = data[1]
    filenames = []
    for s in sources:
        if hasattr(s, '__call__'):
            continue
        if is_local_src_dir(s):
            filenames.extend(list(general_source_files(s)))
        elif is_string(s):
            if os.path.isfile(s):
                filenames.append(s)
            else:
                print('Not existing data file:', s)
        else:
            raise TypeError(repr(s))
    return filenames

    def test_custom_layer_variations(self):
            factor = 2
            layer = CustomLayer(factor=factor)
            x = ops.random.normal(shape=(2, 2))
            y1 = layer(x)
            _, new_layer, _ = self.roundtrip(
                layer,
                custom_objects={"CustomLayer": CustomLayer}
            )
            y2 = new_layer(x)
            self.assertAllClose(y1, y2, atol=1e-5)

            factor_nested = 2
            nested_layer = NestedCustomLayer(factor=factor_nested)
            x_nested = ops.random.normal(shape=(2, 2))
            y3 = nested_layer(x_nested)
            _, new_nested_layer, _ = self.roundtrip(
                nested_layer,
                custom_objects={
                    "NestedCustomLayer": NestedCustomLayer,
                    "custom_fn": custom_fn
                }
            )
            new_nested_layer.set_weights(nested_layer.get_weights())
            y4 = new_nested_layer(x_nested)
            self.assertAllClose(y3, y4, atol=1e-5)

