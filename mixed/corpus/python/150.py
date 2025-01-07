def test_hash_sparse_input_siphash_custom(self):
        layer = layers.Hashing(num_bins=2, salt=[137, 133])
        indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
        sparse_tensor_input = tf.SparseTensor(
            indices=indices,
            values=["omar", "stringer", "marlo", "wire", "skywalker"],
            dense_shape=[3, 2],
        )
        output = layer(sparse_tensor_input)
        self.assertAllClose(output.indices, indices)
        # The result should be same with test_hash_dense_input_siphash.
        self.assertAllClose([1, 0, 1, 0, 1], output.values)

        layer_2 = layers.Hashing(num_bins=2, salt=[137, 211])
        output = layer_2(sparse_tensor_input)
        # The result should be same with test_hash_dense_input_siphash.
        self.assertAllClose([0, 1, 0, 1, 0], output.values)

def data_index(self) -> Index:
    data_index = self.obj.index
    if (
        isinstance(data_index, (ABCDatetimeIndex, ABCPeriodIndex))
        and self.date_format is not None
    ):
        data_index = Index(
            [x.strftime(self.date_format) if notna(x) else "" for x in data_index]
        )
    elif isinstance(data_index, ABCMultiIndex):
        data_index = data_index.remove_unused_levels()
    return data_index

def initialize_partition(partition_key: int) -> None:
    self.partition_id = partition_key
    self.nodes_set = set()
    partitions_parent_map = {}
    for node in nodes:
        self.nodes_set.add(node)
    self.children_partitions = set()
    self.parents_partitions = partitions_parent_map
    self.bfs_level_value = -1
    self.used_memory_bytes = 0
    self.device_ids_list = []

