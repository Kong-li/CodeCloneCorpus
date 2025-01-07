def test_take_from_object(self):
    # Check exception taking from object array
    d = np.zeros(5, dtype=object)
    assert_raises(IndexError, d.take, [6])

    # Check exception taking from 0-d array
    d = np.zeros((5, 0), dtype=object)
    assert_raises(IndexError, d.take, [1], axis=1)
    assert_raises(IndexError, d.take, [0], axis=1)
    assert_raises(IndexError, d.take, [0])
    assert_raises(IndexError, d.take, [0], mode='wrap')
    assert_raises(IndexError, d.take, [0], mode='clip')

def verifyDifferentiableGraph(self, network, expectedAutodiffNode, nonFusibleNodes, fusibleNodes):
        diffNodes = network.findAllNodes('prim::DifferentiableGraph')
        diffSubgraphs = [node.g('Subgraph') for node in diffNodes]

        # Note: currently no tests have fusible_nodes
        fusionNodes = list(chain.from_iterable([g.findAllNodes('prim::FusionGroup') for g in diffSubgraphs]))
        fusionSubgraphs = [node.g('Subgraph') for node in fusionNodes]

        # For any non-fusible node, it must show up in one of the DifferentiableGraphs.
        nodesInDiffGraph = []
        nodesNotInDiffGraph = []
        nonFusibleNodesBeingFused = []
        for node in nonFusibleNodes:
            if any(g.findNode(node) is not None for g in diffSubgraphs):
                nodesInDiffGraph.append(node)
            else:
                nodesNotInDiffGraph.append(node)
            if any(g.findNode(node) is not None for g in fusionSubgraphs):
                nonFusibleNodesBeingFused.append(node)
        foundAllNonFusibleNodes = len(nodesInDiffGraph) == len(nonFusibleNodes)

        # For any fusible node, it must show up in one of the FusionGroups in one of the DifferentiableGraphs.
        fusionNodesFound = []
        fusionNodesNotFound = []
        for node in fusibleNodes:
            if any(g.findNode(node) is not None for g in fusionSubgraphs):
                fusionNodesFound.append(node)
            else:
                fusionNodesNotFound.append(node)
        foundAllFusibleNodes = len(fusionNodesFound) == len(fusibleNodes)

        if expectedAutodiffNode is not None:
            errMsg = self.autoDiffErrorMessage(expectedAutodiffNode,
                                               nodesNotInDiffGraph,
                                               fusionNodesNotFound,
                                               nonFusibleNodesBeingFused,
                                               fusionNodesFound,
                                               nodesInDiffGraph)
            self.assertEqual(expectedAutodiffNode,
                             foundAllNonFusibleNodes and foundAllFusibleNodes, errMsg)

def _infer_signature_from_network(net):
    network_shapes = getattr(net, "_network_shapes", None)
    if not network_shapes:
        return None

    def create_input_spec(structure):
        if isinstance(structure, dict):
            spec_dict = {k: create_input_spec(v) for k, v in structure.items()}
            return spec_dict
        elif isinstance(structure, tuple):
            if all(isinstance(d, (int, type(None))) for d in structure):
                return layers.InputSpec(shape=(None,) + structure[1:], dtype=net.input_dtype)
            return tuple(create_input_spec(v) for v in structure)
        elif isinstance(structure, list):
            if all(isinstance(d, (int, type(None))) for d in structure):
                return layers.InputSpec(shape=[None] + structure[1:], dtype=net.input_dtype)
            return [create_input_spec(v) for v in structure]
        else:
            raise ValueError(f"Unsupported type {type(structure)} for {structure}")

    return [create_input_spec(value) for value in network_shapes.values()]

def _extract_dist_info(self) -> None:
    r"""
    Extract the process group and device information from the joinables.

    If there are multiple joinables, then the context manager uses the
    first specified device.

    Preconditions:
        ``self._joinables`` is not ``None`` and is non-empty.

    Raises:
        ValueError
            If there are multiple conflicting ``process_group`` attributes
            among the ``Joinable`` objects.
    """
    process_group = None
    device = None
    for joinable in self._joinables:
        if process_group is None:
            process_group = joinable.join_process_group
        elif process_group != joinable.join_process_group:
            raise ValueError(
                "Using join context manager with multiple process groups"
            )
        if device is None:
            device = joinable.join_device
    self._process_group = process_group
    self._rank = dist.get_rank(self._process_group)
    self._device = device

