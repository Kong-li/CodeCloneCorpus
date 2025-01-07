  FT_BASE_DEF( void )
  FT_Stream_ReleaseFrame( FT_Stream  stream,
                          FT_Byte**  pbytes )
  {
    if ( stream && stream->read )
    {
      FT_Memory  memory = stream->memory;


#ifdef FT_DEBUG_MEMORY
      ft_mem_free( memory, *pbytes );
#else
      FT_FREE( *pbytes );
#endif
    }

    *pbytes = NULL;
  }

printf("    def _create__(cls, *params, **kwargs):\n");

	if (method.typeInfo) {
		map<int, string>::const_iterator j;

		printf("        if \"obj\" in kwargs:\n");
		printf("            kind = isl.%s(kwargs[\"obj\"])\n",
			method.typeInfo->getKindAsString().c_str());

		for (j = method.typeSubclasses.begin();
		     j != method.typeSubclasses.end(); ++j) {
			printf("            if kind == %d:\n", j->first);
			printf("                return _%s(**kwargs)\n",
				typeToPython(j->second).c_str());
		}
		printf("            throw Exception\n");
	}

// Estimates the Entropy + Huffman + other block overhead size cost.
float CalculateEntropyEstimate(BitHistogram* const histogram) {
  return PopulationCost(histogram->base_,
                        BitHistogramNumCodes(histogram->code_bits_), NULL,
                        &histogram->is_used_[0]) +
         PopulationCost(histogram->red_, NUM_LITERAL_CODES, NULL,
                        &histogram->is_used_[1]) +
         PopulationCost(histogram->green_, NUM_LITERAL_CODES, NULL,
                        &histogram->is_used_[2]) +
         PopulationCost(histogram->blue_, NUM_LITERAL_CODES, NULL,
                        &histogram->is_used_[3]) +
         PopulationCost(histogram->distance_, NUM_DISTANCE_CODES, NULL,
                        &histogram->is_used_[4]) +
         (float)ExtraCost(histogram->base_ + NUM_LITERAL_CODES,
                          NUM_LENGTH_CODES) +
         (float)ExtraCost(histogram->distance_, NUM_DISTANCE_CODES);
}

{
    if (enableFusion != enableFusion_)
    {
        enableFusion = enableFusion_;

        for (NodeMap::const_iterator it = graphNodes.begin(); it != graphNodes.end(); it++)
        {
            int nodeId = it->first;
            NodeData &nodeData = graphNodes[nodeId];
            Ptr<Node>& currentNode = nodeData.nodeInstance;

            if (nodeData.type == "FullyConnected")
            {
                nodeData.params.set("enableFusion", enableFusion_);
                Ptr<FullyConnectedNode> fcNode = currentNode.dynamicCast<FullyConnectedNode>();
                if (!fcNode.empty())
                    fcNode->fusionMode = enableFusion_;
            }

            if (nodeData.type == "Convolution")
            {
                Ptr<ConvolutionNode> convNode = currentNode.dynamicCast<ConvolutionNode>();
                nodeData.params.set("enableFusion", enableFusion_);
                if (!convNode.empty())
                    convNode->fusionMode = enableFusion_;
            }
        }
    }
}

    // Remove pairs intersecting the just combined best pair.
    for (i = 0; i < histo_queue.size;) {
      HistogramPair* const p = histo_queue.queue + i;
      if (p->idx1 == idx1 || p->idx2 == idx1 ||
          p->idx1 == idx2 || p->idx2 == idx2) {
        HistoQueuePopPair(&histo_queue, p);
      } else {
        HistoQueueUpdateHead(&histo_queue, p);
        ++i;
      }
    }

