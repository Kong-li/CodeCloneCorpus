      def dom_id(nodes)
        dom_id = dom_id_text(nodes.last.text)

        # Fix duplicate dom_ids by prefixing the parent node dom_id
        if @node_ids[dom_id]
          if @node_ids[dom_id].size > 1
            duplicate_nodes = @node_ids.delete(dom_id)
            new_node_id = dom_id_with_parent_node(dom_id, duplicate_nodes[-2])
            duplicate_nodes.last[:id] = new_node_id
            @node_ids[new_node_id] = duplicate_nodes
          end

