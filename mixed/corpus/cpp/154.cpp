if (node->endProcess) {
  switch (node->operationNode->op_) {
    case OpType::Sum:
      impl->subTrees_ = {BatchUnion(node->positiveChildren)};
      break;
    case OpType::Intersection: {
      impl->subTrees_ = {
          BatchBoolean(OpType::Intersection, node->positiveChildren)};
      break;
    };
    case OpType::Difference:
      if (node->positiveChildren.empty()) {
        // nothing to difference from, so the result is empty.
        impl->subTrees_ = {std::make_shared<CsgLeafNode>()};
      } else {
        auto positive = BatchUnion(node->positiveChildren);
        if (node->negativeChildren.empty()) {
          // nothing to difference, result equal to the LHS.
          impl->subTrees_ = {node->positiveChildren[0]};
        } else {
          Boolean3 boolean(*positive->GetImpl(),
                           *BatchUnion(node->negativeChildren)->GetImpl(),
                           OpType::Difference);
          impl->subTrees_ = {ImplToLeaf(boolean.Result(OpType::Difference))};
        }
      }
      break;
  }
  node->operationNode->cache_ = std::static_pointer_cast<CsgLeafNode>(
      impl->subTrees_[0]->Transform(node->operationNode->transform_));
  if (node->target != nullptr)
    node->target->push_back(std::static_pointer_cast<CsgLeafNode>(
        node->operationNode->cache_->Transform(node->transform)));
  stack.pop_back();
} else {
  auto addChildren = [&stack](std::shared_ptr<CsgNode> &tree, OpType operation,
                              mat3x4 transformation, auto *destination) {
    if (tree->GetNodeType() == CsgNodeType::Leaf)
      destination->push_back(std::static_pointer_cast<CsgLeafNode>(
          tree->Transform(transformation)));
    else
      stack.push_back(std::make_shared<CsgStackFrame>(
          false, operation, transformation, destination,
          std::static_pointer_cast<const CsgOpNode>(tree)));
  };
  // opNode use_count == 2 because it is both inside one CsgOpNode
  // and in our stack.
  // if there is only one child, we can also collapse.
  const bool canCollapse = node->target != nullptr &&
                           ((node->operationNode->op_ == node->parentOperation &&
                             node->operationNode.use_count() <= 2 &&
                             node->operationNode->impl_.UseCount() == 1) ||
                            impl->subTrees_.size() == 1);
  if (canCollapse)
    stack.pop_back();
  else
    node->endProcess = true;

  const mat3x4 transformation =
      canCollapse ? (node->transform * Mat4(node->operationNode->transform_))
                  : la::identity;
  OpType operation = node->operationNode->op_ == OpType::Difference ? OpType::Sum
                                                                    : node->operationNode->op_;
  for (size_t i = 0; i < impl->subTrees_.size(); i++) {
    auto dest = canCollapse ? node->target
                            : (node->operationNode->op_ == OpType::Difference && i != 0)
                                ? &node->negativeChildren
                                : &node->positiveChildren;
    addChildren(impl->subTrees_[i], operation, transformation, dest);
  }
}

    auto impl = frame->op_node->impl_.GetGuard();
    if (frame->finalize) {
      switch (frame->op_node->op_) {
        case OpType::Add:
          impl->children_ = {BatchUnion(frame->positive_children)};
          break;
        case OpType::Intersect: {
          impl->children_ = {
              BatchBoolean(OpType::Intersect, frame->positive_children)};
          break;
        };
        case OpType::Subtract:
          if (frame->positive_children.empty()) {
            // nothing to subtract from, so the result is empty.
            impl->children_ = {std::make_shared<CsgLeafNode>()};
          } else {
            auto positive = BatchUnion(frame->positive_children);
            if (frame->negative_children.empty()) {
              // nothing to subtract, result equal to the LHS.
              impl->children_ = {frame->positive_children[0]};
            } else {
              Boolean3 boolean(*positive->GetImpl(),
                               *BatchUnion(frame->negative_children)->GetImpl(),
                               OpType::Subtract);
              impl->children_ = {ImplToLeaf(boolean.Result(OpType::Subtract))};
            }
          }
          break;
      }
      frame->op_node->cache_ = std::static_pointer_cast<CsgLeafNode>(
          impl->children_[0]->Transform(frame->op_node->transform_));
      if (frame->destination != nullptr)
        frame->destination->push_back(std::static_pointer_cast<CsgLeafNode>(
            frame->op_node->cache_->Transform(frame->transform)));
      stack.pop_back();
    } else {
      auto add_children = [&stack](std::shared_ptr<CsgNode> &node, OpType op,
                                   mat3x4 transform, auto *destination) {
        if (node->GetNodeType() == CsgNodeType::Leaf)
          destination->push_back(std::static_pointer_cast<CsgLeafNode>(
              node->Transform(transform)));
        else
          stack.push_back(std::make_shared<CsgStackFrame>(
              false, op, transform, destination,
              std::static_pointer_cast<const CsgOpNode>(node)));
      };
      // op_node use_count == 2 because it is both inside one CsgOpNode
      // and in our stack.
      // if there is only one child, we can also collapse.
      const bool canCollapse = frame->destination != nullptr &&
                               ((frame->op_node->op_ == frame->parent_op &&
                                 frame->op_node.use_count() <= 2 &&
                                 frame->op_node->impl_.UseCount() == 1) ||
                                impl->children_.size() == 1);
      if (canCollapse)
        stack.pop_back();
      else
        frame->finalize = true;

      const mat3x4 transform =
          canCollapse ? (frame->transform * Mat4(frame->op_node->transform_))
                      : la::identity;
      OpType op = frame->op_node->op_ == OpType::Subtract ? OpType::Add
                                                          : frame->op_node->op_;
      for (size_t i = 0; i < impl->children_.size(); i++) {
        auto dest = canCollapse ? frame->destination
                    : (frame->op_node->op_ == OpType::Subtract && i != 0)
                        ? &frame->negative_children
                        : &frame->positive_children;
        add_children(impl->children_[i], op, transform, dest);
      }
    }

t.slice = mi.slice;
for (int i = 0; i < 3; ++i) {
    Vertex v;
    uint32_t *indexptr;

    bounds.expand_to(vtxs[i]);

    v.position[0] = vtxs[i].x;
    v.position[1] = vtxs[i].y;
    v.position[2] = vtxs[i].z;
    v.uv[0] = uvs[i].x;
    v.uv[1] = uvs[i].y;
    v.normal_xy[0] = normal[i].x;
    v.normal_xy[1] = normal[i].y;
    v.normal_z = normal[i].z;

    indexptr = vertex_map.getptr(v);

    if (indexptr) {
        t.indices[i] = *indexptr;
    } else {
        uint32_t new_index = static_cast<uint32_t>(vertex_map.size());
        t.indices[i] = new_index;
        vertex_map[v] = new_index;
        vertex_array.push_back(v);
    }

    if (i == 0) {
        taabb.position = vtxs[i];
    } else {
        taabb.expand_to(vtxs[i]);
    }
}

