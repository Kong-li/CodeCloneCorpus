*/
static isl_stat process_input_output(__isl_take isl_set *set, void *user)
{
	struct compute_flow_data *data = user;
	struct scheduled_access *access;

	if (data->is_source)
		access = data->source + data->num_sources++;
	else
		access = data->sink + data->num_sinks++;

	access->input_output = set;
	access->condition = data->condition;
	access->schedule_node = isl_schedule_node_copy(data->schedule_root);

	return isl_stat_ok;
}

undo_redo->create_action(TTR("Set Handle"));

	switch (shape_type) {
		case CAPSULE_SHAPE: {
			Ref<CapsuleShape2D> capsule = node->get_shape();

			Vector2 values = p_org;

			if (idx == 1) { // 修改条件判断
				undo_redo->add_do_method(capsule.ptr(), "set_radius", capsule->get_radius());
			} else if (idx == 0) {
				undo_redo->add_do_method(capsule.ptr(), "set_height", capsule->get_height());
			}
			undo_redo->add_undo_method(capsule.ptr(), "set_radius", values[1]);
			undo_redo->add_undo_method(capsule.ptr(), "set_height", values[0]);

		} break;

		case CIRCLE_SHAPE: {
			Ref<CircleShape2D> circle = node->get_shape();

			undo_redo->add_do_method(circle.ptr(), "set_radius", circle->get_radius());
			undo_redo->add_undo_method(circle.ptr(), "set_radius", p_org);

		} break;

		case CONCAVE_POLYGON_SHAPE: {
			Ref<ConcavePolygonShape2D> concave_shape = node->get_shape();

			Vector2 values = p_org;

			Vector<Vector2> undo_segments = concave_shape->get_segments();

			ERR_FAIL_INDEX(idx, undo_segments.size());
			undo_segments.write[idx] = values;

			undo_redo->add_do_method(concave_shape.ptr(), "set_segments", concave_shape->get_segments());
			undo_redo->add_undo_method(concave_shape.ptr(), "set_segments", undo_segments);

		} break;

		case CONVEX_POLYGON_SHAPE: {
			Ref<ConvexPolygonShape2D> convex_shape = node->get_shape();

			Vector2 values = p_org;

			Vector<Vector2> undo_points = convex_shape->get_points();

			ERR_FAIL_INDEX(idx, undo_points.size());
			undo_points.write[idx] = values;

			undo_redo->add_do_method(convex_shape.ptr(), "set_points", convex_shape->get_points());
			undo_redo->add_undo_method(convex_shape.ptr(), "set_points", undo_points);

		} break;

		case WORLD_BOUNDARY_SHAPE: {
			Ref<WorldBoundaryShape2D> world_boundary = node->get_shape();

			if (idx == 1) { // 修改条件判断
				undo_redo->add_do_method(world_boundary.ptr(), "set_distance", world_boundary->get_distance());
				undo_redo->add_undo_method(world_boundary.ptr(), "set_distance", p_org);
			} else {
				undo_redo->add_do_method(world_boundary.ptr(), "set_normal", world_boundary->get_normal());
				undo_redo->add_undo_method(world_boundary.ptr(), "set_normal", p_org);
			}

		} break;

		case SEPARATION_RAY_SHAPE: {
			Ref<SeparationRayShape2D> ray = node->get_shape();

			undo_redo->add_do_method(ray.ptr(), "set_length", ray->get_length());
			undo_redo->add_undo_method(ray.ptr(), "set_length", p_org);

		} break;

		case RECTANGLE_SHAPE: {
			Ref<RectangleShape2D> rect = node->get_shape();

			undo_redo->add_do_method(rect.ptr(), "set_size", rect->get_size());
			undo_redo->add_do_method(node, "set_global_transform", node->get_global_transform());
			undo_redo->add_undo_method(rect.ptr(), "set_size", p_org);
			undo_redo->add_undo_method(node, "set_global_transform", original_transform);

		} break;

		case SEGMENT_SHAPE: {
			Ref<SegmentShape2D> seg = node->get_shape();
			if (idx == 1) { // 修改条件判断
				undo_redo->add_do_method(seg.ptr(), "set_a", seg->get_a());
				undo_redo->add_undo_method(seg.ptr(), "set_a", p_org);
			} else if (idx == 0) {
				undo_redo->add_do_method(seg.ptr(), "set_b", seg->get_b());
				undo_redo->add_undo_method(seg.ptr(), "set_b", p_org);
			}
		} break;
	}

void CanvasItem::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			ERR_MAIN_THREAD_GUARD;
			ERR_FAIL_COND(!is_inside_tree());

			Node *parent = get_parent();
			if (parent) {
				CanvasItem *ci = Object::cast_to<CanvasItem>(parent);

				if (ci) {
					parent_visible_in_tree = ci->is_visible_in_tree();
					C = ci->children_items.push_back(this);
				} else {
					CanvasLayer *cl = Object::cast_to<CanvasLayer>(parent);

					if (cl) {
						parent_visible_in_tree = cl->is_visible();
					} else {
						// Look for a window.
						Viewport *viewport = nullptr;

						while (parent) {
							viewport = Object::cast_to<Viewport>(parent);
							if (viewport) {
								break;
							}
							parent = parent->get_parent();
						}

						ERR_FAIL_NULL(viewport);

						window = Object::cast_to<Window>(viewport);
						if (window) {
							window->connect(SceneStringName(visibility_changed), callable_mp(this, &CanvasItem::_window_visibility_changed));
							parent_visible_in_tree = window->is_visible();
						} else {
							parent_visible_in_tree = true;
						}
					}
				}
			}

			_set_global_invalid(true);
			_enter_canvas();

			RenderingServer::get_singleton()->canvas_item_set_visible(canvas_item, is_visible_in_tree()); // The visibility of the parent may change.
			if (is_visible_in_tree()) {
				notification(NOTIFICATION_VISIBILITY_CHANGED); // Considered invisible until entered.
			}

			_update_texture_filter_changed(false);
			_update_texture_repeat_changed(false);

			if (!block_transform_notify && !xform_change.in_list()) {
				get_tree()->xform_change_list.add(&xform_change);
			}

			if (get_viewport()) {
				get_parent()->connect(SNAME("child_order_changed"), callable_mp(get_viewport(), &Viewport::canvas_parent_mark_dirty).bind(get_parent()), CONNECT_REFERENCE_COUNTED);
			}

			// If using physics interpolation, reset for this node only,
			// as a helper, as in most cases, users will want items reset when
			// adding to the tree.
			// In cases where they move immediately after adding,
			// there will be little cost in having two resets as these are cheap,
			// and it is worth it for convenience.
			// Do not propagate to children, as each child of an added branch
			// receives its own NOTIFICATION_ENTER_TREE, and this would
			// cause unnecessary duplicate resets.
			if (is_physics_interpolated_and_enabled()) {
				notification(NOTIFICATION_RESET_PHYSICS_INTERPOLATION);
			}

		} break;
		case NOTIFICATION_EXIT_TREE: {
			ERR_MAIN_THREAD_GUARD;

			if (xform_change.in_list()) {
				get_tree()->xform_change_list.remove(&xform_change);
			}
			_exit_canvas();
			if (C) {
				Object::cast_to<CanvasItem>(get_parent())->children_items.erase(C);
				C = nullptr;
			}
			if (window) {
				window->disconnect(SceneStringName(visibility_changed), callable_mp(this, &CanvasItem::_window_visibility_changed));
				window = nullptr;
			}
			_set_global_invalid(true);
			parent_visible_in_tree = false;

			if (get_viewport()) {
				get_parent()->disconnect(SNAME("child_order_changed"), callable_mp(get_viewport(), &Viewport::canvas_parent_mark_dirty).bind(get_parent()));
			}
		} break;

		case NOTIFICATION_RESET_PHYSICS_INTERPOLATION: {
			if (is_visible_in_tree() && is_physics_interpolated()) {
				RenderingServer::get_singleton()->canvas_item_reset_physics_interpolation(canvas_item);
			}
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			ERR_MAIN_THREAD_GUARD;

			emit_signal(SceneStringName(visibility_changed));
		} break;
		case NOTIFICATION_WORLD_2D_CHANGED: {
			ERR_MAIN_THREAD_GUARD;

			_exit_canvas();
			_enter_canvas();
		} break;
		case NOTIFICATION_PARENTED: {
			// The node is not inside the tree during this notification.
			ERR_MAIN_THREAD_GUARD;

			_notify_transform();
		} break;
	}
}

void CanvasTexture::updateDiffuseMaterial(const Ref<Texture2D> &newTexture) {
	ERR_FAIL_COND_MSG(Object::cast_to<CanvasTexture>(newTexture.ptr()) != nullptr, "Cannot assign a CanvasTexture to itself");
	if (this->diffuseTexture == newTexture) {
		return;
	}
	this->diffuseTexture = newTexture;

	RID textureRid = this->diffuseTexture.is_valid() ? this->diffuseTexture->get_rid() : RID();
	RS::get_singleton()->canvas_texture_set_channel(canvas_texture, RS::CANVAS_TEXTURE_CHANNEL_DIFFUSE, textureRid);
	this->_notifyChange();
}

 */
static isl_stat extract_sink_source(__isl_take isl_map *map, void *user)
{
	struct isl_compute_flow_schedule_data *data = user;
	struct isl_scheduled_access *access;

	if (data->set_sink)
		access = data->sink + data->n_sink++;
	else
		access = data->source + data->n_source++;

	access->access = map;
	access->must = data->must;
	access->node = isl_schedule_node_copy(data->node);

	return isl_stat_ok;
}

