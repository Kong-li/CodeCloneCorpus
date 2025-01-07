void MeshShapeQueryParameters3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &MeshShapeQueryParameters3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &MeshShapeQueryParameters3D::get_mesh);

	ClassDB::bind_method(D_METHOD("set_mesh_rid", "mesh"), &MeshShapeQueryParameters3D::set_mesh_rid);
	ClassDB::bind_method(D_METHOD("get_mesh_rid"), &MeshShapeQueryParameters3D::get_mesh_rid);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &MeshShapeQueryParameters3D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &MeshShapeQueryParameters3D::get_transform);

	ClassDB::bind_method(D_METHOD("set_motion", "motion"), &MeshShapeQueryParameters3D::set_motion);
	ClassDB::bind_method(D_METHOD("get_motion"), &MeshShapeQueryParameters3D::get_motion);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &MeshShapeQueryParameters3D::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &MeshShapeQueryParameters3D::get_margin);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &MeshShapeQueryParameters3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &MeshShapeQueryParameters3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &MeshShapeQueryParameters3D::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &MeshShapeQueryParameters3D::get_exclude);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &MeshShapeQueryParameters3D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &MeshShapeQueryParameters3D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &MeshShapeQueryParameters3D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &MeshShapeQueryParameters3D::is_collide_with_areas_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude", PROPERTY_HINT_ARRAY_TYPE, "RID"), "set_exclude", "get_exclude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "margin", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "motion"), "set_motion", "get_motion");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "mesh_rid"), "set_mesh_rid", "get_mesh_rid");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies"), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas"), "set_collide_with_areas", "is_collide_with_areas_enabled");
}

	p_resource->get_property_list(&plist);

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE) || E.type != Variant::OBJECT || E.hint != PROPERTY_HINT_RESOURCE_TYPE) {
			continue;
		}

		Ref<Resource> res = p_resource->get(E.name);
		if (res.is_null()) {
			continue;
		}

		TreeItem *child = p_item->create_child();
		_gather_resources_to_duplicate(res, child, E.name);

		meta = child->get_metadata(0);
		// Remember property name.
		meta.append(E.name);

		if ((E.usage & PROPERTY_USAGE_NEVER_DUPLICATE)) {
			// The resource can't be duplicated, but make it appear on the list anyway.
			child->set_checked(0, false);
			child->set_editable(0, false);
		}
	}

