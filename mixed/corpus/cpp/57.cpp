{
    if (b1->x == 32 && b1->y == 32) {
        if (cbstyle & JPEG_CCP_CBSTYLE_HSC) {
            dec_sigpass_mqc_32x32_hsc(b1, blockno);
        } else {
            dec_sigpass_mqc_32x32_nohsc(b1, blockno);
        }
    } else {
        if (cbstyle & JPEG_CCP_CBSTYLE_HSC) {
            dec_sigpass_mqc_generic_hsc(b1, blockno);
        } else {
            dec_sigpass_mqc_generic_nohsc(b1, blockno);
        }
    }
}

void CSGShape3DGizmoPlugin::updateHandles(const EditorNode3DGizmo *gizmo, int id, bool isSecondary, Camera3D *camera, const Point2 &point) {
	CSGShape3D *shape = Object::cast_to<CSGShape3D>(gizmo->get_node_3d());

	if (Object::cast_to<CSGSphere3D>(shape)) {
		CSGSphere3D *sphere = Object::cast_to<CSGSphere3D>(shape);
		Vector3 segmentA, segmentB;
		helper->computeSegments(camera, point, &segmentA, &segmentB);

		Vector3 sphereRadiusA, sphereRadiusB;
		Geometry3D::getClosestPointsBetweenSegments(Vector3(), Vector3(4096, 0, 0), segmentA, segmentB, sphereRadiusA, sphereRadiusB);
		float radius = sphereRadiusA.x;

		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			radius = Math::snapped(radius, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (radius < 0.001) {
			radius = 0.001;
		}

		sphere->set_radius(radius);
	}

	if (Object::cast_to<CSGBox3D>(shape)) {
		CSGBox3D *box = Object::cast_to<CSGBox3D>(shape);
		Vector3 size, position;
		helper->calculateBoxHandle(&segmentA, &segmentB, id, size, &position);
		box->set_size(size);
		box->set_global_position(position);
	}

	if (Object::cast_to<CSGCylinder3D>(shape)) {
		CSGCylinder3D *cylinder = Object::cast_to<CSGCylinder3D>(shape);

		real_t height = cylinder->get_height();
		real_t radius = cylinder->get_radius();
		Vector3 position;
		helper->calculateCylinderHandle(&segmentA, &segmentB, id, height, radius, &position);
		cylinder->set_height(height);
		cylinder->set_radius(radius);
		cylinder->set_global_position(position);
	}

	if (Object::cast_to<CSGTorus3D>(shape)) {
		CSGTorus3D *torus = Object::cast_to<CSGTorus3D>(shape);

		Vector3 axis;
		axis[0] = 1.0;

		real_t innerRadius, outerRadius;
		Geometry3D::getClosestPointsBetweenSegments(Vector3(), axis * 4096, segmentA, segmentB, &innerRadius, &outerRadius);
		float distance = axis.dot(innerRadius);

		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			distance = Math::snapped(distance, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (distance < 0.001) {
			distance = 0.001;
		}

		if (id == 0) {
			torus->set_inner_radius(distance);
		} else if (id == 1) {
			torus->set_outer_radius(distance);
		}
	}
}

// clang-format on
for (auto test : twogig_max) {
    auto user_addr = test.user.addr;
    auto user_size = test.user.size;
    size_t min_byte_size = 1;
    size_t max_byte_size = INT32_MAX;
    size_t address_byte_size = 8;

    bool result = WatchpointAlgorithmsTest::PowerOf2Watchpoints(
        user_addr, user_size, min_byte_size, max_byte_size, address_byte_size);

    check_testcase(test, !result, min_byte_size, max_byte_size,
                   address_byte_size);
}

for (k = 0; k < (h & ~3u); k += 4, data += 3*l_w, flagsp += 2) { \
                for (i = 0; i < l_w; ++i, ++data, ++flagsp) { \
                        opj_flag_t flags = *flagsp; \
                        if( flags != 0 ) { \
                            opj_t2_dec_sigpass_step_mqc_macro( \
                                flags, flagsp, flags_stride, data, \
                                l_w, 0, mqc, curctx, v, a, c, ct, oneplushalf, vsc); \
                            opj_t2_dec_sigpass_step_mqc_macro( \
                                flags, flagsp, flags_stride, data, \
                                l_w, 1, mqc, curctx, v, a, c, ct, oneplushalf, OPJ_FALSE); \
                            opj_t2_dec_sigpass_step_mqc_macro( \
                                flags, flagsp, flags_stride, data, \
                                l_w, 2, mqc, curctx, v, a, c, ct, oneplushalf, OPJ_FALSE); \
                            opj_t2_dec_sigpass_step_mqc_macro( \
                                flags, flagsp, flags_stride, data, \
                                l_w, 3, mqc, curctx, v, a, c, ct, oneplushalf, OPJ_FALSE); \
                            *flagsp = flags; \
                        } \
                } \
        } \

