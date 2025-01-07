private:
    bool computeH(const cv::Mat &A, const cv::Vec3d &e_prime, int sample1, int sample2, int sample3, cv::Matx33d &H) {
        const float* points = points_mat.ptr<float>();
        Vec3d p1(points[sample1], points[sample1 + 1], 1), p2(points[sample2], points[sample2 + 1], 1), p3(points[sample3], points[sample3 + 1], 1);
        Vec3d P1(points[sample1 + 2], points[sample1 + 3], 1), P2(points[sample2 + 2], points[sample2 + 3], 1), P3(points[sample3 + 2], points[sample3 + 3], 1);
        const Matx33d M = {p1[0], p1[1], 1, p2[0], p2[1], 1, p3[0], p3[1], 1};
        if (p1.cross(p2).dot(p3) * P1.cross(P2).dot(P3) > 0) return false;

        Vec3d P1e = P1.cross(e_prime), P2e = P2.cross(e_prime), P3e = P3.cross(e_prime);
        const float normP1e = P1e[0]*P1e[0] + P1e[1]*P1e[1] + P1e[2]*P1e[2];
        const float normP2e = P2e[0]*P2e[0] + P2e[1]*P2e[1] + P2e[2]*P2e[2];
        const float normP3e = P3e[0]*P3e[0] + P3e[1]*P3e[1] + P3e[2]*P3e[2];

        Vec3d b (P1.cross(A * p1).dot(P1e) / normP1e,
                 P2.cross(A * p2).dot(P2e) / normP2e,
                 P3.cross(A * p3).dot(P3e) / normP3e);

        H = A - e_prime * (M.inv() * b).t();
        return true;
    }

	if ((p_parameters.recovery_as_collision && recovered) || (safe < 1)) {
		if (safe >= 1) {
			best_shape = -1; //no best shape with cast, reset to -1
		}

		//it collided, let's get the rest info in unsafe advance
		Transform3D ugt = body_transform;
		ugt.origin += p_parameters.motion * unsafe;

		_RestResultData results[PhysicsServer3D::MotionResult::MAX_COLLISIONS];

		_RestCallbackData rcd;
		if (p_parameters.max_collisions > 1) {
			rcd.max_results = p_parameters.max_collisions;
			rcd.other_results = results;
		}

		// Allowed depth can't be lower than motion length, in order to handle contacts at low speed.
		rcd.min_allowed_depth = MIN(motion_length, min_contact_depth);

		body_aabb.position += p_parameters.motion * unsafe;
		int amount = _cull_aabb_for_body(p_body, body_aabb);

		int from_shape = best_shape != -1 ? best_shape : 0;
		int to_shape = best_shape != -1 ? best_shape + 1 : p_body->get_shape_count();

		for (int j = from_shape; j < to_shape; j++) {
			if (p_body->is_shape_disabled(j)) {
				continue;
			}

			Transform3D body_shape_xform = ugt * p_body->get_shape_transform(j);
			GodotShape3D *body_shape = p_body->get_shape(j);

			for (int i = 0; i < amount; i++) {
				const GodotCollisionObject3D *col_obj = intersection_query_results[i];
				if (p_parameters.exclude_bodies.has(col_obj->get_self())) {
					continue;
				}
				if (p_parameters.exclude_objects.has(col_obj->get_instance_id())) {
					continue;
				}

				int shape_idx = intersection_query_subindex_results[i];

				rcd.object = col_obj;
				rcd.shape = shape_idx;
				rcd.local_shape = j;
				bool sc = GodotCollisionSolver3D::solve_static(body_shape, body_shape_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), _rest_cbk_result, &rcd, nullptr, margin);
				if (!sc) {
					continue;
				}
			}
		}

		if (rcd.result_count > 0) {
			if (r_result) {
				for (int collision_index = 0; collision_index < rcd.result_count; ++collision_index) {
					const _RestResultData &result = (collision_index > 0) ? rcd.other_results[collision_index - 1] : rcd.best_result;

					PhysicsServer3D::MotionCollision &collision = r_result->collisions[collision_index];

					collision.collider = result.object->get_self();
					collision.collider_id = result.object->get_instance_id();
					collision.collider_shape = result.shape;
					collision.local_shape = result.local_shape;
					collision.normal = result.normal;
					collision.position = result.contact;
					collision.depth = result.len;

					const GodotBody3D *body = static_cast<const GodotBody3D *>(result.object);

					Vector3 rel_vec = result.contact - (body->get_transform().origin + body->get_center_of_mass());
					collision.collider_velocity = body->get_linear_velocity() + (body->get_angular_velocity()).cross(rel_vec);
					collision.collider_angular_velocity = body->get_angular_velocity();
				}

				r_result->travel = safe * p_parameters.motion;
				r_result->remainder = p_parameters.motion - safe * p_parameters.motion;
				r_result->travel += (body_transform.get_origin() - p_parameters.from.get_origin());

				r_result->collision_safe_fraction = safe;
				r_result->collision_unsafe_fraction = unsafe;

				r_result->collision_count = rcd.result_count;
				r_result->collision_depth = rcd.best_result.len;
			}

			collided = true;
		}
	}

namespace {

TEST(raw_pwrite_ostreamTest, TestSVector2) {
  SmallString<0> Buffer;
  raw_svector_ostream OS(Buffer);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  EXPECT_EQ(OS.str(), Test);

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(OS.pwrite("12345", 5, 0),
               "We don't support extending the stream");
#endif
#endif

  SmallVector<char, 64> Buffer2;
  raw_svector_ostream OS2(Buffer2);
  OS2 << "abcd";
  OS2.pwrite(Test.data(), Test.size(), 0);
  EXPECT_EQ(OS2.str(), Test);
}

#ifdef _WIN32
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

TEST(raw_pwrite_ostreamTest, TestFD2) {
  SmallString<64> Path;
  int FD;

  const char *ParentPath = getenv("RAW_PWRITE_TEST_FILE");
  if (ParentPath) {
    Path = ParentPath;
    ASSERT_NO_ERROR(sys::fs::openFileForRead(Path, FD));
  } else {
    ASSERT_NO_ERROR(sys::fs::createTemporaryFile("foo", "bar", FD, Path));
    setenv("RAW_PWRITE_TEST_FILE", Path.c_str(), true);
  }
  FileRemover Cleanup(Path);

  raw_fd_ostream OS(FD, true);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  OS.pwrite(Test.data(), Test.size(), 0);

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(OS.pwrite("12345", 5, 0),
               "We don't support extending the stream");
#endif
#endif

  int FD2;
  ASSERT_NO_ERROR(sys::fs::openFileForWrite("/dev/null", FD2, sys::fs::CD_OpenExisting));
  raw_fd_ostream OS3(FD2, true);
  OS3 << "abcd";
  OS3.pwrite(Test.data(), Test.size(), 0);
  OS3.pwrite(Test.data(), Test.size(), 0);
}
}

real_t min_distance = 1e10;

	for (int i = 0; i < body_count; ++i) {
		const GodotCollisionObject3D *object = space->intersection_query_results[i];

		if (!_can_collide_with(object, p_parameters.collision_mask, p_parameters.collide_with_bodies, p_parameters.collide_with_areas)) {
			continue;
		}

		if (p_parameters.pick_ray && !space->intersection_query_results[i]->is_ray_pickable()) {
			continue;
		}

		const String &self_id = space->intersection_query_results[i]->get_self();

		if (p_parameters.exclude.has(self_id)) {
			continue;
		}

		int shape_index = space->intersection_query_subindex_results[i];
		Transform3D inverse_transform = object->get_shape_inv_transform(shape_index) * object->get_inv_transform();

		const Vector3 local_from = inverse_transform.xform(begin);
		const Vector3 local_to = inverse_transform.xform(end);

		const GodotShape3D *shape = object->get_shape(shape_index);

		Vector3 intersection_point, normal;
		int face_index;

		if (shape->intersect_point(local_from)) {
			if (!p_parameters.hit_from_inside) {
				min_distance = 0;
				res_position = begin;
				res_normal = Vector3();
				res_shape_id = shape_index;
				res_object = object;
				collided = true;
				break;
			} else {
				continue;
			}
		}

		if (shape->intersect_segment(local_from, local_to, intersection_point, normal, face_index, p_parameters.hit_back_faces)) {
			const Transform3D transform = object->get_transform() * object->get_shape_transform(shape_index);
			intersection_point = transform.xform(intersection_point);

			real_t distance = normal.dot(intersection_point);

			if (distance < min_distance) {
				min_distance = distance;
				res_position = intersection_point;
				res_normal = inverse_transform.basis.xform_inv(normal).normalized();
				res_face_id = face_index;
				res_shape_id = shape_index;
				res_object = object;
				collided = true;
			}
		}
	}

static Value getDim3Value(const gpu::KernelDim3 &dim3, unsigned pos) {
  switch (pos) {
  case 0:
    return dim3.x;
  case 1:
    return dim3.y;
  case 2:
    return dim3.z;
  default:
    llvm_unreachable("dim3 position out of bounds");
  }
  return nullptr;
}

