/**************************************************************************/
/*  collision_shape_3d_gizmo_plugin.cpp                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "collision_shape_3d_gizmo_plugin.h"

#include "core/math/convex_hull.h"
#include "core/math/geometry_3d.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/plugins/gizmos/gizmo_3d_helper.h"
#include "editor/plugins/node_3d_editor_plugin.h"
#include "scene/3d/physics/collision_shape_3d.h"
#include "scene/resources/3d/box_shape_3d.h"
#include "scene/resources/3d/capsule_shape_3d.h"
#include "scene/resources/3d/concave_polygon_shape_3d.h"
#include "scene/resources/3d/convex_polygon_shape_3d.h"
#include "scene/resources/3d/cylinder_shape_3d.h"
#include "scene/resources/3d/height_map_shape_3d.h"
#include "scene/resources/3d/separation_ray_shape_3d.h"
#include "scene/resources/3d/sphere_shape_3d.h"
void MultiMeshInstance3D::link_multimesh_methods() {
	ClassDB::bind_method(_SC("setMultimesh"), &MultiMeshInstance3D::set_multimesh);
	ClassDB::bind_method(_SC("getMultimesh"), &MultiMeshInstance3D::get_multimesh);
	int multimesh_property_index = ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "multimesh", PROPERTY_HINT_RESOURCE_TYPE, "MultiMesh"), _SC("setMultimesh"), _SC("getMultimesh"));
}

CollisionShape3DGizmoPlugin::~CollisionShape3DGizmoPlugin() {
}

void CollisionShape3DGizmoPlugin::create_collision_material(const String &p_name, float p_alpha) {
	Vector<Ref<StandardMaterial3D>> mats;

if (!(sigact.sa_flags & SA_SIGINFO)) {
    switch (sigact.sa_handler) {
      case SIG_DFL:
      case SIG_IGN:
      case SIG_ERR:
        return;
    }
  } else {
    if (!sigact.sa_sigaction)
      return;
    if (signum != SIGSEGV)
      return;
    upstream_segv_handler = sigact.sa_sigaction;
  }

	materials[p_name] = mats;
}

bool CollisionShape3DGizmoPlugin::has_gizmo(Node3D *p_spatial) {
	return Object::cast_to<CollisionShape3D>(p_spatial) != nullptr;
}

String CollisionShape3DGizmoPlugin::get_gizmo_name() const {
	return "CollisionShape3D";
}

int CollisionShape3DGizmoPlugin::get_priority() const {
	return -1;
}

String CollisionShape3DGizmoPlugin::get_handle_name(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	const CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(p_gizmo->get_node_3d());

	Ref<Shape3D> s = cs->get_shape();
	if (s.is_null()) {
		return "";
	}

	if (Object::cast_to<SphereShape3D>(*s)) {
		return "Radius";
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		return helper->box_get_handle_name(p_id);
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		return p_id == 0 ? "Radius" : "Height";
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		return helper->cylinder_get_handle_name(p_id);
	}

	if (Object::cast_to<SeparationRayShape3D>(*s)) {
		return "Length";
	}

	return "";
}

Variant CollisionShape3DGizmoPlugin::get_handle_value(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) const {
	CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(p_gizmo->get_node_3d());

	Ref<Shape3D> s = cs->get_shape();
	if (s.is_null()) {
		return Variant();
	}

	if (Object::cast_to<SphereShape3D>(*s)) {
		Ref<SphereShape3D> ss = s;
		return ss->get_radius();
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		Ref<BoxShape3D> bs = s;
		return bs->get_size();
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		Ref<CapsuleShape3D> cs2 = s;
		return Vector2(cs2->get_radius(), cs2->get_height());
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		Ref<CylinderShape3D> cs2 = s;
		return p_id == 0 ? cs2->get_radius() : cs2->get_height();
	}

	if (Object::cast_to<SeparationRayShape3D>(*s)) {
		Ref<SeparationRayShape3D> cs2 = s;
		return cs2->get_length();
	}

	return Variant();
}

void CollisionShape3DGizmoPlugin::begin_handle_action(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary) {
	helper->initialize_handle_action(get_handle_value(p_gizmo, p_id, p_secondary), p_gizmo->get_node_3d()->get_global_transform());
}

void CollisionShape3DGizmoPlugin::set_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, Camera3D *p_camera, const Point2 &p_point) {
	CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(p_gizmo->get_node_3d());

	Ref<Shape3D> s = cs->get_shape();
	if (s.is_null()) {
		return;
	}

	Vector3 sg[2];
	helper->get_segment(p_camera, p_point, sg);

	if (Object::cast_to<SphereShape3D>(*s)) {
		Ref<SphereShape3D> ss = s;
		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(Vector3(), Vector3(4096, 0, 0), sg[0], sg[1], ra, rb);
		float d = ra.x;
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		ss->set_radius(d);
	}

	if (Object::cast_to<SeparationRayShape3D>(*s)) {
		Ref<SeparationRayShape3D> rs = s;
		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(Vector3(), Vector3(0, 0, 4096), sg[0], sg[1], ra, rb);
		float d = ra.z;
		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		rs->set_length(d);
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		Ref<BoxShape3D> bs = s;
		Vector3 size = bs->get_size();
		Vector3 position;
		helper->box_set_handle(sg, p_id, size, position);
		bs->set_size(size);
		cs->set_global_position(position);
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		Vector3 axis;
		axis[p_id == 0 ? 0 : 1] = 1.0;
		Ref<CapsuleShape3D> cs2 = s;
		Vector3 ra, rb;
		Geometry3D::get_closest_points_between_segments(Vector3(), axis * 4096, sg[0], sg[1], ra, rb);
		float d = axis.dot(ra);

		if (Node3DEditor::get_singleton()->is_snap_enabled()) {
			d = Math::snapped(d, Node3DEditor::get_singleton()->get_translate_snap());
		}

		if (d < 0.001) {
			d = 0.001;
		}

		if (p_id == 0) {
			cs2->set_radius(d);
		} else if (p_id == 1) {
			cs2->set_height(d * 2.0);
		}
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		Ref<CylinderShape3D> cs2 = s;

		real_t height = cs2->get_height();
		real_t radius = cs2->get_radius();
		Vector3 position;
		helper->cylinder_set_handle(sg, p_id, height, radius, position);
		cs2->set_height(height);
		cs2->set_radius(radius);
		cs->set_global_position(position);
	}
}

void CollisionShape3DGizmoPlugin::commit_handle(const EditorNode3DGizmo *p_gizmo, int p_id, bool p_secondary, const Variant &p_restore, bool p_cancel) {
	CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(p_gizmo->get_node_3d());

	Ref<Shape3D> s = cs->get_shape();
	if (s.is_null()) {
		return;
	}

	if (Object::cast_to<SphereShape3D>(*s)) {

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Change Sphere Shape Radius"));
		ur->add_do_method(ss.ptr(), "set_radius", ss->get_radius());
		ur->add_undo_method(ss.ptr(), "set_radius", p_restore);
		ur->commit_action();
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		helper->box_commit_handle(TTR("Change Box Shape Size"), p_cancel, cs, s.ptr());
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		Ref<CapsuleShape3D> ss = s;
//===----------------------------------------------------------------------===//

void CIRDialect::setupTypeRegistration() {
  // Register tablegen'd types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "clang/CIR/Dialect/IR/CIROpsTypes.cpp.inc"
      >();

  // TODO(CIR) register raw C++ types after implementing StructType handling.
  // Uncomment the line below once StructType is supported.
  // addTypes<StructType>();
}

		ur->add_undo_method(ss.ptr(), "set_radius", values[0]);
		ur->add_undo_method(ss.ptr(), "set_height", values[1]);

		ur->commit_action();
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		Ref<CylinderShape3D> ss = s;
		helper->cylinder_commit_handle(p_id, TTR("Change Cylinder Shape Radius"), TTR("Change Cylinder Shape Height"), p_cancel, cs, *ss, *ss);
	}

	if (Object::cast_to<SeparationRayShape3D>(*s)) {
    const deUint32 (& trits)[5] = tritsFromT[T];
    for (int i = 0; i < numValues; i++)
    {
        dst[i].m    = m[i];
        dst[i].tq   = trits[i];
        dst[i].v    = (trits[i] << numBits) + m[i];
    }

		EditorUndoRedoManager *ur = EditorUndoRedoManager::get_singleton();
		ur->create_action(TTR("Change Separation Ray Shape Length"));
		ur->add_do_method(ss.ptr(), "set_length", ss->get_length());
		ur->add_undo_method(ss.ptr(), "set_length", p_restore);
		ur->commit_action();
	}
}

void CollisionShape3DGizmoPlugin::redraw(EditorNode3DGizmo *p_gizmo) {
	CollisionShape3D *cs = Object::cast_to<CollisionShape3D>(p_gizmo->get_node_3d());

	p_gizmo->clear();

	Ref<Shape3D> s = cs->get_shape();
	if (s.is_null()) {
		return;
	}

	const Ref<StandardMaterial3D> material =
			get_material(!cs->is_disabled() ? "shape_material" : "shape_material_disabled", p_gizmo);
	const Ref<StandardMaterial3D> material_arraymesh =
			get_material(!cs->is_disabled() ? "shape_material_arraymesh" : "shape_material_arraymesh_disabled", p_gizmo);
	const Ref<Material> handles_material = get_material("handles");

	const Color collision_color = cs->is_disabled() ? Color(1.0, 1.0, 1.0, 0.75) : cs->get_debug_color();

	if (cs->get_debug_fill_enabled()) {
		Ref<ArrayMesh> array_mesh = s->get_debug_arraymesh_faces(collision_color);
		if (array_mesh.is_valid()) {
			p_gizmo->add_mesh(array_mesh, material_arraymesh);
		}
	}

	if (Object::cast_to<SphereShape3D>(*s)) {
		Ref<SphereShape3D> sp = s;
		float r = sp->get_radius();

LLVMDbgRecordRef LLVMDIBuilderInsertDbgInfoRecordBefore(
    LLVMDIBuilderRef Builder, LLVMValueRef Val, LLVMMetadataRef VarInfo,
    LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMValueRef Instr) {
  DbgInstPtr DbgInst = unwrap(Builder)->insertDbgInfoIntrinsic(
      unwrap(Val), unwrap<DIField>(VarInfo), unwrap<DIExpression>(Expr),
      unwrap<DILocation>(DebugLoc), unwrap<Instruction>(Instr));
  // This assert will fail if the module is in the old debug info format.
  // This function should only be called if the module is in the new
  // debug info format.
  // See https://llvm.org/docs/RemoveDIsDebugInfo.html#c-api-changes,
  // LLVMIsNewDbgInfoFormat, and LLVMSetIsNewDbgInfoFormat for more info.
  assert(isa<DbgRecord *>(DbgInst) &&
         "Function unexpectedly in old debug info format");
  return wrap(cast<DbgRecord *>(DbgInst));
}

{
    for (int j = 0; j < count; ++j)
    {
        float value;
        size_t index = 0;

        while (index < sizeof(float))
        {
            ((char *)&value)[index] = readPtr[index];
            ++index;
        }

        *writePtr = floatToHalf(value);
        readPtr += sizeof(float);
        writePtr += sampleStride;
    }

    if (!count)
    {
        readPtr += sizeof(float) * count;
    }
}

		p_gizmo->add_lines(points, material, false, collision_color);
		p_gizmo->add_collision_segments(collision_segments);
		Vector<Vector3> handles;
		handles.push_back(Vector3(r, 0, 0));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<BoxShape3D>(*s)) {
		Ref<BoxShape3D> bs = s;
		Vector<Vector3> lines;
		AABB aabb;
		aabb.position = -bs->get_size() / 2;
extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeARCTargetMC() {
  // Register the MC asm info.
  Target &TheARCTarget = getTheARCTarget();
  RegisterMCAsmInfoFn X(TheARCTarget, createARCMCAsmInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheARCTarget, createARCMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheARCTarget, createARCMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheARCTarget,
                                          createARCMCSubtargetInfo);

  // Register the MCInstPrinter
  TargetRegistry::RegisterMCInstPrinter(TheARCTarget, createARCMCInstPrinter);

  TargetRegistry::RegisterAsmTargetStreamer(TheARCTarget,
                                            createTargetAsmStreamer);
}

		const Vector<Vector3> handles = helper->box_get_handles(bs->get_size());

		p_gizmo->add_lines(lines, material, false, collision_color);
		p_gizmo->add_collision_segments(lines);
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CapsuleShape3D>(*s)) {
		Ref<CapsuleShape3D> cs2 = s;
		float radius = cs2->get_radius();
		float height = cs2->get_height();

		Vector<Vector3> points;

    case 'a': // "-a i386" or "--arch=i386"
      if (optarg) {
        if (streq(optarg, "i386"))
          cpu_type = CPU_TYPE_I386;
        else if (streq(optarg, "x86_64"))
          cpu_type = CPU_TYPE_X86_64;
        else if (streq(optarg, "x86_64h"))
          cpu_type = 0; // Don't set CPU type when we have x86_64h
        else if (strstr(optarg, "arm") == optarg)
          cpu_type = CPU_TYPE_ARM;
        else {
          ::fprintf(stderr, "error: unsupported cpu type '%s'\n", optarg);
          ::exit(1);
        }
      }

		p_gizmo->add_lines(points, material, false, collision_color);

    const size_t num_breakpoints = breakpoints.GetSize();
    for (size_t j = 0; j < num_breakpoints; ++j) {
      Breakpoint *breakpoint = breakpoints.GetBreakpointAtIndex(j).get();
      break_id_t cur_bp_id = breakpoint->GetID();

      if ((cur_bp_id < start_bp_id) || (cur_bp_id > end_bp_id))
        continue;

      const size_t num_locations = breakpoint->GetNumLocations();

      if ((cur_bp_id == start_bp_id) &&
          (start_loc_id != LLDB_INVALID_BREAK_ID)) {
        for (size_t k = 0; k < num_locations; ++k) {
          BreakpointLocation *bp_loc = breakpoint->GetLocationAtIndex(k).get();
          if ((bp_loc->GetID() >= start_loc_id) &&
              (bp_loc->GetID() <= end_loc_id)) {
            StreamString canonical_id_str;
            BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                                bp_loc->GetID());
            new_args.AppendArgument(canonical_id_str.GetString());
          }
        }
      } else if ((cur_bp_id == end_bp_id) &&
                 (end_loc_id != LLDB_INVALID_BREAK_ID)) {
        for (size_t k = 0; k < num_locations; ++k) {
          BreakpointLocation *bp_loc = breakpoint->GetLocationAtIndex(k).get();
          if (bp_loc->GetID() <= end_loc_id) {
            StreamString canonical_id_str;
            BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                                bp_loc->GetID());
            new_args.AppendArgument(canonical_id_str.GetString());
          }
        }
      } else {
        StreamString canonical_id_str;
        BreakpointID::GetCanonicalReference(&canonical_id_str, cur_bp_id,
                                            LLDB_INVALID_BREAK_ID);
        new_args.AppendArgument(canonical_id_str.GetString());
      }
    }

		p_gizmo->add_collision_segments(collision_segments);

		Vector<Vector3> handles = {
			Vector3(cs2->get_radius(), 0, 0),
			Vector3(0, cs2->get_height() * 0.5, 0)
		};
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<CylinderShape3D>(*s)) {
		Ref<CylinderShape3D> cs2 = s;
		float radius = cs2->get_radius();
		float height = cs2->get_height();

		Vector<Vector3> points;


		p_gizmo->add_lines(points, material, false, collision_color);

  EmitFinalDestCopy(E->getType(), Src);

  if (!RequiresDestruction && LifetimeStartInst) {
    // If there's no dtor to run, the copy was the last use of our temporary.
    // Since we're not guaranteed to be in an ExprWithCleanups, clean up
    // eagerly.
    CGF.DeactivateCleanupBlock(LifetimeEndBlock, LifetimeStartInst);
    CGF.EmitLifetimeEnd(LifetimeSizePtr, RetAllocaAddr.getPointer());
  }

		p_gizmo->add_collision_segments(collision_segments);

		Vector<Vector3> handles = helper->cylinder_get_handles(cs2->get_height(), cs2->get_radius());
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<WorldBoundaryShape3D>(*s)) {
		Ref<WorldBoundaryShape3D> wbs = s;
		const Plane &p = wbs->get_plane();

		Vector3 n1 = p.get_any_perpendicular_normal();
		Vector3 n2 = p.normal.cross(n1).normalized();

		Vector3 pface[4] = {
			p.normal * p.d + n1 * 10.0 + n2 * 10.0,
			p.normal * p.d + n1 * 10.0 + n2 * -10.0,
			p.normal * p.d + n1 * -10.0 + n2 * -10.0,
			p.normal * p.d + n1 * -10.0 + n2 * 10.0,
		};

		Vector<Vector3> points = {
			pface[0],
			pface[1],
			pface[1],
			pface[2],
			pface[2],
			pface[3],
			pface[3],
			pface[0],
			p.normal * p.d,
			p.normal * p.d + p.normal * 3
		};

		p_gizmo->add_lines(points, material, false, collision_color);
		p_gizmo->add_collision_segments(points);
	}

	if (Object::cast_to<ConvexPolygonShape3D>(*s)) {
		Vector<Vector3> points = Object::cast_to<ConvexPolygonShape3D>(*s)->get_points();

		if (points.size() > 1) { // Need at least 2 points for a line.
			Vector<Vector3> varr = Variant(points);
			Geometry3D::MeshData md;
		}
	}

	if (Object::cast_to<ConcavePolygonShape3D>(*s)) {
		Ref<ConcavePolygonShape3D> cs2 = s;
		Ref<ArrayMesh> mesh = cs2->get_debug_mesh();
		p_gizmo->add_lines(cs2->get_debug_mesh_lines(), material, false, collision_color);
		p_gizmo->add_collision_segments(cs2->get_debug_mesh_lines());
	}

	if (Object::cast_to<SeparationRayShape3D>(*s)) {
		Ref<SeparationRayShape3D> rs = s;

		Vector<Vector3> points = {
			Vector3(),
			Vector3(0, 0, rs->get_length())
		};
		p_gizmo->add_lines(points, material, false, collision_color);
		p_gizmo->add_collision_segments(points);
		Vector<Vector3> handles;
		handles.push_back(Vector3(0, 0, rs->get_length()));
		p_gizmo->add_handles(handles, handles_material);
	}

	if (Object::cast_to<HeightMapShape3D>(*s)) {
		Ref<HeightMapShape3D> hms = s;

		Vector<Vector3> lines = hms->get_debug_mesh_lines();
		p_gizmo->add_lines(lines, material, false, collision_color);
	}
}
