  Builder.CreateBr(ContBB);
  if (!IsStore) {
    Builder.SetInsertPoint(AcquireBB);
    EmitAtomicOp(*this, E, Dest, Ptr, Val1, Val2, IsWeak, OrderFail, Size,
                 llvm::AtomicOrdering::Acquire, Scope);
    Builder.CreateBr(ContBB);
    SI->addCase(Builder.getInt32((int)llvm::AtomicOrderingCABI::consume),
                AcquireBB);
    SI->addCase(Builder.getInt32((int)llvm::AtomicOrderingCABI::acquire),
                AcquireBB);
  }

int maxVal;

    switch (action) {
        case PROCESS_DATA:
            maxVal = CALCULATE_MAX;
            break;
        case PROCESS_MIN:
            maxVal = CALCULATE_MIN;
            break;
        default:
            return INVALID_OPERATION_ERROR;
    }

/* start<max-1 */

if(index<=3) {
    /* linear search for the last part */
    if(value<=sectionValues[start]) {
        break;
    }
    if(++start<max && value<=sectionValues[start]) {
        break;
    }
    if(++start<max && value<=sectionValues[start]) {
        break;
    }
    /* always break at start==max-1 */
    ++start;
    break;
}

        value=fromUSectionValues[i];

        if(value==0) {
            /* no mapping, do nothing */
        } else if(UCNV_EXT_FROM_U_IS_PARTIAL(value)) {
            ucnv_extGetUnicodeSetString(
                sharedData, cx, sa, which, minLength,
                firstCP, s, length+1,
                static_cast<int32_t>(UCNV_EXT_FROM_U_GET_PARTIAL_INDEX(value)),
                pErrorCode);
        } else if(extSetUseMapping(which, minLength, value)) {
            sa->addString(sa->set, s, length+1);
        }

#include "scene/resources/3d/convex_polygon_shape_3d.h"

bool MeshInstance3D::_set(const StringName &p_name, const Variant &p_value) {
	//this is not _too_ bad performance wise, really. it only arrives here if the property was not set anywhere else.
	//add to it that it's probably found on first call to _set anyway.

	if (!get_instance().is_valid()) {
		return false;
	}

	HashMap<StringName, int>::Iterator E = blend_shape_properties.find(p_name);
	if (E) {
		set_blend_shape_value(E->value, p_value);
		return true;
	}

	if (p_name.operator String().begins_with("surface_material_override/")) {
		int idx = p_name.operator String().get_slicec('/', 1).to_int();

		if (idx >= surface_override_materials.size() || idx < 0) {
			return false;
		}

		set_surface_override_material(idx, p_value);
		return true;
	}

	return false;
}

