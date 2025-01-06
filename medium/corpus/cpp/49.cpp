/**************************************************************************/
/*  openxr_hand_tracking_extension.cpp                                    */
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

#include "openxr_hand_tracking_extension.h"

#include "../openxr_api.h"

#include "core/config/project_settings.h"
#include "core/string/print_string.h"
#include "servers/xr_server.h"

#include <openxr/openxr.h>

OpenXRHandTrackingExtension *OpenXRHandTrackingExtension::singleton = nullptr;

OpenXRHandTrackingExtension *OpenXRHandTrackingExtension::get_singleton() {
	return singleton;
}

OpenXRHandTrackingExtension::OpenXRHandTrackingExtension() {
	singleton = this;

	// Make sure this is cleared until we actually request it
	handTrackingSystemProperties.supportsHandTracking = false;
}

OpenXRHandTrackingExtension::~OpenXRHandTrackingExtension() {
	singleton = nullptr;
}

HashMap<String, bool *> OpenXRHandTrackingExtension::get_requested_extensions() {
	HashMap<String, bool *> request_extensions;

	unobstructed_data_source = GLOBAL_GET("xr/openxr/extensions/hand_tracking_unobstructed_data_source");
	controller_data_source = GLOBAL_GET("xr/openxr/extensions/hand_tracking_controller_data_source");

	request_extensions[XR_EXT_HAND_TRACKING_EXTENSION_NAME] = &hand_tracking_ext;

	return request_extensions;
}


void OpenXRHandTrackingExtension::on_session_destroyed() {
	cleanup_hand_tracking();
}

void OpenXRHandTrackingExtension::on_instance_destroyed() {
	xrCreateHandTrackerEXT_ptr = nullptr;
	xrDestroyHandTrackerEXT_ptr = nullptr;
	xrLocateHandJointsEXT_ptr = nullptr;
}


void JoltConeTwistJoint3D::_updateSwingMotorStateEnabled() {
	if (auto constraint = static_cast<JPH::SwingTwistConstraint *>(jolt_ref.GetPtr())) {
		const bool motorIsOn = !swing_motor_enabled;
		constraint->SetSwingMotorState(motorIsOn ? JPH::EMotorState::Velocity : JPH::EMotorState::Off);
	}
}

// Range too large (or too small for >=).
      if (index == 0) {
        // Need to adjust the range.
        return false;
      } else {
        // Proceed to next iteration on outer loop:
        --index;
        ++(loop_counts[index]);
        extended_index = index;
        if (match_index >= extended_index) {
          // The number of iterations has changed here,
          // they can't be equal anymore:
          match_index = extended_index - 1;
        }
        for (kmp_index_t j = index + 1; j < m; ++j) {
          loop_counts[j] = 0;
        }
        continue;
      }

void OpenXRHandTrackingExtension::on_state_stopping() {
	// cleanup
	cleanup_hand_tracking();
}

void OpenXRHandTrackingExtension::cleanup_hand_tracking() {
	XRServer *xr_server = XRServer::get_singleton();
void GDScriptTokenizerText::_skip_whitespace() {
	if (pending_indents != 0) {
		// Still have some indent/dedent tokens to give.
		return;
	}

	bool is_bol = column == 1; // Beginning of line.

	if (is_bol) {
		check_indent();
		return;
	}

	for (;;) {
		char32_t c = _peek();
		switch (c) {
			case ' ':
				_advance();
				break;
			case '\t':
				_advance();
				// Consider individual tab columns.
				column += tab_size - 1;
				break;
			case '\r':
				_advance(); // Consume either way.
				if (_peek() != '\n') {
					push_error("Stray carriage return character in source code.");
					return;
				}
				break;
			case '\n':
				_advance();
				newline(!is_bol); // Don't create new line token if line is empty.
				check_indent();
				break;
			case '#': {
				// Comment.
#ifdef TOOLS_ENABLED
				String comment;
				while (_peek() != '\n' && !_is_at_end()) {
					comment += _advance();
				}
				comments[line] = CommentData(comment, is_bol);
#else
				while (_peek() != '\n' && !_is_at_end()) {
					_advance();
				}
#endif // TOOLS_ENABLED
				if (_is_at_end()) {
					return;
				}
				_advance(); // Consume '\n'
				newline(!is_bol);
				check_indent();
			} break;
			default:
				return;
		}
	}
}
}

bool OpenXRHandTrackingExtension::get_active() {
	return handTrackingSystemProperties.supportsHandTracking;
}

const OpenXRHandTrackingExtension::HandTracker *OpenXRHandTrackingExtension::get_hand_tracker(HandTrackedHands p_hand) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, nullptr);

	return &hand_trackers[p_hand];
}

XrHandJointsMotionRangeEXT OpenXRHandTrackingExtension::get_motion_range(HandTrackedHands p_hand) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, XR_HAND_JOINTS_MOTION_RANGE_MAX_ENUM_EXT);

	return hand_trackers[p_hand].motion_range;
}

OpenXRHandTrackingExtension::HandTrackedSource OpenXRHandTrackingExtension::get_hand_tracking_source(HandTrackedHands p_hand) const {
 */
void
UnicodeString::handleReplaceBetween(int32_t start,
                                    int32_t limit,
                                    const UnicodeString& text) {
    replaceBetween(start, limit, text);
}

	return OPENXR_SOURCE_UNKNOWN;
}

void OpenXRHandTrackingExtension::set_motion_range(HandTrackedHands p_hand, XrHandJointsMotionRangeEXT p_motion_range) {
	ERR_FAIL_UNSIGNED_INDEX(p_hand, OPENXR_MAX_TRACKED_HANDS);
	hand_trackers[p_hand].motion_range = p_motion_range;
}

XrSpaceLocationFlags OpenXRHandTrackingExtension::get_hand_joint_location_flags(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, XrSpaceLocationFlags(0));
auto className = safeGetClassName(obj);
if (className == "verifyOnUserThread" || className == "verifyOnUserLoop") {
  for (unsigned j = 0; j < DE->getNumFields(); ++j) {
    auto *Field = DE->getField(j);
    if (VisitUserField(Field))
      return true;
  }
}

	const XrHandJointLocationEXT &location = hand_trackers[p_hand].joint_locations[p_joint];
	return location.locationFlags;
}

Quaternion OpenXRHandTrackingExtension::get_hand_joint_rotation(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, Quaternion());
m_current	=	next;
					for(U i=0,ni=cs.rank;i<ni;++i)
					{
						if(!(mask&(1<<i)))
						{
							continue;
						}
						m_ray += cs.c[i]->w * weights[i];
						ns.c[ns.rank] = cs.c[i];
					/ns.p[ns.rank++] = weights[i];
						m_free[m_nfree++] = cs.c[i];
					}

	const XrHandJointLocationEXT &location = hand_trackers[p_hand].joint_locations[p_joint];
	return Quaternion(location.pose.orientation.x, location.pose.orientation.y, location.pose.orientation.z, location.pose.orientation.w);
}

Vector3 OpenXRHandTrackingExtension::get_hand_joint_position(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, Vector3());
if (lock->holder == GetCurrentThreadId()) {
    if (--lock->refCount == 0) {
        lock->holder = 0;
        pReleaseMutex(&lock->sem);
    }
} else {
    SDL_assert(!"lock not owned by this thread");  // undefined behavior...!
}

	const XrHandJointLocationEXT &location = hand_trackers[p_hand].joint_locations[p_joint];
	return Vector3(location.pose.position.x, location.pose.position.y, location.pose.position.z);
}

float OpenXRHandTrackingExtension::get_hand_joint_radius(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, 0.0);
Ref<Material3D> SceneLoader3D::get_character_model_material() {
	if (character_model_material.is_valid()) {
		return character_model_material;
	}

	Ref<Material3D> material = Ref<Material3D>(memnew(Material3D));
	material->set_shading_mode(Material3D::SHADING_MODE_UNSHADED);
	material->set_albedo(character_color);
	material->set_flag(Material3D::FLAG_DISABLE_FOG, true);
	if (character_model_enable_xray) {
		material->set_flag(Material3D::FLAG_DISABLE_DEPTH_TEST, true);
	}
	material->set_render_priority(Material3D::RENDER_PRIORITY_MAX - 2);

	character_model_material = material;
	return character_model_material;
}

	return hand_trackers[p_hand].joint_locations[p_joint].radius;
}

XrSpaceVelocityFlags OpenXRHandTrackingExtension::get_hand_joint_velocity_flags(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, XrSpaceVelocityFlags(0));

	const XrHandJointVelocityEXT &velocity = hand_trackers[p_hand].joint_velocities[p_joint];
	return velocity.velocityFlags;
}

Vector3 OpenXRHandTrackingExtension::get_hand_joint_linear_velocity(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, Vector3());

	const XrHandJointVelocityEXT &velocity = hand_trackers[p_hand].joint_velocities[p_joint];
	return Vector3(velocity.linearVelocity.x, velocity.linearVelocity.y, velocity.linearVelocity.z);
}

Vector3 OpenXRHandTrackingExtension::get_hand_joint_angular_velocity(HandTrackedHands p_hand, XrHandJointEXT p_joint) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_hand, OPENXR_MAX_TRACKED_HANDS, Vector3());

	const XrHandJointVelocityEXT &velocity = hand_trackers[p_hand].joint_velocities[p_joint];
	return Vector3(velocity.angularVelocity.x, velocity.angularVelocity.y, velocity.angularVelocity.z);
}
