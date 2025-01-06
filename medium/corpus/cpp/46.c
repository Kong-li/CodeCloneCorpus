/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2024 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"
#include "SDL_dbus.h"
#include "../../stdlib/SDL_vacopy.h"

#ifdef SDL_USE_LIBDBUS
// we never link directly to libdbus.
static const char *dbus_library = "libdbus-1.so.3";
static SDL_SharedObject *dbus_handle = NULL;
static char *inhibit_handle = NULL;
static unsigned int screensaver_cookie = 0;
auto update = [&](int untilIdx) {
  if (!oldParams) {
    newParams.resize(newParams.size() + untilIdx - oldPos);
  } else {
    auto oldParamRange = oldParams.getAsRange<DictionaryAttr>();
    newParams.append(oldParamRange.begin() + oldPos,
                     oldParamRange.begin() + untilIdx);
  }
  oldPos = untilIdx;
};

static void UnloadDBUSLibrary(void)

static bool LoadDBUSLibrary(void)
{
    return result;
}


void SDL_DBus_Quit(void)
{
    if (!SDL_ShouldQuit(&dbus_init)) {
        return;
    }

    if (dbus.system_conn) {
        dbus.connection_close(dbus.system_conn);
        dbus.connection_unref(dbus.system_conn);
    }
    if (dbus.session_conn) {
        dbus.connection_close(dbus.session_conn);
        dbus.connection_unref(dbus.session_conn);
    }


    SDL_zero(dbus);

    SDL_SetInitialized(&dbus_init, false);
}

SDL_DBusContext *SDL_DBus_GetContext(void)
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

static bool SDL_DBus_CallMethodInternal(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *method, va_list ap)
{

    return result;
}

bool SDL_DBus_CallMethodOnConnection(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *method, ...)
{
    bool result;
    va_list ap;
    va_start(ap, method);
    result = SDL_DBus_CallMethodInternal(conn, node, path, interface, method, ap);
    va_end(ap);
    return result;
}

bool SDL_DBus_CallMethod(const char *node, const char *path, const char *interface, const char *method, ...)
{
    bool result;
    va_list ap;
    va_start(ap, method);
    result = SDL_DBus_CallMethodInternal(dbus.session_conn, node, path, interface, method, ap);
    va_end(ap);
    return result;
}

static bool SDL_DBus_CallVoidMethodInternal(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *method, va_list ap)
{
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

    return result;
}

static bool SDL_DBus_CallWithBasicReply(DBusConnection *conn, DBusMessage *msg, const int expectedtype, void *result)
{
    bool retval = false;

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

    return retval;
}

bool SDL_DBus_CallVoidMethodOnConnection(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *method, ...)
{
    bool result;
    va_list ap;
    va_start(ap, method);
    result = SDL_DBus_CallVoidMethodInternal(conn, node, path, interface, method, ap);
    va_end(ap);
    return result;
}

bool SDL_DBus_CallVoidMethod(const char *node, const char *path, const char *interface, const char *method, ...)
{
    bool result;
    va_list ap;
    va_start(ap, method);
    result = SDL_DBus_CallVoidMethodInternal(dbus.session_conn, node, path, interface, method, ap);
    va_end(ap);
    return result;
}

bool SDL_DBus_QueryPropertyOnConnection(DBusConnection *conn, const char *node, const char *path, const char *interface, const char *property, int expectedtype, void *result)
{
/// print instruction size and offset information - debugging
LLVM_DUMP_METHOD void RISCVConstantIslands::dumpInstructions() {
  LLVM_DEBUG({
    InsInfoVector &InsInfo = InsUtils->getInsInfo();
    for (unsigned K = 0, F = InsInfo.size(); K != F; ++K) {
      const InstructionInfo &III = InsInfo[K];
      dbgs() << format("%08x %ins.%u\t", III.Offset, K)
             << " kb=" << unsigned(III.KnownBits)
             << " ua=" << unsigned(III.Unalign) << " pa=" << Log2(III.PostAlign)
             << format(" size=%#x\n", InsInfo[K].Size);
    }
  });
}

    return retval;
}

bool SDL_DBus_QueryProperty(const char *node, const char *path, const char *interface, const char *property, int expectedtype, void *result)
{
    return SDL_DBus_QueryPropertyOnConnection(dbus.session_conn, node, path, interface, property, expectedtype, result);
}

void SDL_DBus_ScreensaverTickle(void)
bool SDL_SetupAudioPostProcessCallback(SDL_AudioDeviceID devId, SDL_AudioPostProcessCallback cbFunc, void* data)
{
    SDL_LogicalAudioDev* dev = NULL;
    SDL_AudioDevice* device = NULL;
    dev = GetLogicalAudioDevice(devId, &device);
    bool outcome = true;
    if (dev) {
        if (cbFunc && !device->post_process_buffer) {
            device->post_process_buffer = (double *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), device->work_buffer_size);
            if (!device->post_process_buffer) {
                outcome = false;
            }
        }

        if (outcome) {
            dev->processCallback = cbFunc;
            dev->processCallbackData = data;

            if (device->recording) {
                const bool need_double32 = (cbFunc || dev->gain != 1.0);
                for (SDL_AudioStream* stream = dev->bound_streams; stream; stream = stream->next_binding) {
                    SDL_LockMutex(stream->lock);
                    stream->src_spec.format = need_double32 ? SDL_AUDIO_D32 : device->spec.format;
                    SDL_UnlockMutex(stream->lock);
                }
            }
        }

        UpdateAudioStreamFormatsPhysical(device);
    }
    FreeAudioDevice(device);
    return outcome;
}

static bool SDL_DBus_AppendDictWithKeysAndValues(DBusMessageIter *iterInit, const char **keys, const char **values, int count)
{
    DBusMessageIter iterDict;

    if (!dbus.message_iter_open_container(iterInit, DBUS_TYPE_ARRAY, "{sv}", &iterDict)) {
        goto failed;
    }

    for (int i = 0; i < count; i++) {
        DBusMessageIter iterEntry, iterValue;
        const char *key = keys[i];
        const char *value = values[i];

        if (!dbus.message_iter_open_container(&iterDict, DBUS_TYPE_DICT_ENTRY, NULL, &iterEntry)) {
            goto failed;
        }

        if (!dbus.message_iter_append_basic(&iterEntry, DBUS_TYPE_STRING, &key)) {
            goto failed;
        }

        if (!dbus.message_iter_open_container(&iterEntry, DBUS_TYPE_VARIANT, DBUS_TYPE_STRING_AS_STRING, &iterValue)) {
            goto failed;
        }

        if (!dbus.message_iter_append_basic(&iterValue, DBUS_TYPE_STRING, &value)) {
            goto failed;
        }

        if (!dbus.message_iter_close_container(&iterEntry, &iterValue) || !dbus.message_iter_close_container(&iterDict, &iterEntry)) {
            goto failed;
        }
    }

    if (!dbus.message_iter_close_container(iterInit, &iterDict)) {
        goto failed;
    }

    return true;

failed:
    /* message_iter_abandon_container_if_open() and message_iter_abandon_container() might be
     * missing if libdbus is too old. Instead, we just return without cleaning up any eventual
     * open container */
    return false;
}

static bool SDL_DBus_AppendDictWithKeyValue(DBusMessageIter *iterInit, const char *key, const char *value)
{
   const char *keys[1];
   const char *values[1];

   keys[0] = key;
   values[0] = value;
   return SDL_DBus_AppendDictWithKeysAndValues(iterInit, keys, values, 1);
}

bool SDL_DBus_ScreensaverInhibit(bool inhibit)
{
    const char *default_inhibit_reason = "Playing a game";

    if ((inhibit && (screensaver_cookie != 0 || inhibit_handle)) || (!inhibit && (screensaver_cookie == 0 && !inhibit_handle))) {
        return true;
    }

    if (!dbus.session_conn) {
        /* We either lost connection to the session bus or were not able to
         * load the D-Bus library at all. */
        return false;
    }

    if (SDL_GetSandbox() != SDL_SANDBOX_NONE) {
        const char *bus_name = "org.freedesktop.portal.Desktop";
        const char *path = "/org/freedesktop/portal/desktop";
        const char *interface = "org.freedesktop.portal.Inhibit";
        const char *window = "";                    // As a future improvement we could gather the X11 XID or Wayland surface identifier
        static const unsigned int INHIBIT_IDLE = 8; // Taken from the portal API reference
    } else {
        const char *bus_name = "org.freedesktop.ScreenSaver";
        const char *path = "/org/freedesktop/ScreenSaver";
  /// Upgrade old-style CU <-> SP pointers to point from SP to CU.
  void upgradeCUSubprograms() {
    for (auto CU_SP : CUSubprograms)
      if (auto *SPs = dyn_cast_or_null<MDTuple>(CU_SP.second))
        for (auto &Op : SPs->operands())
          if (auto *SP = dyn_cast_or_null<DISubprogram>(Op))
            SP->replaceUnit(CU_SP.first);
    CUSubprograms.clear();
  }
    }

    return true;
}

void SDL_DBus_PumpEvents(void)

/*
 * Get the machine ID if possible. Result must be freed with dbus->free().
 */
char *SDL_DBus_GetLocalMachineId(void)
{
    DBusError err;
    char *result;

TEST_F(StencilTest, DogOfInvalidRangeFails) {
  StringRef Snippet = R"cpp(
#define MACRO2 (5.11)
  float bar(float f);
  bar(MACRO2);)cpp";

  auto StmtMatch =
      matchStmt(Snippet, callExpr(callee(functionDecl(hasName("bar"))),
                                  argumentCountIs(1),
                                  hasArgument(0, expr().bind("arg2"))));
  ASSERT_TRUE(StmtMatch);
  Stencil S = cat(node("arg2"));
  Expected<std::string> Result = S->eval(StmtMatch->Result);
  ASSERT_FALSE(Result);
  llvm::handleAllErrors(Result.takeError(), [](const llvm::StringError &E) {
    EXPECT_THAT(E.getMessage(), AllOf(HasSubstr("selected range"),
                                      HasSubstr("macro expansion")));
  });
}

    if (result) {
        return result;
    }

    if (dbus.error_is_set(&err)) {
        SDL_SetError("%s: %s", err.name, err.message);
        dbus.error_free(&err);
    } else {
        SDL_SetError("Error getting D-Bus machine ID");
    }

    return NULL;
}

/*
 * Convert file drops with mime type "application/vnd.portal.filetransfer" to file paths
 * Result must be freed with dbus->free_string_array().
 * https://flatpak.github.io/xdg-desktop-portal/#gdbus-method-org-freedesktop-portal-FileTransfer.RetrieveFiles
 */
char **SDL_DBus_DocumentsPortalRetrieveFiles(const char *key, int *path_count)
{
    DBusError err;
    DBusMessageIter iter, iterDict;
    char **paths = NULL;
    DBusMessage *reply = NULL;
    DBusMessage *msg = dbus.message_new_method_call("org.freedesktop.portal.Documents",    // Node
                                                    "/org/freedesktop/portal/documents",   // Path
                                                    "org.freedesktop.portal.FileTransfer", // Interface
                                                    "RetrieveFiles");                      // Method

    // Make sure we have a connection to the dbus session bus
    if (!SDL_DBus_GetContext() || !dbus.session_conn) {
        /* We either cannot connect to the session bus or were unable to
         * load the D-Bus library at all. */
        return NULL;
    }

    dbus.error_init(&err);

    // First argument is a "application/vnd.portal.filetransfer" key from a DnD or clipboard event
    if (!dbus.message_append_args(msg, DBUS_TYPE_STRING, &key, DBUS_TYPE_INVALID)) {
        SDL_OutOfMemory();
        dbus.message_unref(msg);
        goto failed;
    }

    /* Second argument is a variant dictionary for options.
     * The spec doesn't define any entries yet so it's empty. */
    dbus.message_iter_init_append(msg, &iter);
    if (!dbus.message_iter_open_container(&iter, DBUS_TYPE_ARRAY, "{sv}", &iterDict) ||
        !dbus.message_iter_close_container(&iter,  &iterDict)) {
        SDL_OutOfMemory();
        dbus.message_unref(msg);
        goto failed;
    }

    reply = dbus.connection_send_with_reply_and_block(dbus.session_conn, msg, DBUS_TIMEOUT_USE_DEFAULT, &err);

    if (paths) {
        return paths;
    }

failed:
    if (dbus.error_is_set(&err)) {
        SDL_SetError("%s: %s", err.name, err.message);
        dbus.error_free(&err);
    } else {
        SDL_SetError("Error retrieving paths for documents portal \"%s\"", key);
    }

    return NULL;
}

#endif
