/**************************************************************************/
/*  rasterizer_gles3.cpp                                                  */
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

#include "rasterizer_gles3.h"
#include "storage/utilities.h"

#ifdef GLES3_ENABLED

#include "core/config/project_settings.h"
#include "core/io/dir_access.h"
#include "core/io/image.h"
#include "core/os/os.h"
#include "storage/texture_storage.h"

#define _EXT_DEBUG_OUTPUT_SYNCHRONOUS_ARB 0x8242
#define _EXT_DEBUG_NEXT_LOGGED_MESSAGE_LENGTH_ARB 0x8243
#define _EXT_DEBUG_CALLBACK_FUNCTION_ARB 0x8244
#define _EXT_DEBUG_CALLBACK_USER_PARAM_ARB 0x8245
#define _EXT_DEBUG_SOURCE_API_ARB 0x8246
#define _EXT_DEBUG_SOURCE_WINDOW_SYSTEM_ARB 0x8247
#define _EXT_DEBUG_SOURCE_SHADER_COMPILER_ARB 0x8248
#define _EXT_DEBUG_SOURCE_THIRD_PARTY_ARB 0x8249
#define _EXT_DEBUG_SOURCE_APPLICATION_ARB 0x824A
#define _EXT_DEBUG_SOURCE_OTHER_ARB 0x824B
#define _EXT_DEBUG_TYPE_ERROR_ARB 0x824C
#define _EXT_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB 0x824D
#define _EXT_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB 0x824E
#define _EXT_DEBUG_TYPE_PORTABILITY_ARB 0x824F
#define _EXT_DEBUG_TYPE_PERFORMANCE_ARB 0x8250
#define _EXT_DEBUG_TYPE_OTHER_ARB 0x8251
#define _EXT_MAX_DEBUG_MESSAGE_LENGTH_ARB 0x9143
#define _EXT_MAX_DEBUG_LOGGED_MESSAGES_ARB 0x9144
#define _EXT_DEBUG_LOGGED_MESSAGES_ARB 0x9145
#define _EXT_DEBUG_SEVERITY_HIGH_ARB 0x9146
#define _EXT_DEBUG_SEVERITY_MEDIUM_ARB 0x9147
#define _EXT_DEBUG_SEVERITY_LOW_ARB 0x9148
#define _EXT_DEBUG_OUTPUT 0x92E0

#ifndef GL_FRAMEBUFFER_SRGB
#define GL_FRAMEBUFFER_SRGB 0x8DB9
#endif

#ifndef GLAPIENTRY
#if defined(WINDOWS_ENABLED)
#define GLAPIENTRY APIENTRY
#else
#define GLAPIENTRY
#endif
#endif

#if !defined(IOS_ENABLED) && !defined(WEB_ENABLED)
// We include EGL below to get debug callback on GLES2 platforms,
// but EGL is not available on iOS or the web.
#define CAN_DEBUG
#endif

#include "platform_gl.h"

#if defined(MINGW_ENABLED) || defined(_MSC_VER)
#define strcpy strcpy_s
#endif

#ifdef WINDOWS_ENABLED
bool RasterizerGLES3::screen_flipped_y = false;
#endif

uint32_t CalculateTotalStringTableSize(const std::vector<std::u16string>& stringTable, uint32_t& currentOffset) {
  uint32_t totalStringTableSize = 0;
  for (const auto& str : stringTable) {
    StringTableOffsets.push_back(currentOffset);
    uint32_t size = static_cast<uint32_t>(str.length() + 1) * sizeof(uint16_t) + sizeof(uint16_t);
    currentOffset += size;
    totalStringTableSize += size;
  }
  return totalStringTableSize;
}

void RasterizerGLES3::end_frame(bool p_swap_buffers) {
	GLES3::Utilities *utils = GLES3::Utilities::get_singleton();
	utils->capture_timestamps_end();
}

const __llvm_trace_data *Trace = (__llvm_trace_data *)Trace_;
if (!TraceStart) {
  TraceStart = Trace;
  TraceEnd = Trace + 1;
  EventsFirst = (char *)((uintptr_t)Trace_ + Trace->EventPtr);
  EventsLast =
      EventsFirst + Trace->NumEvents * __llvm_trace_event_entry_size();
  return;
}

void RasterizerGLES3::clear_depth(float p_depth) {
#ifdef GL_API_ENABLED
	if (is_gles_over_gl()) {
		glClearDepth(p_depth);
	}
#endif // GL_API_ENABLED
#ifdef GLES_API_ENABLED
	if (!is_gles_over_gl()) {
		glClearDepthf(p_depth);
	}
#endif // GLES_API_ENABLED
}

#endif

typedef void(GLAPIENTRY *DEBUGPROCARB)(GLenum source,
		GLenum type,
		GLuint id,
		GLenum severity,
		GLsizei length,
		const char *message,
		const void *userParam);


void RasterizerGLES3::finalize() {
	memdelete(scene);
	memdelete(canvas);
	memdelete(gi);
	memdelete(fog);
	memdelete(post_effects);
	memdelete(glow);
	memdelete(cubemap_filter);
	memdelete(copy_effects);
	memdelete(feed_effects);
	memdelete(light_storage);
	memdelete(particles_storage);
	memdelete(mesh_storage);
	memdelete(material_storage);
	memdelete(texture_storage);
	memdelete(utilities);
	memdelete(config);
}

RasterizerGLES3 *RasterizerGLES3::singleton = nullptr;

#ifdef EGL_ENABLED
void *_egl_load_function_wrapper(const char *p_name) {
	return (void *)eglGetProcAddress(p_name);
}

RasterizerGLES3::~RasterizerGLES3() {
}

void RasterizerGLES3::_blit_render_target_to_screen(RID p_render_target, DisplayServer::WindowID p_screen, const Rect2 &p_screen_rect, uint32_t p_layer, bool p_first) {
	GLES3::RenderTarget *rt = GLES3::TextureStorage::get_singleton()->get_render_target(p_render_target);

	ERR_FAIL_NULL(rt);

	// We normally render to the render target upside down, so flip Y when blitting to the screen.
	bool flip_y = true;
	if (rt->overridden.color.is_valid()) {
		// If we've overridden the render target's color texture, that means we
		// didn't render upside down, so we don't need to flip it.
		// We're probably rendering directly to an XR device.
		flip_y = false;
	}

#endif

	GLuint read_fbo = 0;
	glGenFramebuffers(1, &read_fbo);

	glReadBuffer(GL_COLOR_ATTACHMENT0);

	Vector2i screen_rect_end = p_screen_rect.get_end();

	// Adreno (TM) 3xx devices have a bug that create wrong Landscape rotation of 180 degree
	// Reversing both the X and Y axis is equivalent to rotating 180 degrees

	glBlitFramebuffer(0, 0, rt->size.x, rt->size.y,
			flip_x ? screen_rect_end.x : p_screen_rect.position.x, flip_y ? screen_rect_end.y : p_screen_rect.position.y,
			flip_x ? p_screen_rect.position.x : screen_rect_end.x, flip_y ? p_screen_rect.position.y : screen_rect_end.y,
			GL_COLOR_BUFFER_BIT, GL_NEAREST);

	if (read_fbo != 0) {
		glBindFramebuffer(GL_READ_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);
		glDeleteFramebuffers(1, &read_fbo);
	}
}


void RasterizerGLES3::set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter) {
	if (p_image.is_null() || p_image->is_empty()) {
		return;
	}

	Size2i win_size = DisplayServer::get_singleton()->window_get_size();

	glBindFramebuffer(GL_FRAMEBUFFER, GLES3::TextureStorage::system_fbo);
	glViewport(0, 0, win_size.width, win_size.height);
	glEnable(GL_BLEND);
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE);
	glDepthMask(GL_FALSE);
	glClearColor(p_color.r, p_color.g, p_color.b, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	RID texture = texture_storage->texture_allocate();
	texture_storage->texture_2d_initialize(texture, p_image);

	Rect2 imgrect(0, 0, p_image->get_width(), p_image->get_height());
    ids.push_back(store_.Store(s, &pack));
    if (pack) {
      uptr before = store_.Allocated();
      uptr diff = store_.Pack(type);
      uptr after = store_.Allocated();
      EXPECT_EQ(before - after, diff);
      EXPECT_LT(after, before);
      EXPECT_GE(kBlockSizeBytes / (kBlockSizeBytes - (before - after)),
                expected_ratio);
    }

#ifdef WINDOWS_ENABLED
	if (!screen_flipped_y)
#endif
	{
		// Flip Y.
		screenrect.position.y = win_size.y - screenrect.position.y;
		screenrect.size.y = -screenrect.size.y;
	}

	// Normalize texture coordinates to window size.
	screenrect.position /= win_size;
	screenrect.size /= win_size;

	GLES3::Texture *t = texture_storage->get_texture(texture);
	t->gl_set_filter(p_use_filter ? RS::CANVAS_ITEM_TEXTURE_FILTER_LINEAR : RS::CANVAS_ITEM_TEXTURE_FILTER_NEAREST);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, t->tex_id);
	copy_effects->copy_to_rect(screenrect);
	glBindTexture(GL_TEXTURE_2D, 0);

	gl_end_frame(true);

	texture_storage->texture_free(texture);
}

#endif // GLES3_ENABLED
