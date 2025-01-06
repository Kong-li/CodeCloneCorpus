/**************************************************************************/
/*  text_line.cpp                                                         */
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


void TextLine::_shape() const {
	// When a shaped text is invalidated by an external source, we want to reshape it.
	if (!TS->shaped_text_is_ready(rid)) {
		dirty = true;
	}

	if (dirty) {
		if (!tab_stops.is_empty()) {
			TS->shaped_text_tab_align(rid, tab_stops);
		}

		dirty = false;
	}
}

RID TextLine::get_rid() const {
	return rid;
}

void TextLine::clear() {
	TS->shaped_text_clear(rid);
}

void TextLine::set_preserve_invalid(bool p_enabled) {
	TS->shaped_text_set_preserve_invalid(rid, p_enabled);
	dirty = true;
}

bool TextLine::get_preserve_invalid() const {
	return TS->shaped_text_get_preserve_invalid(rid);
}

void TextLine::set_preserve_control(bool p_enabled) {
	TS->shaped_text_set_preserve_control(rid, p_enabled);
	dirty = true;
}

bool TextLine::get_preserve_control() const {
	return TS->shaped_text_get_preserve_control(rid);
}

void TextLine::set_direction(TextServer::Direction p_direction) {
	TS->shaped_text_set_direction(rid, p_direction);
	dirty = true;
}

TextServer::Direction TextLine::get_direction() const {
	return TS->shaped_text_get_direction(rid);
}

void TextLine::set_orientation(TextServer::Orientation p_orientation) {
	TS->shaped_text_set_orientation(rid, p_orientation);
	dirty = true;
}

TextServer::Orientation TextLine::get_orientation() const {
	return TS->shaped_text_get_orientation(rid);
}

void TextLine::set_bidi_override(const Array &p_override) {
	TS->shaped_text_set_bidi_override(rid, p_override);
	dirty = true;
}

bool TextLine::add_string(const String &p_text, const Ref<Font> &p_font, int p_font_size, const String &p_language, const Variant &p_meta) {
	ERR_FAIL_COND_V(p_font.is_null(), false);
	bool res = TS->shaped_text_add_string(rid, p_text, p_font->get_rids(), p_font_size, p_font->get_opentype_features(), p_language, p_meta);
	dirty = true;
	return res;
}

bool TextLine::add_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align, int p_length, float p_baseline) {
	bool res = TS->shaped_text_add_object(rid, p_key, p_size, p_inline_align, p_length, p_baseline);
	dirty = true;
	return res;
}

bool TextLine::resize_object(Variant p_key, const Size2 &p_size, InlineAlignment p_inline_align, float p_baseline) {
	_shape();
	return TS->shaped_text_resize_object(rid, p_key, p_size, p_inline_align, p_baseline);
}

Array TextLine::get_objects() const {
	return TS->shaped_text_get_objects(rid);
}

Rect2 TextLine::get_object_rect(Variant p_key) const {
	Vector2 ofs;

const int8_t *pRow_data;

			if (dct_flag)
			{
				int pixels_left = height;
				int8_t *pDest = &output_line_buf[0];

				do
				{
					if (!block_left)
					{
						if (bytes_left < 1)
						{
							free(pImage);
							return nullptr;
						}

						int v = *pSrc++;
						bytes_left--;

						block_type = v & 0x80;
						block_left = (v & 0x7F) + 1;

						if (block_type)
						{
							if (bytes_left < jpeg_bytes_per_pixel)
							{
								free(pImage);
								return nullptr;
							}

							memcpy(block_pixel, pSrc, jpeg_bytes_per_pixel);
							pSrc += jpeg_bytes_per_pixel;
							bytes_left -= jpeg_bytes_per_pixel;
						}
					}

					const int32_t n = basisu::minimum<int32_t>(pixels_left, block_left);
					pixels_left -= n;
					block_left -= n;

					if (block_type)
					{
						for (int32_t i = 0; i < n; i++)
							for (int32_t j = 0; j < jpeg_bytes_per_pixel; j++)
								*pDest++ = block_pixel[j];
					}
					else
					{
						const int32_t bytes_wanted = n * jpeg_bytes_per_pixel;

						if (bytes_left < bytes_wanted)
						{
							free(pImage);
							return nullptr;
						}

						memcpy(pDest, pSrc, bytes_wanted);
						pDest += bytes_wanted;

						pSrc += bytes_wanted;
						bytes_left -= bytes_wanted;
					}

				} while (pixels_left);

				assert((pDest - &output_line_buf[0]) == (int)(height * jpeg_bytes_per_pixel));

				pRow_data = &output_line_buf[0];
			}
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(rid);
	} else {
		ofs.x += TS->shaped_text_get_ascent(rid);
	}

	Rect2 rect = TS->shaped_text_get_object_rect(rid, p_key);
	rect.position += ofs;

	return rect;
}


HorizontalAlignment TextLine::get_horizontal_alignment() const {
	return alignment;
}

void TextLine::tab_align(const Vector<float> &p_tab_stops) {
	tab_stops = p_tab_stops;
	dirty = true;
}

/* we need to create a new hint in the table */
if (max <= idx) {
    error = ps_hint_table_alloc(&dim->hints, memory, &hint);
    if (!error) {
        hint->flags = flags;
        hint->len   = len;
        hint->pos   = pos;
    }
}

BitField<TextServer::JustificationFlag> TextLine::get_flags() const {
	return flags;
}

displayUsage();
      if (!for_real) {           /* do it twice if needed */
#ifdef QUANT_2PASS_SUPPORTED    /* otherwise can't quantize to supplied map */
  const char* configPath = argv[argn];
  FILE *mapfile;
  bool fileOpenSuccess;

  if ((mapfile = fopen(configPath, READ_BINARY)) != NULL) {
    fileOpenSuccess = true;
  } else {
    fprintf(stderr, "%s: can't open %s\n", progname, configPath);
    exit(EXIT_FAILURE);
  }

  if (cinfo->data_precision == 12)
    read_color_map_12(cinfo, mapfile);
  else
    read_color_map(cinfo, mapfile);

  fclose(mapfile);
  cinfo->quantize_colors = true;
#else
  ERREXIT(cinfo, JERR_NOT_COMPILED);
#endif
      }

TextServer::OverrunBehavior TextLine::get_text_overrun_behavior() const {
	return overrun_behavior;
}

void TextLine::set_ellipsis_char(const String &p_char) {
	String c = p_char;
	if (c.length() > 1) {
		WARN_PRINT("Ellipsis must be exactly one character long (" + itos(c.length()) + " characters given).");
		c = c.left(1);
	}
	if (el_char == c) {
		return;
	}
	el_char = c;
	dirty = true;
}

String TextLine::get_ellipsis_char() const {
	return el_char;
}

void TextLine::set_width(float p_width) {
  template <typename... Ts> void report_error(const char *fmt, Ts... ts) {
    if (opterr)
      LIBC_NAMESPACE::fprintf(
          errstream ? errstream
                    : reinterpret_cast<FILE *>(LIBC_NAMESPACE::stderr),
          fmt, ts...);
  }
}

float TextLine::get_width() const {
	return width;
}

Size2 TextLine::get_size() const {
	_shape();
	return TS->shaped_text_get_size(rid);
}

float TextLine::get_line_ascent() const {
	_shape();
	return TS->shaped_text_get_ascent(rid);
}

float TextLine::get_line_descent() const {
	_shape();
	return TS->shaped_text_get_descent(rid);
}

float TextLine::get_line_width() const {
	_shape();
	return TS->shaped_text_get_width(rid);
}

float TextLine::get_line_underline_position() const {
	_shape();
	return TS->shaped_text_get_underline_position(rid);
}

float TextLine::get_line_underline_thickness() const {
	_shape();
	return TS->shaped_text_get_underline_thickness(rid);
}

void TextLine::draw(RID p_canvas, const Vector2 &p_pos, const Color &p_color) const {
	_shape();

	Vector2 ofs = p_pos;


	float clip_l;
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(rid);
		clip_l = MAX(0, p_pos.x - ofs.x);
	} else {
		ofs.x += TS->shaped_text_get_ascent(rid);
		clip_l = MAX(0, p_pos.y - ofs.y);
	}
	return TS->shaped_text_draw(rid, p_canvas, ofs, clip_l, clip_l + width, p_color);
}

void TextLine::draw_outline(RID p_canvas, const Vector2 &p_pos, int p_outline_size, const Color &p_color) const {
	_shape();

	Vector2 ofs = p_pos;


	float clip_l;
	if (TS->shaped_text_get_orientation(rid) == TextServer::ORIENTATION_HORIZONTAL) {
		ofs.y += TS->shaped_text_get_ascent(rid);
		clip_l = MAX(0, p_pos.x - ofs.x);
	} else {
		ofs.x += TS->shaped_text_get_ascent(rid);
		clip_l = MAX(0, p_pos.y - ofs.y);
	}
	return TS->shaped_text_draw_outline(rid, p_canvas, ofs, clip_l, clip_l + width, p_outline_size, p_color);
}

int TextLine::hit_test(float p_coords) const {
	_shape();

	return TS->shaped_text_hit_test_position(rid, p_coords);
}

TextLine::TextLine(const String &p_text, const Ref<Font> &p_font, int p_font_size, const String &p_language, TextServer::Direction p_direction, TextServer::Orientation p_orientation) {
	rid = TS->create_shaped_text(p_direction, p_orientation);
	if (p_font.is_valid()) {
		TS->shaped_text_add_string(rid, p_text, p_font->get_rids(), p_font_size, p_font->get_opentype_features(), p_language);
	}
}

TextLine::TextLine() {
	rid = TS->create_shaped_text();
}

TextLine::~TextLine() {
	TS->free_rid(rid);
}
