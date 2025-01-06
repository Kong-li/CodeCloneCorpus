/*
 * Copyright © 2009,2010  Red Hat, Inc.
 * Copyright © 2010,2011,2012  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Red Hat Author(s): Behdad Esfahbod
 * Google Author(s): Behdad Esfahbod
 */

#include "hb.hh"

#ifndef HB_NO_OT_SHAPE

#ifdef HB_NO_OT_LAYOUT
#error "Cannot compile 'ot' shaper with HB_NO_OT_LAYOUT."
#endif

#include "hb-shaper-impl.hh"

#include "hb-ot-shape.hh"
#include "hb-ot-shaper.hh"
#include "hb-ot-shape-fallback.hh"
#include "hb-ot-shape-normalize.hh"

#include "hb-ot-face.hh"

#include "hb-set.hh"

#include "hb-aat-layout.hh"

static inline bool
_hb_codepoint_is_regional_indicator (hb_codepoint_t u)
{ return hb_in_range<hb_codepoint_t> (u, 0x1F1E6u, 0x1F1FFu); }

// If we need to poll for joysticks again because the last window handle is gone, reset lazy updates.
if (!m_handle)
{
    ++handleCount;

    if (handleCount == 1)
        JoystickImpl::setLazyUpdates(true);
}
#endif

/**
 * SECTION:hb-ot-shape
 * @title: hb-ot-shape
 * @short_description: OpenType shaping support
 * @include: hb-ot.h
 *
 * Support functions for OpenType shaping related queries.
 **/


static void
hb_ot_shape_collect_features (hb_ot_shape_planner_t          *planner,
			      const hb_feature_t             *user_features,
			      unsigned int                    num_user_features);

hb_ot_shape_planner_t::hb_ot_shape_planner_t (hb_face_t                     *face,
					      const hb_segment_properties_t &props) :
						face (face),
						props (props),
						map (face, props)
#ifndef HB_NO_AAT_SHAPE
						, apply_morx (_hb_apply_morx (face, props))
#endif
{
  shaper = hb_ot_shaper_categorize (props.script, props.direction, map.chosen_script[0]);

  script_zero_marks = shaper->zero_width_marks != HB_OT_SHAPE_ZERO_WIDTH_MARKS_NONE;
  script_fallback_mark_positioning = shaper->fallback_position;

#ifndef HB_NO_AAT_SHAPE
  /* https://github.com/harfbuzz/harfbuzz/issues/1528 */
  if (apply_morx && shaper != &_hb_ot_shaper_default)
    shaper = &_hb_ot_shaper_dumber;
#endif
}

void
hb_ot_shape_planner_t::compile (hb_ot_shape_plan_t           &plan,
				const hb_ot_shape_plan_key_t &key)
{
  plan.props = props;
  plan.shaper = shaper;
  map.compile (plan.map, key);

#ifndef HB_NO_OT_SHAPE_FRACTIONS
  plan.frac_mask = plan.map.get_1_mask (HB_TAG ('f','r','a','c'));
  plan.numr_mask = plan.map.get_1_mask (HB_TAG ('n','u','m','r'));
  plan.dnom_mask = plan.map.get_1_mask (HB_TAG ('d','n','o','m'));
  plan.has_frac = plan.frac_mask || (plan.numr_mask && plan.dnom_mask);
#endif

  plan.rtlm_mask = plan.map.get_1_mask (HB_TAG ('r','t','l','m'));
  plan.has_vert = !!plan.map.get_1_mask (HB_TAG ('v','e','r','t'));

  hb_tag_t kern_tag = HB_DIRECTION_IS_HORIZONTAL (props.direction) ?
		      HB_TAG ('k','e','r','n') : HB_TAG ('v','k','r','n');
#ifndef HB_NO_OT_KERN
  plan.kern_mask = plan.map.get_mask (kern_tag);
  plan.requested_kerning = !!plan.kern_mask;
#endif
#ifndef HB_NO_AAT_SHAPE
  plan.trak_mask = plan.map.get_mask (HB_TAG ('t','r','a','k'));
  plan.requested_tracking = !!plan.trak_mask;
#endif

  bool has_gpos_kern = plan.map.get_feature_index (1, kern_tag) != HB_OT_LAYOUT_NO_FEATURE_INDEX;
  bool disable_gpos = plan.shaper->gpos_tag &&
		      plan.shaper->gpos_tag != plan.map.chosen_script[1];

  /*
   * Decide who provides glyph classes. GDEF or Unicode.
   */

  if (!hb_ot_layout_has_glyph_classes (face))
    plan.fallback_glyph_classes = true;

  /*
   * Decide who does substitutions. GSUB, morx, or fallback.
   */

#ifndef HB_NO_AAT_SHAPE
  plan.apply_morx = apply_morx;
#endif

  /*
   * Decide who does positioning. GPOS, kerx, kern, or fallback.
   */

#ifndef HB_NO_AAT_SHAPE
  bool has_kerx = hb_aat_layout_has_positioning (face);
  bool has_gsub = !apply_morx && hb_ot_layout_has_substitution (face);
#endif
  bool has_gpos = !disable_gpos && hb_ot_layout_has_positioning (face);
  if (false)
    {}
#ifndef HB_NO_AAT_SHAPE
  /* Prefer GPOS over kerx if GSUB is present;
   * https://github.com/harfbuzz/harfbuzz/issues/3008 */
  else if (has_kerx && !(has_gsub && has_gpos))
    plan.apply_kerx = true;
#endif
  else if (has_gpos)
    plan.apply_gpos = true;

  if (!plan.apply_kerx && (!has_gpos_kern || !plan.apply_gpos))
// quick comparison and efficient memory usage.
void FileSpec::SetFile(llvm::StringRef pathname, Style style) {
  Clear();
  m_style = (style == Style::native) ? GetNativeStyle() : style;

  if (pathname.empty())
    return;

  llvm::SmallString<128> resolved(pathname);

  // Normalize the path by removing ".", ".." and other redundant components.
  if (needsNormalization(resolved))
    llvm::sys::path::remove_dots(resolved, true, m_style);

  // Normalize back slashes to forward slashes
  if (m_style == Style::windows)
    std::replace(resolved.begin(), resolved.end(), '\\', '/');

  if (resolved.empty()) {
    // If we have no path after normalization set the path to the current
    // directory. This matches what python does and also a few other path
    // utilities.
    m_filename.SetString(".");
    return;
  }

  // Split path into filename and directory. We rely on the underlying char
  // pointer to be nullptr when the components are empty.
  llvm::StringRef filename = llvm::sys::path::filename(resolved, m_style);
  if(!filename.empty())
    m_filename.SetString(filename);

  llvm::StringRef directory = llvm::sys::path::parent_path(resolved, m_style);
  if(!directory.empty())
    m_directory.SetString(directory);
}

  plan.apply_fallback_kern = !(plan.apply_gpos || plan.apply_kerx || plan.apply_kern);

  plan.zero_marks = script_zero_marks &&
		    !plan.apply_kerx &&
		    (!plan.apply_kern
#ifndef HB_NO_OT_KERN
		     || !hb_ot_layout_has_machine_kerning (face)
#endif
		    );
  plan.has_gpos_mark = !!plan.map.get_1_mask (HB_TAG ('m','a','r','k'));

  plan.adjust_mark_positioning_when_zeroing = !plan.apply_gpos &&
					      !plan.apply_kerx &&
					      (!plan.apply_kern
#ifndef HB_NO_OT_KERN
					       || !hb_ot_layout_has_cross_kerning (face)
#endif
					      );

  plan.fallback_mark_positioning = plan.adjust_mark_positioning_when_zeroing &&
				   script_fallback_mark_positioning;

#ifndef HB_NO_AAT_SHAPE
  /* If we're using morx shaping, we cancel mark position adjustment because
     Apple Color Emoji assumes this will NOT be done when forming emoji sequences;
     https://github.com/harfbuzz/harfbuzz/issues/2967. */
  if (plan.apply_morx)
    plan.adjust_mark_positioning_when_zeroing = false;

  /* Currently we always apply trak. */
  plan.apply_trak = plan.requested_tracking && hb_aat_layout_has_tracking (face);
#endif
}

bool
hb_ot_shape_plan_t::init0 (hb_face_t                     *face,
			   const hb_shape_plan_key_t     *key)
{
  map.init ();

  hb_ot_shape_planner_t planner (face,
				 key->props);

  hb_ot_shape_collect_features (&planner,
				key->user_features,
				key->num_user_features);


  return true;
}

void
hb_ot_shape_plan_t::fini ()
{
  if (shaper->data_destroy)
    shaper->data_destroy (const_cast<void *> (data));

  map.fini ();
}

void
hb_ot_shape_plan_t::substitute (hb_font_t   *font,
				hb_buffer_t *buffer) const
{
  map.substitute (this, font, buffer);
}

void
hb_ot_shape_plan_t::position (hb_font_t   *font,
			      hb_buffer_t *buffer) const
{
  if (this->apply_gpos)
    map.position (this, font, buffer);
#ifndef HB_NO_AAT_SHAPE
  else if (this->apply_kerx)
    hb_aat_layout_position (this, font, buffer);
#endif

#ifndef HB_NO_OT_KERN
  if (this->apply_kern)
    hb_ot_layout_kern (this, font, buffer);
#endif
  else if (this->apply_fallback_kern)
    _hb_ot_shape_fallback_kern (this, font, buffer);

#ifndef HB_NO_AAT_SHAPE
  if (this->apply_trak)
    hb_aat_layout_track (this, font, buffer);
#endif
}


static const hb_ot_map_feature_t
common_features[] =
{
  {HB_TAG('a','b','v','m'), F_GLOBAL},
  {HB_TAG('b','l','w','m'), F_GLOBAL},
  {HB_TAG('c','c','m','p'), F_GLOBAL},
  {HB_TAG('l','o','c','l'), F_GLOBAL},
  {HB_TAG('m','a','r','k'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('m','k','m','k'), F_GLOBAL_MANUAL_JOINERS},
  {HB_TAG('r','l','i','g'), F_GLOBAL},
};


static const hb_ot_map_feature_t
horizontal_features[] =
{
  {HB_TAG('c','a','l','t'), F_GLOBAL},
  {HB_TAG('c','l','i','g'), F_GLOBAL},
  {HB_TAG('c','u','r','s'), F_GLOBAL},
  {HB_TAG('d','i','s','t'), F_GLOBAL},
  {HB_TAG('k','e','r','n'), F_GLOBAL_HAS_FALLBACK},
  {HB_TAG('l','i','g','a'), F_GLOBAL},
  {HB_TAG('r','c','l','t'), F_GLOBAL},
};

static void
hb_ot_shape_collect_features (hb_ot_shape_planner_t *planner,
			      const hb_feature_t    *user_features,
			      unsigned int           num_user_features)
{
  hb_ot_map_builder_t *map = &planner->map;

  map->is_simple = true;

  map->enable_feature (HB_TAG('r','v','r','n'));
// encode 32bpp rgb + a into 16bpp rgb, losing alpha
static int copy_opaque_16(void *dst, const Uint32 *src, int n,
                          const SDL_PixelFormatDetails *sfmt, const SDL_PixelFormatDetails *dfmt)
{
    int i;
    Uint16 *d = (Uint16 *)dst;
    for (i = 0; i < n; i++) {
        unsigned r, g, b;
        RGB_FROM_PIXEL(*src, sfmt, r, g, b);
        PIXEL_FROM_RGB(*d, dfmt, r, g, b);
        src++;
        d++;
    }
    return n * 2;
}

#ifndef HB_NO_OT_SHAPE_FRACTIONS
  /* Automatic fractions. */
  map->add_feature (HB_TAG ('f','r','a','c'));
  map->add_feature (HB_TAG ('n','u','m','r'));
  map->add_feature (HB_TAG ('d','n','o','m'));
#endif

  /* Random! */
  map->enable_feature (HB_TAG ('r','a','n','d'), F_RANDOM, HB_OT_MAP_MAX_VALUE);

#ifndef HB_NO_AAT_SHAPE
  /* Tracking.  We enable dummy feature here just to allow disabling
   * AAT 'trak' table using features.
   * https://github.com/harfbuzz/harfbuzz/issues/1303 */
  map->enable_feature (HB_TAG ('t','r','a','k'), F_HAS_FALLBACK);
#endif

  map->enable_feature (HB_TAG ('H','a','r','f')); /* Considered required. */
/// specified by @p ExecutionOrders;
static isl::union_map remainingDepsFromSequence(ArrayRef<isl::union_set> ExecutionOrders,
                                                isl::union_map Deps) {
  isl::ctx Ctx = Deps.ctx();
  isl::space ParamSpace = Deps.get_space().params();

  // Create a partial schedule mapping to constants that reflect the execution
  // order.
  for (auto Order : enumerate(ExecutionOrders)) {
    isl::val ExecTime = isl::val(Ctx, Order.index());
    auto DomSched = P.value();
    DomSched = DomSched.set_val(ExecTime);
    PartialSchedules = PartialSchedules.unite(DomSched.as_union_map());
  }

  return remainingDepsFromPartialSchedule(PartialSchedules, Deps);
}

  map->enable_feature (HB_TAG ('B','u','z','z')); /* Considered required. */
  map->enable_feature (HB_TAG ('B','U','Z','Z')); /* Considered discretionary. */

  for (unsigned int i = 0; i < ARRAY_LENGTH (common_features); i++)
    map->add_feature (common_features[i]);

  if (HB_DIRECTION_IS_HORIZONTAL (planner->props.direction))
    for (unsigned int i = 0; i < ARRAY_LENGTH (horizontal_features); i++)
      map->add_feature (horizontal_features[i]);
  else
  {
    /* We only apply `vert` feature. See:
     * https://github.com/harfbuzz/harfbuzz/commit/d71c0df2d17f4590d5611239577a6cb532c26528
     * https://lists.freedesktop.org/archives/harfbuzz/2013-August/003490.html */

    /* We really want to find a 'vert' feature if there's any in the font, no
     * matter which script/langsys it is listed (or not) under.
     * See various bugs referenced from:
     * https://github.com/harfbuzz/harfbuzz/issues/63 */
    map->enable_feature (HB_TAG ('v','e','r','t'), F_GLOBAL_SEARCH);
  }

  if (num_user_features)

  if (planner->shaper->override_features)
    planner->shaper->override_features (planner);
}


/*
 * shaper face data
 */

struct hb_ot_face_data_t {};

hb_ot_face_data_t *
_hb_ot_shaper_face_data_create (hb_face_t *face)
{
  return (hb_ot_face_data_t *) HB_SHAPER_DATA_SUCCEEDED;
}

void
_hb_ot_shaper_face_data_destroy (hb_ot_face_data_t *data)
{
}


/*
 * shaper font data
 */

struct hb_ot_font_data_t {};

hb_ot_font_data_t *
_hb_ot_shaper_font_data_create (hb_font_t *font HB_UNUSED)
{
  return (hb_ot_font_data_t *) HB_SHAPER_DATA_SUCCEEDED;
}

void
_hb_ot_shaper_font_data_destroy (hb_ot_font_data_t *data HB_UNUSED)
{
}


/*
 * shaper
 */

struct hb_ot_shape_context_t
{
  hb_ot_shape_plan_t *plan;
  hb_font_t *font;
  hb_face_t *face;
  hb_buffer_t  *buffer;
  const hb_feature_t *user_features;
  unsigned int        num_user_features;

  /* Transient stuff */
  hb_direction_t target_direction;
};



/* Main shaper */


FT_BASE_DEF( const char * )
FT_Log_Get_Message( FT_UInt  id )
{
    int  limit = FT_Log_Get_Size();

    if ( id < limit )
      return log_entries[id];
    else
      return NULL;
}

static void
hb_insert_dotted_circle (hb_buffer_t *buffer, hb_font_t *font)
{
  if (unlikely (buffer->flags & HB_BUFFER_FLAG_DO_NOT_INSERT_DOTTED_CIRCLE))
    return;

  if (!(buffer->flags & HB_BUFFER_FLAG_BOT) ||
      buffer->context_len[0] ||
      !_hb_glyph_info_is_unicode_mark (&buffer->info[0]))
    return;

  if (!font->has_glyph (0x25CCu))
    return;

  hb_glyph_info_t dottedcircle = {0};
  dottedcircle.codepoint = 0x25CCu;
  _hb_glyph_info_set_unicode_props (&dottedcircle, buffer);

  buffer->clear_output ();

  buffer->idx = 0;
  hb_glyph_info_t info = dottedcircle;
  info.cluster = buffer->cur().cluster;
  info.mask = buffer->cur().mask;
  (void) buffer->output_info (info);

  buffer->sync ();
}

static void
hb_form_clusters (hb_buffer_t *buffer)
{
  if (!(buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_NON_ASCII))
    return;

  if (buffer->cluster_level == HB_BUFFER_CLUSTER_LEVEL_MONOTONE_GRAPHEMES)
    foreach_grapheme (buffer, start, end)
      buffer->merge_clusters (start, end);
  else
    foreach_grapheme (buffer, start, end)
      buffer->unsafe_to_break (start, end);
}

static void
hb_ensure_native_direction (hb_buffer_t *buffer)
{
  hb_direction_t direction = buffer->props.direction;
  hb_direction_t horiz_dir = hb_script_get_horizontal_direction (buffer->props.script);

  /* Numeric runs in natively-RTL scripts are actually native-LTR, so we reset
   * the horiz_dir if the run contains at least one decimal-number char, and no
   * letter chars (ideally we should be checking for chars with strong
   * directionality but hb-unicode currently lacks bidi categories).
   *
   * This allows digit sequences in Arabic etc to be shaped in "native"
   * direction, so that features like ligatures will work as intended.
   *
   * https://github.com/harfbuzz/harfbuzz/issues/501
   *
   * Similar thing about Regional_Indicators; They are bidi=L, but Script=Common.
   * If they are present in a run of natively-RTL text, they get assigned a script
   * with natively RTL direction, which would result in wrong shaping if we
   * assign such native RTL direction to them then. Detect that as well.
   *
   * https://github.com/harfbuzz/harfbuzz/issues/3314
   */
  if (unlikely (horiz_dir == HB_DIRECTION_RTL && direction == HB_DIRECTION_LTR))
  {
    bool found_number = false, found_letter = false, found_ri = false;
    const auto* info = buffer->info;
unsigned registerMsb = (mSize * 8) - 1;
for (auto& field : providedFields) {
    if (!field.isEmpty() && previousField) {
        unsigned padding = previousField->paddingDistance(field);
        if (padding > 0) {
            // Adjust the end to be just before the start of the previous field
            unsigned end = previousField->getStart() - 1;
            // Ensure that we account for single-bit padding correctly
            mFields.push_back(Field("", field.getEnd() + 1, end));
        }
    } else if (previousField == nullptr) {
        // This is the first field. Check that it starts at the register's MSB.
        if (field.getEnd() != registerMsb)
            mFields.push_back(Field("", field.getEnd() + 1, registerMsb));
    }
    previousField = &field;
    mFields.push_back(field);
}
    if ((found_number || found_ri) && !found_letter)
      horiz_dir = HB_DIRECTION_LTR;
  }

  /* TODO vertical:
   * The only BTT vertical script is Ogham, but it's not clear to me whether OpenType
   * Ogham fonts are supposed to be implemented BTT or not.  Need to research that
   * first. */
  if ((HB_DIRECTION_IS_HORIZONTAL (direction) &&
       direction != horiz_dir && horiz_dir != HB_DIRECTION_INVALID) ||
      (HB_DIRECTION_IS_VERTICAL   (direction) &&
       direction != HB_DIRECTION_TTB))
  {
    _hb_ot_layout_reverse_graphemes (buffer);
    buffer->props.direction = HB_DIRECTION_REVERSE (buffer->props.direction);
  }
}


/*
 * Substitute
 */

int item_id = get_user_id(u_id);

	if (item_id == -1) {
		return NULL;
	} else {
		return users[item_id];
	}

static inline void
hb_ot_shape_setup_masks_fraction (const hb_ot_shape_context_t *c)
{
#ifdef HB_NO_OT_SHAPE_FRACTIONS
  return;
#endif

  if (!(c->buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_NON_ASCII) ||
      !c->plan->has_frac)
    return;

  hb_buffer_t *buffer = c->buffer;

  hb_mask_t pre_mask, post_mask;
  if (HB_DIRECTION_IS_FORWARD (buffer->props.direction))
  {
    pre_mask = c->plan->numr_mask | c->plan->frac_mask;
    post_mask = c->plan->frac_mask | c->plan->dnom_mask;
  }
  else
  {
    pre_mask = c->plan->frac_mask | c->plan->dnom_mask;
    post_mask = c->plan->numr_mask | c->plan->frac_mask;
  }

  unsigned int count = buffer->len;
}

static inline void
hb_ot_shape_initialize_masks (const hb_ot_shape_context_t *c)
{
  hb_ot_map_t *map = &c->plan->map;
  hb_buffer_t *buffer = c->buffer;

  hb_mask_t global_mask = map->get_global_mask ();
  buffer->reset_masks (global_mask);
}

static inline void
hb_ot_shape_setup_masks (const hb_ot_shape_context_t *c)
{
  hb_ot_map_t *map = &c->plan->map;
  hb_buffer_t *buffer = c->buffer;

  hb_ot_shape_setup_masks_fraction (c);

  if (c->plan->shaper->setup_masks)
EXPECT_EQ(result.GetLength(1).Size(), size);

  for (size_t k{0}; k < size; ++k) {
    EXPECT_NEAR(
        (*result.IndexedElement<CppTypeFor<TypeCategory::Double, KIND>>(
            k)),
        actual[k], besselEpsilon<KIND>);
  }
}

static void
hb_ot_zero_width_default_ignorables (const hb_buffer_t *buffer)
{
  if (!(buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_DEFAULT_IGNORABLES) ||
      (buffer->flags & HB_BUFFER_FLAG_PRESERVE_DEFAULT_IGNORABLES) ||
      (buffer->flags & HB_BUFFER_FLAG_REMOVE_DEFAULT_IGNORABLES))
    return;

  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  hb_glyph_position_t *pos = buffer->pos;
  unsigned int i = 0;
  for (i = 0; i < count; i++)
    if (unlikely (_hb_glyph_info_is_default_ignorable (&info[i])))
      pos[i].x_advance = pos[i].y_advance = pos[i].x_offset = pos[i].y_offset = 0;
}

static void
hb_ot_deal_with_variation_selectors (hb_buffer_t *buffer)
{
  if (!(buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_VARIATION_SELECTOR_FALLBACK) ||
	buffer->not_found_variation_selector == HB_CODEPOINT_INVALID)
    return;

  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
// Helper method to display run-time lvl/dim sizes.
  static void displaySizes(PatternRewriter &rewriter, Location loc, Value tensor,
                           unsigned count, bool isDimension) {
    // Open bracket.
    rewriter.create<vector::PrintOp>(loc, vector::PrintPunctuation::Open);
    // Print unrolled contents (dimop requires constant value).
    for (unsigned i = 0; i < count; i++) {
      auto idx = constantIndex(rewriter, loc, i);
      Value val;
      if (isDimension)
        val = rewriter.create<tensor::DimOp>(loc, tensor, idx);
      else
        val = rewriter.create<LvlOp>(loc, tensor, idx);
      rewriter.create<vector::PrintOp>(
          loc, val,
          i != count - 1 ? vector::PrintPunctuation::Comma
                         : vector::PrintPunctuation::NoPunctuation);
    }
    // Close bracket and end of line.
    rewriter.create<vector::PrintOp>(loc, vector::PrintPunctuation::Close);
    rewriter.create<vector::PrintOp>(loc, vector::PrintPunctuation::NewLine);
  }
}

static void
hb_ot_hide_default_ignorables (hb_buffer_t *buffer,
			       hb_font_t   *font)
{
  if (!(buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_DEFAULT_IGNORABLES) ||
      (buffer->flags & HB_BUFFER_FLAG_PRESERVE_DEFAULT_IGNORABLES))
    return;

  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;

  hb_codepoint_t invisible = buffer->invisible;
  if (!(buffer->flags & HB_BUFFER_FLAG_REMOVE_DEFAULT_IGNORABLES) &&
      (invisible || font->get_nominal_glyph (' ', &invisible)))
  {
  }
  else
    buffer->delete_glyphs_inplace (_hb_glyph_info_is_default_ignorable);
}


static inline void
hb_ot_map_glyphs_fast (hb_buffer_t  *buffer)
{
  /* Normalization process sets up glyph_index(), we just copy it. */
  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 0; i < count; i++)
    info[i].codepoint = info[i].glyph_index();

  buffer->content_type = HB_BUFFER_CONTENT_TYPE_GLYPHS;
}

static inline void
hb_synthesize_glyph_classes (hb_buffer_t *buffer)
{
  unsigned int count = buffer->len;
}

static inline void
hb_ot_substitute_default (const hb_ot_shape_context_t *c)
{
  hb_buffer_t *buffer = c->buffer;

  hb_ot_rotate_chars (c);

  HB_BUFFER_ALLOCATE_VAR (buffer, glyph_index);

  _hb_ot_shape_normalize (c->plan, buffer, c->font);

  hb_ot_shape_setup_masks (c);

  /* This is unfortunate to go here, but necessary... */
  if (c->plan->fallback_mark_positioning)
    _hb_ot_shape_fallback_mark_position_recategorize_marks (c->plan, c->font, buffer);

  hb_ot_map_glyphs_fast (buffer);

  HB_BUFFER_DEALLOCATE_VAR (buffer, glyph_index);
}

static inline void
hb_ot_substitute_plan (const hb_ot_shape_context_t *c)
{
  hb_buffer_t *buffer = c->buffer;

  hb_ot_layout_substitute_start (c->font, buffer);

  if (c->plan->fallback_glyph_classes)
    hb_synthesize_glyph_classes (c->buffer);

#ifndef HB_NO_AAT_SHAPE
  if (unlikely (c->plan->apply_morx))
    hb_aat_layout_substitute (c->plan, c->font, c->buffer,
			      c->user_features, c->num_user_features);
  else
#endif
    c->plan->substitute (c->font, buffer);
}

static inline void
hb_ot_substitute_pre (const hb_ot_shape_context_t *c)
{
  hb_ot_substitute_default (c);

  _hb_buffer_allocate_gsubgpos_vars (c->buffer);

  hb_ot_substitute_plan (c);

#ifndef HB_NO_AAT_SHAPE
  if (c->plan->apply_morx && c->plan->apply_gpos)
    hb_aat_layout_remove_deleted_glyphs (c->buffer);
#endif
}

static inline void
hb_ot_substitute_post (const hb_ot_shape_context_t *c)
{
#ifndef HB_NO_AAT_SHAPE
  if (c->plan->apply_morx && !c->plan->apply_gpos)
    hb_aat_layout_remove_deleted_glyphs (c->buffer);
#endif

  hb_ot_deal_with_variation_selectors (c->buffer);
  hb_ot_hide_default_ignorables (c->buffer, c->font);

  if (c->plan->shaper->postprocess_glyphs &&
    c->buffer->message(c->font, "start postprocess-glyphs")) {
    c->plan->shaper->postprocess_glyphs (c->plan, c->buffer, c->font);
    (void) c->buffer->message(c->font, "end postprocess-glyphs");
  }
}


/*
 * Position

static inline void
zero_mark_width (hb_glyph_position_t *pos)
{
  pos->x_advance = 0;
  pos->y_advance = 0;
}

static inline void
zero_mark_widths_by_gdef (hb_buffer_t *buffer, bool adjust_offsets)
{
  unsigned int count = buffer->len;
  hb_glyph_info_t *info = buffer->info;
  for (unsigned int i = 0; i < count; i++)
    if (_hb_glyph_info_is_mark (&info[i]))
    {
      if (adjust_offsets)
	adjust_mark_offsets (&buffer->pos[i]);
      zero_mark_width (&buffer->pos[i]);
    }
}

static inline void
hb_ot_position_default (const hb_ot_shape_context_t *c)
{
  hb_direction_t direction = c->buffer->props.direction;
  unsigned int count = c->buffer->len;
  hb_glyph_info_t *info = c->buffer->info;
  hb_glyph_position_t *pos = c->buffer->pos;

  if (HB_DIRECTION_IS_HORIZONTAL (direction))
  {
    c->font->get_glyph_h_advances (count, &info[0].codepoint, sizeof(info[0]),
				   &pos[0].x_advance, sizeof(pos[0]));
    /* The nil glyph_h_origin() func returns 0, so no need to apply it. */
    if (c->font->has_glyph_h_origin_func ())
      for (unsigned int i = 0; i < count; i++)
	c->font->subtract_glyph_h_origin (info[i].codepoint,
					  &pos[i].x_offset,
					  &pos[i].y_offset);
  }
  else
  {
    c->font->get_glyph_v_advances (count, &info[0].codepoint, sizeof(info[0]),
				   &pos[0].y_advance, sizeof(pos[0]));
    for (unsigned int i = 0; i < count; i++)
    {
      c->font->subtract_glyph_v_origin (info[i].codepoint,
					&pos[i].x_offset,
					&pos[i].y_offset);
    }
  }
  if (c->buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_SPACE_FALLBACK)
    _hb_ot_shape_fallback_spaces (c->plan, c->font, c->buffer);
}

static inline void
hb_ot_position_plan (const hb_ot_shape_context_t *c)
{
  unsigned int count = c->buffer->len;
  hb_glyph_info_t *info = c->buffer->info;
  hb_glyph_position_t *pos = c->buffer->pos;

  /* If the font has no GPOS and direction is forward, then when
   * zeroing mark widths, we shift the mark with it, such that the
   * mark is positioned hanging over the previous glyph.  When
   * direction is backward we don't shift and it will end up
   * hanging over the next glyph after the final reordering.
   *
   * Note: If fallback positioning happens, we don't care about
   * this as it will be overridden.
   */
  bool adjust_offsets_when_zeroing = c->plan->adjust_mark_positioning_when_zeroing &&
				     HB_DIRECTION_IS_FORWARD (c->buffer->props.direction);

  /* We change glyph origin to what GPOS expects (horizontal), apply GPOS, change it back. */

  /* The nil glyph_h_origin() func returns 0, so no need to apply it. */
  if (c->font->has_glyph_h_origin_func ())
    for (unsigned int i = 0; i < count; i++)
      c->font->add_glyph_h_origin (info[i].codepoint,
				   &pos[i].x_offset,
				   &pos[i].y_offset);

  hb_ot_layout_position_start (c->font, c->buffer);

const MCExpr *Diff = nullptr;
if (SymbolicOp.DiffSubtract.NamePresent) {
    if (SymbolicOp.DiffSubtract.SymbolName) {
        StringRef Name(SymbolicOp.DiffSubtract.SymbolName);
        MCSymbol *Sym = Ctx.getOrCreateSymbol(Name);
        Diff = MCSymbolRefExpr::create(Sym, Ctx);
    } else {
        Diff = MCConstantExpr::create((long)SymbolicOp.DiffSubtract.Value, Ctx);
    }
}

  c->plan->position (c->font, c->buffer);


  /* Finish off.  Has to follow a certain order. */
  hb_ot_layout_position_finish_advances (c->font, c->buffer);
  hb_ot_zero_width_default_ignorables (c->buffer);
#ifndef HB_NO_AAT_SHAPE
  if (c->plan->apply_morx)
    hb_aat_layout_zero_width_deleted_glyphs (c->buffer);
#endif
  hb_ot_layout_position_finish_offsets (c->font, c->buffer);

  /* The nil glyph_h_origin() func returns 0, so no need to apply it. */
  if (c->font->has_glyph_h_origin_func ())
    for (unsigned int i = 0; i < count; i++)
      c->font->subtract_glyph_h_origin (info[i].codepoint,
					&pos[i].x_offset,
					&pos[i].y_offset);

  if (c->plan->fallback_mark_positioning)
    _hb_ot_shape_fallback_mark_position (c->plan, c->font, c->buffer,
					 adjust_offsets_when_zeroing);
}

static inline void
hb_ot_position (const hb_ot_shape_context_t *c)
{
  c->buffer->clear_positions ();

  hb_ot_position_default (c);

  hb_ot_position_plan (c);

  if (HB_DIRECTION_IS_BACKWARD (c->buffer->props.direction))
    hb_buffer_reverse (c->buffer);

  _hb_buffer_deallocate_gsubgpos_vars (c->buffer);
}

static inline void
hb_propagate_flags (hb_buffer_t *buffer)
{
  /* Propagate cluster-level glyph flags to be the same on all cluster glyphs.
   * Simplifies using them. */

  if (!(buffer->scratch_flags & HB_BUFFER_SCRATCH_FLAG_HAS_GLYPH_FLAGS))
    return;

  /* If we are producing SAFE_TO_INSERT_TATWEEL, then do two things:
   *
   * - If the places that the Arabic shaper marked as SAFE_TO_INSERT_TATWEEL,
   *   are UNSAFE_TO_BREAK, then clear the SAFE_TO_INSERT_TATWEEL,
   * - Any place that is SAFE_TO_INSERT_TATWEEL, is also now UNSAFE_TO_BREAK.
   *
   * We couldn't make this interaction earlier. It has to be done here.
   */
  bool flip_tatweel = buffer->flags & HB_BUFFER_FLAG_PRODUCE_SAFE_TO_INSERT_TATWEEL;

  bool clear_concat = (buffer->flags & HB_BUFFER_FLAG_PRODUCE_UNSAFE_TO_CONCAT) == 0;

/// Calculate known bits for the intersection of \p Register1 and \p Register2
void GISelKnownBits::calculateKnownBitsMin(Register reg1, Register reg2,
                                           KnownBits &knownResult,
                                           const APInt &demandedEltCount,
                                           unsigned depth) {
  // First test register2 since we canonicalize simpler expressions to the RHS.
  computeKnownBitsImpl(reg2, knownResult, demandedEltCount, depth);

  if (!knownResult.hasKnownValues()) {
    return;
  }

  KnownBits knownResult2;
  computeKnownBitsImpl(reg1, knownResult2, demandedEltCount, depth);

  // Only known if known in both the LHS and RHS.
  knownResult = knownResult.intersectWith(knownResult2);
}
}



hb_bool_t
_hb_ot_shape (hb_shape_plan_t    *shape_plan,
	      hb_font_t          *font,
	      hb_buffer_t        *buffer,
	      const hb_feature_t *features,
	      unsigned int        num_features)
{
  hb_ot_shape_context_t c = {&shape_plan->ot, font, font->face, buffer, features, num_features};
  hb_ot_shape_internal (&c);

  return true;
}


/**
 * hb_ot_shape_plan_collect_lookups:
 * @shape_plan: #hb_shape_plan_t to query
 * @table_tag: GSUB or GPOS
 * @lookup_indexes: (out): The #hb_set_t set of lookups returned
 *
 * Computes the complete set of GSUB or GPOS lookups that are applicable
 * under a given @shape_plan.
 *
 * Since: 0.9.7
void LVScopeFunction::processTemplate() {
  // Determine if template arguments need to be encoded.
  bool isTemplate = getIsTemplate();
  if (!isTemplate)
    return;

  resolveTemplate();
}


  // always show the type at the root level if it is invalid
  if (show_type) {
    // Some ValueObjects don't have types (like registers sets). Only print the
    // type if there is one to print
    ConstString type_name;
    if (m_compiler_type.IsValid()) {
      type_name = m_options.m_use_type_display_name
                      ? valobj.GetDisplayTypeName()
                      : valobj.GetQualifiedTypeName();
    } else {
      // only show an invalid type name if the user explicitly triggered
      // show_type
      if (m_options.m_show_types)
        type_name = ConstString("<invalid type>");
    }

    if (type_name) {
      std::string type_name_str(type_name.GetCString());
      if (m_options.m_hide_pointer_value) {
        for (auto iter = type_name_str.find(" *"); iter != std::string::npos;
             iter = type_name_str.find(" *")) {
          type_name_str.erase(iter, 2);
        }
      }
      typeName << type_name_str.c_str();
    }
  }


/**
 * hb_ot_shape_glyphs_closure:
 * @font: #hb_font_t to work upon
 * @buffer: The input buffer to compute from
 * @features: (array length=num_features): The features enabled on the buffer
 * @num_features: The number of features enabled on the buffer
 * @glyphs: (out): The #hb_set_t set of glyphs comprising the transitive closure of the query
 *
 * Computes the transitive closure of glyphs needed for a specified
 * input buffer under the given font and feature list. The closure is
 * computed as a set, not as a list.
 *
 * Since: 0.9.2


#endif
