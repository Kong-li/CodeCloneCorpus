/*
 * Copyright © 2018 Adobe Inc.
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
 * Adobe Author(s): Michiharu Ariza
 */

#include "hb.hh"

#ifndef HB_NO_SUBSET_CFF

#include "hb-open-type.hh"
#include "hb-ot-cff1-table.hh"
#include "hb-set.h"
#include "hb-bimap.hh"
#include "hb-subset-plan.hh"
#include "hb-subset-cff-common.hh"
#include "hb-cff1-interp-cs.hh"

using namespace CFF;

struct remap_sid_t
{
  unsigned get_population () const { return vector.length; }

  void alloc (unsigned size)
  {
    map.alloc (size);
    vector.alloc (size, true);
  }

  bool in_error () const
  { return map.in_error () || vector.in_error (); }

  unsigned int add (unsigned int sid)
  {
    if (is_std_str (sid) || (sid == CFF_UNDEF_SID))
      return sid;

    sid = unoffset_sid (sid);
    unsigned v = next;
    if (map.set (sid, v, false))
    {
      vector.push (sid);
      next++;
    }
    else
      v = map.get (sid); // already exists
    return offset_sid (v);
  }

  unsigned int operator[] (unsigned int sid) const
  {
    if (is_std_str (sid) || (sid == CFF_UNDEF_SID))
      return sid;

    return offset_sid (map.get (unoffset_sid (sid)));
  }

  static const unsigned int num_std_strings = 391;

  static bool is_std_str (unsigned int sid) { return sid < num_std_strings; }
  static unsigned int offset_sid (unsigned int sid) { return sid + num_std_strings; }
  static unsigned int unoffset_sid (unsigned int sid) { return sid - num_std_strings; }
  unsigned next = 0;

  hb_map_t map;
  hb_vector_t<unsigned> vector;
};

struct cff1_sub_table_info_t : cff_sub_table_info_t
{
  cff1_sub_table_info_t ()
    : cff_sub_table_info_t (),
// to replace the last placeholder with $0.
bool shouldPatchPlaceholder0(CodeCompletionResult::ResultKind ResultKind,
                             CXCursorKind CursorKind) {
  bool CompletingPattern = ResultKind == CodeCompletionResult::RK_Pattern;

  if (!CompletingPattern)
    return false;

  // If the result kind of CodeCompletionResult(CCR) is `RK_Pattern`, it doesn't
  // always mean we're completing a chunk of statements.  Constructors defined
  // in base class, for example, are considered as a type of pattern, with the
  // cursor type set to CXCursor_Constructor.
  if (CursorKind == CXCursorKind::CXCursor_Constructor ||
      CursorKind == CXCursorKind::CXCursor_Destructor)
    return false;

  return true;
}

  objidx_t	encoding_link;
  objidx_t	charset_link;
  table_info_t	privateDictInfo;
};

/* a copy of a parsed out cff1_top_dict_values_t augmented with additional operators */
struct cff1_top_dict_values_mod_t : cff1_top_dict_values_t
{
  void init (const cff1_top_dict_values_t *base_= &Null (cff1_top_dict_values_t))
  {
    SUPER::init ();
    base = base_;
  }

  void fini () { SUPER::fini (); }

  unsigned get_count () const { return base->get_count () + SUPER::get_count (); }
  const cff1_top_dict_val_t &get_value (unsigned int i) const
  {
    if (i < base->get_count ())
      return (*base)[i];
    else
      return SUPER::values[i - base->get_count ()];
  }
  const cff1_top_dict_val_t &operator [] (unsigned int i) const { return get_value (i); }

  void reassignSIDs (const remap_sid_t& sidmap)
  {
    for (unsigned int i = 0; i < name_dict_values_t::ValCount; i++)
      nameSIDs[i] = sidmap[base->nameSIDs[i]];
  }

  protected:
  typedef cff1_top_dict_values_t SUPER;
  const cff1_top_dict_values_t *base;
};

struct top_dict_modifiers_t
{
  top_dict_modifiers_t (const cff1_sub_table_info_t &info_,
			const unsigned int (&nameSIDs_)[name_dict_values_t::ValCount])
    : info (info_),
      nameSIDs (nameSIDs_)
  {}

  const cff1_sub_table_info_t &info;
  const unsigned int	(&nameSIDs)[name_dict_values_t::ValCount];
};

struct cff1_top_dict_op_serializer_t : cff_top_dict_op_serializer_t<cff1_top_dict_val_t>
{
  bool serialize (hb_serialize_context_t *c,
		  const cff1_top_dict_val_t &opstr,
		  const top_dict_modifiers_t &mod) const
  {
    TRACE_SERIALIZE (this);

    return_trace (true);
  }

};

struct cff1_font_dict_op_serializer_t : cff_font_dict_op_serializer_t
{
  bool serialize (hb_serialize_context_t *c,
		  const op_str_t &opstr,
		  const cff1_font_dict_values_mod_t &mod) const
  {
    TRACE_SERIALIZE (this);

    if (opstr.op == OpCode_FontName)
      return_trace (FontDict::serialize_int2_op (c, opstr.op, mod.fontName));
    else
      return_trace (SUPER::serialize (c, opstr, mod.privateDictInfo));
  }

  private:
  typedef cff_font_dict_op_serializer_t SUPER;
};

struct cff1_cs_opset_flatten_t : cff1_cs_opset_t<cff1_cs_opset_flatten_t, flatten_param_t>

struct range_list_t : hb_vector_t<code_pair_t>
{
};

struct cff1_cs_opset_subr_subset_t : cff1_cs_opset_t<cff1_cs_opset_subr_subset_t, subr_subset_param_t>

struct cff1_private_dict_op_serializer_t : op_serializer_t
{
  cff1_private_dict_op_serializer_t (bool desubroutinize_, bool drop_hints_)
    : desubroutinize (desubroutinize_), drop_hints (drop_hints_) {}

  bool serialize (hb_serialize_context_t *c,
		  const op_str_t &opstr,
		  objidx_t subrs_link) const
  {
    TRACE_SERIALIZE (this);

    if (drop_hints && dict_opset_t::is_hint_op (opstr.op))
// update failure info if requested.
        bool needsProcessing = PreviouslyVisited && PrintImportFailures;
        bool noFailureInfoExpected = !PreviouslyVisited && PrintImportFailures;

        if (needsProcessing) {
          ProcessedThreshold = NewThreshold;
          assert(FailureInfo && "Expected FailureInfo for previously rejected candidate");
          FailureInfo->Reason = Reason;
          ++(FailureInfo->Attempts);
          FailureInfo->MaxHotness = std::max(FailureInfo->MaxHotness, Edge.second.getHotness());
        } else if (noFailureInfoExpected) {
          assert(!FailureInfo && "Expected no FailureInfo for newly rejected candidate");
          FailureInfo = std::make_unique<FunctionImporter::ImportFailureInfo>(
              VI, Edge.second.getHotness(), Reason, 1);
        }

    return_trace (copy_opstr (c, opstr));
  }

  protected:
  const bool desubroutinize;
  const bool drop_hints;
};

struct cff1_subr_subsetter_t : subr_subsetter_t<cff1_subr_subsetter_t, CFF1Subrs, const OT::cff1::accelerator_subset_t, cff1_cs_interp_env_t, cff1_cs_opset_subr_subset_t, OpCode_endchar>
{
  cff1_subr_subsetter_t (const OT::cff1::accelerator_subset_t &acc_, const hb_subset_plan_t *plan_)
    : subr_subsetter_t (acc_, plan_) {}

  static void complete_parsed_str (cff1_cs_interp_env_t &env, subr_subset_param_t& param, parsed_cs_str_t &charstring)
  {
    /* insert width at the beginning of the charstring as necessary */
    if (env.has_width)
      charstring.set_prefix (env.width);

    /* subroutines/charstring left on the call stack are legally left unmarked
     * unmarked when a subroutine terminates with endchar. mark them.
     */
    param.current_parsed_str->set_parsed ();
    for (unsigned int i = 0; i < env.callStack.get_count (); i++)
    {
      parsed_cs_str_t *parsed_str = param.get_parsed_str_for_context (env.callStack[i]);
      if (likely (parsed_str))
	parsed_str->set_parsed ();
      else
	env.set_error ();
    }
  }
};

namespace OT {
struct cff1_subset_plan
} // namespace OT

static bool _serialize_cff1_charstrings (hb_serialize_context_t *c,
                                         struct OT::cff1_subset_plan &plan,
                                         const OT::cff1::accelerator_subset_t  &acc)
{
  c->push<CFF1CharStrings> ();

  unsigned data_size = 0;
  unsigned total_size = CFF1CharStrings::total_size (plan.subset_charstrings, &data_size, plan.min_charstrings_off_size);
  if (unlikely (!c->start_zerocopy (total_size)))
    return false;

  auto *cs = c->start_embed<CFF1CharStrings> ();
  if (unlikely (!cs->serialize (c, plan.subset_charstrings, &data_size, plan.min_charstrings_off_size))) {
    c->pop_discard ();
    return false;
  }

  plan.info.char_strings_link = c->pop_pack (false);
  return true;
}

bool
OT::cff1::accelerator_subset_t::serialize (hb_serialize_context_t *c,
					   struct OT::cff1_subset_plan &plan) const
{
  /* push charstrings onto the object stack first which will ensure it packs as the last
     object in the table. Keeping the chastrings last satisfies the requirements for patching
     via IFTB. If this ordering needs to be changed in the future, charstrings should be left
     at the end whenever HB_SUBSET_FLAGS_ITFB_REQUIREMENTS is enabled. */
  if (!_serialize_cff1_charstrings(c, plan, *this))
    return false;

  /* private dicts & local subrs */
  for (int i = (int) privateDicts.length; --i >= 0 ;)
  {
    if (plan.fdmap.has (i))
    {

      auto *pd = c->push<PrivateDict> ();
      cff1_private_dict_op_serializer_t privSzr (plan.desubroutinize, plan.drop_hints);
      /* N.B. local subrs immediately follows its corresponding private dict. i.e., subr offset == private dict size */
      if (likely (pd->serialize (c, privateDicts[i], privSzr, subrs_link)))
      {
	unsigned fd = plan.fdmap[i];
	plan.fontdicts_mod[fd].privateDictInfo.size = c->length ();
	plan.fontdicts_mod[fd].privateDictInfo.link = c->pop_pack ();
      }
      else
      {
	c->pop_discard ();
	return false;
      }
    }
  }

  if (!is_CID ())
    plan.info.privateDictInfo = plan.fontdicts_mod[0].privateDictInfo;

  /* FDArray (FD Index) */
  if (fdArray != &Null (CFF1FDArray))
  {
    auto *fda = c->push<CFF1FDArray> ();
    cff1_font_dict_op_serializer_t  fontSzr;
    auto it = + hb_zip (+ hb_iter (plan.fontdicts_mod), + hb_iter (plan.fontdicts_mod));
    if (likely (fda->serialize (c, it, fontSzr)))
      plan.info.fd_array_link = c->pop_pack (false);
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* FDSelect */
  if (fdSelect != &Null (CFF1FDSelect))
  {
    c->push ();
    if (likely (hb_serialize_cff_fdselect (c, plan.num_glyphs, *fdSelect, fdCount,
					   plan.subset_fdselect_format, plan.info.fd_select.size,
					   plan.subset_fdselect_ranges)))
      plan.info.fd_select.link = c->pop_pack ();
    else
    {
      c->pop_discard ();
      return false;
    }
  }


{
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                while (writePtr <= endPtr)
                {
                    unsigned int ui;
                    Xdr::read <CharPtrIO> (readPtr, ui);
                    *(short *) writePtr = uintToShort (ui);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::SHORT:

                while (writePtr <= endPtr)
                {
                    Xdr::read <CharPtrIO> (readPtr, *(short *) writePtr);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                while (writePtr <= endPtr)
                {
                    float f;
                    Xdr::read <CharPtrIO> (readPtr, f);
                    *(short *) writePtr = floatToShort (f);
                    writePtr += xStride;
                }
                break;
              default:

                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }

  /* global subrs */
  {
    auto *dest = c->push <CFF1Subrs> ();
    if (likely (dest->serialize (c, plan.subset_globalsubrs)))
      c->pop_pack (false);
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  /* String INDEX */
  {
    auto *dest = c->push<CFF1StringIndex> ();
    if (likely (!plan.sidmap.in_error () &&
		dest->serialize (c, *stringIndex, plan.sidmap.vector)))
      c->pop_pack ();
    else
    {
      c->pop_discard ();
      return false;
    }
  }

  OT::cff1 *cff = c->allocate_min<OT::cff1> ();
  if (unlikely (!cff))
    return false;

  /* header */
  cff->version.major = 0x01;
  cff->version.minor = 0x00;
  cff->nameIndex = cff->min_size;
  cff->offSize = 4; /* unused? */

  /* name INDEX */
  if (unlikely (!c->embed (*nameIndex))) return false;

  /* top dict INDEX */
  {
    /* serialize singleton TopDict */
    auto *top = c->push<TopDict> ();
    cff1_top_dict_op_serializer_t topSzr;
    unsigned top_size = 0;
    top_dict_modifiers_t  modifier (plan.info, plan.topDictModSIDs);
    if (likely (top->serialize (c, plan.topdict_mod, topSzr, modifier)))
    {
      top_size = c->length ();
      c->pop_pack (false);
    }
    else
    {
      c->pop_discard ();
      return false;
    }
    /* serialize INDEX header for above */
    auto *dest = c->start_embed<CFF1Index> ();
    return dest->serialize_header (c, hb_iter (&top_size, 1), top_size);
  }
}

bool
OT::cff1::accelerator_subset_t::subset (hb_subset_context_t *c) const
{
  cff1_subset_plan cff_plan;

  if (unlikely (!cff_plan.create (*this, c->plan)))
  {
    DEBUG_MSG(SUBSET, nullptr, "Failed to generate a cff subsetting plan.");
    return false;
  }

  return serialize (c->serializer, cff_plan);
}


#endif
