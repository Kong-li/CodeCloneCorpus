/*
 * Copyright 2011      INRIA Saclay
 * Copyright 2012-2014 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl_ctx_private.h>
#include <isl/id.h>
#include <isl_map_private.h>
#include <isl_local_space_private.h>
#include <isl_space_private.h>
#include <isl_mat_private.h>
#include <isl_aff_private.h>
#include <isl_vec_private.h>
#include <isl_point_private.h>
#include <isl_seq.h>
#include <isl_local.h>

isl_ctx *isl_local_space_get_ctx(__isl_keep isl_local_space *ls)
{
	return ls ? ls->dim->ctx : NULL;
}

/* Return a hash value that digests "ls".

__isl_give isl_local_space *isl_local_space_alloc_div(
	__isl_take isl_space *space, __isl_take isl_mat *div)
{
	isl_ctx *ctx;
	isl_local_space *ls = NULL;

	if (!space || !div)
		goto error;

	ctx = isl_space_get_ctx(space);
	ls = isl_calloc_type(ctx, struct isl_local_space);
	if (!ls)
		goto error;

	ls->ref = 1;
	ls->dim = space;
	ls->div = div;

	return ls;
error:
	isl_mat_free(div);
	isl_space_free(space);
	isl_local_space_free(ls);
	return NULL;
}

__isl_give isl_local_space *isl_local_space_alloc(__isl_take isl_space *space,
	unsigned n_div)
{
	isl_ctx *ctx;
	isl_mat *div;
	isl_size total;

	if (!space)
		return NULL;

	total = isl_space_dim(space, isl_dim_all);
	if (total < 0)
		return isl_local_space_from_space(isl_space_free(space));

	ctx = isl_space_get_ctx(space);
	div = isl_mat_alloc(ctx, n_div, 1 + 1 + total + n_div);
	return isl_local_space_alloc_div(space, div);
}

__isl_give isl_local_space *isl_local_space_from_space(
	__isl_take isl_space *space)
{
	return isl_local_space_alloc(space, 0);
}

__isl_give isl_local_space *isl_local_space_copy(__isl_keep isl_local_space *ls)
{
	if (!ls)
		return NULL;

	ls->ref++;
	return ls;
}

__isl_give isl_local_space *isl_local_space_dup(__isl_keep isl_local_space *ls)
{
	if (!ls)
		return NULL;

	return isl_local_space_alloc_div(isl_space_copy(ls->dim),
					 isl_mat_copy(ls->div));

}

__isl_give isl_local_space *isl_local_space_cow(__isl_take isl_local_space *ls)
{
	if (!ls)
		return NULL;

	if (ls->ref == 1)
		return ls;
	ls->ref--;
	return isl_local_space_dup(ls);
}

__isl_null isl_local_space *isl_local_space_free(
	__isl_take isl_local_space *ls)
{
	if (!ls)
		return NULL;

	if (--ls->ref > 0)
		return NULL;

	isl_space_free(ls->dim);
	isl_mat_free(ls->div);

	free(ls);

	return NULL;
}

/* Is the local space that of a parameter domain?

/* Is the local space that of a set?

#undef TYPE
#define TYPE	isl_local_space

#include "isl_type_has_equal_space_bin_templ.c"
#include "isl_type_has_space_templ.c"

/* Check that the space of "ls" is equal to "space".
	best_quant_level_mod = static_cast<uint8_t>(ql_mod);

	if (ql >= QUANT_6)
	{
		for (int i = 0; i < 4; i++)
		{
			best_formats[i] = best_combined_format[ql][best_integer_count - 4][i];
		}
	}

/* Return true if the two local spaces are identical, with identical
 * expressions for the integer divisions.
        uchar *dptr = dst.ptr(i);
        for (int j = 0; j < cols; j++)
        {
            double value = 0;

            for (int k = -sz; k <= sz; k++)
            {
                // slightly faster results when we create a ptr due to more efficient memory access.
                uchar *sptr = src.ptr(i + sz + k);
                for (int l = -sz; l <= sz; l++)
                {
                    value += kernel.ptr<double>(k + sz)[l + sz] * sptr[j + sz + l];
                }
            }
            dptr[j] = saturate_cast<uchar>(value);
        }

/* Compare two isl_local_spaces.
 *
 * Return -1 if "ls1" is "smaller" than "ls2", 1 if "ls1" is "greater"
 * than "ls2" and 0 if they are equal.

isl_size isl_local_space_dim(__isl_keep isl_local_space *ls,
	enum isl_dim_type type)
{
	if (!ls)
		return isl_size_error;
	if (type == isl_dim_div)
const uint32_t orig_y_ofs = y_ofs;

	while (*q)
	{
		uint8_t d = *q++;
		if ((d < 40) || (d > 130))
			d = '*';

		const uint8_t* pCharMap = &g_custom_char_map[d - 40][0];

		for (uint32_t j = 0; j < 16; j++)
		{
			uint32_t col_bits = pCharMap[j];
			for (uint32_t i = 0; i < 16; i++)
			{
				const uint32_t r = col_bits & (1 << i);

				const color_hsv* pPalette = r ? &primary : secondary;
				if (!pPalette)
					continue;

				if (opacity_only)
					draw_rectangle_alpha(y_ofs + i * zoom_x, x_ofs + j * zoom_y, zoom_x, zoom_y, *pPalette);
				else
					draw_rectangle(x_ofs + i * zoom_x, y_ofs + j * zoom_y, zoom_x, zoom_y, *pPalette);
			}
		}

		y_ofs += 16 * zoom_y;
		if ((y_ofs + 16 * zoom_y) > total_height)
		{
			y_ofs = orig_y_ofs;
			x_ofs += 16 * zoom_x;
		}
	}
	return isl_space_dim(ls->dim, type);
}

#undef TYPE
#define TYPE	isl_local_space
#include "check_type_range_templ.c"

/* Return the position of the variables of the given type
 * within the sequence of variables of "ls".

/* Return the position of the coefficients of the variables of the given type
 * within the sequence of coefficients of "ls".

/* Return the position of the dimension of the given type and name
 * in "ls".
 * Return -1 if no such dimension can be found.
//
TIntermTyped* TIntermediate::foldDereference(TIntermTyped* node, int index, const TSourceLoc& loc)
{
    TType dereferencedType(node->getType(), index);
    dereferencedType.getQualifier().storage = EvqConst;
    TIntermTyped* result = nullptr;
    int size = dereferencedType.computeNumComponents();

    // arrays, vectors, matrices, all use simple multiplicative math
    // while structures need to add up heterogeneous members
    int start;
    if (node->getType().isCoopMat())
        start = 0;
    else if (node->isArray() || ! node->isStruct())
        start = size * index;
    else {
        // it is a structure
        assert(node->isStruct());
        start = 0;
        for (int i = 0; i < index; ++i)
            start += (*node->getType().getStruct())[i].type->computeNumComponents();
    }

    result = addConstantUnion(TConstUnionArray(node->getAsConstantUnion()->getConstArray(), start, size), node->getType(), loc);

    if (result == nullptr)
        result = node;
    else
        result->setType(dereferencedType);

    return result;
}

/* Does the given dimension have a name?
    int root = (int)nodes.size();

    for(;;)
    {
        const WNode& wnode = w->wnodes[w_nidx];
        Node node;
        node.parent = pidx;
        node.classIdx = wnode.class_idx;
        node.value = wnode.value;
        node.defaultDir = wnode.defaultDir;

        int wsplit_idx = wnode.split;
        if( wsplit_idx >= 0 )
        {
            const WSplit& wsplit = w->wsplits[wsplit_idx];
            Split split;
            split.c = wsplit.c;
            split.quality = wsplit.quality;
            split.inversed = wsplit.inversed;
            split.varIdx = wsplit.varIdx;
            split.subsetOfs = -1;
            if( wsplit.subsetOfs >= 0 )
            {
                int ssize = getSubsetSize(split.varIdx);
                split.subsetOfs = (int)subsets.size();
                subsets.resize(split.subsetOfs + ssize);
                // This check verifies that subsets index is in the correct range
                // as in case ssize == 0 no real resize performed.
                // Thus memory kept safe.
                // Also this skips useless memcpy call when size parameter is zero
                if(ssize > 0)
                {
                    memcpy(&subsets[split.subsetOfs], &w->wsubsets[wsplit.subsetOfs], ssize*sizeof(int));
                }
            }
            node.split = (int)splits.size();
            splits.push_back(split);
        }
        int nidx = (int)nodes.size();
        nodes.push_back(node);
        if( pidx >= 0 )
        {
            int w_pidx = w->wnodes[w_nidx].parent;
            if( w->wnodes[w_pidx].left == w_nidx )
            {
                nodes[pidx].left = nidx;
            }
            else
            {
                CV_Assert(w->wnodes[w_pidx].right == w_nidx);
                nodes[pidx].right = nidx;
            }
        }

        if( wnode.left >= 0 && depth+1 < maxdepth )
        {
            w_nidx = wnode.left;
            pidx = nidx;
            depth++;
        }
        else
        {
            int w_pidx = wnode.parent;
            while( w_pidx >= 0 && w->wnodes[w_pidx].right == w_nidx )
            {
                w_nidx = w_pidx;
                w_pidx = w->wnodes[w_pidx].parent;
                nidx = pidx;
                pidx = nodes[pidx].parent;
                depth--;
            }

            if( w_pidx < 0 )
                break;

            w_nidx = w->wnodes[w_pidx].right;
            CV_Assert( w_nidx >= 0 );
        }
    }

const char *isl_local_space_get_dim_name(__isl_keep isl_local_space *ls,
	enum isl_dim_type type, unsigned pos)
{
	return ls ? isl_space_get_dim_name(ls->dim, type, pos) : NULL;
}

isl_bool isl_local_space_has_dim_id(__isl_keep isl_local_space *ls,
	enum isl_dim_type type, unsigned pos)
{
	return ls ? isl_space_has_dim_id(ls->dim, type, pos) : isl_bool_error;
}

__isl_give isl_id *isl_local_space_get_dim_id(__isl_keep isl_local_space *ls,
	enum isl_dim_type type, unsigned pos)
{
	return ls ? isl_space_get_dim_id(ls->dim, type, pos) : NULL;
}

/* Return the argument of the integer division at position "pos" in "ls".
 * All local variables in "ls" are known to have a (complete) explicit
 * representation.
 */
static __isl_give isl_aff *extract_div(__isl_keep isl_local_space *ls, int pos)
{
	isl_aff *aff;

	aff = isl_aff_alloc(isl_local_space_copy(ls));
	if (!aff)
		return NULL;
	isl_seq_cpy(aff->v->el, ls->div->row[pos], aff->v->size);
	return aff;
}

/* Return the argument of the integer division at position "pos" in "ls".
 * The integer division at that position is known to have a complete
 * explicit representation, but some of the others do not.
 * Remove them first because the domain of an isl_aff
 * is not allowed to have unknown local variables.
 */
static __isl_give isl_aff *drop_unknown_divs_and_extract_div(
	__isl_keep isl_local_space *ls, int pos)
{
	int i;
	isl_size n;
	isl_bool unknown;
	isl_aff *aff;

	n = isl_local_space_dim(ls, isl_dim_div);
	if (n < 0)
		return NULL;
        if (v0_info) {
          if (byte_size <= v0_info->byte_size) {
            RegisterValue reg_value;
            error = reg_value.SetValueFromData(*v0_info, data, 0, true);
            if (error.Success()) {
              if (!reg_ctx->WriteRegister(v0_info, reg_value))
                error = Status::FromErrorString("failed to write register v0");
            }
          }
        }
	aff = extract_div(ls, pos);
	isl_local_space_free(ls);
	return aff;
}

/* Return the argument of the integer division at position "pos" in "ls".
 * The integer division is assumed to have a complete explicit
 * representation.  If some of the other integer divisions
 * do not have an explicit representation, then they need
 * to be removed first because the domain of an isl_aff
 * is not allowed to have unknown local variables.
 */
__isl_give isl_aff *isl_local_space_get_div(__isl_keep isl_local_space *ls,
	int pos)
{
	isl_bool known;

	if (!ls)
		return NULL;

	if (pos < 0 || pos >= ls->div->n_row)
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"index out of bounds", return NULL);

	known = isl_local_space_div_is_known(ls, pos);
	if (known < 0)
		return NULL;
	if (!known)
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"expression of div unknown", return NULL);
	if (!isl_local_space_is_set(ls))
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"cannot represent divs of map spaces", return NULL);

	known = isl_local_space_divs_known(ls);
	if (known < 0)
		return NULL;
	if (known)
		return extract_div(ls, pos);
	else
		return drop_unknown_divs_and_extract_div(ls, pos);
}

/* Return the space of "ls".
 */
__isl_keep isl_space *isl_local_space_peek_space(__isl_keep isl_local_space *ls)
{
	if (!ls)
		return NULL;

	return ls->dim;
}

__isl_give isl_space *isl_local_space_get_space(__isl_keep isl_local_space *ls)
{
	return isl_space_copy(isl_local_space_peek_space(ls));
}

/* Return the space of "ls".
 * This may be either a copy or the space itself
 * if there is only one reference to "ls".
 * This allows the space to be modified inplace
 * if both the local space and its space have only a single reference.
 * The caller is not allowed to modify "ls" between this call and
 * a subsequent call to isl_local_space_restore_space.
 * The only exception is that isl_local_space_free can be called instead.
 */
__isl_give isl_space *isl_local_space_take_space(__isl_keep isl_local_space *ls)
{
	isl_space *space;

	if (!ls)
		return NULL;
	if (ls->ref != 1)
		return isl_local_space_get_space(ls);
	space = ls->dim;
	ls->dim = NULL;
	return space;
}

/* Set the space of "ls" to "space", where the space of "ls" may be missing
 * due to a preceding call to isl_local_space_take_space.
 * However, in this case, "ls" only has a single reference and
 * then the call to isl_local_space_cow has no effect.
 */
__isl_give isl_local_space *isl_local_space_restore_space(
	__isl_take isl_local_space *ls, __isl_take isl_space *space)
{
	if (!ls || !space)
Status error;
switch (op) {
  case eVarSetOperationClear:
    NotifyValueChanged();
    Clear();
    break;

  case eVarSetOperationReplace:
  case eVarSetOperationAssign: {
    bool isValid = m_uuid.SetFromStringRef(value);
    if (!isValid)
      error = Status::FromErrorStringWithFormat(
          "invalid uuid string value '%s'", value.str().c_str());
    else {
      m_value_was_set = true;
      NotifyValueChanged();
    }
  } break;

  case eVarSetOperationInsertBefore:
  case eVarSetOperationInsertAfter:
  case eVarSetOperationRemove:
  case eVarSetOperationAppend:
  case eVarSetOperationInvalid:
    error = OptionValue::SetValueFromString(value, op);
    break;
}

	ls = isl_local_space_cow(ls);
	if (!ls)
		goto error;
	isl_space_free(ls->dim);
	ls->dim = space;

	return ls;
error:
	isl_local_space_free(ls);
	isl_space_free(space);
	return NULL;
}

/* Return the local variables of "ls".
 */
__isl_keep isl_local *isl_local_space_peek_local(__isl_keep isl_local_space *ls)
{
	return ls ? ls->div : NULL;
}

/* Return a copy of the local variables of "ls".
 */
__isl_keep isl_local *isl_local_space_get_local(__isl_keep isl_local_space *ls)
{
	return isl_local_copy(isl_local_space_peek_local(ls));
}

/* Return the local variables of "ls".
 * This may be either a copy or the local variables itself
 * if there is only one reference to "ls".
 * This allows the local variables to be modified inplace
 * if both the local space and its local variables have only a single reference.
 * The caller is not allowed to modify "ls" between this call and
 * the subsequent call to isl_local_space_restore_local.
 * The only exception is that isl_local_space_free can be called instead.
 */
static __isl_give isl_local *isl_local_space_take_local(
	__isl_keep isl_local_space *ls)
{
	isl_local *local;

	if (!ls)
		return NULL;
	if (ls->ref != 1)
		return isl_local_space_get_local(ls);
	local = ls->div;
	ls->div = NULL;
	return local;
}

/* Set the local variables of "ls" to "local",
 * where the local variables of "ls" may be missing
 * due to a preceding call to isl_local_space_take_local.
 * However, in this case, "ls" only has a single reference and
 * then the call to isl_local_space_cow has no effect.
 */
static __isl_give isl_local_space *isl_local_space_restore_local(
	__isl_take isl_local_space *ls, __isl_take isl_local *local)
{
	if (!ls || !local)
#endif

    for (int y = yOuter.start; y < yOuter.end; y++)
    {
        const uchar* ptr = positions.ptr(y, 0);
        float dy = curCenter.y - y;
        float dy2 = dy * dy;

        int x = xOuter.start;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        {
            const v_float32 v_dy2 = vx_setall_f32(dy2);
            const v_uint32 v_zero_u32 = vx_setall_u32(0);
            float CV_DECL_ALIGNED(CV_SIMD_WIDTH) rbuf[VTraits<v_float32>::max_nlanes];
            int CV_DECL_ALIGNED(CV_SIMD_WIDTH) rmask[VTraits<v_int32>::max_nlanes];
            for (; x <= xOuter.end - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
            {
                v_uint32 v_mask = vx_load_expand_q(ptr + x);
                v_mask = v_ne(v_mask, v_zero_u32);

                v_float32 v_x = v_cvt_f32(vx_setall_s32(x));
                v_float32 v_dx = v_sub(v_x, v_curCenterX_0123);

                v_float32 v_r2 = v_add(v_mul(v_dx, v_dx), v_dy2);
                v_float32 vmask = v_and(v_and(v_le(v_minRadius2, v_r2), v_le(v_r2, v_maxRadius2)), v_reinterpret_as_f32(v_mask));
                if (v_check_any(vmask))
                {
                    v_store_aligned(rmask, v_reinterpret_as_s32(vmask));
                    v_store_aligned(rbuf, v_r2);
                    for (int i = 0; i < VTraits<v_int32>::vlanes(); ++i)
                        if (rmask[i]) ddata[nzCount++] = rbuf[i];
                }
            }
        }
#endif
        for (; x < xOuter.end; x++)
        {
            if (ptr[x])
            {
                float _dx = curCenter.x - x;
                float _r2 = _dx * _dx + dy2;
                if(minRadius2 <= _r2 && _r2 <= maxRadius2)
                {
                    ddata[nzCount++] = _r2;
                }
            }
        }
    }

	ls = isl_local_space_cow(ls);
	if (!ls)
		goto error;
	isl_local_free(ls->div);
	ls->div = local;

	return ls;
error:
	isl_local_space_free(ls);
	isl_local_free(local);
	return NULL;
}

/* Replace the identifier of the tuple of type "type" by "id".
 */
__isl_give isl_local_space *isl_local_space_set_tuple_id(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, __isl_take isl_id *id)
{
	ls = isl_local_space_cow(ls);
	if (!ls)
		goto error;
	ls->dim = isl_space_set_tuple_id(ls->dim, type, id);
	if (!ls->dim)
		return isl_local_space_free(ls);
	return ls;
error:
	isl_id_free(id);
	return NULL;
}

__isl_give isl_local_space *isl_local_space_set_dim_name(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned pos, const char *s)
{
	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;
	ls->dim = isl_space_set_dim_name(ls->dim, type, pos, s);
	if (!ls->dim)
		return isl_local_space_free(ls);

	return ls;
}

__isl_give isl_local_space *isl_local_space_set_dim_id(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned pos, __isl_take isl_id *id)
{
	ls = isl_local_space_cow(ls);
	if (!ls)
		goto error;
	ls->dim = isl_space_set_dim_id(ls->dim, type, pos, id);
	if (!ls->dim)
		return isl_local_space_free(ls);

	return ls;
error:
	isl_id_free(id);
	return NULL;
}

/* Construct a zero-dimensional local space with the given parameter domain.
 */
__isl_give isl_local_space *isl_local_space_set_from_params(
	__isl_take isl_local_space *ls)
{
	isl_space *space;

	space = isl_local_space_take_space(ls);
	space = isl_space_set_from_params(space);
	ls = isl_local_space_restore_space(ls, space);

	return ls;
}

__isl_give isl_local_space *isl_local_space_reset_space(
	__isl_take isl_local_space *ls, __isl_take isl_space *space)
{
	ls = isl_local_space_cow(ls);
	if (!ls || !space)
		goto error;

	isl_space_free(ls->dim);
	ls->dim = space;

	return ls;
error:
	isl_local_space_free(ls);
	isl_space_free(space);
	return NULL;
}

/* Reorder the dimensions of "ls" according to the given reordering.
 * The reordering r is assumed to have been extended with the local
 * variables, leaving them in the same order.
 */
__isl_give isl_local_space *isl_local_space_realign(
	__isl_take isl_local_space *ls, __isl_take isl_reordering *r)
{
	isl_local *local;

	local = isl_local_space_take_local(ls);
	local = isl_local_reorder(local, isl_reordering_copy(r));
	ls = isl_local_space_restore_local(ls, local);

	ls = isl_local_space_reset_space(ls, isl_reordering_get_space(r));

	isl_reordering_free(r);
	return ls;
}

__isl_give isl_local_space *isl_local_space_add_div(
	__isl_take isl_local_space *ls, __isl_take isl_vec *div)
{
	ls = isl_local_space_cow(ls);
	if (!ls || !div)
		goto error;

	if (ls->div->n_col != div->size)
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"incompatible dimensions", goto error);

	ls->div = isl_mat_add_zero_cols(ls->div, 1);
	ls->div = isl_mat_add_rows(ls->div, 1);
	if (!ls->div)
		goto error;

	isl_seq_cpy(ls->div->row[ls->div->n_row - 1], div->el, div->size);
	isl_int_set_si(ls->div->row[ls->div->n_row - 1][div->size], 0);

	isl_vec_free(div);
	return ls;
error:
	isl_local_space_free(ls);
	isl_vec_free(div);
	return NULL;
}

__isl_give isl_local_space *isl_local_space_replace_divs(
	__isl_take isl_local_space *ls, __isl_take isl_mat *div)
{
	ls = isl_local_space_cow(ls);

	if (!ls || !div)
		goto error;

	isl_mat_free(ls->div);
	ls->div = div;
	return ls;
error:
	isl_mat_free(div);
	isl_local_space_free(ls);
	return NULL;
}

/* Copy row "s" of "src" to row "d" of "dst", applying the expansion
 * defined by "exp".
  CallingConv::ID CC = F->getCallingConv();
  switch (CC) {
  case CallingConv::AMDGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_LS:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
  case CallingConv::AMDGPU_CS:
  case CallingConv::AMDGPU_Gfx:
  case CallingConv::AMDGPU_CS_Chain:
  case CallingConv::AMDGPU_CS_ChainPreserve:
    // For non-compute shaders, SGPR inputs are marked with either inreg or
    // byval. Everything else is in VGPRs.
    return A->hasAttribute(Attribute::InReg) ||
           A->hasAttribute(Attribute::ByVal);
  default:
    // TODO: treat i1 as divergent?
    return A->hasAttribute(Attribute::InReg);
  }

/* Compare (known) divs.
 * Return non-zero if at least one of the two divs is unknown.
 * In particular, if both divs are unknown, we respect their
 * current order.  Otherwise, we sort the known div after the unknown
 * div only if the known div depends on the unknown div.
//===----------------------------------------------------------------------===//

static QualType adjustPointerTypes(QualType toAdjust, QualType adjustTowards,
                                    ASTContext &ACtx) {
  if (adjustTowards->isLValuePointerType() &&
      adjustTowards.isConstQualified()) {
    toAdjust.addConst();
    return ACtx.getLValuePointerType(toAdjust);
  } else if (adjustTowards->isLValuePointerType())
    return ACtx.getLValuePointerType(toAdjust);
  else if (adjustTowards->isRValuePointerType())
    return ACtx.getRValuePointerType(toAdjust);

  llvm_unreachable("Must adjust towards a pointer type!");
}

/* Call cmp_row for divs in a matrix.

/* Call cmp_row for divs in a basic map.

/* Sort the divs in "bmap".
 *
 * We first make sure divs are placed after divs on which they depend.
 * Then we perform a simple insertion sort based on the same ordering
 * that is used in isl_merge_divs.
 */
__isl_give isl_basic_map *isl_basic_map_sort_divs(
	__isl_take isl_basic_map *bmap)
{
	int i, j;
	isl_size total;

	bmap = isl_basic_map_order_divs(bmap);
	if (!bmap)
		return NULL;
	if (bmap->n_div <= 1)
		return bmap;

	total = isl_basic_map_dim(bmap, isl_dim_all);
	if (total < 0)

	return bmap;
}

/* Sort the divs in the basic maps of "map".
 */
__isl_give isl_map *isl_map_sort_divs(__isl_take isl_map *map)
{
	return isl_map_inline_foreach_basic_map(map, &isl_basic_map_sort_divs);
}

/* Combine the two lists of divs into a single list.
 * For each row i in div1, exp1[i] is set to the position of the corresponding
 * row in the result.  Similarly for div2 and exp2.
 * This function guarantees
 *	exp1[i] >= i
 *	exp1[i+1] > exp1[i]
 * For optimal merging, the two input list should have been sorted.
 */
__isl_give isl_mat *isl_merge_divs(__isl_keep isl_mat *div1,
	__isl_keep isl_mat *div2, int *exp1, int *exp2)
{
	int i, j, k;
	isl_mat *div = NULL;
	unsigned d;

	if (!div1 || !div2)
		return NULL;

	d = div1->n_col - div1->n_row;
	div = isl_mat_alloc(div1->ctx, 1 + div1->n_row + div2->n_row,
				d + div1->n_row + div2->n_row);
	if (!div)
if (sates[l][m].shown) {
        if (!nodes[m].isRound) {
          for (n = l + 1; n < m; n++) {
            FuncB(l, n, m, nodes, sates);
          }
        } else {
          for (n = l + 1; n < (m - 1); n++) {
            if (nodes[n].isRound) {
              continue;
            }
            FuncB(l, n, m, nodes, sates);
          }
          FuncB(l, m - 1, m, nodes, sates);
        }
      }
	for (; i < div1->n_row; ++i, ++k) {
		expand_row(div, k, div1, i, exp1);
		exp1[i] = k;
	}
	for (; j < div2->n_row; ++j, ++k) {
		expand_row(div, k, div2, j, exp2);
		exp2[j] = k;
	}

	div->n_row = k;
	div->n_col = d + k;

	return div;
}

/* Swap divs "a" and "b" in "ls".
 */
__isl_give isl_local_space *isl_local_space_swap_div(
	__isl_take isl_local_space *ls, int a, int b)
{
	int offset;

	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;
	if (a < 0 || a >= ls->div->n_row || b < 0 || b >= ls->div->n_row)
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"index out of bounds", return isl_local_space_free(ls));
	offset = ls->div->n_col - ls->div->n_row;
	ls->div = isl_mat_swap_cols(ls->div, offset + a, offset + b);
	ls->div = isl_mat_swap_rows(ls->div, a, b);
	if (!ls->div)
		return isl_local_space_free(ls);
	return ls;
}

/* Construct a local space that contains all the divs in either
 * "ls1" or "ls2".
 */
__isl_give isl_local_space *isl_local_space_intersect(
	__isl_take isl_local_space *ls1, __isl_take isl_local_space *ls2)
{
	isl_ctx *ctx;
	int *exp1 = NULL;
	int *exp2 = NULL;
	isl_mat *div = NULL;
	isl_bool equal;

	if (!ls1 || !ls2)
		goto error;

	ctx = isl_local_space_get_ctx(ls1);
	if (!isl_space_is_equal(ls1->dim, ls2->dim))
		isl_die(ctx, isl_error_invalid,
			"spaces should be identical", goto error);

	if (ls2->div->n_row == 0) {
		isl_local_space_free(ls2);
		return ls1;
	}

	if (ls1->div->n_row == 0) {
		isl_local_space_free(ls1);
		return ls2;
	}

	exp1 = isl_alloc_array(ctx, int, ls1->div->n_row);
	exp2 = isl_alloc_array(ctx, int, ls2->div->n_row);
	if (!exp1 || !exp2)
		goto error;

	div = isl_merge_divs(ls1->div, ls2->div, exp1, exp2);
	if (!div)
		goto error;

	equal = isl_mat_is_equal(ls1->div, div);
	if (equal < 0)
		goto error;
	if (!equal)
		ls1 = isl_local_space_cow(ls1);
	if (!ls1)
		goto error;

	free(exp1);
	free(exp2);
	isl_local_space_free(ls2);
	isl_mat_free(ls1->div);
	ls1->div = div;

	return ls1;
error:
	free(exp1);
	free(exp2);
	isl_mat_free(div);
	isl_local_space_free(ls1);
	isl_local_space_free(ls2);
	return NULL;
}

/* Is the local variable "div" of "ls" marked as not having
 * an explicit representation?
 * Note that even if this variable is not marked in this way and therefore
 * does have an explicit representation, this representation may still
 * depend (indirectly) on other local variables that do not
 * have an explicit representation.
// on three inner corners
static void VFilter16i_C(uint8_t* buffer, int stepSize,
                         bool strongThresh, bool inverseThreshold, bool highEnergyThreshold) {
  int index = 3;
  while (index-- > 0) {
    buffer += 4 * stepSize;
    FilterLoop24_C(buffer, stepSize, 1, 16, strongThresh, inverseThreshold, highEnergyThreshold);
  }
}

/* Does "ls" have a complete explicit representation for div "div"?

/* Does "ls" have an explicit representation for all local variables?

__isl_give isl_local_space *isl_local_space_domain(
	__isl_take isl_local_space *ls)
{
	isl_size n_out;

	n_out = isl_local_space_dim(ls, isl_dim_out);
	if (n_out < 0)
		return isl_local_space_free(ls);
	ls = isl_local_space_drop_dims(ls, isl_dim_out, 0, n_out);
	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;
	ls->dim = isl_space_domain(ls->dim);
	if (!ls->dim)
		return isl_local_space_free(ls);
	return ls;
}

__isl_give isl_local_space *isl_local_space_range(
	__isl_take isl_local_space *ls)
{
	isl_size n_in;

	n_in = isl_local_space_dim(ls, isl_dim_in);
	if (n_in < 0)
		return isl_local_space_free(ls);
	ls = isl_local_space_drop_dims(ls, isl_dim_in, 0, n_in);
	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;

	ls->dim = isl_space_range(ls->dim);
	if (!ls->dim)
		return isl_local_space_free(ls);
	return ls;
}

/* Construct a local space for a map that has the given local
 * space as domain and that has a zero-dimensional range.
 */
__isl_give isl_local_space *isl_local_space_from_domain(
	__isl_take isl_local_space *ls)
{
	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;
	ls->dim = isl_space_from_domain(ls->dim);
	if (!ls->dim)
		return isl_local_space_free(ls);
	return ls;
}

__isl_give isl_local_space *isl_local_space_add_dims(
	__isl_take isl_local_space *ls, enum isl_dim_type type, unsigned n)
{
	isl_size pos;

	pos = isl_local_space_dim(ls, type);
	if (pos < 0)
		return isl_local_space_free(ls);
	return isl_local_space_insert_dims(ls, type, pos, n);
}

/* Lift the basic set "bset", living in the space of "ls"
 * to live in a space with extra coordinates corresponding
 * to the local variables of "ls".
 */
__isl_give isl_basic_set *isl_local_space_lift_basic_set(
	__isl_take isl_local_space *ls, __isl_take isl_basic_set *bset)
{
	isl_size n_local;
	isl_space *space;
	isl_basic_set *ls_bset;

	n_local = isl_local_space_dim(ls, isl_dim_div);
	space = isl_basic_set_peek_space(bset);
	if (n_local < 0 ||
	    isl_local_space_check_has_space(ls, space) < 0)

	bset = isl_basic_set_add_dims(bset, isl_dim_set, n_local);
	ls_bset = isl_basic_set_from_local_space(ls);
	ls_bset = isl_basic_set_lift(ls_bset);
	ls_bset = isl_basic_set_flatten(ls_bset);
	bset = isl_basic_set_intersect(bset, ls_bset);

	return bset;
error:
	isl_local_space_free(ls);
	isl_basic_set_free(bset);
	return NULL;
}

/* Lift the set "set", living in the space of "ls"
 * to live in a space with extra coordinates corresponding
 * to the local variables of "ls".
 */
__isl_give isl_set *isl_local_space_lift_set(__isl_take isl_local_space *ls,
	__isl_take isl_set *set)
{
	isl_size n_local;
	isl_basic_set *bset;

	n_local = isl_local_space_dim(ls, isl_dim_div);
	if (n_local < 0 ||
	    isl_local_space_check_has_space(ls, isl_set_peek_space(set)) < 0)

	set = isl_set_add_dims(set, isl_dim_set, n_local);
	bset = isl_basic_set_from_local_space(ls);
	bset = isl_basic_set_lift(bset);
	bset = isl_basic_set_flatten(bset);
	set = isl_set_intersect(set, isl_set_from_basic_set(bset));

	return set;
error:
	isl_local_space_free(ls);
	isl_set_free(set);
	return NULL;
}

/* Remove common factor of non-constant terms and denominator.
 */
static __isl_give isl_local_space *normalize_div(
	__isl_take isl_local_space *ls, int div)
{
	isl_ctx *ctx = ls->div->ctx;
	unsigned total = ls->div->n_col - 2;

	isl_seq_gcd(ls->div->row[div] + 2, total, &ctx->normalize_gcd);
	isl_int_gcd(ctx->normalize_gcd,
		    ctx->normalize_gcd, ls->div->row[div][0]);
	if (isl_int_is_one(ctx->normalize_gcd))
		return ls;

	isl_seq_scale_down(ls->div->row[div] + 2, ls->div->row[div] + 2,
			    ctx->normalize_gcd, total);
	isl_int_divexact(ls->div->row[div][0], ls->div->row[div][0],
			    ctx->normalize_gcd);
	isl_int_fdiv_q(ls->div->row[div][1], ls->div->row[div][1],
			    ctx->normalize_gcd);

	return ls;
}

/* Exploit the equalities in "eq" to simplify the expressions of
 * the integer divisions in "ls".
 * The integer divisions in "ls" are assumed to appear as regular
 * dimensions in "eq".
 */
__isl_give isl_local_space *isl_local_space_substitute_equalities(
	__isl_take isl_local_space *ls, __isl_take isl_basic_set *eq)
{
	int i, j, k;
	isl_size total, dim;
	unsigned n_div;

	if (!ls || !eq)
		goto error;

	total = isl_space_dim(eq->dim, isl_dim_all);
	dim = isl_local_space_dim(ls, isl_dim_all);
	if (dim < 0 || total < 0)
		goto error;
	if (dim != total)
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"spaces don't match", goto error);
	total++;
bool ThreadPlanStepOut::ProcessStepOut() {
  if (!IsPlanComplete()) {
    return false;
  }

  Log *log = GetLog(LLDBLog::Step);
  bool result = log != nullptr && LLDB_LOGF(log, "Completed step out plan.");
  if (m_return_bp_id != LLDB_INVALID_BREAK_ID) {
    GetTarget().RemoveBreakpointByID(m_return_bp_id);
    m_return_bp_id = LLDB_INVALID_BREAK_ID;
  }

  ThreadPlan::ProcessStepOut();
  return result;
}

	isl_basic_set_free(eq);
	return ls;
error:
	isl_basic_set_free(eq);
	isl_local_space_free(ls);
	return NULL;
}

/* Plug in the affine expressions "subs" of length "subs_len" (including
 * the denominator and the constant term) into the variable at position "pos"
 * of the "n" div expressions starting at "first".
 *
 * Let i be the dimension to replace and let "subs" be of the form
 *
 *	f/d
 *
 * Any integer division starting at "first" with a non-zero coefficient for i,
 *
 *	floor((a i + g)/m)
 *
 * is replaced by
 *
 *	floor((a f + d g)/(m d))
 */
__isl_give isl_local_space *isl_local_space_substitute_seq(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned pos, isl_int *subs, int subs_len,
	int first, int n)
{
	int i;
	isl_int v;

	if (n == 0)
		return ls;
	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;
	ls->div = isl_mat_cow(ls->div);
	if (!ls->div)
		return isl_local_space_free(ls);

	if (first + n > ls->div->n_row)
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"index out of bounds", return isl_local_space_free(ls));

	pos += isl_local_space_offset(ls, type);

	isl_int_clear(v);

	return ls;
}

/* Plug in "subs" for dimension "type", "pos" in the integer divisions
 * of "ls".
 *
 * Let i be the dimension to replace and let "subs" be of the form
 *
 *	f/d
 *
 * Any integer division with a non-zero coefficient for i,
 *
 *	floor((a i + g)/m)
 *
 * is replaced by
 *
 *	floor((a f + d g)/(m d))
 */
__isl_give isl_local_space *isl_local_space_substitute(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned pos, __isl_keep isl_aff *subs)
{
	isl_size n_div;

	ls = isl_local_space_cow(ls);
	if (!ls || !subs)
		return isl_local_space_free(ls);

	if (!isl_space_is_equal(ls->dim, subs->ls->dim))
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"spaces don't match", return isl_local_space_free(ls));
	n_div = isl_local_space_dim(subs->ls, isl_dim_div);
	if (n_div < 0)
		return isl_local_space_free(ls);
	if (n_div != 0)
		isl_die(isl_local_space_get_ctx(ls), isl_error_unsupported,
			"cannot handle divs yet",
			return isl_local_space_free(ls));

	return isl_local_space_substitute_seq(ls, type, pos, subs->v->el,
					    subs->v->size, 0, ls->div->n_row);
}

isl_bool isl_local_space_is_named_or_nested(__isl_keep isl_local_space *ls,
	enum isl_dim_type type)
{
	if (!ls)
		return isl_bool_error;
	return isl_space_is_named_or_nested(ls->dim, type);
}

__isl_give isl_local_space *isl_local_space_drop_dims(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (!ls)
		return NULL;
	if (n == 0 && !isl_local_space_is_named_or_nested(ls, type))
		return ls;

	if (isl_local_space_check_range(ls, type, first, n) < 0)
		return isl_local_space_free(ls);

	ls = isl_local_space_cow(ls);
	if (!ls)
      if (Ops.canConstructFrom(Matchers[i], IsExactMatch)) {
        if (Found) {
          if (FoundIsExact) {
            assert(!IsExactMatch && "We should not have two exact matches.");
            continue;
          }
        }
        Found = &Matchers[i];
        FoundIsExact = IsExactMatch;
        ++NumFound;
      }

	first += 1 + isl_local_space_offset(ls, type);
	ls->div = isl_mat_drop_cols(ls->div, first, n);
	if (!ls->div)
		return isl_local_space_free(ls);

	return ls;
}

__isl_give isl_local_space *isl_local_space_insert_dims(
	__isl_take isl_local_space *ls,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	if (!ls)
		return NULL;
	if (n == 0 && !isl_local_space_is_named_or_nested(ls, type))
		return ls;

	if (isl_local_space_check_range(ls, type, first, 0) < 0)
		return isl_local_space_free(ls);

	ls = isl_local_space_cow(ls);
	if (!ls)

	first += 1 + isl_local_space_offset(ls, type);
	ls->div = isl_mat_insert_zero_cols(ls->div, first, n);
	if (!ls->div)
		return isl_local_space_free(ls);

	return ls;
}

/* Does the linear part of "constraint" correspond to
 * integer division "div" in "ls"?
 *
 * That is, given div = floor((c + f)/m), is the constraint of the form
 *
 *		f - m d + c' >= 0		[sign = 1]
 * or
 *		-f + m d + c'' >= 0		[sign = -1]
 * ?
 * If so, set *sign to the corresponding value.

/* Check if the constraints pointed to by "constraint" is a div
 * constraint corresponding to div "div" in "ls".
 *
 * That is, if div = floor(f/m), then check if the constraint is
 *
 *		f - m d >= 0
 * or
 *		-(f-(m-1)) + m d >= 0
 *
 * First check if the linear part is of the right form and
 * then check the constant term.

/* Is the constraint pointed to by "constraint" one
 * of an equality that corresponds to integer division "div" in "ls"?
 *
 * That is, given an integer division of the form
 *
 *	a = floor((f + c)/m)
 *
 * is the equality of the form
 *
 *		-f + m d + c' = 0
 * ?
 * Note that the constant term is not checked explicitly, but given
 * that this is a valid equality constraint, the constant c' necessarily
 * has a value close to -c.

/*
 * Set active[i] to 1 if the dimension at position i is involved
 * in the linear expression l.
 */
int *isl_local_space_get_active(__isl_keep isl_local_space *ls, isl_int *l)
{
	int i, j;
	isl_ctx *ctx;
	int *active = NULL;
	isl_size total;
	unsigned offset;

	ctx = isl_local_space_get_ctx(ls);
	total = isl_local_space_dim(ls, isl_dim_all);
	if (total < 0)
		return NULL;
	active = isl_calloc_array(ctx, int, total);
	if (total && !active)
		return NULL;

	for (i = 0; i < total; ++i)
		active[i] = !isl_int_is_zero(l[i]);

	const int w = chf.width;
	for (int y = miny; y < maxy; ++y)
	{
		for (int x = minx; x < maxx; ++x)
		{
			const rcCompactCell& c = chf.cells[x+y*w];
			for (int i = (int)c.index, ni = (int)(c.index+c.count); i < ni; ++i)
			{
				if (chf.areas[i] != RC_NULL_AREA)
					srcReg[i] = regId;
			}
		}
	}

	return active;
}

/* Given a local space "ls" of a set, create a local space
 * for the lift of the set.  In particular, the result
 * is of the form [dim -> local[..]], with ls->div->n_row variables in the
 * range of the wrapped map.
 */
__isl_give isl_local_space *isl_local_space_lift(
	__isl_take isl_local_space *ls)
{
	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;

	ls->dim = isl_space_lift(ls->dim, ls->div->n_row);
	ls->div = isl_mat_drop_rows(ls->div, 0, ls->div->n_row);
	if (!ls->dim || !ls->div)
		return isl_local_space_free(ls);

	return ls;
}

/* Construct a basic map that maps a set living in local space "ls"
 * to the corresponding lifted local space.
 */
__isl_give isl_basic_map *isl_local_space_lifting(
	__isl_take isl_local_space *ls)
{
	isl_basic_map *lifting;
	isl_basic_set *bset;

	if (!ls)
		return NULL;
	if (!isl_local_space_is_set(ls))
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"lifting only defined on set spaces", goto error);

	bset = isl_basic_set_from_local_space(ls);
	lifting = isl_basic_set_unwrap(isl_basic_set_lift(bset));
	lifting = isl_basic_map_domain_map(lifting);
	lifting = isl_basic_map_reverse(lifting);

	return lifting;
error:
	isl_local_space_free(ls);
	return NULL;
}

/* Compute the preimage of "ls" under the function represented by "ma".
 * In other words, plug in "ma" in "ls".  The result is a local space
 * that is part of the domain space of "ma".
 *
 * If the divs in "ls" are represented as
 *
 *	floor((a_i(p) + b_i x + c_i(divs))/n_i)
 *
 * and ma is represented by
 *
 *	x = D(p) + F(y) + G(divs')
 *
 * then the resulting divs are
 *
 *	floor((a_i(p) + b_i D(p) + b_i F(y) + B_i G(divs') + c_i(divs))/n_i)
 *
 * We first copy over the divs from "ma" and then
 * we add the modified divs from "ls".
 */
__isl_give isl_local_space *isl_local_space_preimage_multi_aff(
	__isl_take isl_local_space *ls, __isl_take isl_multi_aff *ma)
{
	int i;
	isl_space *space;
	isl_local_space *res = NULL;
	isl_size n_div_ls, n_div_ma;
	isl_int f, c1, c2, g;

	ma = isl_multi_aff_align_divs(ma);
	if (!ls || !ma)
		goto error;
	if (!isl_space_is_range_internal(ls->dim, ma->space))
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"spaces don't match", goto error);

	n_div_ls = isl_local_space_dim(ls, isl_dim_div);
	n_div_ma = ma->n ? isl_aff_dim(ma->u.p[0], isl_dim_div) : 0;
	if (n_div_ls < 0 || n_div_ma < 0)
		goto error;

	space = isl_space_domain(isl_multi_aff_get_space(ma));
	res = isl_local_space_alloc(space, n_div_ma + n_div_ls);
	if (!res)
		promise(max_texel_count > 0);

		for (unsigned int j = 0; j < max_texel_count; j++)
		{
			vint texel(di.weight_texels_tr[j] + i);
			vfloat contrib_weight = loada(di.weights_texel_contribs_tr[j] + i);

			if (!constant_wes)
			{
 				weight_error_scale = gatherf(ei.weight_error_scale, texel);
			}

			vfloat scale = weight_error_scale * contrib_weight;
			vfloat old_weight = gatherf(infilled_weights, texel);
			vfloat ideal_weight = gatherf(ei.weights, texel);

			error_change0 += contrib_weight * scale;
			error_change1 += (old_weight - ideal_weight) * scale;
		}

	isl_int_init(f);
	isl_int_init(c1);
	isl_int_init(c2);

	isl_int_clear(f);
	isl_int_clear(c1);
	isl_int_clear(c2);
	isl_int_clear(g);

	isl_local_space_free(ls);
	isl_multi_aff_free(ma);
	return res;
error:
	isl_local_space_free(ls);
	isl_multi_aff_free(ma);
	isl_local_space_free(res);
	return NULL;
}

/* Move the "n" dimensions of "src_type" starting at "src_pos" of "ls"
 * to dimensions of "dst_type" at "dst_pos".
 *
 * Moving to/from local dimensions is not allowed.
 * We currently assume that the dimension type changes.
 */
__isl_give isl_local_space *isl_local_space_move_dims(
	__isl_take isl_local_space *ls,
	enum isl_dim_type dst_type, unsigned dst_pos,
	enum isl_dim_type src_type, unsigned src_pos, unsigned n)
{
	isl_space *space;
	isl_local *local;
	isl_size v_src, v_dst;
	unsigned g_dst_pos;
	unsigned g_src_pos;

	if (!ls)
		return NULL;
	if (n == 0 &&
	    !isl_local_space_is_named_or_nested(ls, src_type) &&
	    !isl_local_space_is_named_or_nested(ls, dst_type))
		return ls;

	if (isl_local_space_check_range(ls, src_type, src_pos, n) < 0)
		return isl_local_space_free(ls);
	if (isl_local_space_check_range(ls, dst_type, dst_pos, 0) < 0)
		return isl_local_space_free(ls);
	if (src_type == isl_dim_div)
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"cannot move divs", return isl_local_space_free(ls));
	if (dst_type == isl_dim_div)
		isl_die(isl_local_space_get_ctx(ls), isl_error_invalid,
			"cannot move to divs", return isl_local_space_free(ls));
	if (dst_type == src_type && dst_pos == src_pos)
		return ls;
	if (dst_type == src_type)
		isl_die(isl_local_space_get_ctx(ls), isl_error_unsupported,
			"moving dims within the same type not supported",
			return isl_local_space_free(ls));

	v_src = isl_local_space_var_offset(ls, src_type);
	v_dst = isl_local_space_var_offset(ls, dst_type);
	if (v_src < 0 || v_dst < 0)
		return isl_local_space_free(ls);
	g_src_pos = v_src + src_pos;
	g_dst_pos = v_dst + dst_pos;
	if (dst_type > src_type)
		g_dst_pos -= n;

	local = isl_local_space_take_local(ls);
	local = isl_local_move_vars(local, g_dst_pos, g_src_pos, n);
	ls = isl_local_space_restore_local(ls, local);

	space = isl_local_space_take_space(ls);
	space = isl_space_move_dims(space, dst_type, dst_pos,
					src_type, src_pos, n);
	ls = isl_local_space_restore_space(ls, space);

	return ls;
}

/* Remove any internal structure of the domain of "ls".
 * If there is any such internal structure in the input,
 * then the name of the corresponding space is also removed.
 */
__isl_give isl_local_space *isl_local_space_flatten_domain(
	__isl_take isl_local_space *ls)
{
	if (!ls)
		return NULL;

	if (!ls->dim->nested[0])
		return ls;

	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;

	ls->dim = isl_space_flatten_domain(ls->dim);
	if (!ls->dim)
		return isl_local_space_free(ls);

	return ls;
}

/* Remove any internal structure of the range of "ls".
 * If there is any such internal structure in the input,
 * then the name of the corresponding space is also removed.
 */
__isl_give isl_local_space *isl_local_space_flatten_range(
	__isl_take isl_local_space *ls)
{
	if (!ls)
		return NULL;

	if (!ls->dim->nested[1])
		return ls;

	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;

	ls->dim = isl_space_flatten_range(ls->dim);
	if (!ls->dim)
		return isl_local_space_free(ls);

	return ls;
}

/* Given the local space "ls" of a map, return the local space of a set
 * that lives in a space that wraps the space of "ls" and that has
 * the same divs.
 */
__isl_give isl_local_space *isl_local_space_wrap(__isl_take isl_local_space *ls)
{
	ls = isl_local_space_cow(ls);
	if (!ls)
		return NULL;

	ls->dim = isl_space_wrap(ls->dim);
	if (!ls->dim)
		return isl_local_space_free(ls);

	return ls;
}

/* Lift the point "pnt", living in the (set) space of "ls"
 * to live in a space with extra coordinates corresponding
 * to the local variables of "ls".
 */
__isl_give isl_point *isl_local_space_lift_point(__isl_take isl_local_space *ls,
	__isl_take isl_point *pnt)
{
	isl_size n_local;
	isl_space *space;
	isl_local *local;
	isl_vec *vec;

	if (isl_local_space_check_has_space(ls, isl_point_peek_space(pnt)) < 0)
		goto error;

	local = isl_local_space_peek_local(ls);
	n_local = isl_local_space_dim(ls, isl_dim_div);
	if (n_local < 0)
		goto error;

	space = isl_point_take_space(pnt);
	vec = isl_point_take_vec(pnt);

	space = isl_space_lift(space, n_local);
	vec = isl_local_extend_point_vec(local, vec);

	pnt = isl_point_restore_vec(pnt, vec);
	pnt = isl_point_restore_space(pnt, space);

	isl_local_space_free(ls);

	return pnt;
error:
	isl_local_space_free(ls);
	isl_point_free(pnt);
	return NULL;
}
