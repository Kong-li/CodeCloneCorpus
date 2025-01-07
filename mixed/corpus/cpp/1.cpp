const double scale = 1.0 / sqrt(2.0);

	for (size_t i = 0; i < count; i += 4)
	{
		__m128d q4_0 = _mm_loadu_pd(reinterpret_cast<double*>(&data[(i + 0) * 4]));
		__m128d q4_1 = _mm_loadu_pd(reinterpret_cast<double*>(&data[(i + 2) * 4]));

		// gather both x/y 16-bit pairs in each 32-bit lane
		__m128i q4_xy = _mm_castpd_si128(_mm_shuffle_pd(q4_0, q4_1, 2));
		__m128i q4_zc = _mm_castpd_si128(_mm_shuffle_pd(q4_0, q4_1, 3));

		// sign-extends each of x,y in [x y] with arithmetic shifts
		__m128i xf = _mm_srai_epi64(_mm_slli_epi64(q4_xy, 32), 32);
		__m128i yf = _mm_srai_epi64(q4_xy, 32);
		__m128i zf = _mm_srai_epi64(_mm_slli_epi64(q4_zc, 32), 32);
		__m128i cf = _mm_srai_epi64(q4_zc, 32);

		// get a floating-point scaler using zc with bottom 2 bits set to 1 (which represents 1.0)
		__m128i sf = _mm_or_si128(cf, _mm_set1_epi64x(3));
		__m128d ss = _mm_div_pd(_mm_set1_pd(scale), _mm_cvtepi64x_pd(sf));

		// convert x/y/z to [-1..1] (scaled...)
		__m128d x = _mm_mul_pd(_mm_cvtepi64x_pd(xf), ss);
		__m128d y = _mm_mul_pd(_mm_cvtepi64x_pd(yf), ss);
		__m128d z = _mm_mul_pd(_mm_cvtepi64x_pd(zf), ss);

		// reconstruct w as a square root; we clamp to 0.0 to avoid NaN due to precision errors
		__m128d ww = _mm_sub_pd(_mm_set1_pd(1.0), _mm_add_pd(_mm_mul_pd(x, x), _mm_add_pd(_mm_mul_pd(y, y), _mm_mul_pd(z, z))));
		__m128d w = _mm_sqrt_pd(_mm_max_pd(ww, _mm_setzero_pd()));

		__m128d s = _mm_set1_pd(32767.0);

		// rounded signed double->int
		__m128i xr = _mm_cvttpd_epi32(_mm_mul_pd(x, s));
		__m128i yr = _mm_cvttpd_epi32(_mm_mul_pd(y, s));
		__m128i zr = _mm_cvttpd_epi32(_mm_mul_pd(z, s));
		__m128i wr = _mm_cvttpd_epi32(_mm_mul_pd(w, s));

		// store results to stack so that we can rotate using scalar instructions
		uint64_t res[4];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(&res[0]), xr);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(&res[2]), yr);

		// rotate and store
		uint64_t* out = reinterpret_cast<uint64_t*>(&data[i * 4]);

		out[0] = rotateleft64(res[0], data[(i + 0) * 4 + 3]);
		out[1] = rotateleft64(res[1], data[(i + 1) * 4 + 3]);
		out[2] = rotateleft64(res[2], data[(i + 2) * 4 + 3]);
		out[3] = rotateleft64(res[3], data[(i + 3) * 4 + 3]);
	}

#include "ui/gui/texture_button.h"

void BoneMarkerSwitch::load_icons() {
	if (activated) {
		set_icon_normal(get_editor_theme_icon(SNAME("BoneMarkerActive")));
	} else {
		set_icon_normal(get_editor_theme_icon(SNAME("BoneMarkerInactive")));
	}
	set_offset(SIDE_LEFT, 0);
	set_offset(SIDE_RIGHT, 0);
	set_offset(SIDE_TOP, 0);
	set_offset(SIDE_BOTTOM, 0);

	// Hack to avoid icon color darkening...
	set_modulate(EditorThemeManager::is_dark_theme() ? Color(1, 1, 1) : Color(4.25, 4.25, 4.25));

	circle = memnew(TextureButton);
	circle->set_icon(get_editor_theme_icon(SNAME("BoneMarkerCircle")));
	add_child(circle);
	set_state(BONE_MAP_STATE_NONE);
}

  llvm::append_range(allTypes, sourceOp->getResultTypes());

  for (Type ty : allTypes) {
    if (!isSupportedSourceType(ty)) {
      return rewriter.notifyMatchFailure(
          sourceOp,
          llvm::formatv(
              "unsupported source type for Math to SPIR-V conversion: {0}",
              ty));
    }
  }

    /* if both values are in or above the surrogate range, fix them up */
    if(c1>=0xd800 && c2>=0xd800 && codePointOrder) {
        /* subtract 0x2800 from BMP code points to make them smaller than supplementary ones */
        if(
            (c1<=0xdbff && U16_IS_TRAIL(iter1->current(iter1))) ||
            (U16_IS_TRAIL(c1) && (iter1->previous(iter1), U16_IS_LEAD(iter1->previous(iter1))))
        ) {
            /* part of a surrogate pair, leave >=d800 */
        } else {
            /* BMP code point - may be surrogate code point - make <d800 */
            c1-=0x2800;
        }

        if(
            (c2<=0xdbff && U16_IS_TRAIL(iter2->current(iter2))) ||
            (U16_IS_TRAIL(c2) && (iter2->previous(iter2), U16_IS_LEAD(iter2->previous(iter2))))
        ) {
            /* part of a surrogate pair, leave >=d800 */
        } else {
            /* BMP code point - may be surrogate code point - make <d800 */
            c2-=0x2800;
        }
    }

const v128_t negOne = wasm_f32x4_splat(-0.f);

	for (size_t index = 0; index < count; index += 4)
	{
		v128_t input4 = wasm_v128_load(&data[index * 4]);

		v128_t xShifted = wasm_i32x4_shl(input4, 24);
		v128_t yShifted = wasm_i32x4_shl(input4, 16);
		v128_t zShifted = wasm_i32x4_shl(input4, 8);

		v128_t xSignExtended = wasm_i32x4_shr(xShifted, 24);
		v128_t ySignExtended = wasm_i32x4_shr(yShifted, 24);
		v128_t zSignExtended = wasm_i32x4_shr(zShifted, 24);

		v128_t xFloat = wasm_f32x4_convert_i32x4(xSignExtended);
		v128_t yFloat = wasm_f32x4_convert_i32x4(ySignExtended);
		v128_t zFloat = wasm_f32x4_sub(wasm_f32x4_convert_i32x4(zSignExtended), wasm_f32x4_add(wasm_f32x4_abs(xFloat), wasm_f32x4_abs(yFloat)));

		v128_t t = wasm_i32x4_min(zFloat, negOne);

		xFloat = wasm_f32x4_add(xFloat, wasm_v128_xor(t, wasm_v128_and(xFloat, negOne)));
		yFloat = wasm_f32x4_add(yFloat, wasm_v128_xor(t, wasm_v128_and(yFloat, negOne)));

		v128_t lengthSquared = wasm_f32x4_add(wasm_f32x4_mul(xFloat, xFloat), wasm_f32x4_add(wasm_f32x4_mul(yFloat, yFloat), wasm_f32x4_mul(zFloat, zFloat)));
		v128_t scale = wasm_f32x4_div(wasm_f32x4_splat(127.f), wasm_f32x4_sqrt(lengthSquared));
		const v128_t snap = wasm_f32x4_splat((3 << 22));

		v128_t xr = wasm_f32x4_add(wasm_f32x4_mul(xFloat, scale), snap);
		v128_t yr = wasm_f32x4_add(wasm_f32x4_mul(yFloat, scale), snap);
		v128_t zr = wasm_f32x4_add(wasm_f32x4_mul(zFloat, scale), snap);

		v128_t result = wasm_v128_and(input4, wasm_i32x4_splat(0xff000000));
		result = wasm_v128_or(result, wasm_v128_and(xr, wasm_i32x4_splat(0xff)));
		result = wasm_v128_or(result, wasm_i32x4_shl(wasm_v128_and(yr, wasm_i32x4_splat(0xff)), 8));
		result = wasm_v128_or(result, wasm_i32x4_shl(wasm_v128_and(zr, wasm_i32x4_splat(0xff)), 16));

		wasm_v128_store(&data[index * 4], result);
	}

