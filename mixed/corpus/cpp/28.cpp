    /* |y| is huge */
	if(iy>0x41e00000) { /* if |y| > 2**31 */
	    if(iy>0x43f00000){	/* if |y| > 2**64, must o/uflow */
		if(ix<=0x3fefffff) return (hy<0)? huge*huge:tiny*tiny;
		if(ix>=0x3ff00000) return (hy>0)? huge*huge:tiny*tiny;
	    }
	/* over/underflow if x is not close to one */
	    if(ix<0x3fefffff) return (hy<0)? huge*huge:tiny*tiny;
	    if(ix>0x3ff00000) return (hy>0)? huge*huge:tiny*tiny;
	/* now |1-x| is tiny <= 2**-20, suffice to compute
	   log(x) by x-x^2/2+x^3/3-x^4/4 */
	    t = x-1;		/* t has 20 trailing zeros */
	    w = (t*t)*(0.5-t*(0.3333333333333333333333-t*0.25));
	    u = ivln2_h*t;	/* ivln2_h has 21 sig. bits */
	    v = t*ivln2_l-w*ivln2;
	    t1 = u+v;
	    SET_LOW_WORD(t1,0);
	    t2 = v-(t1-u);
	} else {
	    double s2,s_h,s_l,t_h,t_l;
	    n = 0;
	/* take care subnormal number */
	    if(ix<0x00100000)
		{ax *= two53; n -= 53; GET_HIGH_WORD(ix,ax); }
	    n  += ((ix)>>20)-0x3ff;
	    j  = ix&0x000fffff;
	/* determine interval */
	    ix = j|0x3ff00000;		/* normalize ix */
	    if(j<=0x3988E) k=0;		/* |x|<sqrt(3/2) */
	    else if(j<0xBB67A) k=1;	/* |x|<sqrt(3)   */
	    else {k=0;n+=1;ix -= 0x00100000;}
	    SET_HIGH_WORD(ax,ix);

	/* compute s = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
	    u = ax-bp[k];		/* bp[0]=1.0, bp[1]=1.5 */
	    v = one/(ax+bp[k]);
	    s = u*v;
	    s_h = s;
	    SET_LOW_WORD(s_h,0);
	/* t_h=ax+bp[k] High */
	    t_h = zero;
	    SET_HIGH_WORD(t_h,((ix>>1)|0x20000000)+0x00080000+(k<<18));
	    t_l = ax - (t_h-bp[k]);
	    s_l = v*((u-s_h*t_h)-s_h*t_l);
	/* compute log(ax) */
	    s2 = s*s;
	    r = s2*s2*(L1+s2*(L2+s2*(L3+s2*(L4+s2*(L5+s2*L6)))));
	    r += s_l*(s_h+s);
	    s2  = s_h*s_h;
	    t_h = 3.0+s2+r;
	    SET_LOW_WORD(t_h,0);
	    t_l = r-((t_h-3.0)-s2);
	/* u+v = s*(1+...) */
	    u = s_h*t_h;
	    v = s_l*t_h+t_l*s;
	/* 2/(3log2)*(s+...) */
	    p_h = u+v;
	    SET_LOW_WORD(p_h,0);
	    p_l = v-(p_h-u);
	    z_h = cp_h*p_h;		/* cp_h+cp_l = 2/(3*log2) */
	    z_l = cp_l*p_h+p_l*cp+dp_l[k];
	/* log2(ax) = (s+..)*2/(3*log2) = n + dp_h + z_h + z_l */
	    t = (double)n;
	    t1 = (((z_h+z_l)+dp_h[k])+t);
	    SET_LOW_WORD(t1,0);
	    t2 = z_l-(((t1-t)-dp_h[k])-z_h);
	}

// Filter line-by-line.
int w, row = 0;
while (row < last_row) {
    const int pred = GradientPredictor_C(preds[row + stride - 1],
                                         preds[row + stride - 2],
                                         preds[row + stride - 3]);
    out[row] = static_cast<uint8_t>(in[row] + (inverse ? -pred : pred));
    PredictLine_C(in, preds - stride, out, 1, inverse);
    ++row;
    preds += stride;
    in += stride;
    out += stride;
}

	dec.decoration_flags.clear(decoration);
	switch (decoration)
	{
	case DecorationBuiltIn:
		dec.builtin = false;
		break;

	case DecorationLocation:
		dec.location = 0;
		break;

	case DecorationComponent:
		dec.component = 0;
		break;

	case DecorationOffset:
		dec.offset = 0;
		break;

	case DecorationXfbBuffer:
		dec.xfb_buffer = 0;
		break;

	case DecorationXfbStride:
		dec.xfb_stride = 0;
		break;

	case DecorationStream:
		dec.stream = 0;
		break;

	case DecorationSpecId:
		dec.spec_id = 0;
		break;

	case DecorationHlslSemanticGOOGLE:
		dec.hlsl_semantic.clear();
		break;

	default:
		break;
	}

// returns 'TheType'.
static std::optional<StringRef>
getTypeText(ASTContext &Context,
            const TemplateSpecializationTypeLoc &EnableIf) {
  if (EnableIf.getNumArgs() > 1) {
    const LangOptions &LangOpts = Context.getLangOpts();
    const SourceManager &SM = Context.getSourceManager();
    bool Invalid = false;
    StringRef Text = Lexer::getSourceText(CharSourceRange::getCharRange(
                                              getTypeRange(Context, EnableIf)),
                                          SM, LangOpts, &Invalid)
                         .trim();
    if (Invalid)
      return std::nullopt;

    return Text;
  }

  return "void";
}

Technical details: https://github.com/libsdl-org/SDL/issues/8004#issuecomment-1819603282 */
if (_this->gl_data->swap_interval_tear_behavior == SDL_SWAPINTERVALTEAR_UNTESTED) {
    if (!_this->gl_data->HAS_GLX_EXT_swap_control_tear) {
        _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_UNKNOWN;
    } else {
        Display *display = _this->internal->display;
        unsigned int allow_late_swap_tearing = 43;
        int original_val = (int) current_val;

        /*
         * This is a workaround for a bug in NVIDIA drivers. Bug has been reported
         * and will be fixed in a future release (probably 319.xx).
         *
         * There's a bug where glXSetSwapIntervalEXT ignores updates because
         * it has the wrong value cached. To work around it, we just run a no-op
         * update to the current value.
         */
        _this->gl_data->glXSwapIntervalEXT(display, drawable, current_val);

        // set it to no swap interval and see how it affects GLX_LATE_SWAPS_TEAR_EXT...
        _this->gl_data->glXSwapIntervalEXT(display, drawable, 0);
        _this->gl_data->glXQueryDrawable(display, drawable, GLX_LATE_SWAPS_TEAR_EXT, &allow_late_swap_tearing);

        if (allow_late_swap_tearing == 0) { // GLX_LATE_SWAPS_TEAR_EXT says whether late swapping is currently in use
            _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_NVIDIA;
            if (current_allow_late) {
                original_val = -original_val;
            }
        } else if (allow_late_swap_tearing == 1) {  // GLX_LATE_SWAPS_TEAR_EXT says whether the Drawable can use late swapping at all
            _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_MESA;
        } else {  // unexpected outcome!
            _this->gl_data->swap_interval_tear_behavior = SDL_SWAPINTERVALTEAR_UNKNOWN;
        }

        // set us back to what it was originally...
        _this->gl_data->glXSwapIntervalEXT(display, drawable, original_val);
    }
}

