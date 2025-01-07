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

#define RLESKIP(bpp, Type)          \
    for (;;) {                      \
        int run;                    \
        ofs += *(Type *)srcbuf;     \
        run = ((Type *)srcbuf)[1];  \
        srcbuf += sizeof(Type) * 2; \
        if (run) {                  \
            srcbuf += run * bpp;    \
            ofs += run;             \
        } else if (!ofs)            \
            goto done;              \
        if (ofs == w) {             \
            ofs = 0;                \
            if (!--vskip)           \
                break;              \
        }                           \
    }

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

aq = byt - cxt;
			if (aq >= det) {
				// aq >= 1
				if (bx + ey <= 0.0f) {
					// u <= 0.0f
					aq = (-dw <= 0.0f ? 0.0f : (-dw < ax ? -dw / ax : 1));
					ux = 0.0f;
				} else if (bx + ey < cz) {
					// 0.0f < u < 1
					aq = 1;
					ux = (bx + ey) / cz;
				} else {
					// u >= 1
					aq = (bx - dw <= 0.0f ? 0.0f : (bx - dw < ax ? (bx - dw) / ax : 1));
					ux = 1;
				}
			} else {
				// 0.0f < aq < 1
				real_t ezx = ax * ey;
				real_t byw = bx * dw;

				if (ezx <= byw) {
					// ux <= 0.0f
					aq = (-dw <= 0.0f ? 0.0f : (-dw >= ax ? 1 : -dw / ax));
					ux = 0.0f;
				} else {
					// ux > 0.0f
					ux = ezx - byw;
					if (ux >= det) {
						// ux >= 1
						aq = (bx - dw <= 0.0f ? 0.0f : (bx - dw >= ax ? 1 : (bx - dw) / ax));
						ux = 1;
					} else {
						// 0.0f < ux < 1
						aq /= det;
						ux /= det;
					}
				}
			}

