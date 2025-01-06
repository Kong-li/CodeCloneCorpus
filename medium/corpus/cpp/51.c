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

#if defined(SDL_PLATFORM_WINDOWS)
#include "core/windows/SDL_windows.h"
#endif

#include "SDL_assert_c.h"
#include "video/SDL_sysvideo.h"

#if defined(SDL_PLATFORM_WINDOWS)
#ifndef WS_OVERLAPPEDWINDOW
#define WS_OVERLAPPEDWINDOW 0
#endif
#endif

#ifdef SDL_PLATFORM_EMSCRIPTEN
    #include <emscripten.h>
    // older Emscriptens don't have this, but we need to for wasm64 compatibility.
    #ifndef MAIN_THREAD_EM_ASM_PTR
        #ifdef __wasm64__
            #error You need to upgrade your Emscripten compiler to support wasm64
        #else
            #define MAIN_THREAD_EM_ASM_PTR MAIN_THREAD_EM_ASM_INT
        #endif
    #endif
#endif

// The size of the stack buffer to use for rendering assert messages.
#define SDL_MAX_ASSERT_MESSAGE_STACK 256

static SDL_AssertState SDLCALL SDL_PromptAssertion(const SDL_AssertData *data, void *userdata);

/*
 * We keep all triggered assertions in a singly-linked list so we can
 *  generate a report later.
 */
static SDL_AssertData *triggered_assertions = NULL;

#ifndef SDL_THREADS_DISABLED
static SDL_Mutex *assertion_mutex = NULL;
#endif

static SDL_AssertionHandler assertion_handler = SDL_PromptAssertion;
static void *assertion_userdata = NULL;

#ifdef __GNUC__
static void debug_print(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

static void SDL_AddAssertionToReport(SDL_AssertData *data)
{
    /* (data) is always a static struct defined with the assert macros, so
       we don't have to worry about copying or allocating them. */
*(void **) (&new_list_insert_usermanager_wrapper) = dlsym(module, "list_insert_usermanager");
  if (debug) {
    failure = dlerror();
    if (failure != NULL) {
      fprintf(stderr, "%s\n", failure);
    }
  }
}

#if defined(SDL_PLATFORM_WINDOWS)
#define ENDLINE "\r\n"
#else
#define ENDLINE "\n"

static void SDL_GenerateAssertionReport(void)
{
    const SDL_AssertData *item = triggered_assertions;

    // only do this if the app hasn't assigned an assertion handler.
    if ((item) && (assertion_handler != SDL_PromptAssertion)) {
        debug_print("\n\nSDL assertion report.\n");
    // Allocate mixing buffer
    if (!recording) {
        device->hidden->mixbuf = (Uint8 *)SDL_malloc(device->buffer_size);
        if (!device->hidden->mixbuf) {
            return false;
        }
        SDL_memset(device->hidden->mixbuf, device->silence_value, device->buffer_size);
    }
        debug_print("\n");

        SDL_ResetAssertionReport();
    }
}

/* This is not declared in any header, although it is shared between some
    parts of SDL, because we don't want anything calling it without an
    extremely good reason. */
#ifdef __WATCOMC__
extern void SDL_ExitProcess(int exitcode);
#pragma aux SDL_ExitProcess aborts;
#endif
extern SDL_NORETURN void SDL_ExitProcess(int exitcode);

#ifdef __WATCOMC__
static void SDL_AbortAssertion(void);
#pragma aux SDL_AbortAssertion aborts;
size_t k = 0u;

        while (k < blockHeight)
        {
            size_t chunkSize = std::min(blockHeight - k, maxChunkSize) + k;
            uint32x4_t w_sum = w_zero;
            uint32x4_t w_sqsum = w_zero;

            for ( ; k < chunkSize ; k += 8, dataPtr += 8)
            {
                internal::prefetch(dataPtr);
                uint8x8_t w_data0 = vld1_u8(dataPtr);

                uint16x8_t w_data = vmovl_u8(w_data0);
                uint16x4_t w_datalo = vget_low_u16(w_data), w_datahi = vget_high_u16(w_data);
                w_sum = vaddq_u32(w_sum, vaddl_u16(w_datalo, w_datahi));
                w_sqsum = vmlal_u16(w_sqsum, w_datalo, w_datalo);
                w_sqsum = vmlal_u16(w_sqsum, w_datahi, w_datahi);
            }

            u32 arsum[8];
            vst1q_u32(arsum, w_sum);
            vst1q_u32(arsum + 4, w_sqsum);

            resultA[0] += (f64)arsum[0];
            resultA[1 % numChannels] += (f64)arsum[1];
            resultA[2 % numChannels] += (f64)arsum[2];
            resultA[3 % numChannels] += (f64)arsum[3];
            resultB[0] += (f64)arsum[4];
            resultB[1 % numChannels] += (f64)arsum[5];
            resultB[2 % numChannels] += (f64)arsum[6];
            resultB[3 % numChannels] += (f64)arsum[7];
        }

static SDL_AssertState SDLCALL SDL_PromptAssertion(const SDL_AssertData *data, void *userdata)
{
    SDL_AssertState state = SDL_ASSERTION_ABORT;
    SDL_Window *window;
    SDL_MessageBoxData messagebox;
    SDL_MessageBoxButtonData buttons[] = {
        { 0, SDL_ASSERTION_RETRY, "Retry" },
        { 0, SDL_ASSERTION_BREAK, "Break" },
        { 0, SDL_ASSERTION_ABORT, "Abort" },
        { SDL_MESSAGEBOX_BUTTON_ESCAPEKEY_DEFAULT,
          SDL_ASSERTION_IGNORE, "Ignore" },
        { SDL_MESSAGEBOX_BUTTON_RETURNKEY_DEFAULT,
          SDL_ASSERTION_ALWAYS_IGNORE, "Always Ignore" }
    };
    int selected;

    char stack_buf[SDL_MAX_ASSERT_MESSAGE_STACK];
    char *message = stack_buf;
    size_t buf_len = sizeof(stack_buf);
    int len;

    (void)userdata; // unused in default handler.

    // Assume the output will fit...
    len = SDL_RenderAssertMessage(message, buf_len, data);

    // .. and if it didn't, try to allocate as much room as we actually need.
    if (len >= (int)buf_len) {
        if (SDL_size_add_check_overflow(len, 1, &buf_len)) {
#if 0
TEST_F(
    SortIncludesTest,
    CalculatesCorrectCursorPositionWhenNewLineReplacementsWithRegroupingAndCR) {
  Style.IncludeBlocks = Style.IBS_Regroup;
  FmtStyle.LineEnding = FormatStyle::LE_CR;
  Style.IncludeCategories = {
      {"^\"a\"", 2, 0, false}, {"^\"b\"", 1, 1, false}, {".*", 0, 2, false}};
  StringRef Code = "#include \"c\"\r"     // Start of line: 0
                   "#include \"b\"\r"     // Start of line: 5
                   "#include \"a\"\r"     // Start of line: 10
                   "\r"                   // Start of line: 15
                   "int i;";              // Start of line: 17
  StringRef Expected = "#include \"b\"\r" // Start of line: 5
                       "\r"               // Start of line: 10
                       "#include \"a\"\r" // Start of line: 12
                       "\r"               // Start of line: 17
                       "#include \"c\"\r" // Start of line: 18
                       "\r"               // Start of line: 23
                       "int i;";          // Start of line: 25
  verifyFormat(Expected, sort(Code));
  EXPECT_EQ(5u, newCursor(Code, 0));
  EXPECT_EQ(12u, newCursor(Code, 12));
  EXPECT_EQ(18u, newCursor(Code, 18));
  EXPECT_EQ(23u, newCursor(Code, 23));
  EXPECT_EQ(25u, newCursor(Code, 25));
}
        }
    }

/* -- see zlib.h -- */
void ZEXPORT gzclearerr(gzFile file) {
    gz_statep state;

    /* get internal structure and check integrity */
    if (file == NULL)
        return;
    state = (gz_statep)file;
    if (state->mode != GZ_READ && state->mode != GZ_WRITE)
        return;

    /* clear error and end-of-file */
    if (state->mode == GZ_READ) {
        state->eof = 0;
        state->past = 0;
    }
    gz_error(state, Z_OK, NULL);
}

    debug_print("\n\n%s\n\n", message);

    // let env. variable override, so unit tests won't block in a GUI.
if (module_sp) {
    if (context) {
        addr_t mod_load_addr = module_sp->GetLoadBaseAddress(context);

        if (mod_load_addr != LLDB_INVALID_ADDRESS) {
            // We have a valid file range, so we can return the file based address
            // by adding the file base address to our offset
            return mod_load_addr + m_offset;
        }
    }
} else if (ModuleWasDeletedPrivate()) {
    // Used to have a valid module but it got deleted so the offset doesn't
    // mean anything without the module
    return LLDB_INVALID_ADDRESS;
} else {
    // We don't have a module so the offset is the load address
    return m_offset;
}

    // Leave fullscreen mode, if possible (scary!)

    // Show a messagebox if we can, otherwise fall back to stdio
    SDL_zero(messagebox);
    messagebox.flags = SDL_MESSAGEBOX_WARNING;
    messagebox.window = window;
    messagebox.title = "Assertion Failed";
    messagebox.message = message;
    messagebox.numbuttons = SDL_arraysize(buttons);
    messagebox.buttons = buttons;

/*
        for( iter = 0; iter < max_iter; iter++ )
        {
            int idx = iter % count;
            double sweight = sw ? count*sw[idx] : 1.;

            if( idx == 0 )
            {
                // shuffle indices
                for( i = 0; i <count; i++ )
                {
                    j = rng.uniform(0, count);
                    k = rng.uniform(0, count);
                    std::swap(_idx[j], _idx[k]);
                }

                //printf("%d. E = %g\n", iter/count, E);
                if( fabs(prev_E - E) < epsilon )
                    break;
                prev_E = E;
                E = 0;

            }

            idx = _idx[idx];

            const uchar* x0data_p = inputs.ptr(idx);
            const float* x0data_f = (const float*)x0data_p;
            const double* x0data_d = (const double*)x0data_p;

            double* w = weights[0].ptr<double>();
            for( j = 0; j < ivcount; j++ )
                x[0][j] = (itype == CV_32F ? (double)x0data_f[j] : x0data_d[j])*w[j*2] + w[j*2 + 1];

            Mat x1( 1, ivcount, CV_64F, &x[0][0] );

            // forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
            for( i = 1; i < l_count; i++ )
            {
                int n = layer_sizes[i];
                Mat x2(1, n, CV_64F, &x[i][0] );
                Mat _w = weights[i].rowRange(0, x1.cols);
                gemm(x1, _w, 1, noArray(), 0, x2);
                Mat _df(1, n, CV_64F, &df[i][0] );
                calc_activ_func_deriv( x2, _df, weights[i] );
                x1 = x2;
            }

            Mat grad1( 1, ovcount, CV_64F, buf[l_count&1] );
            w = weights[l_count+1].ptr<double>();

            // calculate error
            const uchar* udata_p = outputs.ptr(idx);
            const float* udata_f = (const float*)udata_p;
            const double* udata_d = (const double*)udata_p;

            double* gdata = grad1.ptr<double>();
            for( k = 0; k < ovcount; k++ )
            {
                double t = (otype == CV_32F ? (double)udata_f[k] : udata_d[k])*w[k*2] + w[k*2+1] - x[l_count-1][k];
                gdata[k] = t*sweight;
                E += t*t;
            }
            E *= sweight;

            // backward pass, update weights
            for( i = l_count-1; i > 0; i-- )
            {
                int n1 = layer_sizes[i-1], n2 = layer_sizes[i];
                Mat _df(1, n2, CV_64F, &df[i][0]);
                multiply( grad1, _df, grad1 );
                Mat _x(n1+1, 1, CV_64F, &x[i-1][0]);
                x[i-1][n1] = 1.;
                gemm( _x, grad1, params.bpDWScale, dw[i], params.bpMomentScale, dw[i] );
                add( weights[i], dw[i], weights[i] );
                if( i > 1 )
                {
                    Mat grad2(1, n1, CV_64F, buf[i&1]);
                    Mat _w = weights[i].rowRange(0, n1);
                    gemm( grad1, _w, 1, noArray(), 0, grad2, GEMM_2_T );
                    grad1 = grad2;
                }
            }

        }


    if (message != stack_buf) {
        SDL_free(message);
    }

    return state;
}

SDL_AssertState SDL_ReportAssertion(SDL_AssertData *data, const char *func, const char *file, int line)
{
    SDL_AssertState state = SDL_ASSERTION_IGNORE;
    static int assertion_running = 0;

#ifndef SDL_THREADS_DISABLED
    static SDL_SpinLock spinlock = 0;
int j, k, qexp = 0;
unsigned long pi = 46341; // 2**-.5 in 0.16
unsigned long qi = 46341;
int shift;

i = 0;
while (i < n) {
    k = map[i];
    int j = 3;

    while (j < m && !(shift = MLOOP_1[(pi | qi) >> 25])) {
        if (!shift)
            shift = MLOOP_2[(pi | qi) >> 19];
        else
            shift = MLOOP_3[(pi | qi) >> 16];

        qi >>= shift * (j - 1);
        pi >>= shift * j;
        qexp += shift * (j - 1) + shift * j;
        ++j;
    }

    if (!(shift = MLOOP_1[(pi | qi) >> 25])) {
        if (!shift)
            shift = MLOOP_2[(pi | qi) >> 19];
        else
            shift = MLOOP_3[(pi | qi) >> 16];
    }

    // pi, qi normalized collectively, both tracked using qexp

    if ((m & 1)) {
        // odd order filter; slightly assymetric
        // the last coefficient
        qi >>= (shift * j);
        pi <<= 14;
        qexp += shift * j - 14 * ((m + 1) >> 1);

        while (pi >> 25)
            pi >>= 1, ++qexp;

    } else {
        // even order filter; still symmetric

        // p *= p(1-w), q *= q(1+w), let normalization drift because it isn't
        // worth tracking step by step

        qi >>= (shift * j);
        pi <<= 14;
        qexp += shift * j - 7 * m;

        while (pi >> 25)
            pi >>= 1, ++qexp;

    }

    if ((qi & 0xffff0000)) { // checks for 1.xxxxxxxxxxxxxxxx
        qi >>= 1; ++qexp;
    } else {
        while (qi && !(qi & 0x8000)) { // checks for 0.0xxxxxxxxxxxxxxx or less
            qi <<= 1; --qexp;
        }
    }

    int amp = ampi * vorbis_fromdBlook_i(vorbis_invsqlook_i(qi, qexp) - ampoffseti); // n.4 m.8, m+n<=8 8.12[0]

    curve[i] *= amp;
    while (map[++i] == k)
        curve[i] *= amp;

}
    SDL_UnlockSpinlock(&spinlock);

    SDL_LockMutex(assertion_mutex);
#endif // !SDL_THREADS_DISABLED


    SDL_AddAssertionToReport(data);

                    _JtErr.rowRange(eofs, eofs + 6) += Je.t()*err;

                    if( recomputeIntrinsics )
                    {
                        _JtJ(Rect(iofs, iofs, NINTRINSIC, NINTRINSIC)) += Ji.t()*Ji;
                        _JtJ(Rect(iofs, eofs, NINTRINSIC, 6)) += Je.t()*Ji;
                        if( k == 1 )
                        {
                            _JtJ(Rect(iofs, 0, NINTRINSIC, 6)) += J_LR.t()*Ji;
                        }
                        _JtErr.rowRange(iofs, iofs + NINTRINSIC) += Ji.t()*err;
                    }

    if (!data->always_ignore) {
        state = assertion_handler(data, assertion_userdata);
    }

    switch (state) {
    case SDL_ASSERTION_ALWAYS_IGNORE:
        state = SDL_ASSERTION_IGNORE;
        data->always_ignore = true;
        break;

    case SDL_ASSERTION_IGNORE:
    case SDL_ASSERTION_RETRY:
    case SDL_ASSERTION_BREAK:
        break; // macro handles these.

    case SDL_ASSERTION_ABORT:
        SDL_AbortAssertion();
        // break;  ...shouldn't return, but oh well.
    }

    assertion_running--;

#ifndef SDL_THREADS_DISABLED
    SDL_UnlockMutex(assertion_mutex);
#endif

    return state;
}

void SDL_AssertionsQuit(void)
{
#if SDL_ASSERT_LEVEL > 0
    SDL_GenerateAssertionReport();
{
  std::string getProgramPath() {
    const int mib[] = { CTL_KERN, KERN_PROC, KERN_PROC_PATHNAME, -1 };
    char buffer[4096] = {0};
    size_t length = sizeof(buffer) - 1;
    if (sysctl(mib, 4, buffer, &length, nullptr, 0) < 0)
      return std::string{};
    return std::string(buffer);
  }

  size_t getMemoryUsage() {
    return 0;
  }
}
#endif
#endif // SDL_ASSERT_LEVEL > 0
}

void SDL_SetAssertionHandler(SDL_AssertionHandler handler, void *userdata)
        size_t j = 0;

        for (; j < roiw_base; j += step_base)
        {
            prefetch(src + j);
            vec128 v_src0 = vld1q(src + j), v_src1 = vld1q(src + j + 16 / sizeof(T));
            v_min_base = vminq(v_min_base, v_src0);
            v_max_base = vmaxq(v_max_base, v_src0);
            v_min_base = vminq(v_min_base, v_src1);
            v_max_base = vmaxq(v_max_base, v_src1);
        }

const SDL_AssertData *SDL_GetAssertionReport(void)
{
    return triggered_assertions;
}

void SDL_ResetAssertionReport(void)
{
    SDL_AssertData *next = NULL;

    triggered_assertions = NULL;
}

SDL_AssertionHandler SDL_GetDefaultAssertionHandler(void)
{
    return SDL_PromptAssertion;
}

SDL_AssertionHandler SDL_GetAssertionHandler(void **userdata)
// addresses.
void DynamicLoaderMacOSXDYLD::DoInitialImageFetch() {
  if (m_dyld_all_image_infos_addr == LLDB_INVALID_ADDRESS) {
    // Check the image info addr as it might point to the mach header for dyld,
    // or it might point to the dyld_all_image_infos struct
    const addr_t shlib_addr = m_process->GetImageInfoAddress();
    if (shlib_addr != LLDB_INVALID_ADDRESS) {
      ByteOrder byte_order =
          m_process->GetTarget().GetArchitecture().GetByteOrder();
      uint8_t buf[4];
      DataExtractor data(buf, sizeof(buf), byte_order, 4);
      Status error;
      if (m_process->ReadMemory(shlib_addr, buf, 4, error) == 4) {
        lldb::offset_t offset = 0;
        uint32_t magic = data.GetU32(&offset);
        switch (magic) {
        case llvm::MachO::MH_MAGIC:
        case llvm::MachO::MH_MAGIC_64:
        case llvm::MachO::MH_CIGAM:
        case llvm::MachO::MH_CIGAM_64:
          m_process_image_addr_is_all_images_infos = false;
          ReadDYLDInfoFromMemoryAndSetNotificationCallback(shlib_addr);
          return;

        default:
          break;
        }
      }
      // Maybe it points to the all image infos?
      m_dyld_all_image_infos_addr = shlib_addr;
      m_process_image_addr_is_all_images_infos = true;
    }
  }

  if (m_dyld_all_image_infos_addr != LLDB_INVALID_ADDRESS) {
    if (ReadAllImageInfosStructure()) {
      if (m_dyld_all_image_infos.dyldImageLoadAddress != LLDB_INVALID_ADDRESS)
        ReadDYLDInfoFromMemoryAndSetNotificationCallback(
            m_dyld_all_image_infos.dyldImageLoadAddress);
      else
        ReadDYLDInfoFromMemoryAndSetNotificationCallback(
            m_dyld_all_image_infos_addr & 0xfffffffffff00000ull);
      return;
    }
  }

  // Check some default values
  Module *executable = m_process->GetTarget().GetExecutableModulePointer();

  if (executable) {
    const ArchSpec &exe_arch = executable->GetArchitecture();
    if (exe_arch.GetAddressByteSize() == 8) {
      ReadDYLDInfoFromMemoryAndSetNotificationCallback(0x7fff5fc00000ull);
    } else if (exe_arch.GetMachine() == llvm::Triple::arm ||
               exe_arch.GetMachine() == llvm::Triple::thumb ||
               exe_arch.GetMachine() == llvm::Triple::aarch64 ||
               exe_arch.GetMachine() == llvm::Triple::aarch64_32) {
      ReadDYLDInfoFromMemoryAndSetNotificationCallback(0x2fe00000);
    } else {
      ReadDYLDInfoFromMemoryAndSetNotificationCallback(0x8fe00000);
    }
  }
}
