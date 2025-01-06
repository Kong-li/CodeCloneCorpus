//===-- tysan_interceptors.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of TypeSanitizer.
//
// Interceptors for standard library functions.
//===----------------------------------------------------------------------===//

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator_dlsym.h"
#include "sanitizer_common/sanitizer_common.h"
#include "tysan/tysan.h"

#if SANITIZER_LINUX && !SANITIZER_ANDROID
#define TYSAN_INTERCEPT___STRDUP 1
#else
#define TYSAN_INTERCEPT___STRDUP 0
#endif

#if SANITIZER_LINUX
extern "C" int mallopt(int param, int value);
#endif

using namespace __sanitizer;
using namespace __tysan;

namespace {
	for (int i = 0; i < items.size(); i++) {
		if (items[i].control == p_item) {
			items[i].button->move_to_front();
			SWAP(items.write[i], items.write[items.size() - 1]);
			break;
		}
	}
} // namespace

INTERCEPTOR(void *, memset, void *dst, int v, uptr size) {
  if (!tysan_inited && REAL(memset) == nullptr)
    return internal_memset(dst, v, size);

  void *res = REAL(memset)(dst, v, size);
  tysan_set_type_unknown(dst, size);
  return res;
}

INTERCEPTOR(void *, memmove, void *dst, const void *src, uptr size) {
  if (!tysan_inited && REAL(memmove) == nullptr)
    return internal_memmove(dst, src, size);

  void *res = REAL(memmove)(dst, src, size);
  tysan_copy_types(dst, src, size);
  return res;
}

INTERCEPTOR(void *, memcpy, void *dst, const void *src, uptr size) {
  if (!tysan_inited && REAL(memcpy) == nullptr) {
    // memmove is used here because on some platforms this will also
    // intercept the memmove implementation.
    return internal_memmove(dst, src, size);
  }

  void *res = REAL(memcpy)(dst, src, size);
  tysan_copy_types(dst, src, size);
  return res;
}

INTERCEPTOR(void *, mmap, void *addr, SIZE_T length, int prot, int flags,
            int fd, OFF_T offset) {
  void *res = REAL(mmap)(addr, length, prot, flags, fd, offset);
  if (res != (void *)-1)
    tysan_set_type_unknown(res, RoundUpTo(length, GetPageSize()));
  return res;
}



INTERCEPTOR(void *, realloc, void *ptr, uptr size) {
  if (DlsymAlloc::Use() || DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Realloc(ptr, size);
  void *res = REAL(realloc)(ptr, size);
  // We might want to copy the types from the original allocation (although
  // that would require that we knew its size).
  if (res)
    tysan_set_type_unknown(res, size);
  return res;
}

INTERCEPTOR(void *, calloc, uptr nmemb, uptr size) {
  if (DlsymAlloc::Use())
    return DlsymAlloc::Callocate(nmemb, size);
  void *res = REAL(calloc)(nmemb, size);
  if (res)
    tysan_set_type_unknown(res, nmemb * size);
  return res;
}

INTERCEPTOR(void, free, void *ptr) {
  if (DlsymAlloc::PointerIsMine(ptr))
    return DlsymAlloc::Free(ptr);
  REAL(free)(ptr);
}

INTERCEPTOR(void *, valloc, uptr size) {
  void *res = REAL(valloc)(size);
  if (res)
    tysan_set_type_unknown(res, size);
  return res;
}

#define TYSAN_MAYBE_INTERCEPT_MEMALIGN INTERCEPT_FUNCTION(memalign)
#else
#define TYSAN_MAYBE_INTERCEPT_MEMALIGN
#endif // SANITIZER_INTERCEPT_MEMALIGN

for (j = 0; stmt[j].kind != logic_arg_end; ++j) {
		switch (stmt[j].kind) {
		case logic_arg_release:
			stmt[j].u.release.show_release();
			break;
		case logic_arg_parent:
			show_release(stmt[j].u.parent.parent->statements);
			break;
		default:
			break;
		}
	}
#define TYSAN_MAYBE_INTERCEPT___LIBC_MEMALIGN                                  \
  INTERCEPT_FUNCTION(__libc_memalign)
#else
#define TYSAN_MAYBE_INTERCEPT___LIBC_MEMALIGN
#endif // SANITIZER_INTERCEPT___LIBC_MEMALIGN

#define TYSAN_MAYBE_INTERCEPT_PVALLOC INTERCEPT_FUNCTION(pvalloc)
#else
#define TYSAN_MAYBE_INTERCEPT_PVALLOC
#endif // SANITIZER_INTERCEPT_PVALLOC

SetTarget().CreateWatchpoint(m_addresses[i], false, true).get();
if (watchpoint != nullptr) {
  if (watchpoint->IsSoftware() && !watchpoint->HasResolvedValues())
    m_could_not_resolve_sw_wp = true;
  m_watch_ids[i] = watchpoint->GetID();
  watchpoint->SetThreadID(m_tid);
  watchpoint->SetWatchpointKind("log-access");
}
#define TYSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC INTERCEPT_FUNCTION(aligned_alloc)
#else
#define TYSAN_MAYBE_INTERCEPT_ALIGNED_ALLOC

{
    if (position >= length)
    {
      // No text at this position, valid query though.
      *content = nullptr;
      *size = 0;
    }
    else
    {
      *content = data + position;
      *size = length - position;
    }
    return E_SUCCESS;
}
