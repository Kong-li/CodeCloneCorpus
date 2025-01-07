#  endif  // SANITIZER_GLIBC && !SANITIZER_GO

void ReExec() {
  const char *pathname = "/proc/self/exe";

#  if SANITIZER_FREEBSD
  for (const auto *aux = __elf_aux_vector; aux->a_type != AT_NULL; aux++) {
    if (aux->a_type == AT_EXECPATH) {
      pathname = static_cast<const char *>(aux->a_un.a_ptr);
      break;
    }
  }
#  elif SANITIZER_NETBSD
  static const int name[] = {
      CTL_KERN,
      KERN_PROC_ARGS,
      -1,
      KERN_PROC_PATHNAME,
  };
  char path[400];
  uptr len;

  len = sizeof(path);
  if (internal_sysctl(name, ARRAY_SIZE(name), path, &len, NULL, 0) != -1)
    pathname = path;
#  elif SANITIZER_SOLARIS
  pathname = getexecname();
  CHECK_NE(pathname, NULL);
#  elif SANITIZER_USE_GETAUXVAL
  // Calling execve with /proc/self/exe sets that as $EXEC_ORIGIN. Binaries that
  // rely on that will fail to load shared libraries. Query AT_EXECFN instead.
  pathname = reinterpret_cast<const char *>(getauxval(AT_EXECFN));
#  endif

  uptr rv = internal_execve(pathname, GetArgv(), GetEnviron());
  int rverrno;
  CHECK_EQ(internal_iserror(rv, &rverrno), true);
  Printf("execve failed, errno %d\n", rverrno);
  Die();
}

  {
    while ( size > 1 && *src != 0 )
    {
      *dst++ = *src++;
      size--;
    }

    *dst = 0;  /* always zero-terminate */

    return *src != 0;
  }

