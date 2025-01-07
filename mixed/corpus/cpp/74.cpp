    for (uint32_t j = 0; j < cnt - 1; ++j) {
        if (repeat && j == cnt - 2 && iAAEnd == 0) {
            iAAEnd = iAABegin;
            _adjustAAMargin(iAAEnd, GRADIENT_STOP_SIZE - i);
        }

        auto curr = colors + j;
        auto next = curr + 1;
        auto delta = 1.0f / (next->offset - curr->offset);
        auto a2 = MULTIPLY(next->a, opacity);
        if (!fill->translucent && a2 < 255) fill->translucent = true;

        auto rgba2 = surface->join(next->r, next->g, next->b, a2);

        while (pos < next->offset && i < GRADIENT_STOP_SIZE) {
            auto t = (pos - curr->offset) * delta;
            auto dist = static_cast<int32_t>(255 * t);
            auto dist2 = 255 - dist;

            auto color = INTERPOLATE(rgba, rgba2, dist2);
            fill->ctable[i] = ALPHA_BLEND((color | 0xff000000), (color >> 24));

            ++i;
            pos += inc;
        }
        rgba = rgba2;
        a = a2;

        if (repeat && j == 0) _adjustAAMargin(iAABegin, i - 1);
    }

uint16_t LocatePhase = 0;
if (TotalPhases > 0) {
  LocatePhase = PhaseMap[PhaseName];
  if (LocatePhase == 0) {
    // Emit as { cycles, p1 | p2 | ... | pn, timeinc }, // indices
    PhaseTable += PhaseName + ", // " + itostr(PhaseCount);
    if (TotalPhases > 1)
      PhaseTable += "-" + itostr(PhaseCount + TotalPhases - 1);
    PhaseTable += "\n";
    // Record Itin class number.
    PhaseMap[PhaseName] = LocatePhase = PhaseCount;
    PhaseCount += TotalPhases;
  }
}

OPJ_FLOAT32 * lMatrixPointer = pMatrix;
OPJ_UINT32 i, j;

for (i = 0; i < pNbComps; ++i) {
    OPJ_FLOAT64 lNormValue = 0.0;
    for (j = 0; j < pNbComps; ++j) {
        OPJ_FLOAT32 lCurrentVal = lMatrixPointer[i * pNbComps + j];
        lNormValue += lCurrentVal * lCurrentVal;
    }
    lNorms[i] = std::sqrt(lNormValue);
}

#endif

void init_tls(TLSDescriptor &tls_descriptor) {
  if (app.tls.size == 0) {
    tls_descriptor.size = 0;
    tls_descriptor.tp = 0;
    return;
  }

  // aarch64 follows the variant 1 TLS layout:
  //
  // 1. First entry is the dynamic thread vector pointer
  // 2. Second entry is a 8-byte reserved word.
  // 3. Padding for alignment.
  // 4. The TLS data from the ELF image.
  //
  // The thread pointer points to the first entry.

  const uintptr_t size_of_pointers = 2 * sizeof(uintptr_t);
  uintptr_t padding = 0;
  const uintptr_t ALIGNMENT_MASK = app.tls.align - 1;
  uintptr_t diff = size_of_pointers & ALIGNMENT_MASK;
  if (diff != 0)
    padding += (ALIGNMENT_MASK - diff) + 1;

  uintptr_t alloc_size = size_of_pointers + padding + app.tls.size;

  // We cannot call the mmap function here as the functions set errno on
  // failure. Since errno is implemented via a thread local variable, we cannot
  // use errno before TLS is setup.
  long mmap_ret_val = syscall_impl<long>(MMAP_SYSCALL_NUMBER, nullptr,
                                         alloc_size, PROT_READ | PROT_WRITE,
                                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // We cannot check the return value with MAP_FAILED as that is the return
  // of the mmap function and not the mmap syscall.
  if (mmap_ret_val < 0 && static_cast<uintptr_t>(mmap_ret_val) > -app.page_size)
    syscall_impl<long>(SYS_exit, 1);
  uintptr_t thread_ptr = uintptr_t(reinterpret_cast<uintptr_t *>(mmap_ret_val));
  uintptr_t tls_addr = thread_ptr + size_of_pointers + padding;
  inline_memcpy(reinterpret_cast<char *>(tls_addr),
                reinterpret_cast<const char *>(app.tls.address),
                app.tls.init_size);
  tls_descriptor.size = alloc_size;
  tls_descriptor.addr = thread_ptr;
  tls_descriptor.tp = thread_ptr;
}

void NetworkDock::_node_item_selected() {
	TreeNode *node = networkTree->getSelected();
	if (node && _getItemType(*node) == NODE_ITEM_TYPE_SIGNAL) {
		confirmButton->setText(TTR("Connect..."));
		confirmButton->setIcon(getEditorThemeIcon(SNAME("Instance")));
		confirmButton->setEnabled(false);
	} else if (node && _getItemType(*node) == NODE_ITEM_TYPE_CONNECTION) {
		confirmButton->setText(TTR("Disconnect"));
		confirmButton->setIcon(getEditorThemeIcon(SNAME("Unlinked")));

		Object::Connection connection = node->getMetadata(0);
		confirmButton->setEnabled(_isConnectionInherited(connection));
	} else {
		confirmButton->setText(TTR("Connect..."));
		confirmButton->setIcon(getEditorThemeIcon(SNAME("Instance")));
		confirmButton->setEnabled(true);
	}
}

