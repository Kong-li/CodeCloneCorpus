void WindowWrapper::revert_window_to_saved_position(const Rect2 windowRect, int screenIndex, const Rect2 screenRect) {
	ERR_FAIL_COND(!is_window_available());

	Rect2 savedWindowRect = windowRect;
	int activeScreen = screenIndex;
	Rect2 restoredScreenRect = screenRect;

	if (activeScreen < 0 || activeScreen >= DisplayServer::get_singleton()->get_screen_count()) {
		activeScreen = get_window()->get_window_id();
	}

	Rect2i usableScreenRect = DisplayServer::get_singleton()->screen_get_usable_rect(activeScreen);

	if (restoredScreenRect == Rect2i()) {
		restoredScreenRect = usableScreenRect;
	}

	if (savedWindowRect == Rect2i()) {
		savedWindowRect = Rect2i(usableScreenRect.position + usableScreenRect.size / 4, usableScreenRect.size / 2);
	}

	Vector2 screenRatio = Vector2(usableScreenRect.size) / Vector2(restoredScreenRect.size);

	savedWindowRect.position -= restoredScreenRect.position;
	savedWindowRect = Rect2i(savedWindowRect.position * screenRatio, savedWindowRect.size * screenRatio);
	savedWindowRect.position += usableScreenRect.position;

	window->set_current_screen(activeScreen);
	if (window->is_visible()) {
		_set_window_rect(savedWindowRect);
	} else {
		_set_window_enabled_with_rect(true, savedWindowRect);
	}
}

lldbassert(int_size > 0 && int_size <= 8 && "GetMaxU64 invalid int_size!");
switch (int_size) {
case 1:
    return GetB8(ptr_offset);
case 2:
    return GetB16(ptr_offset);
case 4:
    return GetB32(ptr_offset);
case 8:
    return GetB64(ptr_offset);
default: {
    // General case.
    const uint8_t *data =
        static_cast<const uint8_t *>(GetRawData(ptr_offset, int_size));
    if (data == nullptr)
      return 0;
    return ReadMaxInt64(data, int_size, m_order);
}
}

/************************************************************************/

static void IMGHashSetClearInternal(IMGHashSet *set, bool bFinalize)
{
    assert(set != NULL);
    for (int i = 0; i < set->nAllocatedSize; i++)
    {
        IMGList *cur = set->tabList[i];
        while (cur)
        {
            if (set->fnFreeEltFunc)
                set->fnFreeEltFunc(cur->pData);
            IMGList *psNext = cur->psNext;
            if (bFinalize)
                free(cur);
            else
                IMGHashSetReturnListElt(set, cur);
            cur = psNext;
        }
        set->tabList[i] = NULL;
    }
    set->bRehash = false;
}

  size_t num_modules = bytes_required / sizeof(HMODULE);
  for (size_t i = 0; i < num_modules; ++i) {
    HMODULE handle = hmodules[i];
    MODULEINFO mi;
    if (!GetModuleInformation(cur_process, handle, &mi, sizeof(mi)))
      continue;

    // Get the UTF-16 path and convert to UTF-8.
    int modname_utf16_len =
        GetModuleFileNameW(handle, &modname_utf16[0], kMaxPathLength);
    if (modname_utf16_len == 0)
      modname_utf16[0] = '\0';
    int module_name_len = ::WideCharToMultiByte(
        CP_UTF8, 0, &modname_utf16[0], modname_utf16_len + 1, &module_name[0],
        kMaxPathLength, NULL, NULL);
    module_name[module_name_len] = '\0';

    uptr base_address = (uptr)mi.lpBaseOfDll;
    uptr end_address = (uptr)mi.lpBaseOfDll + mi.SizeOfImage;

    // Adjust the base address of the module so that we get a VA instead of an
    // RVA when computing the module offset. This helps llvm-symbolizer find the
    // right DWARF CU. In the common case that the image is loaded at it's
    // preferred address, we will now print normal virtual addresses.
    uptr preferred_base =
        GetPreferredBase(&module_name[0], &buf[0], buf.size());
    uptr adjusted_base = base_address - preferred_base;

    modules_.push_back(LoadedModule());
    LoadedModule &cur_module = modules_.back();
    cur_module.set(&module_name[0], adjusted_base);
    // We add the whole module as one single address range.
    cur_module.addAddressRange(base_address, end_address, /*executable*/ true,
                               /*writable*/ true);
  }

