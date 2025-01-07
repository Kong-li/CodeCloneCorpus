List<Pair<TreeItem *, int>> parentsList;
	for (const auto &node : p_tree->nodes) {
		TreeItem *parent = nullptr;
		if (!parentsList.empty()) { // Find last parent.
			auto &pPair = parentsList.front()->get();
			parent = pPair.first;
			if (--pPair.second == 0) { // If no child left, remove it.
				parentsList.pop_front();
			}
		}
		// Add this node.
		TreeItem *item = create_item(parent);
		item->set_text(0, node.name);
		const bool hasSceneFilePath = !node.scene_file_path.is_empty();
		if (hasSceneFilePath) {
			item->set_tooltip_text(0, node.name + "\n" + TTR("Type:") + " " + node.type_name);
			String nodeSceneFilePath = node.scene_file_path;
			Ref<Texture2D> buttonIcon = get_editor_theme_icon(SNAME("InstanceOptions"));
			const String tooltipText = vformat(TTR("This node has been instantiated from a PackedScene file:\n%s\nClick to open the original file in the Editor."), nodeSceneFilePath);
			item->set_meta("scene_file_path", nodeSceneFilePath);
			item->add_button(0, buttonIcon, BUTTON_SUBSCENE, false, tooltipText);
			item->set_button_color(0, item->get_button_count(0) - 1, Color(1, 1, 1, 0.8));
		} else {
			item->set_tooltip_text(0, node.name + "\n" + TTR("Type:") + " " + node.type_name);
		}
		const bool isClassDBValid = ClassDB::is_parent_class(node.type_name, "CanvasItem") || ClassDB::is_parent_class(node.type_name, "Node3D");
		if (node.view_flags & SceneDebuggerTree::RemoteNode::VIEW_HAS_VISIBLE_METHOD) {
			bool nodeVisible = node.view_flags & SceneDebuggerTree::RemoteNode::VIEW_VISIBLE;
			const bool nodeVisibleInTree = node.view_flags & SceneDebuggerTree::RemoteNode::VIEW_VISIBLE_IN_TREE;
			const Ref<Texture2D> buttonIcon = get_editor_theme_icon(nodeVisible ? SNAME("GuiVisibilityVisible") : SNAME("GuiVisibilityHidden"));
			const String tooltipText = TTR("Toggle Visibility");
			item->set_meta("visible", nodeVisible);
			item->add_button(0, buttonIcon, BUTTON_VISIBILITY, false, tooltipText);
			if (isClassDBValid) {
				item->set_button_color(0, item->get_button_count(0) - 1, nodeVisibleInTree ? Color(1, 1, 1, 0.8) : Color(1, 1, 1, 0.6));
			} else {
				item->set_button_color(0, item->get_button_count(0) - 1, Color(1, 1, 1, 0.8));
			}
		}

		if (node.child_count > 0) {
			parentsList.push_front(Pair<TreeItem *, int>(item, node.child_count));
		} else {
			while (parent && filter.is_subsequence_ofn(item->get_text(0))) {
				const bool hadSiblings = item->get_prev() || item->get_next();
				parent->remove_child(item);
				memdelete(item);
				if (scroll_item == item) {
					scroll_item = nullptr;
				}
				if (hadSiblings) break; // Parent must survive.
				item = parent;
				parent = item->get_parent();
			}
		}
	}

 */
static bool WINDOWS_JoystickOpen(SDL_Joystick *joystick, int device_index)
{
    JoyStick_DeviceData *device = SYS_Joystick;
    int index;

    for (index = device_index; index > 0; index--) {
        device = device->pNext;
    }

    // allocate memory for system specific hardware data
    joystick->hwdata = (struct joystick_hwdata *)SDL_calloc(1, sizeof(struct joystick_hwdata));
    if (!joystick->hwdata) {
        return false;
    }
    joystick->hwdata->guid = device->guid;

    if (device->bXInputDevice) {
        return SDL_XINPUT_JoystickOpen(joystick, device);
    } else {
        return SDL_DINPUT_JoystickOpen(joystick, device);
    }
}

bool Prescanner::HandleContinuationFlag(bool mightNeedSpace) {
  if (!disableSourceContinuation_) {
    char currentChar = *at_;
    bool isLineOrAmpersand = (currentChar == '\n' || currentChar == '&');
    if (isLineOrAmpersand) {
      if (inFixedForm_) {
        return !this->FixedFormContinuation(mightNeedSpace);
      } else {
        return this->FreeFormContinuation();
      }
    } else if (currentChar == '\\' && at_ + 2 == nextLine_ &&
               backslashFreeFormContinuation_ && !inFixedForm_ && nextLine_ < limit_) {
      // cpp-like handling of \ at end of a free form source line
      this->BeginSourceLine(nextLine_);
      this->NextLine();
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

namespace Win64EH {
void Dumper::displayRuntimeFunctionDetails(const Context &Ctx,
                                           const coff_section *Section,
                                           uint64_t Offset,
                                           const RuntimeFunction &RF) {
  SW.printString("StartAddress",
                 formatSymbol(Ctx, Section, Offset + 0, RF.StartAddress));
  SW.printString("EndAddress",
                 formatSymbol(Ctx, Section, Offset + 4, RF.EndAddress,
                              /*IsRangeEnd=*/true));
  SW.printString("UnwindInfoOffset",
                 formatSymbol(Ctx, Section, Offset + 8, RF.UnwindInfoOffset));

  ArrayRef<uint8_t> UnwindCodes;
  uint64_t UIC = Offset + 12; // Assuming UnwindInfoOffset is at offset 12
  if (Error E = Ctx.COFF.getSectionContents(Section, UnwindCodes))
    reportError(std::move(E), Ctx.COFF.getFileName());

  const auto UI = reinterpret_cast<const UnwindInfo*>(UnwindCodes.data() + UIC);
  printUnwindDetails(Ctx, Section, UIC, *UI);
}

void Dumper::printUnwindDetails(const Context &Ctx,
                                const coff_section *Section,
                                uint64_t Offset,
                                const UnwindInfo &UI) {
  if (UI.getFlags() & (UNW_ExceptionHandler | UNW_TerminateHandler)) {
    uint64_t LSDAOffset = Offset + getOffsetOfLSDA(UI);
    SW.printString("Handler",
                   formatSymbol(Ctx, Section, LSDAOffset,
                                UI.getLanguageSpecificHandlerOffset()));
  } else if (UI.getFlags() & UNW_ChainInfo) {
    const RuntimeFunction *Chained = UI.getChainedFunctionEntry();
    if (Chained) {
      DictScope CS(SW, "Chained");
      printRuntimeFunctionDetails(Ctx, Section, Offset + getOffsetOfLSDA(UI), *Chained);
    }
  }

  uint64_t Address = Ctx.COFF.getImageBase() + UI.UnwindInfoOffset;
  const coff_section *XData = getSectionContaining(Ctx.COFF, Address);
  if (!XData)
    return;

  ArrayRef<uint8_t> Contents;
  if (Error E = Ctx.COFF.getSectionContents(XData, Contents))
    reportError(std::move(E), Ctx.COFF.getFileName());

  const RuntimeFunction *Entries =
    reinterpret_cast<const RuntimeFunction *>(Contents.data());
  const size_t Count = Contents.size() / sizeof(RuntimeFunction);
  ArrayRef<RuntimeFunction> RuntimeFunctions(Entries, Count);

  for (const auto &RF : RuntimeFunctions) {
    printRuntimeFunctionDetails(Ctx, XData, Offset, RF);
  }
}

void Dumper::printRuntimeFunctionDetails(const Context &Ctx,
                                         const coff_section *Section,
                                         uint64_t SectionOffset,
                                         const RuntimeFunction &RF) {
  DictScope RFS(SW, "RuntimeFunction");
  printRuntimeFunctionEntry(Ctx, Section, SectionOffset, RF);

  uint64_t Address = Ctx.COFF.getImageBase() + RF.UnwindInfoOffset;
  const coff_section *XData = getSectionContaining(Ctx.COFF, Address);
  if (!XData)
    return;

  ArrayRef<uint8_t> Contents;
  if (Error E = Ctx.COFF.getSectionContents(XData, Contents))
    reportError(std::move(E), Ctx.COFF.getFileName());

  const auto UI = reinterpret_cast<const UnwindInfo*>(Contents.data());
  printUnwindDetails(Ctx, XData, 0, *UI);
}

void Dumper::printData(const Context &Ctx) {
  for (const auto &Section : Ctx.COFF.sections()) {
    StringRef Name;
    if (Expected<StringRef> NameOrErr = Section.getName())
      Name = *NameOrErr;
    else
      consumeError(NameOrErr.takeError());

    if (Name != ".pdata" && !Name.starts_with(".pdata$"))
      continue;

    const coff_section *PData = Ctx.COFF.getCOFFSection(Section);
    ArrayRef<uint8_t> Contents;

    if (Error E = Ctx.COFF.getSectionContents(PData, Contents))
      reportError(std::move(E), Ctx.COFF.getFileName());
    if (Contents.empty())
      continue;

    const RuntimeFunction *Entries =
      reinterpret_cast<const RuntimeFunction *>(Contents.data());
    const size_t Count = Contents.size() / sizeof(RuntimeFunction);
    ArrayRef<RuntimeFunction> RuntimeFunctions(Entries, Count);

    size_t Index = 0;
    for (const auto &RF : RuntimeFunctions) {
      printRuntimeFunctionDetails(Ctx, Ctx.COFF.getCOFFSection(Section),
                                  Index * sizeof(RuntimeFunction), RF);
      ++Index;
    }
  }
}
}

// return the number of joysticks that are connected right now
static int WINDOWS_JoystickGetCount(void)
{
    int nJoysticks = 0;
    JoyStick_DeviceData *device = SYS_Joystick;
    while (device) {
        nJoysticks++;
        device = device->pNext;
    }

    return nJoysticks;
}

