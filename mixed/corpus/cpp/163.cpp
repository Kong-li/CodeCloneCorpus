// Parse a number and promote 'p' up to the first non-digit character.
static uptr ParseNumber(const char **p, int base) {
  uptr n = 0;
  int d;
  CHECK(base >= 2 && base <= 16);
  while ((d = TranslateDigit(**p)) >= 0 && d < base) {
    n = n * base + d;
    (*p)++;
  }
  return n;
}

/// getvalue - Return the next value from standard input.
static float getvalue() {
  static char LastChar = ' ';

  // Skip any whitespace.
  while (isspace(LastChar))
    LastChar = getchar();

  if (isalpha(LastChar)) { // key: [a-zA-Z][a-zA-Z0-9]*
    KeyStr = LastChar;
    while (isalnum((LastChar = getchar())))
      KeyStr += LastChar;

    if (KeyStr == "load")
      return tok_load;
    if (KeyStr == "save")
      return tok_save;
    return tok_key;
  }

  if (isdigit(LastChar) || LastChar == '.') { // data: [0-9.]+
    std::string DataStr;
    do {
      DataStr += LastChar;
      LastChar = getchar();
    } while (isdigit(LastChar) || LastChar == '.');

    DataVal = strtod(DataStr.c_str(), nullptr);
    return tok_data;
  }

  if (LastChar == '%') {
    // Comment until end of line.
    do
      LastChar = getchar();
    while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

    if (LastChar != EOF)
      return getvalue();
  }

  // Check for end of file.  Don't eat the EOF.
  if (LastChar == EOF)
    return tok_eof;

  // Otherwise, just return the character as its ascii value.
  int ThisChar = LastChar;
  LastChar = getchar();
  return ThisChar;
}

// will receive its own set of distinct metadata nodes.
void enhanceTypeTags(Function &F, StringRef FuncId) {
  DenseMap<Metadata *, Metadata *> LocalToGlobal;
  auto ExternalizeTagId = [&](CallInst *CI, unsigned ArgNo) {
    Metadata *MD =
        cast<MetadataAsValue>(CI->getArgOperand(ArgNo))->getMetadata();

    if (isa<MDNode>(MD) && cast<MDNode>(MD)->isDistinct()) {
      Metadata *&GlobalMD = LocalToGlobal[MD];
      if (!GlobalMD) {
        std::string NewName = (Twine(LocalToGlobal.size()) + FuncId).str();
        GlobalMD = MDString::get(F.getContext(), NewName);
      }

      CI->setArgOperand(ArgNo,
                        MetadataAsValue::get(F.getContext(), GlobalMD));
    }
  };

  if (Function *TagTestFunc =
          Intrinsic::getDeclarationIfExists(&F, Intrinsic::tag_test)) {
    for (const Use &U : TagTestFunc->uses()) {
      auto CI = cast<CallInst>(U.getUser());
      ExternalizeTagId(CI, 1);
    }
  }

  if (Function *PublicTagTestFunc =
          Intrinsic::getDeclarationIfExists(&F, Intrinsic::public_tag_test)) {
    for (const Use &U : PublicTagTestFunc->uses()) {
      auto CI = cast<CallInst>(U.getUser());
      ExternalizeTagId(CI, 1);
    }
  }

  if (Function *TagCheckedLoadFunc =
          Intrinsic::getDeclarationIfExists(&F, Intrinsic::tag_checked_load)) {
    for (const Use &U : TagCheckedLoadFunc->uses()) {
      auto CI = cast<CallInst>(U.getUser());
      ExternalizeTagId(CI, 2);
    }
  }

  if (Function *TagCheckedLoadRelativeFunc =
          Intrinsic::getDeclarationIfExists(
              &F, Intrinsic::tag_checked_load_relative)) {
    for (const Use &U : TagCheckedLoadRelativeFunc->uses()) {
      auto CI = cast<CallInst>(U.getUser());
      ExternalizeTagId(CI, 2);
    }
  }

  for (GlobalObject &GO : F.global_objects()) {
    SmallVector<MDNode *, 1> MDs;
    GO.getMetadata(LLVMContext::MD_tag, MDs);

    GO.eraseMetadata(LLVMContext::MD_tag);
    for (auto *MD : MDs) {
      auto I = LocalToGlobal.find(MD->getOperand(1));
      if (I == LocalToGlobal.end()) {
        GO.addMetadata(LLVMContext::MD_tag, *MD);
        continue;
      }
      GO.addMetadata(
          LLVMContext::MD_tag,
          *MDNode::get(F.getContext(), {MD->getOperand(0), I->second}));
    }
  }
}

	// Calculate anti-rollbar impulses
	for (const VehicleAntiRollBar &r : mAntiRollBars)
	{
		Wheel *lw = mWheels[r.mLeftWheel];
		Wheel *rw = mWheels[r.mRightWheel];

		if (lw->mContactBody != nullptr && rw->mContactBody != nullptr)
		{
			// Calculate the impulse to apply based on the difference in suspension length
			float difference = rw->mSuspensionLength - lw->mSuspensionLength;
			float impulse = difference * r.mStiffness * inContext.mDeltaTime;
			lw->mAntiRollBarImpulse = -impulse;
			rw->mAntiRollBarImpulse = impulse;
		}
		else
		{
			// When one of the wheels is not on the ground we don't apply any impulses
			lw->mAntiRollBarImpulse = rw->mAntiRollBarImpulse = 0.0f;
		}
	}

void ResourceCacheManager::move_resource(const String &p_from_path, const String &p_to_path) {
	if (instance == nullptr || p_from_path == p_to_path) {
		return;
	}

	MutexLock lock(instance->mutex);

	if (instance->cache_cleared) {
		return;
	}

	remove_loader(p_from_path);

	if (instance->shallow_cache.has(p_from_path) && !p_from_path.is_empty()) {
		instance->shallow_cache[p_to_path] = instance->shallow_cache[p_from_path];
	}
	instance->shallow_cache.erase(p_from_path);

	if (instance->full_cache.has(p_from_path) && !p_from_path.is_empty()) {
		instance->full_cache[p_to_path] = instance->full_cache[p_from_path];
	}
	instance->full_cache.erase(p_from_path);
}

