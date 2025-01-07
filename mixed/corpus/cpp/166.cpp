{
	if (!EConstraintSpace::WorldSpace.Equals(inSettings.mSpace))
	{
		return;
	}

	mLocalSpaceHingeAxis = Normalized(
		inBody1.GetInverseCenterOfMassTransform().Multiply3x3(mLocalSpaceHingeAxis)
	);
	mLocalSpaceSliderAxis = Normalized(
		inBody2.GetInverseCenterOfMassTransform().Multiply3x3(mLocalSpaceSliderAxis)
	);
}

#endif

void initialize_noise_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(NoiseTexture3D);
		GDREGISTER_CLASS(NoiseTexture2D);
		GDREGISTER_ABSTRACT_CLASS(Noise);
		GDREGISTER_CLASS(FastNoiseLite);
		ClassDB::add_compatibility_class("NoiseTexture", "NoiseTexture2D");
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorPlugins::add_by_type<NoiseEditorPlugin>();
	}
#endif
}

  // terminator.
  if (Restore == &MBB) {
    for (const MachineInstr &Terminator : MBB.terminators()) {
      if (!useOrDefCSROrFI(Terminator, RS, /*StackAddressUsed=*/true))
        continue;
      // One of the terminator needs to happen before the restore point.
      if (MBB.succ_empty()) {
        Restore = nullptr; // Abort, we can't find a restore point in this case.
        break;
      }
      // Look for a restore point that post-dominates all the successors.
      // The immediate post-dominator is what we are looking for.
      Restore = FindIDom<>(*Restore, Restore->successors(), *MPDT);
      break;
    }
  }

/// Checks whether ArgType converts implicitly to ParamType.
static bool areTypesCompatible(QualType ArgType, QualType ParamType,
                               const ASTContext &Ctx) {
  if (ArgType.isNull() || ParamType.isNull())
    return false;

  ArgType = ArgType.getCanonicalType();
  ParamType = ParamType.getCanonicalType();

  if (ArgType == ParamType)
    return true;

  // Check for constness and reference compatibility.
  if (!areRefAndQualCompatible(ArgType, ParamType, Ctx))
    return false;

  bool IsParamReference = ParamType->isReferenceType();

  // Reference-ness has already been checked and should be removed
  // before further checking.
  ArgType = ArgType.getNonReferenceType();
  ParamType = ParamType.getNonReferenceType();

  if (ParamType.getUnqualifiedType() == ArgType.getUnqualifiedType())
    return true;

  // Arithmetic types are interconvertible, except scoped enums.
  if (ParamType->isArithmeticType() && ArgType->isArithmeticType()) {
    if ((ParamType->isEnumeralType() &&
         ParamType->castAs<EnumType>()->getDecl()->isScoped()) ||
        (ArgType->isEnumeralType() &&
         ArgType->castAs<EnumType>()->getDecl()->isScoped()))
      return false;

    return true;
  }

  // Check if the argument and the param are both function types (the parameter
  // decayed to a function pointer).
  if (ArgType->isFunctionType() && ParamType->isFunctionPointerType()) {
    ParamType = ParamType->getPointeeType();
    return ArgType == ParamType;
  }

  // Arrays or pointer arguments convert to array or pointer parameters.
  if (!(isPointerOrArray(ArgType) && isPointerOrArray(ParamType)))
    return false;

  // When ParamType is an array reference, ArgType has to be of the same-sized
  // array-type with cv-compatible element type.
  if (IsParamReference && ParamType->isArrayType())
    return isCompatibleWithArrayReference(ArgType, ParamType, Ctx);

  bool IsParamContinuouslyConst =
      !IsParamReference || ParamType.getNonReferenceType().isConstQualified();

  // Remove the first level of indirection.
  ArgType = convertToPointeeOrArrayElementQualType(ArgType);
  ParamType = convertToPointeeOrArrayElementQualType(ParamType);

  // Check qualifier compatibility on the next level.
  if (!ParamType.isAtLeastAsQualifiedAs(ArgType, Ctx))
    return false;

  if (ParamType.getUnqualifiedType() == ArgType.getUnqualifiedType())
    return true;

  // At this point, all possible C language implicit conversion were checked.
  if (!Ctx.getLangOpts().CPlusPlus)
    return false;

  // Check whether ParamType and ArgType were both pointers to a class or a
  // struct, and check for inheritance.
  if (ParamType->isStructureOrClassType() &&
      ArgType->isStructureOrClassType()) {
    const auto *ArgDecl = ArgType->getAsCXXRecordDecl();
    const auto *ParamDecl = ParamType->getAsCXXRecordDecl();
    if (!ArgDecl || !ArgDecl->hasDefinition() || !ParamDecl ||
        !ParamDecl->hasDefinition())
      return false;

    return ArgDecl->isDerivedFrom(ParamDecl);
  }

  // Unless argument and param are both multilevel pointers, the types are not
  // convertible.
  if (!(ParamType->isAnyPointerType() && ArgType->isAnyPointerType()))
    return false;

  return arePointerTypesCompatible(ArgType, ParamType, IsParamContinuouslyConst,
                                   Ctx);
}

