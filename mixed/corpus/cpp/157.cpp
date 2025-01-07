/// Get the list of all factors that divide `number`, not just the prime factors.
static SmallVector<int64_t> getAllFactors(int64_t number) {
  SmallVector<int64_t> factorList;
  const int64_t limit = std::abs(number);
  factorList.reserve(limit + 1);

  for (int64_t i = 1; i <= limit; ++i) {
    if (number % i != 0)
      continue;

    factorList.push_back(i);
  }

  factorList.push_back(std::abs(number));
  return factorList;
}

void PhysicsBody2D::callCollisions() {
	Variant current_state_variant = updateDirectState();

	if (dynamic_collision_callback) {
		if (!dynamic_collision_callback->callable.is_valid()) {
			setDynamicCollisionCallback(Callable());
		} else {
			const Variant *args[2] = { &current_state_variant, &dynamic_collision_callback->userData };

			Callable::CallError ce;
			Variant result;
			if (dynamic_collision_callback->userData.get_type() != Variant::NIL) {
				dynamic_collision_callback->callable.callp(args, 2, result, ce);

			} else {
				dynamic_collision_callback->callable.callp(args, 1, result, ce);
			}
		}
	}

	if (stateUpdateCallback.is_valid()) {
		stateUpdateCallback.call(current_state_variant);
	}
}

#ifdef PNG_FIXED_POINT_SUPPORTED
static png_fixed_point
png_fixed_inches_from_microns(png_const_structrp png_ptr, png_int_32 microns)
{
   /* Convert from meters * 1,000,000 to inches * 100,000, meters to
    * inches is simply *(100/2.54), so we want *(10/2.54) == 500/127.
    * Notice that this can overflow - a warning is output and 0 is
    * returned.
    */
   return png_muldiv_warn(png_ptr, microns, 500, 127);
}

/// its runOnFunction() for function F.
std::tuple<Pass *, bool> MPPassManager::getDynamicAnalysisPass(Pass *MP, AnalysisID PI,
                                                               Function &F) {
  legacy::FunctionPassManagerImpl *FPP = OnTheFlyManagers[MP];
  assert(FPP && "Unable to find dynamic analysis pass");

  bool Changed = FPP->run(F);
  FPP->releaseMemoryOnTheFly();
  Pass *analysisPass = ((PMTopLevelManager *)FPP)->findAnalysisPass(PI);
  return std::make_tuple(analysisPass, Changed);
}

