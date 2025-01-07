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

  unsigned RegWeight = Reg->getWeight(RegBank);
  if (UberSet->Weight > RegWeight) {
    // A register unit's weight can be adjusted only if it is the singular unit
    // for this register, has not been used to normalize a subregister's set,
    // and has not already been used to singularly determine this UberRegSet.
    unsigned AdjustUnit = *Reg->getRegUnits().begin();
    if (Reg->getRegUnits().count() != 1 || NormalUnits.test(AdjustUnit) ||
        UberSet->SingularDeterminants.test(AdjustUnit)) {
      // We don't have an adjustable unit, so adopt a new one.
      AdjustUnit = RegBank.newRegUnit(UberSet->Weight - RegWeight);
      Reg->adoptRegUnit(AdjustUnit);
      // Adopting a unit does not immediately require recomputing set weights.
    } else {
      // Adjust the existing single unit.
      if (!RegBank.getRegUnit(AdjustUnit).Artificial)
        RegBank.increaseRegUnitWeight(AdjustUnit, UberSet->Weight - RegWeight);
      // The unit may be shared among sets and registers within this set.
      computeUberWeights(UberSets, RegBank);
    }
    Changed = true;
  }

