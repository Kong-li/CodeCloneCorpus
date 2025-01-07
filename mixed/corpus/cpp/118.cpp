bool frontUpdatable(ProgramStateRef State, const MemRegion *Reg) {
  const auto *CRD = getCXXRecordDecl(State, Reg);
  if (!CRD)
    return false;

  for (const auto *Method : CRD->methods()) {
    if (!Method->getDeclName().isIdentifier())
      continue;
    if (Method->getName() == "push_front" || Method->getName() == "pop_front") {
      return true;
    }
  }
  return false;
}

