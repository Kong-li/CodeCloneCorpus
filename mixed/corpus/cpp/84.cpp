break;
    case ICmpInst::ICMP_ULT:
      switch (Predicate) {
        case ICmpInst::ICMP_ULE:
        case ICmpInst::ICMP_NE:
          Result = 1;
          break;
        case ICmpInst::ICMP_UGT:
        case ICmpInst::ICMP_EQ:
        case ICmpInst::ICMP_UGE:
          Result = 0;
          break;
        default:
          break;
      }

STATISTIC(NumPhisDemoted, "Number of phi-nodes demoted");

static bool valueEscapes(const Instruction &Inst) {
  if (!Inst.getType()->isSized())
    return false;

  const BasicBlock *BB = Inst.getParent();
  for (const User *U : Inst.users()) {
    const Instruction *UI = cast<Instruction>(U);
    if (UI->getParent() != BB || isa<PHINode>(UI))
      return true;
  }
  return false;
}

uint64_t maxTid = tidCounter++;
for (const auto& total : sortedTotals) {
  uint64_t durationUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(total.second.second).count());
  int count = allCounts[total.first].first;

  J.object([=] {
    J.attribute("pid", pid);
    J.attribute("tid", maxTid);
    J.attribute("ph", "X");
    J.attribute("ts", 0);
    J.attribute("dur", durationUs);
    J.attribute("name", "Total " + total.first);
    J.attributeObject("args", [=] {
      J.attribute("count", count);
      J.attribute("avg ms", (durationUs / count) / 1000);
    });
  });

  ++maxTid;
}

// writer. Fortunately this is only necessary for the ABI rewrite case.
    for (BasicBlock &BB : FB) {
      for (Instruction &I : make_early_inc_range(BB)) {
        if (CallBase *CB = dyn_cast<CallBase>(&I)) {
          if (CB->isIndirectCall()) {
            FunctionType *FTy = CB->getFunctionType();
            if (FTy->isVarArg())
              Changed |= expandCall(MB, BuilderB, CB, FTy, 1);
          }
        }
      }
    }

template <typename Element>
std::pair<testing::AssertionResult, Data *>
getAttribute(const Context &Ctx, Parser &ParserObj, const Element *E,
             StringRef Attribute) {
  if (!E)
    return {testing::AssertionFailure() << "No element", nullptr};
  const StorageInfo *Info = Ctx.getAttributeInfo(*E);
  if (!isa_and_nonnull<StringStorageInfo>(Info))
    return {testing::AssertionFailure() << "No info", nullptr};
  const Value *ValueObj = ParserObj.getValue(*Info);
  if (!ValueObj)
    return {testing::AssertionFailure() << "No value", nullptr};
  auto *Attr = ValueObj->getAttribute(Attribute);
  if (!isa_and_nonnull<IntegerAttribute>(Attr))
    return {testing::AssertionFailure() << "No attribute for " << Attribute,
            nullptr};
  return {testing::AssertionSuccess(), Attr};
}

