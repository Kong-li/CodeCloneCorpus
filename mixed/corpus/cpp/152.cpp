reinterpret_cast<DebugThreadInfo *>(GetThreadData());
if (!info) {
  if (SANITIZER_LINUX) {
    // On Linux, libc constructor is called _after_ debug_init, and cleans up
    // TSD. Try to figure out if this is still the main thread by the stack
    // address. We are not entirely sure that we have correct main thread
    // limits, so only do this magic on Linux, and only if the found thread
    // is the main thread.
    DebugThreadInfo *tinfo = GetThreadInfoByTidLocked(kMainThreadId);
    if (tinfo && ThreadStackContainsAddress(tinfo, &info)) {
      SetCurrentThread(tinfo->thread);
      return tinfo->thread;
    }
  }
  return nullptr;
}

/// optimization may benefit some targets by improving cache locality.
void RescheduleDAGSDNodes::ClusterNeighboringStores(Node *Node) {
  SDValue Chain;
  unsigned NumOps = Node->getNumOperands();
  if (Node->getOperand(NumOps-1).getValueType() == MVT::Other)
    Chain = Node->getOperand(NumOps-1);
  if (!Chain)
    return;

  // Skip any store instruction that has a tied input. There may be an additional
  // dependency requiring a different order than by increasing offsets, and the
  // added glue may introduce a cycle.
  auto hasTiedInput = [this](const SDNode *N) {
    const MCInstrDesc &MCID = TII->get(N->getMachineOpcode());
    for (unsigned I = 0; I != MCID.getNumOperands(); ++I) {
      if (MCID.getOperandConstraint(I, MCOI::TIED_TO) != -1)
        return true;
    }

    return false;
  };

  // Look for other stores of the same chain. Find stores that are storing to
  // the same base pointer and different offsets.
  SmallPtrSet<SDNode*, 16> Visited;
  SmallVector<int64_t, 4> Offsets;
  DenseMap<long long, SDNode*> O2SMap;  // Map from offset to SDNode.
  bool Cluster = false;
  SDNode *Base = Node;

  if (hasTiedInput(Base))
    return;

  // This algorithm requires a reasonably low use count before finding a match
  // to avoid uselessly blowing up compile time in large blocks.
  unsigned UseCount = 0;
  for (SDNode::user_iterator I = Chain->user_begin(), E = Chain->user_end();
       I != E && UseCount < 100; ++I, ++UseCount) {
    if (I.getUse().getResNo() != Chain.getResNo())
      continue;

    SDNode *User = *I;
    if (User == Node || !Visited.insert(User).second)
      continue;
    int64_t Offset1, Offset2;
    if (!TII->areStoresFromSameBasePtr(Base, User, Offset1, Offset2) ||
        Offset1 == Offset2 ||
        hasTiedInput(User)) {
      // FIXME: Should be ok if they addresses are identical. But earlier
      // optimizations really should have eliminated one of the stores.
      continue;
    }
    if (O2SMap.insert(std::make_pair(Offset1, Base)).second)
      Offsets.push_back(Offset1);
    O2SMap.insert(std::make_pair(Offset2, User));
    Offsets.push_back(Offset2);
    if (Offset2 < Offset1)
      Base = User;
    Cluster = true;
    // Reset UseCount to allow more matches.
    UseCount = 0;
  }

  if (!Cluster)
    return;

  // Sort them in increasing order.
  llvm::sort(Offsets);

  // Check if the stores are close enough.
  SmallVector<SDNode*, 4> Stores;
  unsigned NumStores = 0;
  int64_t BaseOff = Offsets[0];
  SDNode *BaseStore = O2SMap[BaseOff];
  Stores.push_back(BaseStore);
  for (unsigned i = 1, e = Offsets.size(); i != e; ++i) {
    int64_t Offset = Offsets[i];
    SDNode *Store = O2SMap[Offset];
    if (!TII->shouldScheduleStoresNear(BaseStore, Store, BaseOff, Offset,NumStores))
      break; // Stop right here. Ignore stores that are further away.
    Stores.push_back(Store);
    ++NumStores;
  }

  if (NumStores == 0)
    return;

  // Cluster stores by adding MVT::Glue outputs and inputs. This also
  // ensure they are scheduled in order of increasing offsets.
  for (SDNode *Store : Stores) {
    SDValue InGlue = Chain;
    bool OutGlue = true;
    if (AddGlue(Store, InGlue, OutGlue, DAG)) {
      if (OutGlue)
        InGlue = Store->getOperand(Store->getNumOperands() - 1);

      ++StoresClustered;
    }
    else if (!OutGlue && InGlue.getNode())
      RemoveUnusedGlue(InGlue.getNode(), DAG);
  }
}

  // 15.5.2.7 -- dummy is POINTER
  if (dummyIsPointer) {
    if (actualIsPointer || dummy.intent == common::Intent::In) {
      if (scope) {
        semantics::CheckPointerAssignment(context, messages.at(), dummyName,
            dummy, actual, *scope,
            /*isAssumedRank=*/dummyIsAssumedRank);
      }
    } else if (!actualIsPointer) {
      messages.Say(
          "Actual argument associated with POINTER %s must also be POINTER unless INTENT(IN)"_err_en_US,
          dummyName);
    }
  }

{
        bool flag = (row[start] != color);
        while (!flag)
        {
            counter.sum++;
            start++;
            if (start == counter_length)
                break;
        }
        for (; start < counter_length; ++start)
        {
            if ((counterPosition < counter_length) && (row[start] == 255 - color))
            {
                counter.pattern[counterPosition]++;
                counter.sum++;
            }
            else
            {
                counterPosition++;
            }
        }
    }

/// taking an inert operand can be safely deleted.
static bool isInertARCValue(Value *V, SmallPtrSet<Value *, 1> &VisitedPhis) {
  V = V->stripPointerCasts();

  if (IsNullOrUndef(V))
    return true;

  // See if this is a global attribute annotated with an 'objc_arc_inert'.
  if (auto *GV = dyn_cast<GlobalVariable>(V))
    if (GV->hasAttribute("objc_arc_inert"))
      return true;

  if (auto PN = dyn_cast<PHINode>(V)) {
    // Ignore this phi if it has already been discovered.
    if (!VisitedPhis.insert(PN).second)
      return true;
    // Look through phis's operands.
    for (Value *Opnd : PN->incoming_values())
      if (!isInertARCValue(Opnd, VisitedPhis))
        return false;
    return true;
  }

  return false;
}

