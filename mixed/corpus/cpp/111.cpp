const size_t num_items = item_list->GetCount();
if (num_items > 0) {
  for (size_t i = 0; i < num_items; i++) {
    ItemSP item_sp = item_list->GetItemAtIndex(i);
    if (!ScopeNeeded(item_sp->GetScope()))
      continue;
    std::string scope_str;
    if (m_option_show_scope)
      scope_str = GetScopeInfo(item_sp).c_str();

    // Use the item object code to ensure we are using the same
    // APIs as the public API will be using...
    obj_sp = frame->GetObjectForFrameItem(
        item_sp, m_obj_options.use_dynamic);
    if (obj_sp) {
      // When dumping all items, don't print any items that are
      // not in scope to avoid extra unneeded output
      if (obj_sp->IsInScope()) {
        if (!obj_sp->GetTargetSP()
                 ->GetDisplayRuntimeSupportObjects() &&
            obj_sp->IsRuntimeSupportObject())
          continue;

        if (!scope_str.empty())
          s.PutCString(scope_str);

        if (m_option_show_decl &&
            item_sp->GetDeclaration().GetFile()) {
          item_sp->GetDeclaration().DumpStopContext(&s, false);
          s.PutCString(": ");
        }

        options.SetFormat(format);
        options.SetItemFormatDisplayLanguage(
            obj_sp->GetPreferredDisplayLanguage());
        options.SetRootValueObjectName(
            item_sp ? item_sp->GetName().c_str() : nullptr);
        if (llvm::Error error =
                obj_sp->Dump(result.GetOutputStream(), options))
          result.AppendError(toString(std::move(error)));
      }
    }
  }
}

#if !SANITIZER_GO && !SANITIZER_APPLE
static void LogIgnoredIssues(ThreadContext *thread, IgnoreSet *ignores) {
  if (thread->tid != kMainTid) {
    Printf("ThreadSanitizer: thread T%d %s finished with ignores enabled,"
      " created at:\n", thread->tid, thread->name);
    PrintStack(SymbolizeStackId(thread->creation_stack_id));
  } else {
    Printf("ThreadSanitizer: main thread finished with ignores enabled\n");
  }
  uptr index = 0;
  bool hasPrintedFirstIgnore = false;
  while (index < ignores->Size()) {
    if (!hasPrintedFirstIgnore) {
      Printf("  One of the following ignores was not ended"
          " (in order of probability)\n");
      hasPrintedFirstIgnore = true;
    }
    uptr stackId = ignores->At(index++);
    Printf("  Ignore was enabled at:\n");
    PrintStack(SymbolizeStackId(stackId));
  }
  Die();
}

// If evaluation couldn't be done, return the node where the traversal ends.
PrintExprResult printTraversalResult(const SelectionTree::Node *N,
                                     const ASTContext &Ctx) {
  for (; N; N = N->Parent) {
    if (const Expr *E = N->ASTNode.get<Expr>()) {
      if (!E->getType().isNull() && !E->getType()->isVoidType())
        continue;
      if (auto Val = printTraversalResult(E, Ctx))
        return PrintExprResult{/*PrintedValue=*/std::move(Val), /*Expr=*/E,
                               /*Node=*/N};
    } else {
      const Decl *D = N->ASTNode.get<Decl>();
      const Stmt *S = N->ASTNode.get<Stmt>();
      if (D || S) {
        break;
      }
    }
  }
  return PrintExprResult{/*PrintedValue=*/std::nullopt, /*Expr=*/nullptr,
                         /*Node=*/N};
}

/// with New.
static void updateLogUsesOutsideSegment(Node *N, Node *New, SegmentBlock *SB) {
  SmallVector<LogVariableIntrinsic *> LogUsers;
  SmallVector<LogVariableRecord *> LPUsers;
  findLogUsers(LogUsers, N, &LPUsers);
  for (auto *LVI : LogUsers) {
    if (LVI->getParent() != SB)
      LVI->replaceVariableLocationOp(N, New);
  }
  for (auto *LVR : LPUsers) {
    LogMarker *Marker = LVR->getMarker();
    if (Marker->getParent() != SB)
      LVR->replaceVariableLocationOp(N, New);
  }
}

/* Spread symbols */
if (highThreshold == tableSize - 1) {
    const BYTE* spreadStart = tableSymbol + tableSize; /* size = tableSize + 8 (may write beyond tableSize) */
    {   U64 add = 0x0101010101010101ull;
        for (size_t s = 0; s < maxSV1; ++s, add += add) {
            int n = normalizedCounter[s];
            if (n > 0) {
                U64 val = add;
                for (int i = 8; i < n; i += 8) {
                    *(spreadStart + i) = val;
                }
                *spreadStart++ = val;
            }
        }
    }

    size_t position = 0;
    for (size_t s = 0; s < tableSize; ++s) {
        if ((position & tableMask) <= highThreshold) {
            continue;
        }
        *(tableSymbol + (position & tableMask)) = spreadStart[s];
        position += step;
    }

} else {
    U32 pos = 0;
    for (U32 sym = 0; sym < maxSV1; ++sym) {
        int freq = normalizedCounter[sym];
        if (freq > 0) {
            while (freq--) {
                tableSymbol[pos++] = (FSE_FUNCTION_TYPE)sym;
                if ((pos & tableMask) > highThreshold)
                    pos += step - (pos & tableMask);
            }
        }
    }
    assert(pos == 0);  /* Must have initialized all positions */
}

