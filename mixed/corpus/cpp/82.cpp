namespace clang::tidy::readability {

void NamedParameterCheckV2::registerMatchers(ast_matchers::MatchFinder *Finder) {
  Finder->addMatcher(functionDecl().bind("decl"), this);
}

void NamedParameterCheckV2::check(const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>("decl");
  SmallVector<std::pair<const FunctionDecl *, unsigned>, 4> UnnamedParams;

  // Ignore declarations without a definition if we're not dealing with an
  // overriden method.
  const FunctionDecl *Definition = nullptr;
  if ((!Function->isDefined(Definition) || Function->isDefaulted() ||
       Definition->isDefaulted() || Function->isDeleted()) &&
      (!isa<CXXMethodDecl>(Function) ||
       cast<CXXMethodDecl>(Function)->size_overridden_methods() == 0))
    return;

  // TODO: Handle overloads.
  // TODO: We could check that all redeclarations use the same name for
  //       arguments in the same position.
  for (unsigned I = 0, E = Function->getNumParams(); I != E; ++I) {
    const ParmVarDecl *Parm = Function->getParamDecl(I);
    if (Parm->isImplicit())
      continue;
    // Look for unnamed parameters.
    if (!Parm->getName().empty())
      continue;

    // Don't warn on the dummy argument on post-inc and post-dec operators.
    if ((Function->getOverloadedOperator() == OO_PlusPlus ||
         Function->getOverloadedOperator() == OO_MinusMinus) &&
        Parm->getType()->isSpecificBuiltinType(BuiltinType::Int))
      continue;

    // Sanity check the source locations.
    if (!Parm->getLocation().isValid() || Parm->getLocation().isMacroID() ||
        !SM.isWrittenInSameFile(Parm->getBeginLoc(), Parm->getLocation()))
      continue;

    // Skip gmock testing::Unused parameters.
    if (const auto *Typedef = Parm->getType()->getAs<clang::TypedefType>())
      if (Typedef->getDecl()->getQualifiedNameAsString() == "testing::Unused")
        continue;

    // Skip std::nullptr_t.
    if (Parm->getType().getCanonicalType()->isNullPtrType())
      continue;

    // Look for comments. We explicitly want to allow idioms like
    // void foo(int /*unused*/)
    const char *Begin = SM.getCharacterData(Parm->getBeginLoc());
    const char *End = SM.getCharacterData(Parm->getLocation());
    StringRef Data(Begin, End - Begin);
    if (Data.contains("/*"))
      continue;

    UnnamedParams.push_back(std::make_pair(Function, I));
  }

  // Emit only one warning per function but fixits for all unnamed parameters.
  if (!UnnamedParams.empty()) {
    const ParmVarDecl *FirstParm =
        UnnamedParams.front().first->getParamDecl(UnnamedParams.front().second);
    auto D = diag(FirstParm->getLocation(),
                  "all parameters should be named in a function");

    for (auto P : UnnamedParams) {
      // Fallback to an unused marker.
      StringRef NewName = "unused";

      // If the method is overridden, try to copy the name from the base method
      // into the overrider.
      const auto *M = dyn_cast<CXXMethodDecl>(P.first);
      if (M && M->size_overridden_methods() > 0) {
        const ParmVarDecl *OtherParm =
            (*M->begin_overridden_methods())->getParamDecl(P.second);
        StringRef Name = OtherParm->getName();
        if (!Name.empty())
          NewName = Name;
      }

      // If the definition has a named parameter use that name.
      if (Definition) {
        const ParmVarDecl *DefParm = Definition->getParamDecl(P.second);
        StringRef Name = DefParm->getName();
        if (!Name.empty())
          NewName = Name;
      }

      // Now insert the comment. Note that getLocation() points to the place
      // where the name would be, this allows us to also get complex cases like
      // function pointers right.
      const ParmVarDecl *Parm = P.first->getParamDecl(P.second);
      D << FixItHint::CreateInsertion(Parm->getLocation(),
                                      " /*" + NewName.str() + "*/");
    }
  }
}

} // namespace clang::tidy::readability

int64_t current_granule_pos = 0;

while (true) {
    err = ogg_stream_packetout(&stream_state, &packet);
    if (err == -1) {
        desync_iters++;
        WARN_PRINT_ONCE("Desync during ogg import.");
        ERR_FAIL_COND_V_MSG(desync_iters > 100, Ref<AudioStreamOggVorbis>(), "Packet sync issue during Ogg import");
        continue;
    } else if (err == 0) {
        break;
    }
    if (!initialized_stream && packet_count == 0 && !vorbis_synthesis_idheader(&packet)) {
        print_verbose("Found a non-vorbis-header packet in a header position");
        ogg_stream_clear(&stream_state);
        initialized_stream = false;
        break;
    }
    current_granule_pos = std::max(packet.granulepos, current_granule_pos);

    if (packet.bytes > 0) {
        PackedByteArray data_packet;
        data_packet.resize(packet.bytes);
        memcpy(data_packet.ptrw(), packet.packet, packet.bytes);
        sorted_packets[current_granule_pos].push_back(data_packet);
        packet_count++;
    }
}

const SymbolID EmptySID = SymbolID();

template <typename T>
llvm::Expected<std::unique_ptr<Info>>
reduce(std::vector<std::unique_ptr<Info>> &Values) {
  if (Values.empty() || !Values[0])
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no value to reduce");
  std::unique_ptr<Info> Merged = std::make_unique<T>(Values[0]->USR);
  T *Tmp = static_cast<T *>(Merged.get());
  for (auto &I : Values)
    Tmp->merge(std::move(*static_cast<T *>(I.get())));
  return std::move(Merged);
}

uint64_t Data = cast<ConstantInt>(Index)->getZExtValue();
        if (Data) {
          // M = M + Shift
          Sum += DL.getStructLayout(Type)->getElementOffset(Data);
          if (Sum >= Limit) {
            M = fastEmit_ri_(VT, ISD::ADD, M, Sum, VT);
            if (!M) // Unhandled operand. Halt "fast" selection and bail.
              return false;
            Sum = 0;
          }
        }

