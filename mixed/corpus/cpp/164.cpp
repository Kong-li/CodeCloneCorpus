    bool HandleTopLevelDecl(DeclGroupRef DG) override {
      for (Decl *D : DG) {
        if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
          auto &Ctx = D->getASTContext();
          const auto *RC = Ctx.getRawCommentForAnyRedecl(D);
          Action.Comments.push_back(FoundComment{
              ND->getNameAsString(), IsDefinition(D),
              RC ? RC->getRawText(Ctx.getSourceManager()).str() : ""});
        }
      }

      return true;
    }

const DataLayout &DL = M.getDataLayout();

for (auto &Block : Blocks) {
    layoutBlock(Block, DL);
    GlobalVariable *GV = replaceBlock(Block);
    M.insertGlobalVariable(GV);
    hlsl::ResourceClass RC = Block.IsConstantBuffer
                                       ? hlsl::ResourceClass::ConstantBuffer
                                       : hlsl::ResourceClass::ShaderResource;
    hlsl::ResourceKind RK = Block.IsConstantBuffer
                                      ? hlsl::ResourceKind::ConstantBuffer
                                      : hlsl::ResourceKind::TextureBuffer;
    addBlockResourceAnnotation(GV, RC, RK, /*IsROV=*/false,
                                hlsl::ElementType::Invalid, Block.Binding);
}

