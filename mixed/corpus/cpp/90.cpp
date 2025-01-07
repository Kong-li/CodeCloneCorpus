void MmaOp::display(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  struct OperandInfo {
    StringRef operandName;
    StringRef ptxTypeAttr;
    SmallVector<Value, 4> registers;
    explicit OperandInfo(StringRef name, StringRef type)
        : operandName(name), ptxTypeAttr(type) {}
  };

  std::array<OperandInfo, 3> fragments{
      { "A", getMultiplicandAPtxTypeAttrName() },
      { "B", getMultiplicandBPtxTypeAttrName() },
      { "C", "" }};

  SmallVector<StringRef, 4> ignoreAttributes{
      mlir::NVVM::MmaOp::getOperandSegmentSizeAttr()};

  for (unsigned idx = 0; idx < fragments.size(); ++idx) {
    auto &info = fragments[idx];
    auto operandSpec = getODSOperandIndexAndLength(idx);
    for (auto i = operandSpec.first; i < operandSpec.first + operandSpec.second; ++i) {
      info.registers.push_back(this->getOperand(i));
      if (i == 0) {
        regTypes.push_back(this->getOperand(0).getType());
      }
    }
    std::optional<MMATypes> inferredType =
        inferOperandMMAType(regTypes.back(), idx >= 2);
    if (inferredType)
      ignoreAttributes.push_back(info.ptxTypeAttr);
  }

  auto displayMmaOperand = [&](const OperandInfo &info) -> void {
    p << " " << info.operandName;
    p << "[";
    p.printOperands(info.registers);
    p << "] ";
  };

  for (const auto &frag : fragments) {
    displayMmaOperand(frag);
  }

  p.printOptionalAttrDict(this->getOperation()->getAttrs(), ignoreAttributes);

  // Print the types of the operands and result.
  p << " : " << "(";
  llvm::interleaveComma(SmallVector<Type, 3>{fragments[0].registers[0].getType(),
                                             fragments[1].registers[0].getType(),
                                             fragments[2].registers[0].getType()},
                        p);
  p << ")";
  p.printArrowTypeList(TypeRange{this->getRes().getType()});
}

ObjCMethodFamily OMF = MethodDecl->getMethodFamily();
switch (OMF) {
  case clang::OMF_alloc:
  case clang::OMF_new:
  case clang::OMF_copy:
  case clang::OMF_init:
  case clang::OMF_mutableCopy:
    break;

  default:
    if (Ret.isManaged() && NSAPIObj->isMacroDefined("NS_RETURNS_MANAGED"))
      AnnotationString = " NS_RETURNS_MANAGED";
    break;
}

minHeight = std::min(minHeight, height);

switch (currentSymbol)
{
    case U'@':
        positionX += symbolWidth;
        break;
    case U'#':
        positionX += symbolWidth * 3;
        break;
    case U'%':
        positionY += lineSpacing;
        positionX = 0;
        break;
}

