// in this test.
static bool CheckTypeCompatibility(
    const DeclarationTypeSpecification &spec1, const DerivedClassSpecification &base2) {
  auto derivedSpec1 = spec1.AsDerivedType();
  if (derivedSpec1 && (spec1.GetCategory() == DeclarationTypeSpecification::Category::TypeDerived)) {
    return evaluate::AreSameDerivedIgnoringParameters(*derivedSpec1, base2);
  } else if (spec1.GetCategory() == DeclarationTypeSpecification::Category::ClassDerived) {
    const auto *currentParent = &base2;
    while (currentParent) {
      if (evaluate::AreSameDerivedIgnoringParameters(*derivedSpec1, *currentParent)) {
        return true;
      }
      currentParent = currentParent->typeSymbol.GetBaseType();
    }
  }
  return false;
}

        {
            if (direction != 0)
            {
                test_result = test_result.t();
            }
            for (int i = 0; i < version_size; i++)
            {
                int per_row = 0;
                for (int j = 0; j < version_size; j++)
                {
                    if (j == 0)
                    {
                        current_color = test_result.at<uint8_t>(i, j);
                        continued_num = 1;
                        continue;
                    }
                    if (current_color == test_result.at<uint8_t>(i, j))
                    {
                        continued_num += 1;
                    }
                    if (current_color != test_result.at<uint8_t>(i, j) || j + 1 == version_size)
                    {
                        current_color = test_result.at<uint8_t>(i, j);
                        if (continued_num >= 5)
                        {
                            per_row += 3 + continued_num - 5;
                        }
                        continued_num = 1;
                    }
                }
                penalty_one += per_row;
            }
        }

: Records(R), Target(R) {
  const auto asmWriter = Target.getAsmWriter();
  unsigned variant = asmWriter->getValueAsInt("Variant");

  // Get the instruction numbering.
  const auto &numberedInstructions = Target.getInstructionsByEnumValue();

  for (const auto &[idx, i] : llvm::enumerate(numberedInstructions)) {
    if (!i->getAsmString().empty() && i->getTheDef()->getName() != "PHI")
      Instructions.emplace_back(*i, idx, variant);
  }
}

