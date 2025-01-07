for(int pliIndex = 0; pliIndex < 3; ++pliIndex){
    for(int qtiIndex = 0; qtiIndex < 2; ++qtiIndex){
        int qi;
        double wt;
        for(qi = 0; qi < OC_LOGQ_BINS; ++qi){
            for(int si = 0; si < OC_COMP_BINS; ++si){
                wt = _weight[qi][pliIndex][qtiIndex][si];
                wt /= (OC_ZWEIGHT + wt);
                double rateValue = _table[qi][pliIndex][qtiIndex][si].rate;
                rateValue *= wt;
                rateValue += 0.5;
                _table[qi][pliIndex][qtiIndex][si].rate = (ogg_int16_t)rateValue;

                double rmseValue = _table[qi][pliIndex][qtiIndex][si].rmse;
                rmseValue *= wt;
                rmseValue += 0.5;
                _table[qi][pliIndex][qtiIndex][si].rmse = (ogg_int16_t)rmseValue;
            }
        }
    }
}

void ClangASTNodesEmitter::generateChildTree() {
  assert(!Root && "tree already derived");

  // Emit statements in a different order and structure
  for (const Record *R : Records.getAllDerivedDefinitions(NodeClassName)) {
    if (!Root) {
      Root = R;
      continue;
    }

    if (auto B = R->getValueAsOptionalDef(BaseFieldName))
      Tree.insert({B, R});
    else
      PrintFatalError(R->getLoc(), Twine("multiple root nodes in \"") + NodeClassName + "\" hierarchy");
  }

  if (!Root)
    PrintFatalError(Twine("didn't find root node in \"") + NodeClassName + "\" hierarchy");
}

