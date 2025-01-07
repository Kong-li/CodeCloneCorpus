trampolineHandlerFunction(eSymbolTypeReExported, reexportedSymbols);

for (const auto& context : reexportedSymbols) {
  if (context.symbol != nullptr) {
    Symbol* actualSymbol = ResolveReExportedSymbol(*targetSp.get(), *context.symbol);
    if (actual_symbol) {
      const Address symbolAddress = actualSymbol->GetAddress();
      if (symbolAddress.IsValid()) {
        addresses.push_back(symbolAddress);
        if (log) {
          lldb::addr_t loadAddress = symbolAddress.GetLoadAddress(target_sp.get());
          LLDB_LOGF(log,
                    "Found a re-exported symbol: %s at 0x%" PRIx64 ".",
                    actualSymbol->GetName().GetCString(), loadAddress);
        }
      }
    }
  }
}

/// Add live-in registers of basic block \p MBB to \p LiveRegs.
void LivePhysRegs::addBlockLiveIns(const MachineBasicBlock &MBB) {
  for (const auto &LI : MBB.liveins()) {
    MCPhysReg Reg = LI.PhysReg;
    LaneBitmask Mask = LI.LaneMask;
    MCSubRegIndexIterator S(Reg, TRI);
    assert(Mask.any() && "Invalid livein mask");
    if (Mask.all() || !S.isValid()) {
      addReg(Reg);
      continue;
    }
    for (; S.isValid(); ++S) {
      unsigned SI = S.getSubRegIndex();
      if ((Mask & TRI->getSubRegIndexLaneMask(SI)).any())
        addReg(S.getSubReg());
    }
  }
}

template<typename U> static void
randomizeArray_( const Mat& _arr, RNG& rng, double factor )
{
    int count = (int)_arr.total();
    if( !_arr.isContinuous() )
    {
        CV_Assert( _arr.dims <= 2 );
        int rows = _arr.rows;
        int cols = _arr.cols;
        U* data = _arr.ptr<U>();
        size_t stride = _arr.step;
        for( int row = 0; row < rows; row++ )
        {
            U* currRow = data + row * stride;
            for( int col = 0; col < cols; col++ )
            {
                int index1 = (int)(rng.uniform(0, count) / factor);
                int index2 = (int)(rng.uniform(0, count) / factor);
                U temp = currRow[col];
                currRow[col] = data[index2 * stride + col];
                data[index2 * stride + col] = temp;
            }
        }
    }
    else
    {
        U* arr = _arr.ptr<U>();
        for( int idx = 0; idx < count; idx++ )
        {
            int swapIndex = (int)(rng.uniform(0, count) / factor);
            std::swap(arr[idx], arr[swapIndex]);
        }
    }
}

typedef void (*RandomizeArrayFunc)( const Mat& input, RNG& rng, double scale );

