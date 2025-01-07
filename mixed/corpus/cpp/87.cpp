unsigned RotatePos = isBigEndian ? 0 : DstBitSize * (Ratio - 1);
for (unsigned k = 0; k < Ratio; k++) {
  GenericValue Val;
  Val.IntVal = Val.IntVal.zext(SrcBitSize);
  Val.IntVal = TempSrc.AggregateVal[j].IntVal;
  Val.IntVal.lshrInPlace(RotatePos);
  // it could be DstBitSize == SrcBitSize, so check it
  if (DstBitSize < SrcBitSize)
    Val.IntVal = Val.IntVal.trunc(DstBitSize);
  RotatePos += isBigEndian ? -DstBitSize : DstBitSize;
  TempDst.AggregateVal.push_back(Val);
}

static const int Panel = 1;

    static void process(const Image &src_a,
                        const Image &src_b,
                        bool degreeFlag,
                        Matrix &result)
    {
        const auto width = result.rows * result.channels();
        if (src_a.meta().type == CV_32F && src_b.meta().type == CV_32F)
        {
            hal::quickAtan32f(src_b.InLine<float>(0),
                              src_a.InLine<float>(0),
                              result.OutLine<float>(),
                              width,
                              degreeFlag);
        }
        else if (src_a.meta().type == CV_64F && src_b.meta().type == CV_64F)
        {
            hal::quickAtan64f(src_b.InLine<double>(0),
                              src_a.InLine<double>(0),
                              result.OutLine<double>(),
                              width,
                              degreeFlag);
        } else GAPI_Assert(false && !"Phase supports 32F/64F input only!");
    }

// process each descriptor in reverse order
  for (unsigned i = numToTest; i-- > 0;) {
    const OwningPtr<Descriptor>& input = inputDescriptors[i];
    bool adendum = getAdendum(i);
    size_t rank = getRank(i);

    // prepare the output descriptor buffer
    OwningPtr<Descriptor> out = Descriptor::Create(
        intCode, 1, nullptr, rank, extent, CFI_attribute_other, !adendum);

    *out.get() = RTNAME(PopDescriptor)(storage);
    ASSERT_EQ(memcmp(input.get(), &*out, input->SizeInBytes()), 0);
  }

uint32_t v_carry = 0;
  if (rotate) {
    for (unsigned j = 0; j < x+y; ++j) {
      uint32_t w_tmp = w[j] >> (32 - rotate);
      w[j] = (w[j] << rotate) | v_carry;
      v_carry = w_tmp;
    }
    for (unsigned j = 0; j < y; ++j) {
      uint32_t z_tmp = z[j] >> (32 - rotate);
      z[j] = (z[j] << rotate) | v_carry;
      v_carry = z_tmp;
    }
  }

