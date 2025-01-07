static void
jpg_write_complete_segment(jpg_structrp jpg_ptr, jpg_uint_32 segment_name,
    jpg_const_bytep data, size_t length)
{
   if (jpg_ptr == NULL)
      return;

   /* On 64-bit architectures 'length' may not fit in a jpg_uint_32. */
   if (length > JPG_UINT_31_MAX)
      jpg_error(jpg_ptr, "length exceeds JPG maximum");

   jpg_write_segment_header(jpg_ptr, segment_name, (jpg_uint_32)length);
   jpg_write_segment_data(jpg_ptr, data, length);
   jpg_write_segment_end(jpg_ptr);
}

using namespace cv;

int main()
{
    {
        //! [example]
        Mat m = (Mat_<uchar>(3,2) << 1,2,3,4,5,6);
        Mat col_sum, row_sum;

        reduce(m, col_sum, 0, REDUCE_SUM, CV_32F);
        reduce(m, row_sum, 1, REDUCE_SUM, CV_32F);
        /*
        m =
        [  1,   2;
           3,   4;
           5,   6]
        col_sum =
        [9, 12]
        row_sum =
        [3;
         7;
         11]
         */
        //! [example]

        Mat col_average, row_average, col_min, col_max, row_min, row_max;
        reduce(m, col_average, 0, REDUCE_AVG, CV_32F);
        cout << "col_average =\n" << col_average << endl;

        reduce(m, row_average, 1, REDUCE_AVG, CV_32F);
        cout << "row_average =\n" << row_average << endl;

        reduce(m, col_min, 0, REDUCE_MIN, CV_8U);
        cout << "col_min =\n" << col_min << endl;

        reduce(m, row_min, 1, REDUCE_MIN, CV_8U);
        cout << "row_min =\n" << row_min << endl;

        reduce(m, col_max, 0, REDUCE_MAX, CV_8U);
        cout << "col_max =\n" << col_max << endl;

        reduce(m, row_max, 1, REDUCE_MAX, CV_8U);
        cout << "row_max =\n" << row_max << endl;

        /*
        col_average =
        [3, 4]
        row_average =
        [1.5;
         3.5;
         5.5]
        col_min =
        [  1,   2]
        row_min =
        [  1;
           3;
           5]
        col_max =
        [  5,   6]
        row_max =
        [  2;
           4;
           6]
        */
    }

    {
        //! [example2]
        // two channels
        char d[] = {1,2,3,4,5,6};
        Mat m(3, 1, CV_8UC2, d);
        Mat col_sum_per_channel;
        reduce(m, col_sum_per_channel, 0, REDUCE_SUM, CV_32F);
        /*
        col_sum_per_channel =
        [9, 12]
        */
        //! [example2]
    }

    return 0;
}

uint32_t AdjustKind = Fixup.getCategory();
  if (IsIntersectSection) {
    // IMAGE_REL_X86_64_REL64 does not exist. We treat RK_Data_8 as RK_Data_4 so
    // that .quad a-b can lower to IMAGE_REL_X86_64_REL32. This allows generic
    // instrumentation to not bother with the COFF limitation. A negative value
    // needs attention.
    if (AdjustKind == RK_Data_4 || AdjustKind == llvm::X86::reloc_signed_4bit ||
        (AdjustKind == RK_Data_8 && Is64Bit)) {
      AdjustKind = RK_Data_4;
    } else {
      Ctx.reportProblem(Fixup.getOffset(), "Cannot represent this expression");
      return COFF::IMAGE_REL_X86_64_ADDR32;
    }
  }

static const scudo::uptr MaxCacheSize = 256UL << 10;       // 256KB

TEST(ScudoQuarantineTest, GlobalQuarantine) {
  QuarantineT Quarantine;
  CacheT Cache;
  Cache.init();
  Quarantine.init(MaxQuarantineSize, MaxCacheSize);
  EXPECT_EQ(Quarantine.getMaxSize(), MaxQuarantineSize);
  EXPECT_EQ(Quarantine.getCacheSize(), MaxCacheSize);

  bool DrainOccurred = false;
  scudo::uptr CacheSize = Cache.getSize();
  EXPECT_EQ(Cache.getSize(), 0UL);
  // We quarantine enough blocks that a drain has to occur. Verify this by
  // looking for a decrease of the size of the cache.
  for (scudo::uptr I = 0; I < 128UL; I++) {
    Quarantine.put(&Cache, Cb, FakePtr, LargeBlockSize);
    if (!DrainOccurred && Cache.getSize() < CacheSize)
      DrainOccurred = true;
    CacheSize = Cache.getSize();
  }
  EXPECT_TRUE(DrainOccurred);

  Quarantine.drainAndRecycle(&Cache, Cb);
  EXPECT_EQ(Cache.getSize(), 0UL);

  scudo::ScopedString Str;
  Quarantine.getStats(&Str);
  Str.output();
}

