size_t idx = plane_idx;
                    for (int k = 0; k < ndims - 1; ++k)
                    {
                        size_t next_idx = idx % shape[k + 1];
                        int i_k = static_cast<int>((idx / shape[k + 1]) % shape[k]);
                        ptr_ += i_k * step[k];
                        ptr1_ += i_k * step1_[k];
                        ptr2_ += i_k * step2_[k];
                        ptr3_ += i_k * step3_[k];
                        idx = next_idx;
                    }

    remainder = remainder > k_ptrace_word_size ? k_ptrace_word_size : remainder;

    if (remainder == k_ptrace_word_size) {
      unsigned long data = 0;
      memcpy(&data, src, k_ptrace_word_size);

      LLDB_LOG(log, "[{0:x}]:{1:x}", addr, data);
      error = NativeProcessLinux::PtraceWrapper(
          PTRACE_POKEDATA, GetCurrentThreadID(), (void *)addr, (void *)data);
      if (error.Fail())
        return error;
    } else {
      unsigned char buff[8];
      size_t bytes_read;
      error = ReadMemory(addr, buff, k_ptrace_word_size, bytes_read);
      if (error.Fail())
        return error;

      memcpy(buff, src, remainder);

      size_t bytes_written_rec;
      error = WriteMemory(addr, buff, k_ptrace_word_size, bytes_written_rec);
      if (error.Fail())
        return error;

      LLDB_LOG(log, "[{0:x}]:{1:x} ({2:x})", addr, *(const unsigned long *)src,
               *(unsigned long *)buff);
    }

// - unwrapping of template decls
TEST(InsertionPointTests, CXX_newName) {
  Annotations Code(R"cpp(
    class C {
    public:
      $Method^void pubMethodNew();
      $Field^int PubFieldNew;

    $private^private:
      $field^int PrivFieldNew;
      $method^void privMethodNew();
      template <typename T> void privTemplateMethodNew();
    $end^};
  )cpp");

  auto AST = TestTU::withCode(Code.code()).build();
  const CXXRecordDecl &C = cast<CXXRecordDecl>(findDecl(AST, "C"));

  auto IsMethod = [](const Decl *D) { return llvm::isa<CXXMethodDecl>(D); };
  auto Any = [](const Decl *D) { return true; };

  // Test single anchors.
  auto Point = [&](Anchor A, AccessSpecifier Protection) {
    auto Loc = insertionPoint(C, {A}, Protection);
    return sourceLocToPosition(AST.getSourceManager(), Loc);
  };
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_public), Code.point("Method_new"));
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_public), Code.point("Field_new"));
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_public), Code.point("Method_new"));
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_public), Code.point("private_new"));
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_private), Code.point("method_new"));
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_private), Code.point("end_new"));
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_private), Code.point("field_new"));
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_private), Code.point("end_new"));
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_protected), Position{});
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_protected), Position{});
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_protected), Position{});
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_protected), Position{});

  // Edits when there's no match --> end of matching access control section.
  auto Edit = insertDecl("x_new", C, {}, AS_public);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("private_new"));

  Edit = insertDecl("x_new", C, {}, AS_private);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("end_new"));

  Edit = insertDecl("x_new", C, {}, AS_protected);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("end_new"));
  EXPECT_EQ(Edit->getReplacementText(), "protected:\nx_new");
}

