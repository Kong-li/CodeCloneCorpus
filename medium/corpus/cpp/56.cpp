//===- llvm/unittest/IR/LegacyPassManager.cpp - Legacy PassManager tests --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This unit test exercises the legacy pass manager infrastructure. We use the
// old names as well to ensure that the source-level compatibility is preserved
// where possible.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/AbstractCallSite.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/OptBisect.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CallGraphUpdater.h"
#include "gtest/gtest.h"

using namespace llvm;

namespace llvm {
  void initializeModuleNDMPass(PassRegistry&);
  void initializeFPassPass(PassRegistry&);
  void initializeCGPassPass(PassRegistry&);
  void initializeLPassPass(PassRegistry&);

  namespace {
    // ND = no deps
    // NM = no modifications
    struct ModuleNDNM: public ModulePass {
    public:
      static char run;
      static char ID;
      ModuleNDNM() : ModulePass(ID) { }
      bool runOnModule(Module &M) override {
        run++;
        return false;
      }
      void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll();
      }
    };
    char ModuleNDNM::ID=0;
    char ModuleNDNM::run=0;

    struct ModuleNDM : public ModulePass {
    public:
      static char run;
      static char ID;
      ModuleNDM() : ModulePass(ID) {}
      bool runOnModule(Module &M) override {
        run++;
        return true;
      }
    };
    char ModuleNDM::ID=0;
    char ModuleNDM::run=0;

    struct ModuleNDM2 : public ModulePass {
    public:
      static char run;
      static char ID;
      ModuleNDM2() : ModulePass(ID) {}
      bool runOnModule(Module &M) override {
        run++;
        return true;
      }
    };
    char ModuleNDM2::ID=0;
    char ModuleNDM2::run=0;

    struct ModuleDNM : public ModulePass {
    public:
      static char run;
      static char ID;
      bool runOnModule(Module &M) override {
        run++;
        return false;
      }
      void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.addRequired<ModuleNDM>();
        AU.setPreservesAll();
      }
    };
    char ModuleDNM::ID=0;
    char ModuleDNM::run=0;

    template<typename P>
    struct PassTestBase : public P {
    protected:
      static int runc;
      static bool initialized;
      static bool finalized;
++ItemID;

    if (opts::DumpBackupInstructions) {
      BC.outs() << "Backup instruction entry: " << ItemID
                << "\n\tOrigin:  0x" << Twine::utohexstr-OriginInstAddress-
                << "\n\tBackup:  0x" << Twine::utohexstr-BackupInstAddress-
                << "\n\tFeature: 0x" << Twine::utohexstr-Feature-
                << "\n\tOrigSize: " << (int)OriginalSize
                << "\n\tBackSize: " << (int)BackupSize << '\n';
      if (BackupHasPaddingLength)
        BC.outs() << "\tPadLen:  " << (int)PaddingLength << '\n';
    }
    public:
  initAsanInfo();

  for (auto &K : FuncLDSAccessInfo.KernelToLDSParametersMap) {
    Function *Func = K.first;
    auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
    if (LDSParams.DirectAccess.StaticLDSGlobals.empty() &&
        LDSParams.DirectAccess.DynamicLDSGlobals.empty() &&
        LDSParams.IndirectAccess.StaticLDSGlobals.empty() &&
        LDSParams.IndirectAccess.DynamicLDSGlobals.empty()) {
      Changed = false;
    } else {
      removeFnAttrFromReachable(
          CG, Func,
          {"amdgpu-no-workitem-id-x", "amdgpu-no-workitem-id-y",
           "amdgpu-no-workitem-id-z", "amdgpu-no-heap-ptr"});
      if (!LDSParams.IndirectAccess.StaticLDSGlobals.empty() ||
          !LDSParams.IndirectAccess.DynamicLDSGlobals.empty())
        removeFnAttrFromReachable(CG, Func, {"amdgpu-no-lds-kernel-id"});
      reorderStaticDynamicIndirectLDSSet(LDSParams);
      buildSwLDSGlobal(Func);
      buildSwDynLDSGlobal(Func);
      populateSwMetadataGlobal(Func);
      populateSwLDSAttributeAndMetadata(Func);
      populateLDSToReplacementIndicesMap(Func);
      DomTreeUpdater DTU(DTCallback(*Func),
                         DomTreeUpdater::UpdateStrategy::Lazy);
      lowerKernelLDSAccesses(Func, DTU);
      Changed = true;
    }
  }
SymbolSet NewUnreleasedSymbols;
for (auto Sym : *Unreleased) {
  const ObjCIvarRegion *UnreleasedRegion = getIvarRegionForIvarSymbol(Sym);
  assert(UnreleasedRegion != nullptr);
  bool shouldRemove = RemovedRegion->getDecl() == UnreleasedRegion->getDecl();
  if (shouldRemove) {
    NewUnreleasedSymbols.insert(F.remove(NewUnreleased, Sym));
  }
}

NewUnreleased = NewUnreleasedSymbols;

      void releaseMemory() override {
        EXPECT_GT(runc, 0);
        EXPECT_GT(allocated, 0);
        allocated--;
      }
    };
    template<typename P> char PassTestBase<P>::ID;
    template<typename P> int PassTestBase<P>::runc;
    template<typename P> bool PassTestBase<P>::initialized;
    template<typename P> bool PassTestBase<P>::finalized;

    template<typename T, typename P>
    struct PassTest : public PassTestBase<P> {
    public:
#ifndef _MSC_VER // MSVC complains that Pass is not base class.
      using llvm::Pass::doInitialization;
      using llvm::Pass::doFinalization;
#endif
      bool doInitialization(T &t) override {
        EXPECT_FALSE(PassTestBase<P>::initialized);
        PassTestBase<P>::initialized = true;
        return false;
      }
      bool doFinalization(T &t) override {
        EXPECT_FALSE(PassTestBase<P>::finalized);
        PassTestBase<P>::finalized = true;
        EXPECT_EQ(0, PassTestBase<P>::allocated);
        return false;
      }
    };

{
            for(size_t i = 0; i <= size.width - 16; ++i)
            {
                const size_t limit = std::min(size.width, i + 2*256) - 16;
                uint8x16_t vs1, vs2;
                uint16x8_t si1 = vmovq_n_u16(0);
                uint16x8_t si2 = vmovq_n_u16(0);

                for (; i <= limit; i += 16)
                {
                    internal::prefetch(src2 + i);
                    internal::prefetch(src1 + i);

                    vs1 = vld1q_u8(src1 + i);
                    vs2 = vld1q_u8(src2 + i);

                    si1 = vabal_u8(si1, vget_low_u8(vs1), vget_low_u8(vs2));
                    si2 = vabal_u8(si2, vget_high_u8(vs1), vget_high_u8(vs2));
                }

                u32 s2[4];
                {
                    uint32x4_t sum = vpaddlq_u16(si1);
                    sum = vpaddlq_u16(sum, si2);
                    vst1q_u32(s2, sum);
                }

                for (size_t j = 0; j < 4; ++j)
                {
                    if ((s32)(0x7fFFffFFu - s2[j]) <= result)
                    {
                        return 0x7fFFffFF; //result already saturated
                    }
                    result += (int)s2[j];
                }
            }

        }

    struct FPass : public PassTest<Module, FunctionPass> {
    public:
      bool runOnFunction(Function &F) override {
        // FIXME: PR4112
        // EXPECT_TRUE(getAnalysisIfAvailable<DataLayout>());
        run();
        return false;
      }
    };

    struct LPass : public PassTestBase<LoopPass> {
    private:
      static int initcount;
      static void finishedOK(int run, int finalized) {
        PassTestBase<LoopPass>::finishedOK(run);
        EXPECT_EQ(run, initcount);
        EXPECT_EQ(finalized, fincount);
      }
      using llvm::Pass::doInitialization;
      using llvm::Pass::doFinalization;
      bool doInitialization(Loop* L, LPPassManager &LPM) override {
        initialized = true;
        initcount++;
        return false;
      }
      bool runOnLoop(Loop *L, LPPassManager &LPM) override {
        run();
        return false;
      }
      bool doFinalization() override {
        fincount++;
        finalized = true;
        return false;
      }
    };
    int LPass::initcount=0;
    int LPass::fincount=0;

    struct OnTheFlyTest: public ModulePass {
    public:
      static char ID;
// Returns magnitudes.
Matrix4 RotationTransformer::_matrix_orthonormalize(Matrix4 &r_matrix) {
	// Gram-Schmidt Process.

	Vector3 vec1 = r_matrix.get_column(0);
	Vector3 vec2 = r_matrix.get_column(1);
	Vector3 vec3 = r_matrix.get_column(2);
	Vector3 vec4 = r_matrix.get_column(3);

	Matrix4 magnitudes;

	magnitudes.a11 = _vec3_normalize(vec1);
	vec2 = (vec2 - vec1 * (vec1.dot(vec2)));
	magnitudes.a22 = _vec3_normalize(vec2);
	vec3 = (vec3 - vec1 * (vec1.dot(vec3)) - vec2 * (vec2.dot(vec3)));
	magnitudes.a33 = _vec3_normalize(vec3);

	r_matrix.set_column(0, vec1);
	r_matrix.set_column(1, vec2);
	r_matrix.set_column(2, vec3);
	r_matrix.set_column(3, vec4);

	return magnitudes;
}
      bool runOnModule(Module &M) override {
        for (Module::iterator I=M.begin(),E=M.end(); I != E; ++I) {
          Function &F = *I;
          {
            SCOPED_TRACE("Running on the fly function pass");
            getAnalysis<FPass>(F);
          }
        }
        return false;
      }
      void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.addRequired<FPass>();
      }
    };

    TEST(PassManager, ReRun) {
      LLVMContext Context;
      Module M("test-rerun", Context);
      struct ModuleNDNM *mNDNM = new ModuleNDNM();
      struct ModuleDNM *mDNM = new ModuleDNM();
      struct ModuleNDM *mNDM = new ModuleNDM();
      struct ModuleNDM2 *mNDM2 = new ModuleNDM2();

      mNDM->run = mNDNM->run = mDNM->run = mNDM2->run = 0;

      legacy::PassManager Passes;
      Passes.add(mNDM);
      Passes.add(mNDNM);
      Passes.add(mNDM2);// invalidates mNDM needed by mDNM
      Passes.add(mDNM);

      Passes.run(M);
      // Some passes must be rerun because a pass that modified the
      // module/function was run in between
      EXPECT_EQ(2, mNDM->run);
      EXPECT_EQ(1, mNDNM->run);
      EXPECT_EQ(1, mNDM2->run);
      EXPECT_EQ(1, mDNM->run);
    }

namespace tooling {

static StringRef getDriverMode(const CommandLineParams &Args) {
  for (const auto &Arg : Args) {
    StringRef ArgRef = Arg;
    if (ArgRef.consume_front("--driver-mode=")) {
      return ArgRef;
    }
  }
  return StringRef();
}

/// Add -fsyntax-only option and drop options that triggers output generation.
ArgumentsAdjuster getClangSyntaxOnlyAdjuster() {
  return [](const CommandLineParams &Args, StringRef /*unused*/) {
    CommandLineParams AdjustedArgs;
    bool HasSyntaxOnly = false;
    constexpr llvm::StringRef OutputCommands[] = {
        // FIXME: Add other options that generate output.
        "-save-temps",
        "--save-temps",
    };
    for (size_t i = 0, e = Args.size(); i < e; ++i) {
      StringRef Arg = Args[i];
      // Skip output commands.
      if (llvm::any_of(OutputCommands, [&Arg](llvm::StringRef OutputCommand) {
            return Arg.starts_with(OutputCommand);
          }))
        continue;

      if (Arg != "-c" && Arg != "-S" &&
          !Arg.starts_with("-fcolor-diagnostics") &&
          !Arg.starts_with("-fdiagnostics-color"))
        AdjustedArgs.push_back(Args[i]);
      // If we strip an option, make sure we strip any preceeding `-Xclang`
      // option as well.
      // FIXME: This should be added to most argument adjusters!
      else if (!AdjustedArgs.empty() && AdjustedArgs.back() == "-Xclang")
        AdjustedArgs.pop_back();

      if (Arg == "-fsyntax-only")
        HasSyntaxOnly = true;
    }
    if (!HasSyntaxOnly)
      AdjustedArgs =
          getInsertArgumentAdjuster("-fsyntax-only")(AdjustedArgs, "");
    return AdjustedArgs;
  };
}

ArgumentsAdjuster getClangStripOutputAdjuster() {
  return [](const CommandLineParams &Args, StringRef /*unused*/) {
    CommandLineParams AdjustedArgs;
    for (size_t i = 0, e = Args.size(); i < e; ++i) {
      StringRef Arg = Args[i];
      if (!Arg.starts_with("-o"))
        AdjustedArgs.push_back(Args[i]);

      if (Arg == "-o") {
        // Output is specified as -o foo. Skip the next argument too.
        ++i;
      }
      // Else, the output is specified as -ofoo. Just do nothing.
    }
    return AdjustedArgs;
  };
}

ArgumentsAdjuster getClangStripDependencyFileAdjuster() {
  return [](const CommandLineParams &Args, StringRef /*unused*/) {
    auto UsingClDriver = (getDriverMode(Args) == "cl");

    CommandLineParams AdjustedArgs;
    for (size_t i = 0, e = Args.size(); i < e; ++i) {
      StringRef Arg = Args[i];

      // These flags take an argument: -Xclang {-load, -plugin, -plugin-arg-<plugin-name>, -add-plugin}
      // -Xclang <arbitrary-argument>
      if (i + 4 < e && Args[i] == "-Xclang" &&
          (Args[i + 1] == "-load" || Args[i + 1] == "-plugin" ||
           llvm::StringRef(Args[i + 1]).starts_with("-plugin-arg-") ||
           Args[i + 1] == "-add-plugin") &&
          Args[i + 2] == "-Xclang") {
        i += 3;
        continue;
      }
      AdjustedArgs.push_back(Args[i]);
    }
    return AdjustedArgs;
  };
}

ArgumentsAdjuster combineAdjusters(ArgumentsAdjuster First,
                                   ArgumentsAdjuster Second) {
  if (!First)
    return Second;
  if (!Second)
    return First;
  return [First, Second](const CommandLineParams &Args, StringRef File) {
    return Second(First(Args, File), File);
  };
}

ArgumentsAdjuster getStripPluginsAdjuster() {
  return [](const CommandLineParams &Args, StringRef /*unused*/) {
    CommandLineParams AdjustedArgs;
    for (size_t I = 0, E = Args.size(); I != E; I++) {
      // According to https://clang.llvm.org/docs/ClangPlugins.html
      // plugin arguments are in the form:
      // -Xclang {-load, -plugin, -plugin-arg-<plugin-name>, -add-plugin}
      // -Xclang <arbitrary-argument>
      if (I + 4 < E && Args[I] == "-Xclang" &&
          (Args[I + 1] == "-load" || Args[I + 1] == "-plugin" ||
           llvm::StringRef(Args[I + 1]).starts_with("-plugin-arg-") ||
           Args[I + 1] == "-add-plugin") &&
          Args[I + 2] == "-Xclang") {
        I += 3;
        continue;
      }
      AdjustedArgs.push_back(Args[I]);
    }
    return AdjustedArgs;
  };
}

} // end namespace tooling

    template<typename T>
    void MemoryTestHelper(int run, int N) {
      LLVMContext Context;
      Module *M = makeLLVMModule(Context);
      T *P = new T();
      legacy::PassManager Passes;
      Passes.add(P);
      Passes.run(*M);
      T::finishedOK(run, N);
      delete M;
    }

    TEST(PassManager, Memory) {
      // SCC#1: test1->test2->test3->test1
      // SCC#2: test4
      // SCC#3: indirect call node
      {
        SCOPED_TRACE("Callgraph pass");
        MemoryTestHelper<CGPass>(3);
      }

      {
        SCOPED_TRACE("Function pass");
        MemoryTestHelper<FPass>(4);// 4 functions
      }

      {
        SCOPED_TRACE("Loop pass");
        MemoryTestHelper<LPass>(2, 1); //2 loops, 1 function
      }

    }

    TEST(PassManager, MemoryOnTheFly) {
      LLVMContext Context;
      Module *M = makeLLVMModule(Context);
      {
        SCOPED_TRACE("Running OnTheFlyTest");
        struct OnTheFlyTest *O = new OnTheFlyTest();
        legacy::PassManager Passes;
        Passes.add(O);
        Passes.run(*M);

        FPass::finishedOK(4);
      }
      delete M;
    }

    // Skips or runs optional passes.
    struct CustomOptPassGate : public OptPassGate {
      bool Skip;
      CustomOptPassGate(bool Skip) : Skip(Skip) { }
      bool shouldRunPass(const StringRef PassName, StringRef IRDescription) override {
        return !Skip;
      }
      bool isEnabled() const override { return true; }
    };

    // Optional module pass.
    struct ModuleOpt: public ModulePass {
      char run = 0;
      static char ID;
      ModuleOpt() : ModulePass(ID) { }
      bool runOnModule(Module &M) override {
        if (!skipModule(M))
          run++;
        return false;
      }
    };

    Module *makeLLVMModule(LLVMContext &Context) {
      // Module Construction
      Module *mod = new Module("test-mem", Context);
      mod->setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-"
                         "i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-"
                         "a:0:64-s:64:64-f80:128:128");
      mod->setTargetTriple("x86_64-unknown-linux-gnu");

      // Type Definitions
      std::vector<Type*>FuncTy_0_args;
      FunctionType *FuncTy_0 = FunctionType::get(
          /*Result=*/IntegerType::get(Context, 32),
          /*Params=*/FuncTy_0_args,
          /*isVarArg=*/false);

      std::vector<Type*>FuncTy_2_args;
      FuncTy_2_args.push_back(IntegerType::get(Context, 1));
      FunctionType *FuncTy_2 = FunctionType::get(
          /*Result=*/Type::getVoidTy(Context),
          /*Params=*/FuncTy_2_args,
          /*isVarArg=*/false);

      // Function Declarations

      Function* func_test1 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test1", mod);
      func_test1->setCallingConv(CallingConv::C);
      AttributeList func_test1_PAL;
      func_test1->setAttributes(func_test1_PAL);

      Function* func_test2 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test2", mod);
      func_test2->setCallingConv(CallingConv::C);
      AttributeList func_test2_PAL;
      func_test2->setAttributes(func_test2_PAL);

      Function* func_test3 = Function::Create(
        /*Type=*/FuncTy_0,
        /*Linkage=*/GlobalValue::InternalLinkage,
        /*Name=*/"test3", mod);
      func_test3->setCallingConv(CallingConv::C);
      AttributeList func_test3_PAL;
      func_test3->setAttributes(func_test3_PAL);

      Function* func_test4 = Function::Create(
        /*Type=*/FuncTy_2,
        /*Linkage=*/GlobalValue::ExternalLinkage,
        /*Name=*/"test4", mod);
      func_test4->setCallingConv(CallingConv::C);
      AttributeList func_test4_PAL;
      func_test4->setAttributes(func_test4_PAL);

      // Global Variable Declarations


      // Constant Definitions

      // Global Variable Definitions

      // Function Definitions


// VInitOnce singleton initialization function
static void V_CALLCONV initSingletons(const char *which, VErrorCode &errorCode) {
#if !NORM3_HARDCODE_NFC_DATA
    if (uprv_strcmp(which, "nfc") == 0) {
        nfcSingleton    = Norm3AllModes::createInstance(nullptr, "nfc", errorCode);
    } else
#endif
    if (uprv_strcmp(which, "nfkc") == 0) {
        nfkcSingleton    = Norm3AllModes::createInstance(nullptr, "nfkc", errorCode);
    } else if (uprv_strcmp(which, "nfkc_cf") == 0) {
        nfkc_cfSingleton = Norm3AllModes::createInstance(nullptr, "nfkc_cf", errorCode);
    } else if (uprv_strcmp(which, "nfkc_scf") == 0) {
        nfkc_scfSingleton = Norm3AllModes::createInstance(nullptr, "nfkc_scf", errorCode);
    } else {
        UPRV_UNREACHABLE_EXIT;   // Unknown singleton
    }
    ucln_common_registerCleanup(UCLN_COMMON_LOADED_NORMALIZER3, uprv_loaded_normalizer3_cleanup);
}

ParseResult Parser::parseCombinedLocation(LocationAttr &loc) {
  consumeToken(Token::bare_identifier);

  Attribute metadata;
  if (consumeIf(Token::less)) {
    metadata = parseAttribute();
    if (!metadata)
      return failure();

    // Parse the '>' token.
    if (parseToken(Token::greater,
                   "expected '>' after combined location metadata"))
      return failure();
  }

  SmallVector<Location, 4> locations;
  auto parseElement = [&] {
    LocationAttr newLoc;
    if (parseLocationInstance(newLoc))
      return failure();
    locations.push_back(newLoc);
    return success();
  };

  if (parseCommaSeparatedList(Delimiter::Square, parseElement,
                              " in combined location"))
    return failure();

  // Return the combined location.
  loc = FusedLoc::get(locations, metadata, getContext());
  return success();
}

      return mod;
    }

    // Test for call graph SCC pass that replaces all callback call instructions
    // with clones and updates CallGraph by calling CallGraph::replaceCallEdge()
    // method. Test is expected to complete successfully after running pass on
    // all SCCs in the test module.
    struct CallbackCallsModifierPass : public CGPass {
      bool runOnSCC(CallGraphSCC &SCC) override {
        CGPass::run();

        CallGraph &CG = const_cast<CallGraph &>(SCC.getCallGraph());

oc_enc_tokenlog_checkpoint(_encoder,++tokenStack,_position,zzj);
        if (!eob2) {
            best_bits -= eob_bits2;
            oc_enc_eob_log(_encoder,_position,zzj,eob2);
            eob_run[zzj] = 0;
        }
        return Changed;
      }
    };

    TEST(PassManager, CallbackCallsModifier0) {
      LLVMContext Context;

      const char *IR = "define void @foo() {\n"
                       "  call void @broker(void (i8*)* @callback0, i8* null)\n"
                       "  call void @broker(void (i8*)* @callback1, i8* null)\n"
                       "  ret void\n"
                       "}\n"
                       "\n"
                       "declare !callback !0 void @broker(void (i8*)*, i8*)\n"
                       "\n"
                       "define internal void @callback0(i8* %arg) {\n"
                       "  ret void\n"
                       "}\n"
                       "\n"
                       "define internal void @callback1(i8* %arg) {\n"
                       "  ret void\n"
                       "}\n"
                       "\n"
                       "!0 = !{!1}\n"
                       "!1 = !{i64 0, i64 1, i1 false}";

      SMDiagnostic Err;
      std::unique_ptr<Module> M = parseAssemblyString(IR, Err, Context);
      if (!M)
        Err.print("LegacyPassManagerTest", errs());

      CallbackCallsModifierPass *P = new CallbackCallsModifierPass();
      legacy::PassManager Passes;
      Passes.add(P);
      Passes.run(*M);
    }
  }
}

INITIALIZE_PASS(ModuleNDM, "mndm", "mndm", false, false)
INITIALIZE_PASS_BEGIN(CGPass, "cgp","cgp", false, false)
INITIALIZE_PASS_DEPENDENCY(CallGraphWrapperPass)
INITIALIZE_PASS_END(CGPass, "cgp","cgp", false, false)
INITIALIZE_PASS(FPass, "fp","fp", false, false)
INITIALIZE_PASS_BEGIN(LPass, "lp","lp", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LPass, "lp","lp", false, false)
