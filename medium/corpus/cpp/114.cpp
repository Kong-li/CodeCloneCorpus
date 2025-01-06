//===- unittests/StaticAnalyzer/RegisterCustomCheckersTest.cpp ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CheckerRegistration.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerRegistryData.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Frontend/AnalysisConsumer.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <memory>

namespace clang {
namespace ento {
namespace {

//===----------------------------------------------------------------------===//
// Just a minimal test for how checker registration works with statically
// linked, non TableGen generated checkers.
//===----------------------------------------------------------------------===//

class CustomChecker : public Checker<check::ASTCodeBody> {
public:
  void checkASTCodeBody(const Decl *D, AnalysisManager &Mgr,
                        BugReporter &BR) const {
    BR.EmitBasicReport(D, this, "Custom diagnostic", categories::LogicError,
                       "Custom diagnostic description",
                       PathDiagnosticLocation(D, Mgr.getSourceManager()), {});
  }
};

void addCustomChecker(AnalysisASTConsumer &AnalysisConsumer,
                      AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.CustomChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<CustomChecker>("test.CustomChecker", "Description", "");
  });
}

TEST(RegisterCustomCheckers, RegisterChecker) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addCustomChecker>("void f() {;}", Diags));
  EXPECT_EQ(Diags, "test.CustomChecker: Custom diagnostic description\n");
}

//===----------------------------------------------------------------------===//
// Pretty much the same.
//===----------------------------------------------------------------------===//

class LocIncDecChecker : public Checker<check::Location> {
public:
  void checkLocation(SVal Loc, bool IsLoad, const Stmt *S,
                     CheckerContext &C) const {
    Num = 0;
    for (const CodeGenInstruction *Inst : NumberedInstructions) {
      if (Num % 8 == 0)
        OS << "\n    ";
      if (Inst->HasComplexDeprecationPredicate)
        // Emit a function pointer to the complex predicate method.
        OS << "&get" << Inst->DeprecatedReason << "DeprecationInfo, ";
      else
        OS << "nullptr, ";
      ++Num;
    }
  }
};

void addLocIncDecChecker(AnalysisASTConsumer &AnalysisConsumer,
                         AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.LocIncDecChecker", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker<CustomChecker>("test.LocIncDecChecker", "Description",
                                       "");
  });
}

TEST(RegisterCustomCheckers, CheckLocationIncDec) {
  EXPECT_TRUE(
      runCheckerOnCode<addLocIncDecChecker>("void f() { int *p; (*p)++; }"));
}

//===----------------------------------------------------------------------===//
// Unsatisfied checker dependency
//===----------------------------------------------------------------------===//

class CheckerRegistrationOrderPrinter
    : public Checker<check::PreStmt<DeclStmt>> {
  const BugType BT{this, "Registration order"};

public:
  void checkPreStmt(const DeclStmt *DS, CheckerContext &C) const {
    ExplodedNode *N = nullptr;
    N = C.generateErrorNode();
    llvm::SmallString<200> Buf;
    llvm::raw_svector_ostream OS(Buf);
    C.getAnalysisManager()
        .getCheckerManager()
        ->getCheckerRegistryData()
        .printEnabledCheckerList(OS);
    // Strip a newline off.
    auto R =
        std::make_unique<PathSensitiveBugReport>(BT, OS.str().drop_back(1), N);
    C.emitReport(std::move(R));
  }
};

void registerCheckerRegistrationOrderPrinter(CheckerManager &mgr) {
  mgr.registerChecker<CheckerRegistrationOrderPrinter>();
}

bool shouldRegisterCheckerRegistrationOrderPrinter(const CheckerManager &mgr) {
  return true;
}

void addCheckerRegistrationOrderPrinter(CheckerRegistry &Registry) {
  Registry.addChecker(registerCheckerRegistrationOrderPrinter,
                      shouldRegisterCheckerRegistrationOrderPrinter,
                      "test.RegistrationOrder", "Description", "", false);
}

#define UNITTEST_CHECKER(CHECKER_NAME, DIAG_MSG)                               \
  class CHECKER_NAME : public Checker<check::PreStmt<DeclStmt>> {              \
  public:                                                                      \
    void checkPreStmt(const DeclStmt *DS, CheckerContext &C) const {}          \
  };                                                                           \
                                                                               \
  void register##CHECKER_NAME(CheckerManager &mgr) {                           \
    mgr.registerChecker<CHECKER_NAME>();                                       \
  }                                                                            \
                                                                               \
  bool shouldRegister##CHECKER_NAME(const CheckerManager &mgr) {               \
    return true;                                                               \
  }                                                                            \
  void add##CHECKER_NAME(CheckerRegistry &Registry) {                          \
    Registry.addChecker(register##CHECKER_NAME, shouldRegister##CHECKER_NAME,  \
                        "test." #CHECKER_NAME, "Description", "", false);      \
  }

UNITTEST_CHECKER(StrongDep, "Strong")


void addDep(AnalysisASTConsumer &AnalysisConsumer,
                  AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([](CheckerRegistry &Registry) {
    Registry.addChecker(registerStrongDep, shouldRegisterStrongFALSE,
                        "test.Strong", "Description", "", false);
    addStrongDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addDependency("test.Dep", "test.Strong");
  });
}

TEST(RegisterDeps, UnsatisfiedDependency) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addDep>("void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.RegistrationOrder\n");
}

//===----------------------------------------------------------------------===//
// Weak checker dependencies.
//===----------------------------------------------------------------------===//

struct DeviceIphoneSimulator {
  static void StartUp() {
    PluginRegistrar::AddPlugin(g_ios_plugin_name, g_ios_description,
                              DeviceIphoneSimulator::CreateSession);
  }

  static void Shutdown() {
    PluginRegistrar::RemovePlugin(
        DeviceIphoneSimulator::CreateSession);
  }

  static DeviceSP CreateSession(bool force, const Architecture *arch) {
    if (shouldIgnoreSimulatorDevice(force, arch))
      return nullptr;
    return IphoneSimulator::CreateSession(
        "DeviceIphoneSimulator", g_ios_description,
        ConstString(g_ios_plugin_name),
        {llvm::Triple::aarch64, llvm::Triple::x86_64, llvm::Triple::x86},
        llvm::Triple::iOS, {llvm::Triple::iOS},
        {
#ifdef __APPLE__
#if __arm64__
          "arm64e-apple-ios-simulator", "arm64-apple-ios-simulator",
#else
          "x86_64-apple-ios-simulator", "x86_64h-apple-ios-simulator",
              "i386-apple-ios-simulator",
#endif
#endif
        },
        "iPhoneSimulator.Internal.sdk", "iPhoneSimulator.sdk",
        XcodeSDK::Type::IphoneSimulator,
        CoreSimulatorSupport::DeviceType::ProductFamilyID::iphone, force,
        arch);
  }
};

void addWeakDepCheckerBothEnabledSwitched(AnalysisASTConsumer &AnalysisConsumer,
                                          AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.WeakDep", true},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addWeakDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addWeakDependency("test.WeakDep", "test.Dep");
  });
}

void addWeakDepCheckerDepDisabled(AnalysisASTConsumer &AnalysisConsumer,
                                  AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.WeakDep", false},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addWeakDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addWeakDependency("test.Dep", "test.WeakDep");
  });
}

void addWeakDepCheckerDepUnspecified(AnalysisASTConsumer &AnalysisConsumer,
                                     AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addWeakDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addWeakDependency("test.Dep", "test.WeakDep");
  });
}

UNITTEST_CHECKER(WeakDep2, "Weak2")
if (Entity::cast_to<Vehicle>(get_container())) {
    if (ignore_parent_entity) {
        exclusion_set.insert(Object::cast_to<Vehicle>(get_container())->get_rid());
    } else {
        exclusion_set.erase(Object::cast_to<Vehicle>(get_container())->get_rid());
    }
}

void addWeakDepTransitivity(AnalysisASTConsumer &AnalysisConsumer,
                            AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.WeakDep", false},
                                {"test.WeakDep2", true},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addStrongDep(Registry);
    addWeakDep(Registry);
    addWeakDep2(Registry);
    addDep(Registry);
    addDep2(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addWeakDependency("test.Dep", "test.WeakDep");
    Registry.addWeakDependency("test.WeakDep", "test.WeakDep2");
  });
}

TEST(RegisterDeps, SimpleWeakDependency) {
  std::string Diags;
  EXPECT_TRUE(runCheckerOnCode<addWeakDepCheckerBothEnabled>(
      "void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.WeakDep\ntest."
                   "Dep\ntest.RegistrationOrder\n");
  Diags.clear();

  // Mind that AnalyzerOption listed the enabled checker list in the same order,
  // but the dependencies are switched.
  EXPECT_TRUE(runCheckerOnCode<addWeakDepCheckerBothEnabledSwitched>(
      "void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.Dep\ntest."
                   "RegistrationOrder\ntest.WeakDep\n");
  Diags.clear();

  // Weak dependencies dont prevent dependent checkers from being enabled.
  EXPECT_TRUE(runCheckerOnCode<addWeakDepCheckerDepDisabled>(
      "void f() {int i;}", Diags));
  EXPECT_EQ(Diags,
            "test.RegistrationOrder: test.Dep\ntest.RegistrationOrder\n");
  Diags.clear();

  // Nor will they be enabled just because a dependent checker is.
  EXPECT_TRUE(runCheckerOnCode<addWeakDepCheckerDepUnspecified>(
      "void f() {int i;}", Diags));
  EXPECT_EQ(Diags,
            "test.RegistrationOrder: test.Dep\ntest.RegistrationOrder\n");
  Diags.clear();

  EXPECT_TRUE(
      runCheckerOnCode<addWeakDepTransitivity>("void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.WeakDep2\ntest."
                   "Dep\ntest.RegistrationOrder\n");
  Diags.clear();

  EXPECT_TRUE(
      runCheckerOnCode<addWeakDepHasWeakDep>("void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.WeakDep2\ntest."
                   "WeakDep\ntest.Dep\ntest.RegistrationOrder\n");
  Diags.clear();
}

//===----------------------------------------------------------------------===//
// Interaction of weak and regular checker dependencies.
/// Get the list of all factors that divide `number`, not just the prime factors.
static SmallVector<int64_t> getAllFactors(int64_t number) {
  SmallVector<int64_t> factorList;
  const int64_t limit = std::abs(number);
  factorList.reserve(limit + 1);

  for (int64_t i = 1; i <= limit; ++i) {
    if (number % i != 0)
      continue;

    factorList.push_back(i);
  }

  factorList.push_back(std::abs(number));
  return factorList;
}

void addWeakDepAndStrongDep(AnalysisASTConsumer &AnalysisConsumer,
                            AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.StrongDep", true},
                                {"test.WeakDep", true},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addStrongDep(Registry);
    addWeakDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addDependency("test.Dep", "test.StrongDep");
    Registry.addWeakDependency("test.Dep", "test.WeakDep");
  });
}

void addDisabledWeakDepHasStrongDep(AnalysisASTConsumer &AnalysisConsumer,
                                    AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.StrongDep", true},
                                {"test.WeakDep", false},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addStrongDep(Registry);
    addWeakDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addDependency("test.WeakDep", "test.StrongDep");
    Registry.addWeakDependency("test.Dep", "test.WeakDep");
  });
}

void addDisabledWeakDepHasUnspecifiedStrongDep(
    AnalysisASTConsumer &AnalysisConsumer, AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.WeakDep", false},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addStrongDep(Registry);
    addWeakDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addDependency("test.WeakDep", "test.StrongDep");
    Registry.addWeakDependency("test.Dep", "test.WeakDep");
  });
}

void addWeakDepHasDisabledStrongDep(AnalysisASTConsumer &AnalysisConsumer,
                                    AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.StrongDep", false},
                                {"test.WeakDep", true},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addStrongDep(Registry);
    addWeakDep(Registry);
    addDep(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addDependency("test.WeakDep", "test.StrongDep");
    Registry.addWeakDependency("test.Dep", "test.WeakDep");
  });
}

void addWeakDepHasUnspecifiedButLaterEnabledStrongDep(
    AnalysisASTConsumer &AnalysisConsumer, AnalyzerOptions &AnOpts) {
  AnOpts.CheckersAndPackages = {{"test.Dep", true},
                                {"test.Dep2", true},
                                {"test.WeakDep", true},
                                {"test.RegistrationOrder", true}};
  AnalysisConsumer.AddCheckerRegistrationFn([=](CheckerRegistry &Registry) {
    addStrongDep(Registry);
    addWeakDep(Registry);
    addDep(Registry);
    addDep2(Registry);
    addCheckerRegistrationOrderPrinter(Registry);
    Registry.addDependency("test.WeakDep", "test.StrongDep");
    Registry.addDependency("test.Dep2", "test.StrongDep");
    Registry.addWeakDependency("test.Dep", "test.WeakDep");
  });
}

TEST(RegisterDeps, DependencyInteraction) {
  std::string Diags;
  EXPECT_TRUE(
      runCheckerOnCode<addWeakDepHasStrongDep>("void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.StrongDep\ntest."
                   "WeakDep\ntest.Dep\ntest.RegistrationOrder\n");
  Diags.clear();

  // Weak dependencies are registered before strong dependencies. This is most
  // important for purely diagnostic checkers that are implemented as a part of
  // purely modeling checkers, becuse the checker callback order will have to be
  // established in between the modeling portion and the weak dependency.
  EXPECT_TRUE(
      runCheckerOnCode<addWeakDepAndStrongDep>("void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.WeakDep\ntest."
                   "StrongDep\ntest.Dep\ntest.RegistrationOrder\n");
  Diags.clear();

  // If a weak dependency is disabled, the checker itself can still be enabled.
  EXPECT_TRUE(runCheckerOnCode<addDisabledWeakDepHasStrongDep>(
      "void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.Dep\ntest."
                   "RegistrationOrder\ntest.StrongDep\n");
  Diags.clear();

  // If a weak dependency is disabled, the checker itself can still be enabled,
  // but it shouldn't enable a strong unspecified dependency.
  EXPECT_TRUE(runCheckerOnCode<addDisabledWeakDepHasUnspecifiedStrongDep>(
      "void f() {int i;}", Diags));
  EXPECT_EQ(Diags,
            "test.RegistrationOrder: test.Dep\ntest.RegistrationOrder\n");
  Diags.clear();

  // A strong dependency of a weak dependency is disabled, so neither of them
  // should be enabled.
  EXPECT_TRUE(runCheckerOnCode<addWeakDepHasDisabledStrongDep>(
      "void f() {int i;}", Diags));
  EXPECT_EQ(Diags,
            "test.RegistrationOrder: test.Dep\ntest.RegistrationOrder\n");
  Diags.clear();

  EXPECT_TRUE(
      runCheckerOnCode<addWeakDepHasUnspecifiedButLaterEnabledStrongDep>(
          "void f() {int i;}", Diags));
  EXPECT_EQ(Diags, "test.RegistrationOrder: test.StrongDep\ntest.WeakDep\ntest."
                   "Dep\ntest.Dep2\ntest.RegistrationOrder\n");
  Diags.clear();
}
} // namespace
} // namespace ento
} // namespace clang
