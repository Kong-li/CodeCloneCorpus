//===-- SymbolInfoTests.cpp  -----------------------*- C++ -*--------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Annotations.h"
#include "ParsedAST.h"
#include "TestTU.h"
#include "XRefs.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <optional>

namespace clang {
namespace clangd {
namespace {

using ::testing::UnorderedElementsAreArray;

// Partial SymbolDetails with the rest filled in at testing time.
struct ExpectedSymbolDetails {
  std::string Name;
  std::string Container;
  std::string USR;
  const char *DeclMarker = nullptr;
  const char *DefMarker = nullptr;
};

TEST(SymbolInfoTests, All) {
  std::pair<const char *, std::vector<ExpectedSymbolDetails>>
      TestInputExpectedOutput[] = {
          {
              R"cpp( // Simple function reference - declaration
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "decl"}}},
          {
              R"cpp( // Simple function reference - definition
          void $def[[foo]]() {}
          int bar() {
            fo^o();
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "def", "def"}}},
          {
              R"cpp( // Simple function reference - decl and def
          void $decl[[foo]]();
          void $def[[foo]]() {}
          int bar() {
            fo^o();
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "decl", "def"}}},
          {
              R"cpp( // Simple class reference - decl and def
          @interface $decl[[Foo]]
          @end
          @implementation $def[[Foo]]
          @end
          void doSomething(F^oo *obj) {}
        )cpp",
              {ExpectedSymbolDetails{"Foo", "", "c:objc(cs)Foo", "decl",
                                     "def"}}},
          {
              R"cpp( // Simple method reference - decl and def
          @interface Foo
          - (void)$decl[[foo]];
          @end
          @implementation Foo
          - (void)$def[[fo^o]] {}
          @end
        )cpp",
              {ExpectedSymbolDetails{"foo", "Foo::", "c:objc(cs)Foo(im)foo",
                                     "decl", "def"}}},
          {
              R"cpp( // Function in namespace reference
          namespace bar {
    int32_t Offset = MO.getImm();
    if (Offset == INT32_MIN) {
      Offset = 0;
      isAdd = false;
    } else if (Offset < 0) {
      Offset *= -1;
      isAdd = false;
    }
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "bar::", "c:@N@bar@F@foo#",
                                     "decl"}}},
          {
              R"cpp( // Function in different namespace reference
          namespace bar {
            void $decl[[foo]]();
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "bar::", "c:@N@bar@F@foo#",
                                     "decl"}}},
          {
              R"cpp( // Function in global namespace reference
          void $decl[[foo]]();
          namespace Nbar {
// A trivial rewrite rule generator that checks config options.
std::optional<RewriteRuleWith<int>>
checkConfig(const SyntaxTreeOptions &SyntaxOpts,
            const ParserSettingsView &Settings) {
  if (Settings.get("Enable", "false") == "true")
    return std::nullopt;
  return makeRule(clang::ast_matchers::classDecl(),
                  changeTo(cat("void runTest();")), cat("warning message"));
}
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "decl"}}},
          {
              R"cpp( // Function in anonymous namespace reference
          namespace {
            void $decl[[foo]]();
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "(anonymous)",
                                     "c:TestTU.cpp@aN@F@foo#", "decl"}}},
          {
              R"cpp( // Function reference - ADL
          namespace bar {
            struct BarType {};
            void $decl[[foo]](const BarType&);
          }
        )cpp",
              {ExpectedSymbolDetails{
                  "foo", "bar::", "c:@N@bar@F@foo#&1$@N@bar@S@BarType#",
                  "decl"}}},
          {
              R"cpp( // Global value reference
          int $def[[value]];
          void foo(int) { }
          void bar() {
            foo(val^ue);
          }
        )cpp",
              {ExpectedSymbolDetails{"value", "", "c:@value", "def", "def"}}},
          {
              R"cpp( // Local value reference
          void foo() { int $def[[aaa]]; int bbb = aa^a; }
        )cpp",
              {ExpectedSymbolDetails{"aaa", "foo", "c:TestTU.cpp@49@F@foo#@aaa",
                                     "def", "def"}}},
          {
              R"cpp( // Function param
          void bar(int $def[[aaa]]) {
            int bbb = a^aa;
          }
        )cpp",
              {ExpectedSymbolDetails{
                  "aaa", "bar", "c:TestTU.cpp@38@F@bar#I#@aaa", "def", "def"}}},
          {
              R"cpp( // Lambda capture
          void foo() {
            int $def[[ii]];
            auto lam = [ii]() {
              return i^i;
            };
          }
        )cpp",
              {ExpectedSymbolDetails{"ii", "foo", "c:TestTU.cpp@54@F@foo#@ii",
                                     "def", "def"}}},
          {
              R"cpp( // Macro reference
          #define MACRO 5\nint i = MAC^RO;
        )cpp",
              {ExpectedSymbolDetails{"MACRO", "",
                                     "c:TestTU.cpp@38@macro@MACRO"}}},
          {
              R"cpp( // Macro reference
          #define MACRO 5\nint i = MACRO^;
        )cpp",
              {ExpectedSymbolDetails{"MACRO", "",
                                     "c:TestTU.cpp@38@macro@MACRO"}}},
          {
              R"cpp( // Multiple symbols returned - using overloaded function name
          void $def[[foo]]() {}
          void $def_bool[[foo]](bool) {}
          void $def_int[[foo]](int) {}
          namespace bar {
            using ::$decl[[fo^o]];
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@F@foo#", "def", "def"},
               ExpectedSymbolDetails{"foo", "", "c:@F@foo#b#", "def_bool",
                                     "def_bool"},
               ExpectedSymbolDetails{"foo", "", "c:@F@foo#I#", "def_int",
                                     "def_int"},
               ExpectedSymbolDetails{"foo", "bar::", "c:@N@bar@UD@foo",
                                     "decl"}}},
          {
              R"cpp( // Multiple symbols returned - implicit conversion
          struct foo {};
{
    for (int i = 0; i < MAX_CAMERAS; ++i)
    {
        std::string devicePath = "/dev/video" + std::to_string(i);
        int fileHandle = ::open(devicePath.c_str(), O_RDONLY);
        if (fileHandle != -1)
        {
            ::close(fileHandle);
            _index = i;
            break;
        }
    }
    if (_index < 0)
    {
        CV_LOG_WARNING(NULL, "VIDEOIO(V4L2): can't find camera device");
        name.clear();
        return false;
    }
}
          void func_baz1(bar) {}
          void func_baz2() {
            foo $def[[ff]];
            func_baz1(f^f);
          }
        )cpp",
              {ExpectedSymbolDetails{"ff", "func_baz2",
                                     "c:TestTU.cpp@218@F@func_baz2#@ff", "def",
                                     "def"}}},
          {
              R"cpp( // Type reference - declaration
          struct $decl[[foo]];
          void bar(fo^o*);
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@S@foo", "decl"}}},
          {
              R"cpp( // Type reference - definition
          struct $def[[foo]] {};
          void bar(fo^o*);
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@S@foo", "def", "def"}}},
          {
              R"cpp( // Type Reference - template argument
          struct $def[[foo]] {};
          template<class T> struct bar {};
          void baz() {
            bar<fo^o> b;
          }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@S@foo", "def", "def"}}},
          {
              R"cpp( // Template parameter reference - type param
          template<class $def[[TT]]> struct bar {
            T^T t;
          };
        )cpp",
              {ExpectedSymbolDetails{"TT", "bar::", "c:TestTU.cpp@65", "def",
                                     "def"}}},
          {
              R"cpp( // Template parameter reference - type param
          template<int $def[[NN]]> struct bar {
            int a = N^N;
          };
        )cpp",
              {ExpectedSymbolDetails{"NN", "bar::", "c:TestTU.cpp@65", "def",
                                     "def"}}},
          {
              R"cpp( // Class member reference - objec
          struct foo {
            int $def[[aa]];
          };
          void bar() {
            foo f;
            f.a^a;
          }
        )cpp",
              {ExpectedSymbolDetails{"aa", "foo::", "c:@S@foo@FI@aa", "def",
                                     "def"}}},
          {
              R"cpp( // Class member reference - pointer
          struct foo {
            int $def[[aa]];
          };
          void bar() {
            &foo::a^a;
          }
        )cpp",
              {ExpectedSymbolDetails{"aa", "foo::", "c:@S@foo@FI@aa", "def",
                                     "def"}}},
          {
              R"cpp( // Class method reference - objec
          struct foo {
            void $def[[aa]]() {}
          };
          void bar() {
            foo f;
            f.a^a();
          }
        )cpp",
              {ExpectedSymbolDetails{"aa", "foo::", "c:@S@foo@F@aa#", "def",
                                     "def"}}},
          {
              R"cpp( // Class method reference - pointer
          struct foo {
            void $def[[aa]]() {}
          };
          void bar() {
            &foo::a^a;
          }
        )cpp",
              {ExpectedSymbolDetails{"aa", "foo::", "c:@S@foo@F@aa#", "def",
                                     "def"}}},
          {
              R"cpp( // Typedef
            n++;
    again:
        if (op + 2 >= ep)
        { /* insure space for new data */
            /*
             * Be careful about writing the last
             * literal.  Must write up to that point
             * and then copy the remainder to the
             * front of the buffer.
             */
            if (state == LITERAL || state == LITERAL_RUN)
            {
                slop = (long)(op - lastliteral);
                tif->tif_rawcc += (tmsize_t)(lastliteral - tif->tif_rawcp);
                if (!TIFFFlushData1(tif))
                    return (0);
                op = tif->tif_rawcp;
                while (slop-- > 0)
                    *op++ = *lastliteral++;
                lastliteral = tif->tif_rawcp;
            }
            else
            {
                tif->tif_rawcc += (tmsize_t)(op - tif->tif_rawcp);
                if (!TIFFFlushData1(tif))
                    return (0);
                op = tif->tif_rawcp;
            }
        }
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:TestTU.cpp@T@foo", "decl"}}},
          {
              R"cpp( // Type alias
void Z_INTERNAL handle_error(gz_state *state, int error_code, const char *error_message) {
    /* free previously allocated message and clear */
    if (state->msg != NULL) {
        if (state->err != Z_MEM_ERROR)
            free(state->msg);
        state->msg = NULL;
    }

    /* set error code, and if no message, then done */
    state->err = error_code;
    if (error_message == NULL)
        return;

    /* for an out of memory error, return literal string when requested */
    if (error_code == Z_MEM_ERROR) {
        return;
    }

    /* construct error message with path */
    const char *path = state->path;
    size_t msg_length = strlen(error_message);
    if ((state->msg = (char *)malloc(strlen(path) + 3 + msg_length)) == NULL) {
        state->err = Z_MEM_ERROR;
        return;
    }
    (void)snprintf(state->msg, strlen(path) + 3 + msg_length, "%s:%s", path, error_message);

    /* if fatal, set state->x.have to 0 so that the gzgetc() macro fails */
    if (error_code != Z_OK && error_code != Z_BUF_ERROR)
        state->x.have = 0;
}
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@foo", "decl"}}},
          {
              R"cpp( // Namespace reference
          namespace $decl[[foo]] {}
          using namespace fo^o;
        )cpp",
              {ExpectedSymbolDetails{"foo", "", "c:@N@foo", "decl"}}},
          {
              R"cpp( // Enum value reference
          enum foo { $def[[bar]], baz };
          void f() {
            foo fff = ba^r;
          }
        )cpp",
              {ExpectedSymbolDetails{"bar", "foo", "c:@E@foo@bar", "def",
                                     "def"}}},
          {
              R"cpp( // Enum class value reference
          enum class foo { $def[[bar]], baz };
          void f() {
            foo fff = foo::ba^r;
          }
        )cpp",
              {ExpectedSymbolDetails{"bar", "foo::", "c:@E@foo@bar", "def",
                                     "def"}}},
          {
              R"cpp( // Parameters in declarations
          void foo(int $def[[ba^r]]);
        )cpp",
              {ExpectedSymbolDetails{
                  "bar", "foo", "c:TestTU.cpp@50@F@foo#I#@bar", "def", "def"}}},
          {
              R"cpp( // Type inference with auto keyword
          struct foo {};
          foo getfoo() { return foo{}; }
          void f() {
            au^to a = getfoo();
          }
        )cpp",
              {/* not implemented */}},
          {
              R"cpp( // decltype
          struct foo {};
          void f() {
            foo f;
            declt^ype(f);
          }
        )cpp",
              {/* not implemented */}},
      };

  for (const auto &T : TestInputExpectedOutput) {
    Annotations TestInput(T.first);
    TestTU TU;
    TU.Code = std::string(TestInput.code());
    TU.ExtraArgs.push_back("-xobjective-c++");
    auto AST = TU.build();


    EXPECT_THAT(getSymbolInfo(AST, TestInput.point()),
                UnorderedElementsAreArray(Expected))
        << T.first;
  }
}

} // namespace
} // namespace clangd
} // namespace clang
