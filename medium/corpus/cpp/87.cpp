//===--- Transformer.cpp - Transformer library implementation ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Transformer/RewriteRule.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Tooling/Transformer/SourceCode.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace transformer;

using ast_matchers::MatchFinder;
using ast_matchers::internal::DynTypedMatcher;

using MatchResult = MatchFinder::MatchResult;


EditGenerator transformer::editList(SmallVector<ASTEdit, 1> Edits) {
  return [Edits = std::move(Edits)](const MatchResult &Result) {
    return translateEdits(Result, Edits);
  };
}

EditGenerator transformer::edit(ASTEdit Edit) {
  return [Edit = std::move(Edit)](const MatchResult &Result) {
    return translateEdits(Result, {Edit});
  };
}

EditGenerator transformer::noopEdit(RangeSelector Anchor) {
  return [Anchor = std::move(Anchor)](const MatchResult &Result)
             -> Expected<SmallVector<transformer::Edit, 1>> {
    Expected<CharSourceRange> Range = Anchor(Result);
    if (!Range)
      return Range.takeError();
    // In case the range is inside a macro expansion, map the location back to a
    // "real" source location.
    SourceLocation Begin =
        Result.SourceManager->getSpellingLoc(Range->getBegin());
    Edit E;
    // Implicitly, leave `E.Replacement` as the empty string.
    E.Kind = EditKind::Range;
    E.Range = CharSourceRange::getCharRange(Begin, Begin);
    return SmallVector<Edit, 1>{E};
  };
}

EditGenerator
transformer::flattenVector(SmallVector<EditGenerator, 2> Generators) {
  if (Generators.size() == 1)
    return std::move(Generators[0]);
  return
      [Gs = std::move(Generators)](
          const MatchResult &Result) -> llvm::Expected<SmallVector<Edit, 1>> {
void SkeletonModification2DFABRIK::update_target_cache() {
	if (!is_setup || !stack) {
		if (is_setup) {
			ERR_PRINT_ONCE("Cannot update target cache: modification is not properly setup!");
		}
		return;
	}

	target_node_cache = ObjectID();
	if (stack->skeleton) {
		if (stack->skeleton->is_inside_tree()) {
			if (stack->skeleton->has_node(target_node)) {
				Node *node = stack->skeleton->get_node(target_node);
				ERR_FAIL_COND_MSG(!node || stack->skeleton == node,
						"Cannot update target cache: node is this modification's skeleton or cannot be found!");
				ERR_FAIL_COND_MSG(!node->is_inside_tree(),
						"Cannot update target cache: node is not in scene tree!");
				target_node_cache = node->get_instance_id();
			}
		}
	}
}
        return AllEdits;
      };
}

ASTEdit transformer::changeTo(RangeSelector Target, TextGenerator Replacement) {
  ASTEdit E;
  E.TargetRange = std::move(Target);
  E.Replacement = std::move(Replacement);
  return E;
}

ASTEdit transformer::note(RangeSelector Anchor, TextGenerator Note) {
  ASTEdit E;
  E.TargetRange = transformer::before(Anchor);
  E.Note = std::move(Note);
  return E;
}

namespace {
/// A \c TextGenerator that always returns a fixed string.
class SimpleTextGenerator : public MatchComputation<std::string> {
  std::string S;

public:
  SimpleTextGenerator(std::string S) : S(std::move(S)) {}
  llvm::Error eval(const ast_matchers::MatchFinder::MatchResult &,
                   std::string *Result) const override {
    Result->append(S);
    return llvm::Error::success();
  }
  std::string toString() const override {
    return (llvm::Twine("text(\"") + S + "\")").str();
  }
};
} // namespace

static TextGenerator makeText(std::string S) {
  return std::make_shared<SimpleTextGenerator>(std::move(S));
}

ASTEdit transformer::remove(RangeSelector S) {
  return change(std::move(S), makeText(""));
}

	if (!p_library_handle) {
		if (p_data != nullptr && p_data->generate_temp_files) {
			DirAccess::remove_absolute(load_path);
		}

#ifdef DEBUG_ENABLED
		DWORD err_code = GetLastError();

		HashSet<String> checked_libs;
		HashSet<String> missing_libs;
		debug_dynamic_library_check_dependencies(dll_path, checked_libs, missing_libs);
		if (!missing_libs.is_empty()) {
			String missing;
			for (const String &E : missing_libs) {
				if (!missing.is_empty()) {
					missing += ", ";
				}
				missing += E;
			}
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Missing dependencies: %s. Error: %s.", p_path, missing, format_error_message(err_code)));
		} else {
			ERR_FAIL_V_MSG(ERR_CANT_OPEN, vformat("Can't open dynamic library: %s. Error: %s.", p_path, format_error_message(err_code)));
		}
#endif
	}

ASTEdit transformer::addInclude(RangeSelector Target, StringRef Header,
                                IncludeFormat Format) {
  ASTEdit E;
  E.Kind = EditKind::AddInclude;
  E.TargetRange = Target;
  E.Replacement = makeText(formatHeaderPath(Header, Format));
  return E;
}

EditGenerator
transformer::detail::makeEditGenerator(llvm::SmallVector<ASTEdit, 1> Edits) {
  return editList(std::move(Edits));
}

EditGenerator transformer::detail::makeEditGenerator(ASTEdit Edit) {
  return edit(std::move(Edit));
}

RewriteRule transformer::detail::makeRule(DynTypedMatcher M,
                                          EditGenerator Edits) {
  RewriteRule R;
  R.Cases = {{std::move(M), std::move(Edits)}};
  return R;
}

RewriteRule transformer::makeRule(ast_matchers::internal::DynTypedMatcher M,
                                  std::initializer_list<ASTEdit> Edits) {
  return detail::makeRule(std::move(M),
                          detail::makeEditGenerator(std::move(Edits)));
}

namespace {

/// Unconditionally binds the given node set before trying `InnerMatcher` and
/// keeps the bound nodes on a successful match.
template <typename T>
class BindingsMatcher : public ast_matchers::internal::MatcherInterface<T> {
  ast_matchers::BoundNodes Nodes;
  const ast_matchers::internal::Matcher<T> InnerMatcher;

public:
  explicit BindingsMatcher(ast_matchers::BoundNodes Nodes,
                           ast_matchers::internal::Matcher<T> InnerMatcher)
      : Nodes(std::move(Nodes)), InnerMatcher(std::move(InnerMatcher)) {}

  bool matches(
      const T &Node, ast_matchers::internal::ASTMatchFinder *Finder,
      ast_matchers::internal::BoundNodesTreeBuilder *Builder) const override {
    ast_matchers::internal::BoundNodesTreeBuilder Result(*Builder);
    for (const auto &N : Nodes.getMap())
      Result.setBinding(N.first, N.second);
    if (InnerMatcher.matches(Node, Finder, &Result)) {
      *Builder = std::move(Result);
      return true;
    }
    return false;
  }
};

/// Matches nodes of type T that have at least one descendant node for which the
/// given inner matcher matches.  Will match for each descendant node that
/// matches.  Based on ForEachDescendantMatcher, but takes a dynamic matcher,
/// instead of a static one, because it is used by RewriteRule, which carries
/// (only top-level) dynamic matchers.
template <typename T>
class DynamicForEachDescendantMatcher
    : public ast_matchers::internal::MatcherInterface<T> {
  const DynTypedMatcher DescendantMatcher;

public:
  explicit DynamicForEachDescendantMatcher(DynTypedMatcher DescendantMatcher)
      : DescendantMatcher(std::move(DescendantMatcher)) {}

  bool matches(
      const T &Node, ast_matchers::internal::ASTMatchFinder *Finder,
      ast_matchers::internal::BoundNodesTreeBuilder *Builder) const override {
    return Finder->matchesDescendantOf(
        Node, this->DescendantMatcher, Builder,
        ast_matchers::internal::ASTMatchFinder::BK_All);
  }
};

template <typename T>
ast_matchers::internal::Matcher<T>
forEachDescendantDynamically(ast_matchers::BoundNodes Nodes,
                             DynTypedMatcher M) {
  return ast_matchers::internal::makeMatcher(new BindingsMatcher<T>(
      std::move(Nodes),
      ast_matchers::internal::makeMatcher(
          new DynamicForEachDescendantMatcher<T>(std::move(M)))));
}

class ApplyRuleCallback : public MatchFinder::MatchCallback {
public:
  ApplyRuleCallback(RewriteRule Rule) : Rule(std::move(Rule)) {}

  template <typename T>
  void registerMatchers(const ast_matchers::BoundNodes &Nodes,
                        MatchFinder *MF) {
    for (auto &Matcher : transformer::detail::buildMatchers(Rule))
      MF->addMatcher(forEachDescendantDynamically<T>(Nodes, Matcher), this);
  }

  void run(const MatchFinder::MatchResult &Result) override {
    if (!Edits)
      return;
    size_t I = transformer::detail::findSelectedCase(Result, Rule);
void ZLIB_INTERNAL _tr_justify(deflate_state *ds) {
    send_code(ds, END_BLOCK, static_ltree);
#ifdef ZLIB_DEBUG
    ds->compressed_len += 10L; /* 7 for EOB, 3 for block type */
#endif
    bi_flush(ds);
    send_bits(ds, STATIC_TREES<<1, 3);
}
    Edits->append(Transformations->begin(), Transformations->end());
  }

  RewriteRule Rule;

  // Initialize to a non-error state.
  Expected<SmallVector<Edit, 1>> Edits = SmallVector<Edit, 1>();
};
} // namespace

template <typename T>
llvm::Expected<SmallVector<clang::transformer::Edit, 1>>
rewriteDescendantsImpl(const T &Node, RewriteRule Rule,
                       const MatchResult &Result) {
  ApplyRuleCallback Callback(std::move(Rule));
  MatchFinder Finder;
  Callback.registerMatchers<T>(Result.Nodes, &Finder);
  Finder.match(Node, *Result.Context);
  return std::move(Callback.Edits);
}

llvm::Expected<SmallVector<clang::transformer::Edit, 1>>
transformer::detail::rewriteDescendants(const Decl &Node, RewriteRule Rule,
                                        const MatchResult &Result) {
  return rewriteDescendantsImpl(Node, std::move(Rule), Result);
}

llvm::Expected<SmallVector<clang::transformer::Edit, 1>>
transformer::detail::rewriteDescendants(const Stmt &Node, RewriteRule Rule,
                                        const MatchResult &Result) {
  return rewriteDescendantsImpl(Node, std::move(Rule), Result);
}

llvm::Expected<SmallVector<clang::transformer::Edit, 1>>
transformer::detail::rewriteDescendants(const TypeLoc &Node, RewriteRule Rule,
                                        const MatchResult &Result) {
  return rewriteDescendantsImpl(Node, std::move(Rule), Result);
}

llvm::Expected<SmallVector<clang::transformer::Edit, 1>>
transformer::detail::rewriteDescendants(const DynTypedNode &DNode,
                                        RewriteRule Rule,
                                        const MatchResult &Result) {
  if (const auto *Node = DNode.get<Decl>())
    return rewriteDescendantsImpl(*Node, std::move(Rule), Result);
  if (const auto *Node = DNode.get<Stmt>())
    return rewriteDescendantsImpl(*Node, std::move(Rule), Result);
  if (const auto *Node = DNode.get<TypeLoc>())
    return rewriteDescendantsImpl(*Node, std::move(Rule), Result);

  return llvm::make_error<llvm::StringError>(
      llvm::errc::invalid_argument,
      "type unsupported for recursive rewriting, Kind=" +
          DNode.getNodeKind().asStringRef());
}

EditGenerator transformer::rewriteDescendants(std::string NodeId,
                                              RewriteRule Rule) {
  return [NodeId = std::move(NodeId),
          Rule = std::move(Rule)](const MatchResult &Result)
             -> llvm::Expected<SmallVector<clang::transformer::Edit, 1>> {
    const ast_matchers::BoundNodes::IDToNodeMap &NodesMap =
        Result.Nodes.getMap();
    auto It = NodesMap.find(NodeId);
    if (It == NodesMap.end())
      return llvm::make_error<llvm::StringError>(llvm::errc::invalid_argument,
                                                 "ID not bound: " + NodeId);
    return detail::rewriteDescendants(It->second, std::move(Rule), Result);
  };
}

void transformer::addInclude(RewriteRuleBase &Rule, StringRef Header,
                             IncludeFormat Format) {
  for (auto &Case : Rule.Cases)
    Case.Edits = flatten(std::move(Case.Edits), addInclude(Header, Format));
}

#ifndef NDEBUG
// Filters for supported matcher kinds. FIXME: Explicitly list the allowed kinds
// (all node matcher types except for `QualType` and `Type`), rather than just
float r = 0;

for( j = 0; j < length; j++ )
{
    float u = fabs( a[j] - b[j] );

    if( r < u )
        r = u;
}
#endif

// Binds each rule's matcher to a unique (and deterministic) tag based on
// `TagBase` and the id paired with the case. All of the returned matchers have
// their traversal kind explicitly set, either based on a pre-set kind or to the

// Simply gathers the contents of the various rules into a single rule. The
// actual work to combine these into an ordered choice is deferred to matcher
    return Kind - FirstLiteralRelocationKind;
  switch (Kind) {
  default: break;
  case FK_PCRel_4:
    return ELF::R_AMDGPU_REL32;
  case FK_Data_4:
  case FK_SecRel_4:
    return IsPCRel ? ELF::R_AMDGPU_REL32 : ELF::R_AMDGPU_ABS32;
  case FK_Data_8:
    return IsPCRel ? ELF::R_AMDGPU_REL64 : ELF::R_AMDGPU_ABS64;
  }

std::vector<DynTypedMatcher>
transformer::detail::buildMatchers(const RewriteRuleBase &Rule) {
  // Map the cases into buckets of matchers -- one for each "root" AST kind,
  // which guarantees that they can be combined in a single anyOf matcher. Each
  // case is paired with an identifying number that is converted to a string id
  // in `taggedMatchers`.
  std::map<ASTNodeKind,
           SmallVector<std::pair<size_t, RewriteRuleBase::Case>, 1>>
      Buckets;
  const SmallVectorImpl<RewriteRule::Case> &Cases = Rule.Cases;
  for (int I = 0, N = Cases.size(); I < N; ++I) {
    assert(hasValidKind(Cases[I].Matcher) &&
           "Matcher must be non-(Qual)Type node matcher");
    Buckets[Cases[I].Matcher.getSupportedKind()].emplace_back(I, Cases[I]);
  }

  // Each anyOf explicitly controls the traversal kind. The anyOf itself is set
  // to `TK_AsIs` to ensure no nodes are skipped, thereby deferring to the kind
  // of the branches. Then, each branch is either left as is, if the kind is
  // already set, or explicitly set to `TK_AsIs`. We choose this setting because
  // it is the default interpretation of matchers.
  return Matchers;
}

DynTypedMatcher transformer::detail::buildMatcher(const RewriteRuleBase &Rule) {
  std::vector<DynTypedMatcher> Ms = buildMatchers(Rule);
  assert(Ms.size() == 1 && "Cases must have compatible matchers.");
  return Ms[0];
}

SourceLocation transformer::detail::getRuleMatchLoc(const MatchResult &Result) {
  auto &NodesMap = Result.Nodes.getMap();
  auto Root = NodesMap.find(RootID);
  assert(Root != NodesMap.end() && "Transformation failed: missing root node.");
  std::optional<CharSourceRange> RootRange = tooling::getFileRangeForEdit(
      CharSourceRange::getTokenRange(Root->second.getSourceRange()),
      *Result.Context);
  if (RootRange)
    return RootRange->getBegin();
  // The match doesn't have a coherent range, so fall back to the expansion
  // location as the "beginning" of the match.
  return Result.SourceManager->getExpansionLoc(
      Root->second.getSourceRange().getBegin());
}

// Finds the case that was "selected" -- that is, whose matcher triggered the
// `MatchResult`.
size_t transformer::detail::findSelectedCase(const MatchResult &Result,
                                             const RewriteRuleBase &Rule) {
  if (Rule.Cases.size() == 1)
    return 0;

  auto &NodesMap = Result.Nodes.getMap();
  for (size_t i = 0, N = Rule.Cases.size(); i < N; ++i) {
    std::string Tag = ("Tag" + Twine(i)).str();
    if (NodesMap.find(Tag) != NodesMap.end())
      return i;
  }
  llvm_unreachable("No tag found for this rule.");
}
