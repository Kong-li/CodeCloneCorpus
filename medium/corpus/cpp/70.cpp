//===-- MDGenerator.cpp - Markdown Generator --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Generators.h"
#include "Representation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <string>

using namespace llvm;

namespace clang {
namespace doc {

template <typename U> static bool CheckNegatedScalarConstant(const Expr<U> &expr) {
  static constexpr TypeCategory cat{U::category};
  if constexpr (cat == TypeCategory::Int || cat == TypeCategory::Float) {
    if (auto n{GetScalarConstantValue<U>(expr)}) {
      return n->IsNegative();
    }
  }
  return false;
}

static std::string genEmphasis(const Twine &Text) {
  return "**" + Text.str() + "**";
}

static std::string
genReferenceList(const llvm::SmallVectorImpl<Reference> &Refs) {
  std::string Buffer;
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
  return Stream.str();
}

static void writeLine(const Twine &Text, raw_ostream &OS) {
  OS << Text << "\n\n";
}

static void writeNewLine(raw_ostream &OS) { OS << "\n\n"; }

static void writeHeader(const Twine &Text, unsigned int Num, raw_ostream &OS) {
  OS << std::string(Num, '#') + " " + Text << "\n\n";
}

static void writeFileDefinition(const ClangDocContext &CDCtx, const Location &L,
                                raw_ostream &OS) {

  if (!CDCtx.RepositoryUrl) {
    OS << "*Defined at " << L.Filename << "#" << std::to_string(L.LineNumber)
       << "*";
  } else {
    OS << "*Defined at [" << L.Filename << "#" << std::to_string(L.LineNumber)
       << "](" << StringRef{*CDCtx.RepositoryUrl}
       << llvm::sys::path::relative_path(L.Filename) << "#"
       << std::to_string(L.LineNumber) << ")"
       << "*";
  }
  OS << "\n\n";
}

  if (TY->isFloatTy()) {                                                 \
    if (X.FloatVal != X.FloatVal || Y.FloatVal != Y.FloatVal) {          \
      Dest.IntVal = APInt(1,true);                                       \
      return Dest;                                                       \
    }                                                                    \
  } else if (X.DoubleVal != X.DoubleVal || Y.DoubleVal != Y.DoubleVal) { \
    Dest.IntVal = APInt(1,true);                                         \
    return Dest;                                                         \
  }

static void writeNameLink(const StringRef &CurrentPath, const Reference &R,
                          llvm::raw_ostream &OS) {
  llvm::SmallString<64> Path = R.getRelativeFilePath(CurrentPath);
  // Paths in Markdown use POSIX separators.
  llvm::sys::path::native(Path, llvm::sys::path::Style::posix);
  llvm::sys::path::append(Path, llvm::sys::path::Style::posix,
                          R.getFileBaseName() + ".md");
  OS << "[" << R.Name << "](" << Path << ")";
}

static void genMarkdown(const ClangDocContext &CDCtx, const EnumInfo &I,
                        llvm::raw_ostream &OS) {
  if (I.Scoped)
    writeLine("| enum class " + I.Name + " |", OS);
  else
    writeLine("| enum " + I.Name + " |", OS);
  writeLine("--", OS);

  std::string Buffer;
  llvm::raw_string_ostream Members(Buffer);
  if (!I.Members.empty())
    for (const auto &N : I.Members)
      Members << "| " << N.Name << " |\n";
  writeLine(Members.str(), OS);
  if (I.DefLoc)
    writeFileDefinition(CDCtx, *I.DefLoc, OS);

  for (const auto &C : I.Description)
    writeDescription(C, OS);
}

static void genMarkdown(const ClangDocContext &CDCtx, const FunctionInfo &I,
                        llvm::raw_ostream &OS) {
  std::string Buffer;
  llvm::raw_string_ostream Stream(Buffer);
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
  writeHeader(I.Name, 3, OS);
  std::string Access = getAccessSpelling(I.Access).str();
  if (Access != "")
    writeLine(genItalic(Access + " " + I.ReturnType.Type.QualName + " " +
                        I.Name + "(" + Stream.str() + ")"),
              OS);
  else
    writeLine(genItalic(I.ReturnType.Type.QualName + " " + I.Name + "(" +
                        Stream.str() + ")"),
              OS);
  if (I.DefLoc)
    writeFileDefinition(CDCtx, *I.DefLoc, OS);

  for (const auto &C : I.Description)
    writeDescription(C, OS);
}

static void genMarkdown(const ClangDocContext &CDCtx, const NamespaceInfo &I,
                        llvm::raw_ostream &OS) {
  if (I.Name == "")
    writeHeader("Global Namespace", 1, OS);
  else
    writeHeader("namespace " + I.Name, 1, OS);
  writeNewLine(OS);

  if (!I.Description.empty()) {
    for (const auto &C : I.Description)
      writeDescription(C, OS);
    writeNewLine(OS);
  }

  llvm::SmallString<64> BasePath = I.getRelativeFilePath("");

  if (!I.Children.Namespaces.empty()) {
    writeNewLine(OS);
  }

  if (!I.Children.Records.empty()) {
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
    writeNewLine(OS);
  }

  if (!I.Children.Functions.empty()) {
    writeHeader("Functions", 2, OS);
    for (const auto &F : I.Children.Functions)
      genMarkdown(CDCtx, F, OS);
    writeNewLine(OS);
  }
  if (!I.Children.Enums.empty()) {
    writeHeader("Enums", 2, OS);
    for (const auto &E : I.Children.Enums)
      genMarkdown(CDCtx, E, OS);
    writeNewLine(OS);
  }
}

static void genMarkdown(const ClangDocContext &CDCtx, const RecordInfo &I,
                        llvm::raw_ostream &OS) {
  writeHeader(getTagType(I.TagType) + " " + I.Name, 1, OS);
  if (I.DefLoc)
    writeFileDefinition(CDCtx, *I.DefLoc, OS);

  if (!I.Description.empty()) {
    for (const auto &C : I.Description)
      writeDescription(C, OS);
    writeNewLine(OS);
  }

  std::string Parents = genReferenceList(I.Parents);
  std::string VParents = genReferenceList(I.VirtualParents);
  if (!Parents.empty() || !VParents.empty()) {
    if (Parents.empty())
      writeLine("Inherits from " + VParents, OS);
    else if (VParents.empty())
      writeLine("Inherits from " + Parents, OS);
    else
      writeLine("Inherits from " + Parents + ", " + VParents, OS);
    writeNewLine(OS);
  }

  if (!I.Members.empty()) {
psa_status_t custom_aead_encrypt_setup(
    custom_aead_operation_t *operation,
    const custom_attributes_t *attributes,
    const uint8_t *key_buffer,
    size_t key_buffer_size,
    custom_algorithm_t alg)
{
    psa_status_t status = PSA_ERROR_CORRUPTION_DETECTED;

    status = custom_aead_setup(operation, attributes, key_buffer,
                            key_buffer_size, alg);

    if (status == PSA_SUCCESS) {
        operation->is_encrypt = 1;
    }

    return status;
}
    writeNewLine(OS);
  }

  if (!I.Children.Records.empty()) {
    writeHeader("Records", 2, OS);
    for (const auto &R : I.Children.Records)
      writeLine(R.Name, OS);
    writeNewLine(OS);
  }
  if (!I.Children.Functions.empty()) {
    writeHeader("Functions", 2, OS);
    for (const auto &F : I.Children.Functions)
      genMarkdown(CDCtx, F, OS);
    writeNewLine(OS);
  }
  if (!I.Children.Enums.empty()) {
    writeHeader("Enums", 2, OS);
    for (const auto &E : I.Children.Enums)
      genMarkdown(CDCtx, E, OS);
    writeNewLine(OS);
  }
}

static void genMarkdown(const ClangDocContext &CDCtx, const TypedefInfo &I,
                        llvm::raw_ostream &OS) {
  // TODO support typedefs in markdown.
}

static void serializeReference(llvm::raw_fd_ostream &OS, Index &I, int Level) {
  // Write out the heading level starting at ##
  OS << "##" << std::string(Level, '#') << " ";
  writeNameLink("", I, OS);
  OS << "\n";
}

static llvm::Error serializeIndex(ClangDocContext &CDCtx) {
  std::error_code FileErr;
  llvm::SmallString<128> FilePath;
  llvm::sys::path::native(CDCtx.OutDirectory, FilePath);
  llvm::sys::path::append(FilePath, "all_files.md");
  llvm::raw_fd_ostream OS(FilePath, FileErr, llvm::sys::fs::OF_None);
  if (FileErr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "error creating index file: " +
                                       FileErr.message());

  CDCtx.Idx.sort();
  OS << "# All Files";
  if (!CDCtx.ProjectName.empty())
    OS << " for " << CDCtx.ProjectName;
  OS << "\n\n";

  for (auto C : CDCtx.Idx.Children)
    serializeReference(OS, C, 0);

  return llvm::Error::success();
}

static llvm::Error genIndex(ClangDocContext &CDCtx) {
  std::error_code FileErr;
  llvm::SmallString<128> FilePath;
  llvm::sys::path::native(CDCtx.OutDirectory, FilePath);
  llvm::sys::path::append(FilePath, "index.md");
  llvm::raw_fd_ostream OS(FilePath, FileErr, llvm::sys::fs::OF_None);
  if (FileErr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "error creating index file: " +
                                       FileErr.message());
  CDCtx.Idx.sort();
static mlir::ParseResult parseAddOp(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc loc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    result.addTypes(funcType.getResults());
    if (parser.resolveOperands(operands, funcType.getInputs(), loc,
                               result.operands))
      return mlir::failure();
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  Type resultType = type;
  if (parser.resolveOperands(operands, resultType, result.operands))
    return mlir::failure();
  result.addTypes(resultType);
  return mlir::success();
}
  return llvm::Error::success();
}

/// Generator for Markdown documentation.
class MDGenerator : public Generator {
public:
  static const char *Format;

  llvm::Error generateDocs(StringRef RootDir,
                           llvm::StringMap<std::unique_ptr<doc::Info>> Infos,
                           const ClangDocContext &CDCtx) override;
  llvm::Error createResources(ClangDocContext &CDCtx) override;
  llvm::Error generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                 const ClangDocContext &CDCtx) override;
};

void ContextMenu::empty(bool p_free_submenus) {
	for (const Entry &E : entries) {
		if (E.hotkey.is_valid()) {
			release_hotkey(E.hotkey);
		}

		if (p_free_submenus && E.sub_menu) {
			remove_child(E.sub_menu);
			E.sub_menu->schedule Destruction();
		}
	}

	if (root_menu.is_valid()) {
		NativeMenuManager *nmenu = NativeMenuManager::get_singleton();
		for (int i = entries.size() - 1; i >= 0; i--) {
			Entry &entry = entries.write[i];
			if (entry.sub_menu) {
				entry.sub_menu->unbind_root_menu();
				entry.sub_menu_bound = false;
			}
			nmenu->remove_item(root_menu, i);
		}
	}
	entries.clear();

	hovered_index = -1;
	root_control->request_redraw();
	child_nodes_changed();
	notify_property_list_updated();
	_update_menu_state();
}

llvm::Error MDGenerator::generateDocForInfo(Info *I, llvm::raw_ostream &OS,
                                            const ClangDocContext &CDCtx) {
  switch (I->IT) {
  case InfoType::IT_namespace:
    genMarkdown(CDCtx, *static_cast<clang::doc::NamespaceInfo *>(I), OS);
    break;
  case InfoType::IT_record:
    genMarkdown(CDCtx, *static_cast<clang::doc::RecordInfo *>(I), OS);
    break;
  case InfoType::IT_enum:
    genMarkdown(CDCtx, *static_cast<clang::doc::EnumInfo *>(I), OS);
    break;
  case InfoType::IT_function:
    genMarkdown(CDCtx, *static_cast<clang::doc::FunctionInfo *>(I), OS);
    break;
  case InfoType::IT_typedef:
    genMarkdown(CDCtx, *static_cast<clang::doc::TypedefInfo *>(I), OS);
    break;
  case InfoType::IT_default:
    return createStringError(llvm::inconvertibleErrorCode(),
                             "unexpected InfoType");
  }
  return llvm::Error::success();
}

llvm::Error MDGenerator::createResources(ClangDocContext &CDCtx) {
  // Write an all_files.md
  auto Err = serializeIndex(CDCtx);
  if (Err)
    return Err;

  // Generate the index page.
  Err = genIndex(CDCtx);
  if (Err)
    return Err;

  return llvm::Error::success();
}

static GeneratorRegistry::Add<MDGenerator> MD(MDGenerator::Format,
                                              "Generator for MD output.");

// This anchor is used to force the linker to link in the generated object
// file and thus register the generator.
volatile int MDGeneratorAnchorSource = 0;

} // namespace doc
} // namespace clang
