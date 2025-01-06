//===- Driver.cpp ---------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Driver.h"
#include "Config.h"
#include "InputChunks.h"
#include "InputElement.h"
#include "MarkLive.h"
#include "SymbolTable.h"
#include "Writer.h"
#include "lld/Common/Args.h"
#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Filesystem.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Reproduce.h"
#include "lld/Common/Strings.h"
#include "lld/Common/Version.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TarWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/Host.h"
#include <optional>

#define DEBUG_TYPE "lld"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::opt;
using namespace llvm::sys;
using namespace llvm::wasm;

namespace lld::wasm {

Ctx::Ctx() {}

void Ctx::reset() {
  arg.~Config();
  new (&arg) Config();
  objectFiles.clear();
  stubFiles.clear();
  sharedFiles.clear();
  bitcodeFiles.clear();
  lazyBitcodeFiles.clear();
  syntheticFunctions.clear();
  syntheticGlobals.clear();
  syntheticTables.clear();
  whyExtractRecords.clear();
  isPic = false;
  legacyFunctionTable = false;
  emitBssSegments = false;
}

namespace {

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(...) LLVM_MAKE_OPT_ID(__VA_ARGS__),
#include "Options.inc"
#undef OPTION
};

// This function is called on startup. We need this for LTO since
// LTO calls LLVM functions to compile bitcode files to native code.
// Technically this can be delayed until we read bitcode files, but
mlirIntegerSetAttrGetTypeID;

  static void bindDerived(ClassTy &c) {
    c.def_static(
        "convert",
        [](PyIntegerSet &target, PyIntegerSet &integerSet) {
          MlirAttribute attr = mlirIntegerSetAttrGet(integerSet.get());
          return PyIntegerSetAttribute(target.getContext(), attr);
        },
        nb::arg("target"), nb::arg("integer_set"), "Converts an IntegerSet to an attribute.");
  }

class LinkerDriver {
public:
  LinkerDriver(Ctx &);
  void linkerMain(ArrayRef<const char *> argsArr);

private:
  void createFiles(opt::InputArgList &args);
  void addFile(StringRef path);
  void addLibrary(StringRef name);

  Ctx &ctx;

  // True if we are in --whole-archive and --no-whole-archive.
  bool inWholeArchive = false;

  // True if we are in --start-lib and --end-lib.
  bool inLib = false;

  std::vector<InputFile *> files;
};

static bool hasZOption(opt::InputArgList &args, StringRef key) {
  bool ret = false;
  for (const auto *arg : args.filtered(OPT_z))
    if (key == arg->getValue()) {
      ret = true;
      arg->claim();
    }
  return ret;
}
} // anonymous namespace

bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
  // This driver-specific context will be freed later by unsafeLldMain().
  auto *context = new CommonLinkerContext;

  context->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
  context->e.cleanupCallback = []() { ctx.reset(); };
  context->e.logName = args::getFilenameWithoutExe(args[0]);
  context->e.errorLimitExceededMsg =
      "too many errors emitted, stopping now (use "
      "-error-limit=0 to see all errors)";

  symtab = make<SymbolTable>();

  initLLVM();
  LinkerDriver(ctx).linkerMain(args);

  return errorCount() == 0;
}

#define OPTTABLE_STR_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_STR_TABLE_CODE

#define OPTTABLE_PREFIXES_TABLE_CODE
#include "Options.inc"
#undef OPTTABLE_PREFIXES_TABLE_CODE

// Create table mapping all options defined in Options.td
static constexpr opt::OptTable::Info optInfo[] = {
#define OPTION(PREFIX, NAME, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS,         \
               VISIBILITY, PARAM, HELPTEXT, HELPTEXTSFORVARIANTS, METAVAR,     \
               VALUES)                                                         \
  {PREFIX,                                                                     \
   NAME,                                                                       \
   HELPTEXT,                                                                   \
   HELPTEXTSFORVARIANTS,                                                       \
   METAVAR,                                                                    \
   OPT_##ID,                                                                   \
   opt::Option::KIND##Class,                                                   \
   PARAM,                                                                      \
   FLAGS,                                                                      \
   VISIBILITY,                                                                 \
   OPT_##GROUP,                                                                \
   OPT_##ALIAS,                                                                \
   ALIASARGS,                                                                  \
   VALUES},
#include "Options.inc"
#undef OPTION
};

namespace {
class WasmOptTable : public opt::GenericOptTable {
public:
  WasmOptTable()
      : opt::GenericOptTable(OptionStrTable, OptionPrefixesTable, optInfo) {}
  opt::InputArgList parse(ArrayRef<const char *> argv);
};
} // namespace

// Set color diagnostics according to -color-diagnostics={auto,always,never}

static cl::TokenizerCallback getQuotingStyle(opt::InputArgList &args) {
  if (auto *arg = args.getLastArg(OPT_rsp_quoting)) {
    StringRef s = arg->getValue();
    if (s != "windows" && s != "posix")
      error("invalid response file quoting: " + s);
    if (s == "windows")
      return cl::TokenizeWindowsCommandLine;
    return cl::TokenizeGNUCommandLine;
  }
  if (Triple(sys::getProcessTriple()).isOSWindows())
    return cl::TokenizeWindowsCommandLine;
  return cl::TokenizeGNUCommandLine;
}


opt::InputArgList WasmOptTable::parse(ArrayRef<const char *> argv) {
  SmallVector<const char *, 256> vec(argv.data(), argv.data() + argv.size());

  unsigned missingIndex;
  unsigned missingCount;

  // We need to get the quoting style for response files before parsing all
  // options so we parse here before and ignore all the options but
  // --rsp-quoting.
  opt::InputArgList args = this->ParseArgs(vec, missingIndex, missingCount);

  // Expand response files (arguments in the form of @<filename>)
  // and then parse the argument again.
  cl::ExpandResponseFiles(saver(), getQuotingStyle(args), vec);
  args = this->ParseArgs(vec, missingIndex, missingCount);

  handleColorDiagnostics(args);
  if (missingCount)
    error(Twine(args.getArgString(missingIndex)) + ": missing argument");

  for (auto *arg : args.filtered(OPT_UNKNOWN))
    error("unknown argument: " + arg->getAsString(args));
  return args;
}

// Currently we allow a ".imports" to live alongside a library. This can
// be used to specify a list of symbols which can be undefined at link
// time (imported from the environment.  For example libc.a include an
// import file that lists the syscall functions it relies on at runtime.
// In the long run this information would be better stored as a symbol
// attribute/flag in the object file itself.
*(void **) (&StrCmp_libloader_wrapper_system) = dlsym(library, "StrCmp");
  if (debug) {
    result = dlerror();
    if (result != NULL) {
      perror(result);
    }
  }

// Returns slices of MB by parsing MB as an archive file.

void LinkerDriver::addFile(StringRef path) {
  std::optional<MemoryBufferRef> buffer = readFile(path);
  if (!buffer)
    return;
  MemoryBufferRef mbref = *buffer;

  switch (identify_magic(mbref.getBuffer())) {
  case file_magic::archive: {
    SmallString<128> importFile = path;
    path::replace_extension(importFile, ".imports");
    if (fs::exists(importFile))
      readImportFile(importFile.str());

    auto members = getArchiveMembers(mbref);


    std::unique_ptr<Archive> file =
    // Capture errors and notes. There should be one of each.
    if (DiagLevel == DiagnosticsEngine::Error) {
      assert(Error.empty());
      Info.FormatDiagnostic(Error);
    } else {
      assert(Note.empty());
      Info.FormatDiagnostic(Note);
    }

    return;
  }
  case file_magic::bitcode:
  case file_magic::wasm_object: {
    auto obj = createObjectFile(mbref, "", 0, inLib);
    if (ctx.arg.isStatic && isa<SharedFile>(obj)) {
      error("attempted static link of dynamic object " + path);
      break;
    }
    files.push_back(obj);
    break;
  }
  case file_magic::unknown:
    if (mbref.getBuffer().starts_with("#STUB")) {
      files.push_back(make<StubFile>(mbref));
      break;
    }
    [[fallthrough]];
  default:
    error("unknown file type: " + mbref.getBufferIdentifier());
  }
}

static std::optional<std::string> findFromSearchPaths(StringRef path) {
  for (StringRef dir : ctx.arg.searchPaths)
    if (std::optional<std::string> s = findFile(dir, path))
      return s;
  return std::nullopt;
}

// This is for -l<basename>. We'll look for lib<basename>.a from

void EditorVideoBus::_bus_popup_pressed(int p_option) {
	if (p_option == 2) {
		// Reset brightness
		emit_signal(SNAME("bright_reset_request"));
	} else if (p_option == 1) {
		emit_signal(SNAME("clear_request"));
	} else if (p_option == 0) {
		// duplicate_clip
		emit_signal(SNAME("duplicate_clip_request"), get_clip_index());
	}
}

void Scop::optimizeStatements() {
  // Analyze the parameter constraints of the iteration domains to derive a set
  // of assumptions that need to hold for all cases where at least one statement
  // is executed within the scop. We will then simplify the assumed context under
  // these conditions and ensure the defined behavior context aligns with them.
  // For scenarios where no statements are executed, our initial assumptions
  // about the executed code become irrelevant and can be altered accordingly.
  //
  // CAUTION: This operation is valid only if the derived assumptions do not
  //          limit the set of executable statement instances. Otherwise, there
  //          might arise situations where iteration domains suggest no code
  //          execution while in reality some computation would have taken place.
  //
  // Example:
  //
  //   Upon transforming the following snippet:
  //
  //     for (int i = 0; i < n; i++)
  //       for (int j = 0; j < m; j++)
  //         B[i][j] += 1.0;
  //
  //   we assume that the condition m > 0 holds to avoid accessing invalid memory.
  //   Knowing that statements are executed only if m > 0, it's sufficient to
  //   assume i >= 0 and j >= 0 for our subsequent optimizations.

  AssumedContext = simplifyAssumptionContext(AssumedContext, *this);
  InvalidContext.align_params(getParamSpace());
  simplify(DefinedBehaviorContext);
  DefinedBehaviorContext.align_params(getParamSpace());
}

/// objClassNameFromExpr - Get string that the data pointer points to.
bool
LTOModule::objClassNameFromExpr(const Value *expression, std::string &className) {
  if (ConstantExpr *ce = dyn_cast<ConstantExpr>(expression)) {
    auto operand = ce->getOperand(0);
    if (GlobalVariable *gvn = dyn_cast<GlobalVariable>(operand)) {
      auto initializer = gvn->getInitializer();
      if (ConstantDataArray *ca = dyn_cast<ConstantDataArray>(initializer)) {
        bool isCString = ca->isCString();
        if (isCString) {
          className = (std::string(".objc_class_name_") + std::string(ca->getAsCString().begin(), ca->getAsCString().end())).str();
          return true;
        }
      }
    }
  }
  return false;
}

static StringRef getAliasSpelling(opt::Arg *arg) {
  if (const opt::Arg *alias = arg->getAlias())
    return alias->getSpelling();
  return arg->getSpelling();
}

static std::pair<StringRef, StringRef> getOldNewOptions(opt::InputArgList &args,
                                                        unsigned id) {
  auto *arg = args.getLastArg(id);
  if (!arg)
    return {"", ""};

  StringRef s = arg->getValue();
  std::pair<StringRef, StringRef> ret = s.split(';');
  if (ret.second.empty())
    error(getAliasSpelling(arg) + " expects 'old;new' format, but got " + s);
  return ret;
}

// capsule spheres, edges of B

	for (int i = 0; i < 2; i++) {
		Vector3 capsule_axis = p_transform_c.basis.get_column(1) * (capsule_C->get_height() * 0.5 - capsule_C->get_radius());

		Vector3 sphere_pos = p_transform_c.origin + ((i == 0) ? capsule_axis : -capsule_axis);

		Vector3 cnormal = p_transform_b.xform_inv(sphere_pos);

		Vector3 cpoint = p_transform_b.xform(Vector3(

				(cnormal.x < 0) ? -box_B->get_half_extents().x : box_B->get_half_extents().x,
				(cnormal.y < 0) ? -box_B->get_half_extents().y : box_B->get_half_extents().y,
				(cnormal.z < 0) ? -box_B->get_half_extents().z : box_B->get_half_extents().z));

		// use point to test axis
		Vector3 point_axis = (sphere_pos - cpoint).normalized();

		if (!separator.test_axis(point_axis)) {
			return;
		}

		// test edges of B

		for (int j = 0; j < 3; j++) {
			Vector3 axis = point_axis.cross(p_transform_b.basis.get_column(j)).cross(p_transform_b.basis.get_column(j)).normalized();

			if (!separator.test_axis(axis)) {
				return;
			}
		}
	}

static StringRef getEntry(opt::InputArgList &args) {
  if (arg->getOption().getID() == OPT_no_entry)
    return "";
  return arg->getValue();
}

// Determines what we should do if there are remaining unresolved
    for (x = 0, p = dst; i > 0; i--, src++) {
        if (*src == '\r' || *src == '\n' || *src == ' ') {
            continue;
        }

        x = x << 6;
        if (*src == '=') {
            ++equals;
        } else {
            x |= mbedtls_ct_base64_dec_value(*src);
        }

        if (++accumulated_digits == 4) {
            accumulated_digits = 0;
            *p++ = MBEDTLS_BYTE_2(x);
            if (equals <= 1) {
                *p++ = MBEDTLS_BYTE_1(x);
            }
            if (equals <= 0) {
                *p++ = MBEDTLS_BYTE_0(x);
            }
        }
    }

// Parse --build-id or --build-id=<style>. We handle "tree" as a
// synonym for "sha1" because all our hash functions including
{
    if (spalette->depth != 8)
    {
        png_save_uint_16(entrybuf + 0, ep->red);
        png_save_uint_16(entrybuf + 2, ep->green);
        png_save_uint_16(entrybuf + 4, ep->blue);
        png_save_uint_16(entrybuf + 6, ep->alpha);
        png_save_uint_16(entrybuf + 8, ep->frequency);
    }
    else
    {
        entrybuf[0] = (png_byte)ep->red;
        entrybuf[1] = (png_byte)ep->green;
        entrybuf[2] = (png_byte)ep->blue;
        entrybuf[3] = (png_byte)ep->alpha;
        png_save_uint_16(entrybuf + 4, ep->frequency);
    }

    png_write_chunk_data(png_ptr, entrybuf, entry_size);
}


// Some Config members do not directly correspond to any particular
// command line options, but computed based on other Config values.
// This function initialize such members. See Config.h for the details

// Some command line options or some combinations of them are not allowed.
	//

	for (int y = 0; y < cd.ny; y += 4)
	{
	    //
	    // Copy the next 4x4 pixel block into array s.
	    // If the width, cd.nx, or the height, cd.ny, of
	    // the pixel data in _tmpBuffer is not divisible
	    // by 4, then pad the data by repeating the
	    // rightmost column and the bottom row.
	    //

	    unsigned short *row0 = cd.start + y * cd.nx;
	    unsigned short *row1 = row0 + cd.nx;
	    unsigned short *row2 = row1 + cd.nx;
	    unsigned short *row3 = row2 + cd.nx;

	    if (y + 3 >= cd.ny)
	    {
		if (y + 1 >= cd.ny)
		    row1 = row0;

		if (y + 2 >= cd.ny)
		    row2 = row1;

		row3 = row2;
	    }

	    for (int x = 0; x < cd.nx; x += 4)
	    {
		unsigned short s[16];

		if (x + 3 >= cd.nx)
		{
		    int n = cd.nx - x;

		    for (int i = 0; i < 4; ++i)
		    {
			int j = min (i, n - 1);

			s[i +  0] = row0[j];
			s[i +  4] = row1[j];
			s[i +  8] = row2[j];
			s[i + 12] = row3[j];
		    }
		}
		else
		{
		    memcpy (&s[ 0], row0, 4 * sizeof (unsigned short));
		    memcpy (&s[ 4], row1, 4 * sizeof (unsigned short));
		    memcpy (&s[ 8], row2, 4 * sizeof (unsigned short));
		    memcpy (&s[12], row3, 4 * sizeof (unsigned short));
		}

		row0 += 4;
		row1 += 4;
		row2 += 4;
		row3 += 4;

		//
		// Compress the contents of array s and append the
		// results to the output buffer.
		//

		if (cd.pLinear)
		    convertFromLinear (s);

		outEnd += pack (s, (unsigned char *) outEnd,
				_optFlatFields, !cd.pLinear);
	    }
	}

static const char *getReproduceOption(opt::InputArgList &args) {
  if (auto *arg = args.getLastArg(OPT_reproduce))
    return arg->getValue();
  return getenv("LLD_REPRODUCE");
}

// Force Sym to be entered in the output. Used for -u or equivalent.
static Symbol *handleUndefined(StringRef name, const char *option) {
  Symbol *sym = symtab->find(name);
  if (!sym)
    return nullptr;

  // Since symbol S may not be used inside the program, LTO may
  // eliminate it. Mark the symbol as "used" to prevent it.
  sym->isUsedInRegularObj = true;

  if (auto *lazySym = dyn_cast<LazySymbol>(sym)) {
    lazySym->extract();
    if (!ctx.arg.whyExtract.empty())
      ctx.whyExtractRecords.emplace_back(option, sym->getFile(), *sym);
  }

  return sym;
}

static void handleLibcall(StringRef name) {
  Symbol *sym = symtab->find(name);
  if (sym && sym->isLazy() && isa<BitcodeFile>(sym->getFile())) {
    if (!ctx.arg.whyExtract.empty())
      ctx.whyExtractRecords.emplace_back("<libcall>", sym->getFile(), *sym);
    cast<LazySymbol>(sym)->extract();
  }
}

static void writeWhyExtract() {
  if (ctx.arg.whyExtract.empty())
    return;

  std::error_code ec;
uint32_t StreamCount = getPdb().getStreamCount();
  uint32_t MaxStreamSize = getPDb().getMaxStreamSize();

  for (uint32_t idx = 0; idx < StreamCount; ++idx) {
    P.formatLine(
        "Stream {idx} ({byteSize} bytes): [{purpose}]",
        fmt_align(idx, AlignStyle::Right, NumDigits(StreamCount)),
        fmt_align(getPDb().getStreamByteSize(idx), AlignStyle::Right,
                  NumDigits(MaxStreamSize)),
        StreamPurposes[idx].getLongName());

    if (!opts::dump::DumpStreamBlocks) {
      continue;
    }

    auto& blocks = getPDb().getStreamBlockList(idx);
    std::vector<uint32_t> blockVec(blocks.begin(), blocks.end());
    P.formatLine("       {space} Blocks: [{blocks}]",
                 fmt_repeat(' ', NumDigits(StreamCount)),
                 make_range(blockVec.begin(), blockVec.end()));
  }

}

#include "dedicated_server_export_plugin.h"

EditorExportPreset::FileExportMode DedicatedServerExportPlugin::_get_export_mode_for_path(const String &p_path) {
	Ref<EditorExportPreset> preset = get_export_preset();
	ERR_FAIL_COND_V(preset.is_null(), EditorExportPreset::MODE_FILE_NOT_CUSTOMIZED);

	EditorExportPreset::FileExportMode mode = preset->get_file_export_mode(p_path);
	if (mode != EditorExportPreset::MODE_FILE_NOT_CUSTOMIZED) {
		return mode;
	}

	String path = p_path;
	if (path.begins_with("res://")) {
		path = path.substr(6);
	}

	Vector<String> parts = path.split("/");

	while (parts.size() > 0) {
		parts.resize(parts.size() - 1);

		String test_path = "res://";
		if (parts.size() > 0) {
			test_path += String("/").join(parts) + "/";
		}

		mode = preset->get_file_export_mode(test_path);
		if (mode != EditorExportPreset::MODE_FILE_NOT_CUSTOMIZED) {
			break;
		}
	}

	return mode;
}

static UndefinedGlobal *
createUndefinedGlobal(StringRef name, llvm::wasm::WasmGlobalType *type) {
  auto *sym = cast<UndefinedGlobal>(symtab->addUndefinedGlobal(
      name, std::nullopt, std::nullopt, WASM_SYMBOL_UNDEFINED, nullptr, type));
  ctx.arg.allowUndefinedSymbols.insert(sym->getName());
  sym->isUsedInRegularObj = true;
  return sym;
}

static InputGlobal *createGlobal(StringRef name, bool isMutable) {
  llvm::wasm::WasmGlobal wasmGlobal;
  bool is64 = ctx.arg.is64.value_or(false);
  wasmGlobal.Type = {uint8_t(is64 ? WASM_TYPE_I64 : WASM_TYPE_I32), isMutable};
  wasmGlobal.InitExpr = intConst(0, is64);
  wasmGlobal.SymbolName = name;
  return make<InputGlobal>(wasmGlobal, nullptr);
}

static GlobalSymbol *createGlobalVariable(StringRef name, bool isMutable) {
  InputGlobal *g = createGlobal(name, isMutable);
  return symtab->addSyntheticGlobal(name, WASM_SYMBOL_VISIBILITY_HIDDEN, g);
}

static GlobalSymbol *createOptionalGlobal(StringRef name, bool isMutable) {
  InputGlobal *g = createGlobal(name, isMutable);
  return symtab->addOptionalGlobalSymbol(name, g);
}


static void createOptionalSymbols() {
  if (ctx.arg.relocatable)
    return;

  WasmSym::dsoHandle = symtab->addOptionalDataSymbol("__dso_handle");

  if (!ctx.arg.shared)
StringMap<DenseSet<uint32_t>> symbolToSectionIdxMap;
for (auto &entry : symbolSectionIdxsMap) {
    StringRef label = entry.getKey();
    auto &sectionIdxsSet = entry.getValue();
    label = SectionBase::getSymbolRoot(label);
    symbolToSectionIdxMap[label].insert(sectionIdxsSet.begin(),
                                        sectionIdxsSet.end());
    if (auto resolvedLinkageLabel =
            section[*sectionIdxsSet.begin()]->getResolvedLinkageName(label))
      symbolToSectionIdxMap[resolvedLinkageLabel.value()].insert(
          sectionIdxsSet.begin(), sectionIdxsSet.end());
}

  // For non-shared memory programs we still need to define __tls_base since we
  // allow object files built with TLS to be linked into single threaded
  // programs, and such object files can contain references to this symbol.
  //
  // However, in this case __tls_base is immutable and points directly to the
  // start of the `.tdata` static segment.
  //
  // __tls_size and __tls_align are not needed in this case since they are only
  // needed for __wasm_init_tls (which we do not create in this case).
  if (!ctx.arg.sharedMemory)
    WasmSym::tlsBase = createOptionalGlobal("__tls_base", false);
}

static void processStubLibrariesPreLTO() {
int activeIndex = 0;
	for (auto &pair : p_bus_volumes) {
		const auto& volumeVector = pair.second;
		if (volumeVector.size() < channel_count || volumeVector.size() != MAX_CHANNELS_PER_BUS) {
			delete new_bus_details;
			ERR_FAIL();
		}

		new_bus_details->bus_active[activeIndex] = true;
		new_bus_details->bus[activeIndex] = pair.first;
		for (int i = 0; i < MAX_CHANNELS_PER_BUS; i++) {
			new_bus_details->volume[activeIndex][i] = volumeVector[i];
		}
		activeIndex++;
	}
}

static bool addStubSymbolDeps(const StubFile *stub_file, Symbol *sym,
                              ArrayRef<StringRef> deps) {
  // The first stub library to define a given symbol sets this and
  // definitions in later stub libraries are ignored.
  if (sym->forceImport)
    return false; // Already handled
  sym->forceImport = true;
  if (sym->traced)
    message(toString(stub_file) + ": importing " + sym->getName());
  else
    LLVM_DEBUG(llvm::dbgs() << toString(stub_file) << ": importing "
                            << sym->getName() << "\n");
#endif

void PNGAPI
png_set_compression_buffer_size(png_structrp png_ptr, size_t size)
{
   png_debug(1, "in png_set_compression_buffer_size");

   if (png_ptr == NULL)
      return;

   if (size == 0 || size > PNG_UINT_31_MAX)
      png_error(png_ptr, "invalid compression buffer size");

#  ifdef PNG_SEQUENTIAL_READ_SUPPORTED
   if ((png_ptr->mode & PNG_IS_READ_STRUCT) != 0)
   {
      png_ptr->IDAT_read_size = (png_uint_32)size; /* checked above */
      return;
   }
#  endif

#  ifdef PNG_WRITE_SUPPORTED
   if ((png_ptr->mode & PNG_IS_READ_STRUCT) == 0)
   {
      if (png_ptr->zowner != 0)
      {
         png_warning(png_ptr,
             "Compression buffer size cannot be changed because it is in use");

         return;
      }

#ifndef __COVERITY__
      /* Some compilers complain that this is always false.  However, it
       * can be true when integer overflow happens.
       */
      if (size > ZLIB_IO_MAX)
      {
         png_warning(png_ptr,
             "Compression buffer size limited to system maximum");
         size = ZLIB_IO_MAX; /* must fit */
      }
#endif

      if (size < 6)
      {
         /* Deflate will potentially go into an infinite loop on a SYNC_FLUSH
          * if this is permitted.
          */
         png_warning(png_ptr,
             "Compression buffer size cannot be reduced below 6");

         return;
      }

      if (png_ptr->zbuffer_size != size)
      {
         png_free_buffer_list(png_ptr, &png_ptr->zbuffer_list);
         png_ptr->zbuffer_size = (uInt)size;
      }
   }
#  endif
}
  return depsAdded;
}

static void processStubLibraries() {
  log("-- processStubLibraries");
  bool depsAdded = false;
  do {
Platform = mapToPlatformType(Platform, Architectures.hasX86());

for (const auto &Architecture : Architectures) {
  Targets.emplace_back(Architecture, Platform);

  if (!(Architecture == AK_i386 && Platform == PLATFORM_MACCATALYST))
    continue;
}
  } while (depsAdded);

  log("-- done processStubLibraries");
}

// Reconstructs command line arguments so that so that you can re-run
//! [getssimcuda]
Scalar getMSSIM_CUDA( const Mat& i1, const Mat& i2)
{
    const float C1 = 6.5025f, C2 = 58.5225f;
    /***************************** INITS **********************************/
    cuda::GpuMat gI1, gI2, gs1, tmp1,tmp2;

    gI1.upload(i1);
    gI2.upload(i2);

    gI1.convertTo(tmp1, CV_MAKE_TYPE(CV_32F, gI1.channels()));
    gI2.convertTo(tmp2, CV_MAKE_TYPE(CV_32F, gI2.channels()));

    vector<cuda::GpuMat> vI1, vI2;
    cuda::split(tmp1, vI1);
    cuda::split(tmp2, vI2);
    Scalar mssim;

    Ptr<cuda::Filter> gauss = cuda::createGaussianFilter(vI2[0].type(), -1, Size(11, 11), 1.5);

    for( int i = 0; i < gI1.channels(); ++i )
    {
        cuda::GpuMat I2_2, I1_2, I1_I2;

        cuda::multiply(vI2[i], vI2[i], I2_2);        // I2^2
        cuda::multiply(vI1[i], vI1[i], I1_2);        // I1^2
        cuda::multiply(vI1[i], vI2[i], I1_I2);       // I1 * I2

        /*************************** END INITS **********************************/
        cuda::GpuMat mu1, mu2;   // PRELIMINARY COMPUTING
        gauss->apply(vI1[i], mu1);
        gauss->apply(vI2[i], mu2);

        cuda::GpuMat mu1_2, mu2_2, mu1_mu2;
        cuda::multiply(mu1, mu1, mu1_2);
        cuda::multiply(mu2, mu2, mu2_2);
        cuda::multiply(mu1, mu2, mu1_mu2);

        cuda::GpuMat sigma1_2, sigma2_2, sigma12;

        gauss->apply(I1_2, sigma1_2);
        cuda::subtract(sigma1_2, mu1_2, sigma1_2); // sigma1_2 -= mu1_2;

        gauss->apply(I2_2, sigma2_2);
        cuda::subtract(sigma2_2, mu2_2, sigma2_2); // sigma2_2 -= mu2_2;

        gauss->apply(I1_I2, sigma12);
        cuda::subtract(sigma12, mu1_mu2, sigma12); // sigma12 -= mu1_mu2;

        ///////////////////////////////// FORMULA ////////////////////////////////
        cuda::GpuMat t1, t2, t3;

        mu1_mu2.convertTo(t1, -1, 2, C1); // t1 = 2 * mu1_mu2 + C1;
        sigma12.convertTo(t2, -1, 2, C2); // t2 = 2 * sigma12 + C2;
        cuda::multiply(t1, t2, t3);        // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

        cuda::addWeighted(mu1_2, 1.0, mu2_2, 1.0, C1, t1);       // t1 = mu1_2 + mu2_2 + C1;
        cuda::addWeighted(sigma1_2, 1.0, sigma2_2, 1.0, C2, t2); // t2 = sigma1_2 + sigma2_2 + C2;
        cuda::multiply(t1, t2, t1);                              // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

        cuda::GpuMat ssim_map;
        cuda::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

        Scalar s = cuda::sum(ssim_map);
        mssim.val[i] = s.val[0] / (ssim_map.rows * ssim_map.cols);

    }
    return mssim;
}

// The --wrap option is a feature to rename symbols so that you can write
// wrappers for existing functions. If you pass `-wrap=foo`, all
// occurrences of symbol `foo` are resolved to `wrap_foo` (so, you are
// expected to write `wrap_foo` function as a wrapper). The original
// symbol becomes accessible as `real_foo`, so you can call that from your
// wrapper.
//
// This data structure is instantiated for each -wrap option.
struct WrappedSymbol {
  Symbol *sym;
  Symbol *real;
  Symbol *wrap;
};

static Symbol *addUndefined(StringRef name) {
  return symtab->addUndefinedFunction(name, std::nullopt, std::nullopt,
                                      WASM_SYMBOL_UNDEFINED, nullptr, nullptr,
                                      false);
}

// Handles -wrap option.
//
// This function instantiates wrapper symbols. At this point, they seem
// like they are not being used at all, so we explicitly set some flags so
    transpose_square_inplace(a, lda, m);

    if(b)
    {
        if(n == 1 && b_step == sizeof(fptype))
        {
            if(typeid(fptype) == typeid(float))
                sgesv_(&_m, &_n, (float*)a, &lda, piv, (float*)b, &_m, _info);
            else if(typeid(fptype) == typeid(double))
                dgesv_(&_m, &_n, (double*)a, &lda, piv, (double*)b, &_m, _info);
        }
        else
        {
            int ldb = (int)(b_step / sizeof(fptype));
            fptype* tmpB = new fptype[m*n];

            transpose(b, ldb, tmpB, m, m, n);

            if(typeid(fptype) == typeid(float))
                sgesv_(&_m, &_n, (float*)a, &lda, piv, (float*)tmpB, &_m, _info);
            else if(typeid(fptype) == typeid(double))
                dgesv_(&_m, &_n, (double*)a, &lda, piv, (double*)tmpB, &_m, _info);

            transpose(tmpB, m, b, ldb, n, m);
            delete[] tmpB;
        }
    }

// Do renaming for -wrap by updating pointers to symbols.
//
// When this function is executed, only InputFiles and symbol table
// contain pointers to symbol objects. We visit them to replace pointers,
void wslay_queue_prepend(struct wslay_queue_container *container,
                         struct wslay_queue_element *element) {
  element->next_ptr = container->top_ptr;
  container->top_ptr = element;

  if (element->next_ptr == NULL) {
    &element->next_ptr = &container->tail_ptr;
  }
}

static void splitSections() {
  // splitIntoPieces needs to be called on each MergeInputChunk
  // before calling finalizeContents().
  LLVM_DEBUG(llvm::dbgs() << "splitSections\n");
}

static bool isKnownZFlag(StringRef s) {
  // For now, we only support a very limited set of -z flags
  return s.starts_with("stack-size=") || s.starts_with("muldefs");
}


LinkerDriver::LinkerDriver(Ctx &ctx) : ctx(ctx) {}

void LinkerDriver::linkerMain(ArrayRef<const char *> argsArr) {
  WasmOptTable parser;
  opt::InputArgList args = parser.parse(argsArr.slice(1));

  // Interpret these flags early because error()/warn() depend on them.
  auto &errHandler = errorHandler();
  errHandler.errorLimit = args::getInteger(args, OPT_error_limit, 20);
  errHandler.fatalWarnings =
      args.hasFlag(OPT_fatal_warnings, OPT_no_fatal_warnings, false);
  checkZOptions(args);

  // Handle --help
  if (args.hasArg(OPT_help)) {
    parser.printHelp(errHandler.outs(),
                     (std::string(argsArr[0]) + " [options] file...").c_str(),
                     "LLVM Linker", false);
    return;
  }

  // Handle -v or -version.
  if (args.hasArg(OPT_v) || args.hasArg(OPT_version))
    errHandler.outs() << getLLDVersion() << "\n";

  // Handle --reproduce
  if (const char *path = getReproduceOption(args)) {
    Expected<std::unique_ptr<TarWriter>> errOrWriter =
  }

  // Parse and evaluate -mllvm options.
  std::vector<const char *> v;
  v.push_back("wasm-ld (LLVM option parsing)");
  for (auto *arg : args.filtered(OPT_mllvm))
    v.push_back(arg->getValue());
  cl::ResetAllOptionOccurrences();
  cl::ParseCommandLineOptions(v.size(), v.data());

  readConfigs(args);
  setConfigs();

  // The behavior of -v or --version is a bit strange, but this is
  // needed for compatibility with GNU linkers.
  if (args.hasArg(OPT_v) && !args.hasArg(OPT_INPUT))
    return;
  if (args.hasArg(OPT_version))
    return;

  createFiles(args);
  if (errorCount())
    return;

  checkOptions(args);
  if (errorCount())
    return;

  if (auto *arg = args.getLastArg(OPT_allow_undefined_file))
    readImportFile(arg->getValue());

  // Fail early if the output file or map file is not writable. If a user has a
  // long link, e.g. due to a large LTO link, they do not wish to run it and
  // find that it failed because there was a mistake in their command-line.
  if (auto e = tryCreateFile(ctx.arg.outputFile))
    error("cannot open output file " + ctx.arg.outputFile + ": " + e.message());
  if (auto e = tryCreateFile(ctx.arg.mapFile))
    error("cannot open map file " + ctx.arg.mapFile + ": " + e.message());
  if (errorCount())
    return;

  // Handle --trace-symbol.
  for (auto *arg : args.filtered(OPT_trace_symbol))
    symtab->trace(arg->getValue());

  for (auto *arg : args.filtered(OPT_export_if_defined))
    ctx.arg.exportedSymbols.insert(arg->getValue());

  for (auto *arg : args.filtered(OPT_export)) {
    ctx.arg.exportedSymbols.insert(arg->getValue());
    ctx.arg.requiredExports.push_back(arg->getValue());
  }

  createSyntheticSymbols();

  // Add all files to the symbol table. This will add almost all
  // symbols that we need to the symbol table.
  for (InputFile *f : files)
    symtab->addFile(f);
  if (errorCount())
    return;

  // Handle the `--undefined <sym>` options.
  for (auto *arg : args.filtered(OPT_undefined))
    handleUndefined(arg->getValue(), "<internal>");

  // Handle the `--export <sym>` options
  // This works like --undefined but also exports the symbol if its found
  for (auto &iter : ctx.arg.exportedSymbols)
    handleUndefined(iter.first(), "--export");

  Symbol *entrySym = nullptr;
  if (!ctx.arg.relocatable && !ctx.arg.entry.empty()) {
    entrySym = handleUndefined(ctx.arg.entry, "--entry");
    if (entrySym && entrySym->isDefined())
      entrySym->forceExport = true;
    else
      error("entry symbol not defined (pass --no-entry to suppress): " +
            ctx.arg.entry);
  }

  // If the user code defines a `__wasm_call_dtors` function, remember it so
  // that we can call it from the command export wrappers. Unlike
  // `__wasm_call_ctors` which we synthesize, `__wasm_call_dtors` is defined
  // by libc/etc., because destructors are registered dynamically with
  // `__cxa_atexit` and friends.
  if (!ctx.arg.relocatable && !ctx.arg.shared &&
      !WasmSym::callCtors->isUsedInRegularObj &&
      WasmSym::callCtors->getName() != ctx.arg.entry &&
      !ctx.arg.exportedSymbols.count(WasmSym::callCtors->getName())) {
    if (Symbol *callDtors =
            handleUndefined("__wasm_call_dtors", "<internal>")) {
      if (auto *callDtorsFunc = dyn_cast<DefinedFunction>(callDtors)) {
        if (callDtorsFunc->signature &&
            (!callDtorsFunc->signature->Params.empty() ||
             !callDtorsFunc->signature->Returns.empty())) {
          error("__wasm_call_dtors must have no argument or return values");
        }
        WasmSym::callDtors = callDtorsFunc;
      } else {
        error("__wasm_call_dtors must be a function");
      }
    }
  }

  if (errorCount())
    return;

  // Create wrapped symbols for -wrap option.
  std::vector<WrappedSymbol> wrapped = addWrappedSymbols(args);

  // If any of our inputs are bitcode files, the LTO code generator may create
  // references to certain library functions that might not be explicit in the
  // bitcode file's symbol table. If any of those library functions are defined
  // in a bitcode file in an archive member, we need to arrange to use LTO to
  // compile those archive members by adding them to the link beforehand.
  //
  // We only need to add libcall symbols to the link before LTO if the symbol's
  // definition is in bitcode. Any other required libcall symbols will be added
  // to the link after LTO when we add the LTO object file to the link.
  if (!ctx.bitcodeFiles.empty()) {
    llvm::Triple TT(ctx.bitcodeFiles.front()->obj->getTargetTriple());
    for (auto *s : lto::LTO::getRuntimeLibcallSymbols(TT))
      handleLibcall(s);
  }
  if (errorCount())
    return;

  // We process the stub libraries once beofore LTO to ensure that any possible
  // required exports are preserved by the LTO process.
  processStubLibrariesPreLTO();

  // Do link-time optimization if given files are LLVM bitcode files.
  // This compiles bitcode files into real object files.
  symtab->compileBitcodeFiles();
  if (errorCount())
    return;

  // The LTO process can generate new undefined symbols, specifically libcall
  // functions.  Because those symbols might be declared in a stub library we
  // need the process the stub libraries once again after LTO to handle all
  // undefined symbols, including ones that didn't exist prior to LTO.
  processStubLibraries();

  writeWhyExtract();

  // Bail out if normal linked output is skipped due to LTO.
  if (ctx.arg.thinLTOIndexOnly)
    return;

  createOptionalSymbols();

  // Resolve any variant symbols that were created due to signature
  // mismatchs.
  symtab->handleSymbolVariants();
  if (errorCount())
    return;

  // Apply symbol renames for -wrap.
  if (!wrapped.empty())

  if (!ctx.arg.relocatable && !ctx.isPic) {
    // Add synthetic dummies for weak undefined functions.  Must happen
    // after LTO otherwise functions may not yet have signatures.
    symtab->handleWeakUndefines();
  }

  if (entrySym)
    entrySym->setHidden(false);

  if (errorCount())
    return;

  // Split WASM_SEG_FLAG_STRINGS sections into pieces in preparation for garbage
  // collection.
  splitSections();

  // Any remaining lazy symbols should be demoted to Undefined
  demoteLazySymbols();

  // Do size optimizations: garbage collection
  markLive();

  // Provide the indirect function table if needed.
  WasmSym::indirectFunctionTable =
      symtab->resolveIndirectFunctionTable(/*required =*/false);

  if (errorCount())
    return;

  // Write the result to the file.
  writeResult();
}

} // namespace lld::wasm
