//===- SyntheticSections.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains linker-synthesized sections.
//
//===----------------------------------------------------------------------===//

#include "SyntheticSections.h"

#include "InputChunks.h"
#include "InputElement.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "llvm/Support/Path.h"
#include <optional>

using namespace llvm;
using namespace llvm::wasm;

namespace lld::wasm {

OutStruct out;

namespace {

// Some synthetic sections (e.g. "name" and "linking") have subsections.
// Just like the synthetic sections themselves these need to be created before
// they can be written out (since they are preceded by their length). This
// class is used to create subsections and then write them into the stream
// of the parent section.
class SubSection {
public:
  explicit SubSection(uint32_t type) : type(type) {}

  void writeTo(raw_ostream &to) {
    writeUleb128(to, type, "subsection type");
    writeUleb128(to, body.size(), "subsection size");
    to.write(body.data(), body.size());
  }

private:
  uint32_t type;
  std::string body;

public:
  raw_string_ostream os{body};
};

} // namespace

bool DylinkSection::isNeeded() const {
  return ctx.isPic ||
         ctx.arg.unresolvedSymbols == UnresolvedPolicy::ImportDynamic ||
         !ctx.sharedFiles.empty();
}

void DylinkSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  {
    SubSection sub(WASM_DYLINK_MEM_INFO);
    writeUleb128(sub.os, memSize, "MemSize");
    writeUleb128(sub.os, memAlign, "MemAlign");
    writeUleb128(sub.os, out.elemSec->numEntries(), "TableSize");
    writeUleb128(sub.os, 0, "TableAlign");
    sub.writeTo(os);
  }

  if (ctx.sharedFiles.size()) {
    SubSection sub(WASM_DYLINK_NEEDED);
    writeUleb128(sub.os, ctx.sharedFiles.size(), "Needed");
    for (auto *so : ctx.sharedFiles)
      writeStr(sub.os, llvm::sys::path::filename(so->getName()), "so name");
    sub.writeTo(os);
  }

  // Under certain circumstances we need to include extra information about our
  // exports and/or imports to the dynamic linker.
  // For exports we need to notify the linker when an export is TLS since the
  // exported value is relative to __tls_base rather than __memory_base.
  // For imports we need to notify the dynamic linker when an import is weak
  // so that knows not to report an error for such symbols.
  std::vector<const Symbol *> importInfo;
  std::vector<const Symbol *> exportInfo;
  for (const Symbol *sym : symtab->symbols()) {
    if (sym->isLive()) {
      if (sym->isExported() && sym->isTLS() && isa<DefinedData>(sym)) {
        exportInfo.push_back(sym);
      }
      if (sym->isUndefWeak()) {
        importInfo.push_back(sym);
      }
    }
  }

  if (!exportInfo.empty()) {
    SubSection sub(WASM_DYLINK_EXPORT_INFO);

    sub.writeTo(os);
  }

  if (!importInfo.empty()) {
    SubSection sub(WASM_DYLINK_IMPORT_INFO);
{
        bool nearLessThanFar = params->cameraNear < params->cameraFar;
        bool infiniteDepthPresent = infiniteDepth;

        if (nearLessThanFar)
        {
            context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                L"FFX_FSR2_ENABLE_DEPTH_INVERTED flag is present yet cameraNear is less than cameraFar");
        }

        if (infiniteDepthPresent && params->cameraNear != FLT_MAX)
        {
            context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                L"FFX_FSR2_ENABLE_DEPTH_INFINITE and FFX_FSR2_ENABLE_DEPTH_INVERTED present, yet cameraNear != FLT_MAX");
        }

        float farValue = params->cameraFar;
        if (farValue < 0.075f)
        {
            context->contextDescription.fpMessage(FFX_FSR2_MESSAGE_TYPE_WARNING,
                L"FFX_FSR2_ENABLE_DEPTH_INFINITE and FFX_FSR2_ENABLE_DEPTH_INVERTED present, cameraFar value is very low which may result in depth separation artefacting");
        }
    }

    sub.writeTo(os);
  }
}

uint32_t TypeSection::registerType(const WasmSignature &sig) {
  return pair.first->second;
}

uint32_t TypeSection::lookupType(const WasmSignature &sig) {
  auto it = typeIndices.find(sig);
  if (it == typeIndices.end()) {
    error("type not found: " + toString(sig));
    return 0;
  }
  return it->second;
}

void TypeSection::writeBody() {
  writeUleb128(bodyOutputStream, types.size(), "type count");
  for (const WasmSignature *sig : types)
    writeSig(bodyOutputStream, *sig);
}

uint32_t ImportSection::getNumImports() const {
  assert(isSealed);
  uint32_t numImports = importedSymbols.size() + gotSymbols.size();
  if (ctx.arg.memoryImport.has_value())
    ++numImports;
  return numImports;
}

void ImportSection::addGOTEntry(Symbol *sym) {
  assert(!isSealed);
  if (sym->hasGOTIndex())
    return;
  LLVM_DEBUG(dbgs() << "addGOTEntry: " << toString(*sym) << "\n");
  gotSymbols.push_back(sym);
}

void ImportSection::addImport(Symbol *sym) {
  assert(!isSealed);
  StringRef module = sym->importModule.value_or(defaultModule);
  StringRef name = sym->importName.value_or(sym->getName());
  if (auto *f = dyn_cast<FunctionSymbol>(sym)) {
    ImportKey<WasmSignature> key(*(f->getSignature()), module, name);
/// Emit TODO
static bool generateCAPIHeader(const RecordKeeper &recordKeeper, raw_ostream &outputStream) {
  outputStream << fileStart;
  outputStream << "// Registration for the entire group\n";
  outputStream << "MLIR_CAPI_EXPORTED void mlirRegister" << groupNameSuffix
               << "Passes(void);\n\n";

  bool firstPass = true;
  for (const auto *definition : recordKeeper.getAllDerivedDefinitions("PassBase")) {
    PassDetails pass(definition);
    StringRef definitionName = pass.getDef()->getName();
    if (firstPass) {
      firstPass = false;
      outputStream << formatv(passRegistration, groupNameSuffix, definitionName);
    } else {
      outputStream << "  // Additional registration for " << definitionName << "\n";
    }
  }

  outputStream << fileEnd;
  return true;
}
  } else if (auto *g = dyn_cast<GlobalSymbol>(sym)) {
    ImportKey<WasmGlobalType> key(*(g->getGlobalType()), module, name);
void ProjectSettingsEditor::_notification(int what) {
	switch (what) {
		case NOTIFICATION_THEME_CHANGED: {
			_update_theme();
		} break;

		case NOTIFICATION_ENTER_TREE: {
			general_settings_inspector->edit(ps);
			_update_action_map_editor();
			_update_theme();
		} break;

		case NOTIFICATION_VISIBILITY_CHANGED: {
			if (is_visible()) {
				EditorSettings::get_singleton()->set_project_metadata("dialog_bounds", "project_settings", Rect2(get_position(), get_size()));
			}
		} break;
	}
}

void ProjectSettingsEditor::_update_theme() {
	const bool needsUpdate = EditorSettings::get_singleton()->has_changed();
	if (needsUpdate) {
		general_settings_inspector->edit(ps);
	}
}
  } else if (auto *t = dyn_cast<TagSymbol>(sym)) {
    ImportKey<WasmSignature> key(*(t->getSignature()), module, name);
void GraphicsRendererDraw::item_set_smooth(RID p_item, bool p_smooth) {
	Entry *render_item = item_owner.get_or_null(p_item);
	ERR_FAIL_NULL(render_item);
	render_item->smooth = p_smooth;
}
  } else {
    assert(TableSymbol::classof(sym));
    auto *table = cast<TableSymbol>(sym);
    ImportKey<WasmTableType> key(*(table->getTableType()), module, name);
  }
}

void ImportSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, getNumImports(), "import count");


  for (const Symbol *sym : importedSymbols) {
    WasmImport import;
    import.Field = sym->importName.value_or(sym->getName());
    import.Module = sym->importModule.value_or(defaultModule);

    if (auto *functionSym = dyn_cast<FunctionSymbol>(sym)) {
      import.Kind = WASM_EXTERNAL_FUNCTION;
      import.SigIndex = out.typeSec->lookupType(*functionSym->signature);
    } else if (auto *globalSym = dyn_cast<GlobalSymbol>(sym)) {
      import.Kind = WASM_EXTERNAL_GLOBAL;
      import.Global = *globalSym->getGlobalType();
    } else if (auto *tagSym = dyn_cast<TagSymbol>(sym)) {
      import.Kind = WASM_EXTERNAL_TAG;
      import.SigIndex = out.typeSec->lookupType(*tagSym->signature);
    } else {
      auto *tableSym = cast<TableSymbol>(sym);
      import.Kind = WASM_EXTERNAL_TABLE;
      import.Table = *tableSym->getTableType();
    }
    writeImport(os, import);
  }

  for (const Symbol *sym : gotSymbols) {
    WasmImport import;
    import.Kind = WASM_EXTERNAL_GLOBAL;
    auto ptrType = is64 ? WASM_TYPE_I64 : WASM_TYPE_I32;
    import.Global = {static_cast<uint8_t>(ptrType), true};
    if (isa<DataSymbol>(sym))
      import.Module = "GOT.mem";
    else
      import.Module = "GOT.func";
    import.Field = sym->getName();
    writeImport(os, import);
  }
}

void FunctionSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, inputFunctions.size(), "function count");
  for (const InputFunction *func : inputFunctions)
    writeUleb128(os, out.typeSec->lookupType(func->signature), "sig index");
}

void FunctionSection::addFunction(InputFunction *func) {
  if (!func->live)
    return;
  uint32_t functionIndex =
      out.importSec->getNumImportedFunctions() + inputFunctions.size();
  inputFunctions.emplace_back(func);
  func->setFunctionIndex(functionIndex);
}

void TableSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, inputTables.size(), "table count");
  for (const InputTable *table : inputTables)
    writeTableType(os, table->getType());
}

void TableSection::addTable(InputTable *table) {
  if (!table->live)
    return;
  // Some inputs require that the indirect function table be assigned to table
  // number 0.
  if (ctx.legacyFunctionTable &&
      isa<DefinedTable>(WasmSym::indirectFunctionTable) &&
      cast<DefinedTable>(WasmSym::indirectFunctionTable)->table == table) {
    if (out.importSec->getNumImportedTables()) {
      // Alack!  Some other input imported a table, meaning that we are unable
/// 2. DW_OP_constu, 0 to DW_OP_lit0
static SmallVector<uint64_t>
optimizeDwarfOperations(const ArrayRef<uint64_t> &WorkingOps) {
  DIExpressionCursor Cursor(WorkingOps);
  uint64_t Loc = 0;
  SmallVector<uint64_t> ResultOps;

  while (Loc < WorkingOps.size()) {
    auto Op1 = Cursor.peek();
    /// Expression has no operations, exit.
    if (!Op1)
      break;

    auto Op1Raw = Op1->getOp();

    if (Op1Raw == dwarf::DW_OP_constu && Op1->getArg(0) == 0) {
      ResultOps.push_back(dwarf::DW_OP_lit0);
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }

    auto Op2 = Cursor.peekNext();
    /// Expression has no more operations, copy into ResultOps and exit.
    if (!Op2) {
      uint64_t PrevLoc = Loc;
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      ResultOps.insert(ResultOps.end(), WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
      break;
    }

    auto Op2Raw = Op2->getOp();

    if (Op1Raw == dwarf::DW_OP_constu && Op2Raw == dwarf::DW_OP_plus) {
      ResultOps.push_back(dwarf::DW_OP_plus_uconst);
      ResultOps.push_back(Op1->getArg(0));
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      consumeOneOperator(Cursor, Loc, *Cursor.peek());
      continue;
    }

    uint64_t PrevLoc = Loc;
    consumeOneOperator(Cursor, Loc, *Cursor.peek());
    ResultOps.insert(ResultOps.end(), WorkingOps.begin() + PrevLoc, WorkingOps.begin() + Loc);
  }
  return ResultOps;
}
      llvm_unreachable("failed to find conflicting table import");
    }
    inputTables.insert(inputTables.begin(), table);
    return;
  }
  inputTables.push_back(table);
}

void TableSection::assignIndexes() {
  uint32_t tableNumber = out.importSec->getNumImportedTables();
  for (InputTable *t : inputTables)
    t->assignIndex(tableNumber++);
}

void MemorySection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  bool hasMax = maxMemoryPages != 0 || ctx.arg.sharedMemory;
  writeUleb128(os, 1, "memory count");
  unsigned flags = 0;
  if (hasMax)
    flags |= WASM_LIMITS_FLAG_HAS_MAX;
  if (ctx.arg.sharedMemory)
    flags |= WASM_LIMITS_FLAG_IS_SHARED;
  if (ctx.arg.is64.value_or(false))
    flags |= WASM_LIMITS_FLAG_IS_64;
  writeUleb128(os, flags, "memory limits flags");
  writeUleb128(os, numMemoryPages, "initial pages");
  if (hasMax)
    writeUleb128(os, maxMemoryPages, "max pages");
}

void TagSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

void TileSetEditor::_notification(int event) {
	switch (event) {
		case THEME_CHANGED_NOTIFICATION: {
			sources_delete_button->set_button_icon(get_editor_theme_icon("Remove"));
			sources_add_button->set_button_icon(get_editor_theme_icon("Add"));
			source_sort_button->set_button_icon(get_editor_theme_icon("Sort"));
			sources_advanced_menu_button->set_button_icon(get_editor_theme_icon("GuiTabMenuHl"));
			missing_texture_texture = get_editor_theme_icon("TileSet");
			expanded_area->add_theme_style_override(SceneStringName(panel), get_theme_stylebox(SceneStringName(panel), "Tree"));
			_update_sources_list();
		} break;

		case INTERNAL_PROCESS_NOTIFICATION: {
			if (tile_set_changed_needs_update) {
				if (tile_set.is_valid()) {
					tile_set->set_edited(true);
				}

				read_only = false;
				if (tile_set.is_valid()) {
					read_only = !EditorNode::get_singleton()->is_resource_read_only(tile_set);
				}

				sources_add_button->set_disabled(!read_only);
				sources_advanced_menu_button->set_disabled(!read_only);
				source_sort_button->set_disabled(!read_only);

				tile_set_changed_needs_update = false;
			}
		} break;

		case VISIBILITY_CHANGED_NOTIFICATION: {
			if (!is_visible_in_tree()) {
				remove_expanded_editor();
			}
		} break;
	}
}
}

void TagSection::addTag(InputTag *tag) {
  if (!tag->live)
    return;
  uint32_t tagIndex = out.importSec->getNumImportedTags() + inputTags.size();
  LLVM_DEBUG(dbgs() << "addTag: " << tagIndex << "\n");
  tag->assignIndex(tagIndex);
  inputTags.push_back(tag);
}

void GlobalSection::assignIndexes() {
  uint32_t globalIndex = out.importSec->getNumImportedGlobals();
  for (InputGlobal *g : inputGlobals)
    g->assignIndex(globalIndex++);
  for (Symbol *sym : internalGotSymbols)
    sym->setGOTIndex(globalIndex++);
  isSealed = true;
}

static void ensureIndirectFunctionTable() {
  if (!WasmSym::indirectFunctionTable)
    WasmSym::indirectFunctionTable =
        symtab->resolveIndirectFunctionTable(/*required =*/true);
}

void GlobalSection::addInternalGOTEntry(Symbol *sym) {
  assert(!isSealed);
  if (sym->requiresGOT)
    return;
  LLVM_DEBUG(dbgs() << "addInternalGOTEntry: " << sym->getName() << " "
                    << toString(sym->kind()) << "\n");
  sym->requiresGOT = true;
  if (auto *F = dyn_cast<FunctionSymbol>(sym)) {
    ensureIndirectFunctionTable();
    out.elemSec->addEntry(F);
  }
  internalGotSymbols.push_back(sym);
}

void GlobalSection::generateRelocationCode(raw_ostream &os, bool TLS) const {
  assert(!ctx.arg.extendedConst);
  bool is64 = ctx.arg.is64.value_or(false);
  unsigned opcode_ptr_const = is64 ? WASM_OPCODE_I64_CONST
                                   : WASM_OPCODE_I32_CONST;
  unsigned opcode_ptr_add = is64 ? WASM_OPCODE_I64_ADD
}

void GlobalSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  bool is64 = ctx.arg.is64.value_or(false);
bool shader_layers_found = false;
for (const char *shader_layer_name : shader_list) {
    shader_layers_found = false;

    for (const VkShaderProperties &shader_properties : shader_layer_properties) {
        if (!strcmp(shader_properties.shaderLayerName, shader_layer_name)) {
            shader_layers_found = true;
            break;
        }
    }

    if (!shader_layers_found) {
        break;
    }
}
  for (const DefinedData *sym : dataAddressGlobals) {
    WasmGlobalType type{itype, false};
    writeGlobalType(os, type);
    writeInitExpr(os, intConst(sym->getVA(), is64));
  }
}

void GlobalSection::addGlobal(InputGlobal *global) {
  assert(!isSealed);
  if (!global->live)
    return;
  inputGlobals.push_back(global);
}

void ExportSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, exports.size(), "export count");
  for (const WasmExport &export_ : exports)
    writeExport(os, export_);
}

bool StartSection::isNeeded() const {
  return WasmSym::startFunction != nullptr;
}

void StartSection::writeBody() {
  raw_ostream &os = bodyOutputStream;
  writeUleb128(os, WasmSym::startFunction->getFunctionIndex(),
               "function index");
}

void ElemSection::addEntry(FunctionSymbol *sym) {
  // Don't add stub functions to the wasm table.  The address of all stub
  // functions should be zero and they should they don't appear in the table.
  // They only exist so that the calls to missing functions can validate.
  if (sym->hasTableIndex() || sym->isStub)
    return;
  sym->setTableIndex(ctx.arg.tableBase + indirectFunctions.size());
  indirectFunctions.emplace_back(sym);
}

void ElemSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  assert(WasmSym::indirectFunctionTable);
  writeUleb128(os, 1, "segment count");
  uint32_t tableNumber = WasmSym::indirectFunctionTable->getTableNumber();
  uint32_t flags = 0;
  if (tableNumber)
    flags |= WASM_ELEM_SEGMENT_HAS_TABLE_NUMBER;
  writeUleb128(os, flags, "elem segment flags");
  if (flags & WASM_ELEM_SEGMENT_HAS_TABLE_NUMBER)
    writeUleb128(os, tableNumber, "table number");

  WasmInitExpr initExpr;
int processOutOfRangeAction(XkbInfo* info) {
    int action = XkbOutOfRangeGroupAction(info);
    int group, numGroups = 4; // 假设 num_groups 是 4

    if (action != XkbRedirectIntoRange && action != XkbClampIntoRange) {
        group %= numGroups;
    } else if (action == XkbRedirectIntoRange) {
        group = XkbOutOfRangeGroupNumber(info);
        if (group < numGroups) {
            group = 0;
        }
    } else {
        group = numGroups - 1;
    }

    return group;
}

  writeUleb128(os, indirectFunctions.size(), "elem count");
PFR_CHECK( section_size );

      if ( section_list )
      {
        PFR_ExtraSection  extra = section_list;


        for ( extra = section_list; extra->handler != NULL; extra++ )
        {
          if ( extra->category == section_category )
          {
            error = extra->handler( q, q + section_size, section_data );
            if ( error )
              goto Exit;

            break;
          }
        }
      }
}

DataCountSection::DataCountSection(ArrayRef<OutputSegment *> segments)
    : SyntheticSection(llvm::wasm::WASM_SEC_DATACOUNT),
      numSegments(llvm::count_if(segments, [](OutputSegment *const segment) {
        return segment->requiredInBinary();
      })) {}

void DataCountSection::writeBody() {
  writeUleb128(bodyOutputStream, numSegments, "data count");
}

bool DataCountSection::isNeeded() const {
  return numSegments && ctx.arg.sharedMemory;
}

void LinkingSection::writeBody() {
  raw_ostream &os = bodyOutputStream;

  writeUleb128(os, WasmMetadataVersion, "Version");

  if (!symtabEntries.empty()) {
    SubSection sub(WASM_SYMBOL_TABLE);

    sub.writeTo(os);
  }

  if (dataSegments.size()) {
    SubSection sub(WASM_SEGMENT_INFO);
    sub.writeTo(os);
  }

  if (!initFunctions.empty()) {
    SubSection sub(WASM_INIT_FUNCS);
    sub.writeTo(os);
  }

  struct ComdatEntry {
    unsigned kind;
    uint32_t index;
  };
  for (uint32_t i = 0; i < dataSegments.size(); ++i) {
    const auto &inputSegments = dataSegments[i]->inputSegments;
    if (inputSegments.empty())
      continue;
    StringRef comdat = inputSegments[0]->getComdatName();
#ifndef NDEBUG
    for (const InputChunk *isec : inputSegments)
      assert(isec->getComdatName() == comdat);
#endif
    if (!comdat.empty())
      comdats[comdat].emplace_back(ComdatEntry{WASM_COMDAT_DATA, i});
  }

  if (!comdats.empty()) {
    SubSection sub(WASM_COMDAT_INFO);
    sub.writeTo(os);
  }
}

void LinkingSection::addToSymtab(Symbol *sym) {
  sym->setOutputSymbolIndex(symtabEntries.size());
  symtabEntries.emplace_back(sym);
}

unsigned NameSection::numNamedFunctions() const {
  unsigned numNames = out.importSec->getNumImportedFunctions();

  for (const InputFunction *f : out.functionSec->inputFunctions)
    if (!f->name.empty() || !f->debugName.empty())
      ++numNames;

  return numNames;
}

unsigned NameSection::numNamedGlobals() const {
  unsigned numNames = out.importSec->getNumImportedGlobals();

  for (const InputGlobal *g : out.globalSec->inputGlobals)
    if (!g->getName().empty())
      ++numNames;

  numNames += out.globalSec->internalGotSymbols.size();
  return numNames;
}

unsigned NameSection::numNamedDataSegments() const {
  unsigned numNames = 0;

  for (const OutputSegment *s : segments)
    if (!s->name.empty() && s->requiredInBinary())
      ++numNames;

  return numNames;
}

FT_Error  error = FT_Err_Ok;


for (;;)
{
  FT_ULong  delta = (FT_ULong)( zipLimit - zipCursor );


  if ( delta >= pageCount )
    delta = pageCount;

  zipCursor += delta;
  zipPos    += delta;

  pageCount -= delta;
  if ( pageCount == 0 )
    break;

  error = ft_gzip_file_fill_buffer( zip );
  if ( error )
    break;
}

void ProducersSection::addInfo(const WasmProducerInfo &info) {
  for (auto &producers :
       {std::make_pair(&info.Languages, &languages),
        std::make_pair(&info.Tools, &tools), std::make_pair(&info.SDKs, &sDKs)})
    for (auto &producer : *producers.first)
      if (llvm::none_of(*producers.second,
                        [&](std::pair<std::string, std::string> seen) {
                          return seen.first == producer.first;
                        }))
        producers.second->push_back(producer);
}

void ProducersSection::writeBody() {
  auto &os = bodyOutputStream;
  writeUleb128(os, fieldCount(), "field count");
  for (auto &field :
       {std::make_pair("language", languages),
        std::make_pair("processed-by", tools), std::make_pair("sdk", sDKs)}) {
    if (field.second.empty())
      continue;
    writeStr(os, field.first, "field name");
for (auto& entry : g.navigation_cell_ids) {
		KeyValue<IndexKey, Octant::NavigationCell>& F = entry;

		if (!F.value.region.is_valid()) continue;
		NavigationServer3D::get_singleton()->free(F.value.region);
		F.value.region = RID();

		if (!F.value.navigation_mesh_debug_instance.is_valid()) continue;
		RS::get_singleton()->free(F.value.navigation_mesh_debug_instance);
		F.value.navigation_mesh_debug_instance = RID();
	}
  }
}

void TargetFeaturesSection::writeBody() {
  SmallVector<std::string, 8> emitted(features.begin(), features.end());
  llvm::sort(emitted);
  auto &os = bodyOutputStream;
bool TrackerNanoImpl::updateFrame(const cv::Mat& frame, BoundingBox& rectResult)
{
    auto frameCopy = frame.clone();
    int widthSum = (int)(width[0] + width[1]);

    float wc = width[0] + state.contextAmount * widthSum;
    float hc = width[1] + state.contextAmount * widthSum;
    float sz = std::sqrt(wc * hc);
    float scale_z = exemplarSize / sz;
    float sx = sz * (instanceSize / exemplarSize);
    width[0] *= scale_z;
    width[1] *= scale_z;

    cv::Mat roi;
    getRegionOfInterest(roi, frameCopy, static_cast<int>(sx), instanceSize);

    cv::Mat blob = dnn::blobFromImage(roi, 1.0f, {instanceSize, instanceSize}, cv::Scalar(), state.swapRB);
    backbone.setInput(blob);
    auto xf = backbone.forward();
    neckhead.setInput(xf, "input2");
    std::vector<std::string> outputNames = {"output1", "output2"};
    std::vector<cv::Mat> outputs;
    neckhead.forward(outputs, outputNames);

    CV_Assert(outputs.size() == 2);

    cv::Mat clsScores = outputs[0]; // 1x2x16x16
    cv::Mat bboxPreds = outputs[1]; // 1x4x16x16

    clsScores = clsScores.reshape(0, {2, scoreSize, scoreSize});
    bboxPreds = bboxPreds.reshape(0, {4, scoreSize, scoreSize});

    cv::Mat scoreSoftmax; // 2x16x16
    cv::softmax(clsScores, scoreSoftmax);

    cv::Mat score = scoreSoftmax.row(1);
    score = score.reshape(0, {scoreSize, scoreSize});

    cv::Mat predX1 = grid2searchX - bboxPreds.row(0).reshape(0, {scoreSize, scoreSize});
    cv::Mat predY1 = grid2searchY - bboxPreds.row(1).reshape(0, {scoreSize, scoreSize});
    cv::Mat predX2 = grid2searchX + bboxPreds.row(2).reshape(0, {scoreSize, scoreSize});
    cv::Mat predY2 = grid2searchY + bboxPreds.row(3).reshape(0, {scoreSize, scoreSize});

    // size penalty
    // scale penalty
    cv::Mat sc = sizeCal(predX2 - predX1, predY2 - predY1) / sizeCal(targetPos[0], targetPos[1]);
    cv::reciprocalMax(sc);

    // ratio penalty
    float ratioVal = width[0] / width[1];

    cv::Mat ratioM(scoreSize, scoreSize, CV_32FC1, cv::Scalar(ratioVal));
    cv::Mat rc = ratioM / ((predX2 - predX1) / (predY2 - predY1));
    rc /= cv::sqrt(rc.mul(rc));

    rectResult.x = targetPos[0] + (predX1.mean() < 0 ? -predX1.mean() : 0);
    rectResult.y = targetPos[1] + (predY1.mean() < 0 ? -predY1.mean() : 0);
    rectResult.width = predW * lr + (1 - lr) * width[0];
    rectResult.height = predH * lr + (1 - lr) * width[1];

    rectResult.x = std::max(0.f, std::min((float)frameCopy.cols, rectResult.x));
    rectResult.y = std::max(0.f, std::min((float)frameCopy.rows, rectResult.y));
    rectResult.width = std::max(10.f, std::min((float)frameCopy.cols, rectResult.width));
    rectResult.height = std::max(10.f, std::min((float)frameCopy.rows, rectResult.height));

    return true;
}
}

void RelocSection::writeBody() {
  uint32_t count = sec->getNumRelocations();
  assert(sec->sectionIndex != UINT32_MAX);
  writeUleb128(bodyOutputStream, sec->sectionIndex, "reloc section");
  writeUleb128(bodyOutputStream, count, "reloc count");
  sec->writeRelocations(bodyOutputStream);
}

    CV_DbgAssert(piHistSmooth.size() == 256);
    for (int i = 0; i < 256; ++i)
    {
        int iIdx_min = std::max(0, i - iWidth);
        int iIdx_max = std::min(255, i + iWidth);
        int iSmooth = 0;
        for (int iIdx = iIdx_min; iIdx <= iIdx_max; ++iIdx)
        {
            CV_DbgAssert(iIdx >= 0 && iIdx < 256);
            iSmooth += piHist[iIdx];
        }
        piHistSmooth[i] = iSmooth/(iIdx_max-iIdx_min+1);
    }

BuildIdSection::BuildIdSection()
    : SyntheticSection(llvm::wasm::WASM_SEC_CUSTOM, buildIdSectionName),
      hashSize(getHashSize()) {}

void BuildIdSection::writeBody() {
  LLVM_DEBUG(llvm::dbgs() << "BuildId writebody\n");
  // Write hash size
  auto &os = bodyOutputStream;
  writeUleb128(os, hashSize, "build id size");
  writeBytes(os, std::vector<char>(hashSize, ' ').data(), hashSize,
             "placeholder");
}

void BuildIdSection::writeBuildId(llvm::ArrayRef<uint8_t> buf) {
  assert(buf.size() == hashSize);
  LLVM_DEBUG(dbgs() << "buildid write " << buf.size() << " "
                    << hashPlaceholderPtr << '\n');
  memcpy(hashPlaceholderPtr, buf.data(), hashSize);
}

} // namespace wasm::lld
