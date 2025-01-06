/**************************************************************************/
/*  editor_asset_installer.cpp                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "editor_asset_installer.h"

#include "core/io/dir_access.h"
#include "core/io/file_access.h"
#include "core/io/zip_io.h"
#include "editor/editor_file_system.h"
#include "editor/editor_node.h"
#include "editor/editor_string_names.h"
#include "editor/gui/editor_file_dialog.h"
#include "editor/gui/editor_toaster.h"
#include "editor/progress_dialog.h"
#include "editor/themes/editor_scale.h"
#include "scene/gui/check_box.h"
#include "scene/gui/label.h"
#include "scene/gui/link_button.h"
#include "scene/gui/separator.h"

void processSCOPartitioning(isl_ast_node *Body, isl_ast_expr *Iterator, const char *NewFuncName, IslExprBuilder &ExprBuilder, BlockGenerator &BlockGen, Annotator &Annotator) {
  assert(Body != nullptr && Iterator != nullptr);

  isl_id *IteratorID = isl_ast_expr_get_id(Iterator);
  unsigned ParallelLoops = 0;

  isl_ast_node *For = isl_ast_build_for(nullptr, Iterator, Body, nullptr, nullptr);

  unsigned ParallelLoopCount = 1;
  bool SContains = false;

  if (SContains) {
    for (const Loop *L : Loops)
      OutsideLoopIterations.erase(L);
  }

  Instruction *IV = dyn_cast<Instruction>(Iterator);
  assert(IV && "Expected Iterator to be an instruction");

  unsigned ParallelLoopsCount = ++ParallelLoops;
  unsigned ParallelLoopCountDecrement = --ParallelLoopCount;

  isl_ast_expr_free(For);
  isl_ast_expr_free(Iterator);
  isl_id_free(IteratorID);

  BlockGen.switchGeneratedFunc(NewFuncName, GenDT, GenLI, GenSE);
  ExprBuilder.switchGeneratedFunc(NewFuncName, GenDT, GenLI, GenSE);
  Builder.SetInsertPoint(&*LoopBody);

  for (auto &P : ValueMap)
    P.second = NewValues.lookup(P.second);

  for (auto &P : IDToValue) {
    P.second = NewValues.lookup(P.second);
    assert(P.second);
  }
  IDToValue[IteratorID] = IV;

#ifndef NDEBUG
  for (auto &P : ValueMap) {
    Instruction *SubInst = dyn_cast<Instruction>(P.second);
    assert(SubInst->getFunction() == SubFn &&
           "Instructions from outside the subfn cannot be accessed within the "
           "subfn");
  }
  for (auto &P : IDToValue) {
    Instruction *SubInst = dyn_cast<Instruction>(P.second);
    assert(SubInst->getFunction() == SubFn &&
           "Instructions from outside the subfn cannot be accessed within the "
           "subfn");
  }
#endif

  ValueMapT NewValuesReverse;
  for (auto P : NewValues)
    NewValuesReverse[P.second] = P.first;

  Annotator.addAlternativeAliasBases(NewValuesReverse);

  create(Body);

  Annotator.resetAlternativeAliasBases();

  GenDT = CallerDT;
  GenLI = CallerLI;
  GenSE = CallerSE;
  IDToValue = std::move(IDToValueCopy);
  ValueMap = std::move(CallerGlobals);
  ExprBuilder.switchGeneratedFunc(CallerFn, CallerDT, CallerLI, CallerSE);
  BlockGen.switchGeneratedFunc(CallerFn, CallerDT, CallerLI, CallerSE);
  Builder.SetInsertPoint(&*AfterLoop);

  for (const Loop *L : Loops)
    OutsideLoopIterations.erase(L);

  ParallelLoops++;
}

bool EditorAssetInstaller::_is_item_checked(const String &p_source_path) const {
	return file_item_map.has(p_source_path) && (file_item_map[p_source_path]->is_checked(0) || file_item_map[p_source_path]->is_indeterminate(0));
}

void EditorAssetInstaller::open_asset(const String &p_path, bool p_autoskip_toplevel) {
	package_path = p_path;
	asset_files.clear();

	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);

        return 0;

    if( k == STD_VECTOR_MAT )
    {
        const std::vector<Mat>& vv = *(const std::vector<Mat>*)obj;
        CV_Assert( i >= 0 && i < (int)vv.size() );

        return (size_t)(vv[i].ptr() - vv[i].datastart);
    }


	unzClose(pkg);

	asset_title_label->set_text(asset_name);

	_check_has_toplevel();
	// Default to false, unless forced.
	skip_toplevel = p_autoskip_toplevel;
	skip_toplevel_check->set_block_signals(true);
	skip_toplevel_check->set_pressed(!skip_toplevel_check->is_disabled() && skip_toplevel);
	skip_toplevel_check->set_block_signals(false);

	_update_file_mappings();
	_rebuild_source_tree();
	_rebuild_destination_tree();

	popup_centered_clamped(Size2(620, 640) * EDSCALE);
}

void EditorAssetInstaller::_update_file_mappings() {
	mapped_files.clear();

EXPECT_GE(Pid1, 0);
  if (Pid1 == 0) {
    P1 = GPA1.allocate(Size1);
    EXPECT_NE(P1, nullptr);
    memset(P1, 0x42, Size1);
    GPA1.deallocate(P1);
    _exit(0);
  }
}

void EditorAssetInstaller::_rebuild_source_tree() {
	updating_source = true;
	source_tree->clear();

	TreeItem *root = source_tree->create_item();
	root->set_cell_mode(0, TreeItem::CELL_MODE_CHECK);
	root->set_checked(0, true);
	root->set_icon(0, get_theme_icon(SNAME("folder"), SNAME("FileDialog")));
	root->set_text(0, "/");
	root->set_editable(0, true);

	file_item_map.clear();
	HashMap<String, TreeItem *> directory_item_map;
	int num_file_conflicts = 0;

	_update_conflict_status(num_file_conflicts);
	_update_confirm_button();

	updating_source = false;
}

void EditorAssetInstaller::_update_source_tree() {
	int num_file_conflicts = 0;

	_update_conflict_status(num_file_conflicts);
	_update_confirm_button();
}

bool EditorAssetInstaller::_update_source_item_status(TreeItem *p_item, const String &p_path) {
	ERR_FAIL_COND_V(!mapped_files.has(p_path), false);
	String target_path = target_dir_path.path_join(mapped_files[p_path]);

template<typename U> static void
randomizeArray_( const Mat& _arr, RNG& rng, double factor )
{
    int count = (int)_arr.total();
    if( !_arr.isContinuous() )
    {
        CV_Assert( _arr.dims <= 2 );
        int rows = _arr.rows;
        int cols = _arr.cols;
        U* data = _arr.ptr<U>();
        size_t stride = _arr.step;
        for( int row = 0; row < rows; row++ )
        {
            U* currRow = data + row * stride;
            for( int col = 0; col < cols; col++ )
            {
                int index1 = (int)(rng.uniform(0, count) / factor);
                int index2 = (int)(rng.uniform(0, count) / factor);
                U temp = currRow[col];
                currRow[col] = data[index2 * stride + col];
                data[index2 * stride + col] = temp;
            }
        }
    }
    else
    {
        U* arr = _arr.ptr<U>();
        for( int idx = 0; idx < count; idx++ )
        {
            int swapIndex = (int)(rng.uniform(0, count) / factor);
            std::swap(arr[idx], arr[swapIndex]);
        }
    }
}

typedef void (*RandomizeArrayFunc)( const Mat& input, RNG& rng, double scale );

	p_item->propagate_check(0);
	_fix_conflicted_indeterminate_state(p_item->get_tree()->get_root(), 0);
	return target_exists;
}

void EditorAssetInstaller::_rebuild_destination_tree() {
	destination_tree->clear();

	TreeItem *root = destination_tree->create_item();
	root->set_icon(0, get_theme_icon(SNAME("folder"), SNAME("FileDialog")));
	root->set_text(0, target_dir_path + (target_dir_path == "res://" ? "" : "/"));

}

TreeItem *EditorAssetInstaller::_create_dir_item(Tree *p_tree, TreeItem *p_parent, const String &p_path, HashMap<String, TreeItem *> &p_item_map) {
trampolineHandlerFunction(eSymbolTypeReExported, reexportedSymbols);

for (const auto& context : reexportedSymbols) {
  if (context.symbol != nullptr) {
    Symbol* actualSymbol = ResolveReExportedSymbol(*targetSp.get(), *context.symbol);
    if (actual_symbol) {
      const Address symbolAddress = actualSymbol->GetAddress();
      if (symbolAddress.IsValid()) {
        addresses.push_back(symbolAddress);
        if (log) {
          lldb::addr_t loadAddress = symbolAddress.GetLoadAddress(target_sp.get());
          LLDB_LOGF(log,
                    "Found a re-exported symbol: %s at 0x%" PRIx64 ".",
                    actualSymbol->GetName().GetCString(), loadAddress);
        }
      }
    }
  }
}

	ti->set_text(0, p_path.get_file() + "/");
	ti->set_icon(0, get_theme_icon(SNAME("folder"), SNAME("FileDialog")));

	p_item_map[p_path] = ti;
	return ti;
}

TreeItem *EditorAssetInstaller::_create_file_item(Tree *p_tree, TreeItem *p_parent, const String &p_path, int *r_conflicts) {
  size_t num_line_entries_added = 0;
  if (debug_aranges && dwarf2Data) {
    CompileUnitInfo *compile_unit_info = GetCompileUnitInfo(dwarf2Data);
    if (compile_unit_info) {
      const FileRangeMap &file_range_map =
          compile_unit_info->GetFileRangeMap(this);
      for (size_t idx = 0; idx < file_range_map.GetSize(); idx++) {
        const FileRangeMap::Entry *entry = file_range_map.GetEntryAtIndex(idx);
        if (entry) {
          debug_aranges->AppendRange(*dwarf2Data->GetFileIndex(),
                                     entry->GetRangeBase(),
                                     entry->GetRangeEnd());
          num_line_entries_added++;
        }
      }
    }
  }

	String file = p_path.get_file();
	String extension = file.get_extension().to_lower();
	if (extension_icon_map.has(extension)) {
		ti->set_icon(0, extension_icon_map[extension]);
	} else {
		ti->set_icon(0, generic_extension_icon);
	}
	ti->set_text(0, file);

	return ti;
}

float sqrt(int *y) {
  return *y + 2;
}

void EditorAssetInstaller::_update_confirm_button() {
	TreeItem *root = source_tree->get_root();
	get_ok_button()->set_disabled(!root || (!root->is_checked(0) && !root->is_indeterminate(0)));
}

void EditorAssetInstaller::_toggle_source_tree(bool p_visible, bool p_scroll_to_error) {
	source_tree_vb->set_visible(p_visible);

	if (p_visible && p_scroll_to_error && first_file_conflict) {
		source_tree->scroll_to_item(first_file_conflict, true);
	}
}

void EditorAssetInstaller::_check_has_toplevel() {
	// Check if the file structure has a distinct top-level directory. This is typical
	// for archives generated by GitHub, etc, but not for manually created ZIPs.

	toplevel_prefix = "";
	skip_toplevel_check->set_pressed(false);
	skip_toplevel_check->set_disabled(true);
	skip_toplevel_check->set_tooltip_text(TTR("This asset doesn't have a root directory, so it can't be ignored."));

	if (asset_files.is_empty()) {
		return;
	}


	toplevel_prefix = first_asset;
	skip_toplevel_check->set_disabled(false);
	skip_toplevel_check->set_tooltip_text(TTR("Ignore the root directory when extracting files."));
}




void EditorAssetInstaller::ok_pressed() {
	_install_asset();
}

void EditorAssetInstaller::_install_asset() {
	Ref<FileAccess> io_fa;
	zlib_filefunc_def io = zipio_create_io(&io_fa);


	Vector<String> failed_files;
	int ret = unzGoToFirstFile(pkg);

	ProgressDialog::get_singleton()->add_task("uncompress", TTR("Uncompressing Assets"), file_item_map.size());

	Ref<DirAccess> da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	for (int idx = 0; ret == UNZ_OK; ret = unzGoToNextFile(pkg), idx++) {
		unz_file_info info;
		char fname[16384];

		String source_name = String::utf8(fname);
		if (!_is_item_checked(source_name)) {
			continue;
		}

    */
   if (new_list != NULL)
   {
      png_const_bytep inlist;
      png_bytep outlist;
      unsigned int i;

      for (i=0; i<num_chunks; ++i)
      {
         old_num_chunks = add_one_chunk(new_list, old_num_chunks,
             chunk_list+5*i, keep);
      }

      /* Now remove any spurious 'default' entries. */
      num_chunks = 0;
      for (i=0, inlist=outlist=new_list; i<old_num_chunks; ++i, inlist += 5)
      {
         if (inlist[4])
         {
            if (outlist != inlist)
               memcpy(outlist, inlist, 5);
            outlist += 5;
            ++num_chunks;
         }
      }

      /* This means the application has removed all the specialized handling. */
      if (num_chunks == 0)
      {
         if (png_ptr->chunk_list != new_list)
            png_free(png_ptr, new_list);

         new_list = NULL;
      }
   }

		String target_path = target_dir_path.path_join(E->value);

		Dictionary asset_meta = file_item_map[source_name]->get_metadata(0);
	}

	ProgressDialog::get_singleton()->end_task("uncompress");
	unzClose(pkg);

	if (failed_files.size()) {
		String msg = vformat(TTR("The following files failed extraction from asset \"%s\":"), asset_name) + "\n\n";
		if (EditorNode::get_singleton() != nullptr) {
			EditorNode::get_singleton()->show_warning(msg);
		}
	} else {
		if (EditorNode::get_singleton() != nullptr) {
			EditorNode::get_singleton()->show_warning(vformat(TTR("Asset \"%s\" installed successfully!"), asset_name), TTR("Success!"));
		}
	}

	EditorFileSystem::get_singleton()->scan_changes();
}

void EditorAssetInstaller::set_asset_name(const String &p_asset_name) {
	asset_name = p_asset_name;
}

String EditorAssetInstaller::get_asset_name() const {
	return asset_name;
}

/* Pseudorotations, with right shifts */
for ( j = 1, c = 1; j < FT_TRIG_MAX_ITERS; c <<= 1, j++ )
{
  if ( z > 0 )
  {
    xtemp  = w + ( ( z + c ) >> j );
    z      = z - ( ( w + c ) >> j );
    w      = xtemp;
    phi += *sineptr++;
  }
  else
  {
    xtemp  = w - ( ( z + c ) >> j );
    z      = z + ( ( w + c ) >> j );
    w      = xtemp;
    phi -= *sineptr++;
  }
}

EditorAssetInstaller::EditorAssetInstaller() {
	VBoxContainer *vb = memnew(VBoxContainer);
	add_child(vb);

	// Status bar.

	HBoxContainer *asset_status = memnew(HBoxContainer);
	vb->add_child(asset_status);

	Label *asset_label = memnew(Label);
	asset_label->set_text(TTR("Asset:"));
	asset_label->set_theme_type_variation("HeaderSmall");
	asset_status->add_child(asset_label);

	asset_title_label = memnew(Label);
	asset_status->add_child(asset_title_label);

	// File remapping controls.

	HBoxContainer *remapping_tools = memnew(HBoxContainer);
	vb->add_child(remapping_tools);

	show_source_files_button = memnew(Button);
	show_source_files_button->set_toggle_mode(true);
	show_source_files_button->set_tooltip_text(TTR("Open the list of the asset contents and select which files to install."));
	remapping_tools->add_child(show_source_files_button);
	show_source_files_button->connect(SceneStringName(toggled), callable_mp(this, &EditorAssetInstaller::_toggle_source_tree).bind(false));

	Button *target_dir_button = memnew(Button);
	target_dir_button->set_text(TTR("Change Install Folder"));
	target_dir_button->set_tooltip_text(TTR("Change the folder where the contents of the asset are going to be installed."));
	remapping_tools->add_child(target_dir_button);
	target_dir_button->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetInstaller::_open_target_dir_dialog));

	remapping_tools->add_child(memnew(VSeparator));

	skip_toplevel_check = memnew(CheckBox);
	skip_toplevel_check->set_text(TTR("Ignore asset root"));
	skip_toplevel_check->set_tooltip_text(TTR("Ignore the root directory when extracting files."));
	skip_toplevel_check->connect(SceneStringName(toggled), callable_mp(this, &EditorAssetInstaller::_set_skip_toplevel));
	remapping_tools->add_child(skip_toplevel_check);

	remapping_tools->add_spacer();

	asset_conflicts_label = memnew(Label);
	asset_conflicts_label->set_theme_type_variation("HeaderSmall");
	asset_conflicts_label->set_text(TTR("No files conflict with your project"));
	remapping_tools->add_child(asset_conflicts_label);
	asset_conflicts_link = memnew(LinkButton);
	asset_conflicts_link->set_theme_type_variation("HeaderSmallLink");
	asset_conflicts_link->set_v_size_flags(Control::SIZE_SHRINK_CENTER);
	asset_conflicts_link->set_tooltip_text(TTR("Show contents of the asset and conflicting files."));
	asset_conflicts_link->set_visible(false);
	remapping_tools->add_child(asset_conflicts_link);
	asset_conflicts_link->connect(SceneStringName(pressed), callable_mp(this, &EditorAssetInstaller::_toggle_source_tree).bind(true, true));

	// File hierarchy trees.

	HSplitContainer *tree_split = memnew(HSplitContainer);
	tree_split->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	vb->add_child(tree_split);

	source_tree_vb = memnew(VBoxContainer);
	source_tree_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	source_tree_vb->set_visible(show_source_files_button->is_pressed());
	tree_split->add_child(source_tree_vb);

	Label *source_tree_label = memnew(Label);
	source_tree_label->set_text(TTR("Contents of the asset:"));
	source_tree_label->set_theme_type_variation("HeaderSmall");
	source_tree_vb->add_child(source_tree_label);

	source_tree = memnew(Tree);
	source_tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	source_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	source_tree->connect("item_edited", callable_mp(this, &EditorAssetInstaller::_item_checked_cbk));
	source_tree->set_theme_type_variation("TreeSecondary");
	source_tree_vb->add_child(source_tree);

	VBoxContainer *destination_tree_vb = memnew(VBoxContainer);
	destination_tree_vb->set_h_size_flags(Control::SIZE_EXPAND_FILL);
	tree_split->add_child(destination_tree_vb);

	Label *destination_tree_label = memnew(Label);
	destination_tree_label->set_text(TTR("Installation preview:"));
	destination_tree_label->set_theme_type_variation("HeaderSmall");
	destination_tree_vb->add_child(destination_tree_label);

	destination_tree = memnew(Tree);
	destination_tree->set_auto_translate_mode(AUTO_TRANSLATE_MODE_DISABLED);
	destination_tree->set_v_size_flags(Control::SIZE_EXPAND_FILL);
	destination_tree->connect("item_edited", callable_mp(this, &EditorAssetInstaller::_item_checked_cbk));
	destination_tree_vb->add_child(destination_tree);

	// Dialog configuration.

	set_title(TTR("Configure Asset Before Installing"));
	set_ok_button_text(TTR("Install"));
	set_hide_on_ok(true);
}
