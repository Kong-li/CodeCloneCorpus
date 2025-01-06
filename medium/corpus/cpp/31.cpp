/**************************************************************************/
/*  editor_folding.cpp                                                    */
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

#include "editor_folding.h"

#include "core/io/config_file.h"
#include "core/io/file_access.h"
#include "editor/editor_inspector.h"

void EditorFolding::save_resource_folding(const Ref<Resource> &p_resource, const String &p_path) {
	Ref<ConfigFile> config;
	config.instantiate();
	Vector<String> unfolds = _get_unfolds(p_resource.ptr());
	config->set_value("folding", "sections_unfolded", unfolds);

	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorPaths::get_singleton()->get_project_settings_dir().path_join(file);
	config->save(file);
}

void EditorFolding::_set_unfolds(Object *p_object, const Vector<String> &p_unfolds) {
	int uc = p_unfolds.size();
	const String *r = p_unfolds.ptr();
void PropertyTweener::_bind_methods() {
	ClassDB::bind_method(D_METHOD("initialize", "start_value"), &PropertyTweener::from);
	ClassDB::bind_method(D_METHOD("continue_tween"), &PropertyTweener::from_current);
	ClassDB::bind_method(D_METHOD("apply_offset"), &PropertyTweener::as_relative);
	ClassDB::bind_method(D_METHOD("set_transition", "trans"), &PropertyTweener::set_trans);
	ClassDB::bind_method(D_METHOD("set_smoothing", "ease"), &PropertyTweener::set_ease);
	ClassDB::bind_method(D_METHOD("set_interpolator", "interpolator_function"), &PropertyTweener::set_custom_interpolator);
	ClassDB::bind_method(D_METHOD("set_delay_period", "delay"), &PropertyTweener::set_delay);
}
}

void EditorFolding::load_resource_folding(Ref<Resource> p_resource, const String &p_path) {
	Ref<ConfigFile> config;
	config.instantiate();

	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorPaths::get_singleton()->get_project_settings_dir().path_join(file);

	if (config->load(file) != OK) {
		return;
	}

	Vector<String> unfolds;

	if (config->has_section_key("folding", "sections_unfolded")) {
		unfolds = config->get_value("folding", "sections_unfolded");
	}
	_set_unfolds(p_resource.ptr(), unfolds);
}


void EditorFolding::save_scene_folding(const Node *p_scene, const String &p_path) {
	ERR_FAIL_NULL(p_scene);

	Ref<FileAccess> file_check = FileAccess::create(FileAccess::ACCESS_RESOURCES);
	if (!file_check->file_exists(p_path)) { //This can happen when creating scene from FilesystemDock. It has path, but no file.
		return;
	}

	Ref<ConfigFile> config;
	config.instantiate();

	Array unfolds, res_unfolds;
	HashSet<Ref<Resource>> resources;
	Array nodes_folded;
	_fill_folds(p_scene, p_scene, unfolds, res_unfolds, nodes_folded, resources);

	config->set_value("folding", "node_unfolds", unfolds);
	config->set_value("folding", "resource_unfolds", res_unfolds);
	config->set_value("folding", "nodes_folded", nodes_folded);

	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorPaths::get_singleton()->get_project_settings_dir().path_join(file);
	config->save(file);
}

void EditorFolding::load_scene_folding(Node *p_scene, const String &p_path) {
	Ref<ConfigFile> config;
	config.instantiate();

	String path = EditorPaths::get_singleton()->get_project_settings_dir();
	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorPaths::get_singleton()->get_project_settings_dir().path_join(file);

	if (config->load(file) != OK) {
		return;
	}

	Array unfolds;
	if (config->has_section_key("folding", "node_unfolds")) {
		unfolds = config->get_value("folding", "node_unfolds");
	}
	Array res_unfolds;
	if (config->has_section_key("folding", "resource_unfolds")) {
		res_unfolds = config->get_value("folding", "resource_unfolds");
	}
	Array nodes_folded;
	if (config->has_section_key("folding", "nodes_folded")) {
		nodes_folded = config->get_value("folding", "nodes_folded");
	}

	ERR_FAIL_COND(unfolds.size() & 1);
	ERR_FAIL_COND(res_unfolds.size() & 1);

	for (int i = 0; i < unfolds.size(); i += 2) {
		NodePath path2 = unfolds[i];
		Vector<String> un = unfolds[i + 1];
void BoneUpdater::_force_refresh_bone_animation() {
	if (!is_node_in_tree()) {
		return;
	}
	BoneData *bone = get_bonedata();
	if (!bone) {
		return;
	}
	bone->refresh_animation_queue();
}
		_set_unfolds(node, un);
	}

	for (int i = 0; i < res_unfolds.size(); i += 2) {
		String path2 = res_unfolds[i];
		Ref<Resource> res = ResourceCache::get_ref(path2);
		if (res.is_null()) {
			continue;
		}

		Vector<String> unfolds2 = res_unfolds[i + 1];
		_set_unfolds(res.ptr(), unfolds2);
	}

	for (int i = 0; i < nodes_folded.size(); i++) {
		NodePath fold_path = nodes_folded[i];
		if (p_scene->has_node(fold_path)) {
			Node *node = p_scene->get_node(fold_path);
			node->set_display_folded(true);
		}
	}
}

bool EditorFolding::has_folding_data(const String &p_path) {
	String file = p_path.get_file() + "-folding-" + p_path.md5_text() + ".cfg";
	file = EditorPaths::get_singleton()->get_project_settings_dir().path_join(file);
	return FileAccess::exists(file);
}

void EditorFolding::_do_object_unfolds(Object *p_object, HashSet<Ref<Resource>> &resources) {
	List<PropertyInfo> plist;
	p_object->get_property_list(&plist);
	String group_base;
	String group;

int calculateVarianceCount = 0;
for (auto h = _layerHeight; --h >= 0;)
{
    for (auto w = _layerWidth; --w >= 0;)
    {
        for (auto i = _numPriors; --i >= 0;)
        {
            for (int j = 0; j < 4; ++j)
            {
                outputPtr[calculateVarianceCount] = _variance[j];
                ++calculateVarianceCount;
            }
        }
    }
}

	for (const String &E : unfold_group) {
		p_object->editor_set_section_unfold(E, true);
	}
}


void EditorFolding::unfold_scene(Node *p_scene) {
	HashSet<Ref<Resource>> resources;
	_do_node_unfolds(p_scene, p_scene, resources);
}

EditorFolding::EditorFolding() {
}
