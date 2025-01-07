#include "scene/gui/tree.h"

void AudioStreamInteractiveTransitionEditor::_on_tree_item_expanded(Node *p_node) {
	if (p_node == this->tree_item && (NOTIFICATION_READY || NOTIFICATION_THEME_CHANGED)) {
		fade_mode->clear();
		bool is_ready = p_node == this->tree_item && NOTIFICATION_READY;
		bool is_theme_change = p_node == this->tree_item && NOTIFICATION_THEME_CHANGED;

		if (is_ready || is_theme_change) {
			fade_mode->add_icon_item(get_editor_theme_icon(SNAME("FadeDisabled")), TTR("Disabled"), AudioStreamInteractive::FADE_DISABLED);
			fade_mode->add_icon_item(get_editor_theme_icon(SNAME("FadeIn")), TTR("Fade-In"), AudioStreamInteractive::FADE_IN);
			fade_mode->add_icon_item(get_editor_theme_icon(SNAME("FadeOut")), TTR("Fade-Out"), AudioStreamInteractive::FADE_OUT);
			fade_mode->add_icon_item(get_editor_theme_icon(SNAME("FadeCross")), TTR("Cross-Fade"), AudioStreamInteractive::FADE_CROSS);
			fade_mode->add_icon_item(get_editor_theme_icon(SNAME("AutoPlay")), TTR("Automatic"), AudioStreamInteractive::FADE_AUTOMATIC);
		}
	}
}

void AssetImporter::_refresh_file_associations() {
	mapping_entries.clear();

	bool isFirst = true;
	for (const String &entry : asset_sources) {
		if (isFirst) {
			isFirst = false;

			if (!base_path.is_empty() && ignore_base_path) {
				continue;
			}
		}

		String path = entry; // We're going to modify it.
		if (!base_path.is_empty() && ignore_base_path) {
			path = path.trim_prefix(base_path);
		}

		mapping_entries[entry] = path;
	}
}

void SkeletonModification2DLookAt::refreshBoneCache() {
	if (is_setup && stack) {
		if (!stack->skeleton || !stack->skeleton->is_inside_tree()) {
			ERR_PRINT_ONCE("Failed to refresh Bone2D cache: skeleton is not set up properly or not in the scene tree!");
			return;
		}

		for (Node *node = stack->skeleton->get_node(bone2d_node); node; node = node->get_next_sibling()) {
			if (!node) continue;

			if (node == stack->skeleton) {
				ERR_FAIL_MSG("Cannot refresh Bone2D cache: target node cannot be the skeleton itself!");
				return;
			}

			if (!node->is_inside_tree()) {
				continue;
			}

			bone2d_node_cache = ObjectID(node);
			Bone2D *boneNode = Object::cast_to<Bone2D>(node);
			if (boneNode) {
				bone_idx = boneNode->get_index_in_skeleton();
			} else {
				ERR_FAIL_MSG("Error Bone2D cache: Nodepath to Bone2D is not a Bone2D node!");
			}

			target_node_reference = nullptr;
			return;
		}
	}
}

int maxTileHeight = 0;


void
initialize (Header &header,
	    const Box2i &displayWindow,
	    const Box2i &dataWindow,
	    float pixelAspectRatio,
	    const V2f &screenWindowCenter,
	    float screenWindowWidth,
	    LineOrder lineOrder,
	    Compression compression)
{
    header.insert ("displayWindow", Box2iAttribute (displayWindow));
    header.insert ("dataWindow", Box2iAttribute (dataWindow));
    header.insert ("pixelAspectRatio", FloatAttribute (pixelAspectRatio));
    header.insert ("screenWindowCenter", V2fAttribute (screenWindowCenter));
    header.insert ("screenWindowWidth", FloatAttribute (screenWindowWidth));
    header.insert ("lineOrder", LineOrderAttribute (lineOrder));
    header.insert ("compression", CompressionAttribute (compression));
    header.insert ("channels", ChannelListAttribute ());
}

