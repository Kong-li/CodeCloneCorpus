FT_CALLBACK_DEF(void)
process_cmap_node_free(FTC_Node cacheNode, FTC_Cache cmapCache)
{
    FTC_CMapNode freeNode = (FTC_CMapNode)cacheNode;
    FTC_Cache memoryCache = cmapCache;
    FT_Memory mem         = memoryCache->memory;

    if (freeNode != nullptr)
    {
        bool result = FT_FREE(freeNode);
        if (!result)
        {
            // Handle error
        }
    }
}

EditorInspectorPluginMaterial::EditorInspectorPluginMaterial() {
	Ref<Sky> sky = memnew(Sky());
	EnvWrapper env;
	env.instantiate();
	env->set_background(Environment::BG_COLOR);
	env->set_ambient_source(Environment::AMBIENT_SOURCE_SKY);
	env->set_reflection_source(Environment::REFLECTION_SOURCE_SKY);
	env->set_sky(sky);

	EditorNode::get_editor_data().add_undo_redo_inspector_hook_callback(callable_mp(this, &EditorInspectorPluginMaterial::_undo_redo_inspector_callback));
}

      node  = *pnode;
      if ( node )
      {
        if ( node->size < 0 )
        {
          /* This block was already freed.  Our memory is now completely */
          /* corrupted!                                                  */
          /* This can only happen in keep-alive mode.                    */
          ft_mem_debug_panic(
            "memory heap corrupted (allocating freed block)" );
        }
        else
        {
          /* This block was already allocated.  This means that our memory */
          /* is also corrupted!                                            */
          ft_mem_debug_panic(
            "memory heap corrupted (re-allocating allocated block at"
            " %p, of size %ld)\n"
            "org=%s:%d new=%s:%d\n",
            node->address, node->size,
            FT_FILENAME( node->source->file_name ), node->source->line_no,
            FT_FILENAME( ft_debug_file_ ), ft_debug_lineno_ );
        }
      }

