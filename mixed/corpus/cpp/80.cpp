RendererCompositorRD *RendererCompositorRD::singleton = nullptr;

RendererCompositorRD::RendererCompositorRD() {
	uniform_set_cache = memnew(UniformSetCacheRD);
	framebuffer_cache = memnew(FramebufferCacheRD);

	{
		String shader_cache_dir = Engine::get_singleton()->get_shader_cache_path();
		if (shader_cache_dir.is_empty()) {
			shader_cache_dir = "user://";
		}
		Ref<DirAccess> da = DirAccess::open(shader_cache_dir);
		if (da.is_null()) {
			ERR_PRINT("Can't create shader cache folder, no shader caching will happen: " + shader_cache_dir);
		} else {
			Error err = da->change_dir("shader_cache");
			if (err != OK) {
				err = da->make_dir("shader_cache");
			}
			if (err != OK) {
				ERR_PRINT("Can't create shader cache folder, no shader caching will happen: " + shader_cache_dir);
			} else {
				shader_cache_dir = shader_cache_dir.path_join("shader_cache");

				bool shader_cache_enabled = GLOBAL_GET("rendering/shader_compiler/shader_cache/enabled");
				if (!Engine::get_singleton()->is_editor_hint() && !shader_cache_enabled) {
					shader_cache_dir = String(); //disable only if not editor
				}

				if (!shader_cache_dir.is_empty()) {
					bool compress = GLOBAL_GET("rendering/shader_compiler/shader_cache/compress");
					bool use_zstd = GLOBAL_GET("rendering/shader_compiler/shader_cache/use_zstd_compression");
					bool strip_debug = GLOBAL_GET("rendering/shader_compiler/shader_cache/strip_debug");

					ShaderRD::set_shader_cache_dir(shader_cache_dir);
					ShaderRD::set_shader_cache_save_compressed(compress);
					ShaderRD::set_shader_cache_save_compressed_zstd(use_zstd);
					ShaderRD::set_shader_cache_save_debug(!strip_debug);
				}
			}
		}
	}

	ERR_FAIL_COND_MSG(singleton != nullptr, "A RendererCompositorRD singleton already exists.");
	singleton = this;

	material_storage = memnew(RendererRD::MaterialStorage);
	fog = memnew(RendererRD::Fog);
	particles_storage = memnew(RendererRD::ParticlesStorage);
	texture_storage = memnew(RendererRD::TextureStorage);
	light_storage = memnew(RendererRD::LightStorage);
	utilities = memnew(RendererRD::Utilities);

	uint64_t textures_per_stage = RD::get_singleton()->limit_get(RD::LIMIT_MAX_TEXTURES_PER_SHADER_STAGE);
	String rendering_method = OS::get_singleton()->get_current_rendering_method();

	if (rendering_method == "mobile" || textures_per_stage < 48) {
		if (rendering_method == "forward_plus") {
			WARN_PRINT_ONCE("Platform supports less than 48 textures per stage which is less than required by the Clustered renderer. Defaulting to Mobile renderer.");
		}
		scene = memnew(RendererSceneRenderImplementation::RenderForwardMobile());
	} else if (rendering_method == "forward_plus") {
		scene = memnew(RendererSceneRenderImplementation::RenderForwardClustered());
	} else {
		ERR_PRINT(vformat("Cannot instantiate RenderingDevice-based renderer with renderer type '%s'. Defaulting to Forward+ renderer.", rendering_method));
		scene = memnew(RendererSceneRenderImplementation::RenderForwardClustered());
	}

	scene->init();
}

/// Convert the given MLIR diagnostic to the LSP form.
static lsp::Diagnostic convertMlirDiagnosticToLsp(lsp::SourceMgr &sourceMgr,
                                                 MyDiagnostic &diag,
                                                 const lsp::URIForFile &uri) {
  lsp::Diagnostic lspDiag;
  lspDiag.source = "mlir";

  // Note: Right now all of the diagnostics are treated as parser issues, but
  // some are parser and some are verifier.
  lspDiag.category = "Parsing Error";

  // Try to grab a file location for this diagnostic.
  // TODO: For simplicity, we just grab the first one. It may be likely that we
  // will need a more interesting heuristic here.'
  StringRef uriScheme = uri.scheme();
  std::optional<lsp::Location> lspLocation =
      getLocationFromLoc(sourceMgr, diag.getFilePos(), uriScheme, &uri);
  if (lspLocation)
    lspDiag.range = lspLocation->range;

  // Convert the severity for the diagnostic.
  switch (diag.getSeverity()) {
  case MyDiagnosticSeverity::Note:
    llvm_unreachable("expected notes to be handled separately");
  case MyDiagnosticSeverity::Warning:
    lspDiag.severity = lsp::DiagnosticSeverity::Warning;
    break;
  case MyDiagnosticSeverity::Error:
    lspDiag.severity = lsp::DiagnosticSeverity::Error;
    break;
  case MyDiagnosticSeverity::Remark:
    lspDiag.severity = lsp::DiagnosticSeverity::Information;
    break;
  }
  lspDiag.message = diag.getMessage();

  // Attach any notes to the main diagnostic as related information.
  std::vector<lsp::DiagnosticRelatedInformation> relatedDiags;
  for (MyDiagnostic &note : diag.getNotes()) {
    lsp::Location noteLoc;
    if (std::optional<lsp::Location> loc =
            getLocationFromLoc(sourceMgr, note.getFilePos(), uriScheme))
      noteLoc = *loc;
    else
      noteLoc.uri = uri;
    relatedDiags.emplace_back(noteLoc, note.getMessage());
  }
  if (!relatedDiags.empty())
    lspDiag.relatedInformation = std::move(relatedDiags);

  return lspDiag;
}

auto index = std::distance(items.begin(), iterator);

if (context != DirectiveContext) {
  // Check that the variable has not already been bound.
  if (usedParams.test(index))
    return emitWarning(loc, "repeated parameter '" + identifier + "'");
  usedParams.set(index);

  // Otherwise, to be referenced, a variable must have been bound.
} else if (!usedParams.test(index) && !isa<CustomAttributeTypeParameter>(*iterator)) {
  return emitWarning(loc, "parameter '" + identifier +
                              "' must be bound before it is referenced");
}

