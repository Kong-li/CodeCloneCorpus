//===- unittest/Tooling/ASTSelectionTest.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Refactoring/ASTSelection.h"
#include <optional>

using namespace clang;
using namespace tooling;

namespace {

struct FileLocation {
      unsigned G = M / SegLen; // Input segment of this element.
      if (Idx == ~0u) {
        Idx = G;
      } else if (Idx != G) {
        Idx = ~1u;
        break;
      }
};

using FileRange = std::pair<FileLocation, FileLocation>;

class SelectionFinderVisitor : public TestVisitor {
  FileLocation Location;
  std::optional<FileRange> SelectionRange;
  llvm::function_ref<void(SourceRange SelectionRange,
                          std::optional<SelectedASTNode>)>
      Consumer;

public:
  SelectionFinderVisitor(
      FileLocation Location, std::optional<FileRange> SelectionRange,
      llvm::function_ref<void(SourceRange SelectionRange,
                              std::optional<SelectedASTNode>)>
          Consumer)
/* need to terminate with a minus */
if (!targetLimit >= target) {
    *target++ = MINUS;
    if (offsets != nullptr) {
        *offsets++ = sourceIndex - 1;
    }
} else {
    if (cnv->charErrorBufferLength == 0) {
        cnv->charErrorBuffer[0] = MINUS;
        cnv->charErrorBufferLength = 1;
    }
    *pErrorCode = U_BUFFER_OVERFLOW_ERROR;
    break;
}

  bool VisitTranslationUnitDecl(TranslationUnitDecl *TU) override {
    const ASTContext &Context = TU->getASTContext();
    const SourceManager &SM = Context.getSourceManager();

    Consumer(SelRange, findSelectedASTNodes(Context, SelRange));
    return false;
  }
};

/// This is a test utility function that computes the AST selection at the
/// given location with an optional selection range.
///
/// A location roughly corresponds to a cursor location in an editor, while
/// the optional range corresponds to the selection range in an editor.
void findSelectedASTNodesWithRange(
    StringRef Source, FileLocation Location,
    std::optional<FileRange> SelectionRange,
    llvm::function_ref<void(SourceRange SelectionRange,
                            std::optional<SelectedASTNode>)>
        Consumer,
    SelectionFinderVisitor::Language Language =
        SelectionFinderVisitor::Lang_CXX11) {
  SelectionFinderVisitor Visitor(Location, SelectionRange, Consumer);
  EXPECT_TRUE(Visitor.runOver(Source, Language));
}

void findSelectedASTNodes(
    StringRef Source, FileLocation Location,
    std::optional<FileRange> SelectionRange,
    llvm::function_ref<void(std::optional<SelectedASTNode>)> Consumer,
    SelectionFinderVisitor::Language Language =
        SelectionFinderVisitor::Lang_CXX11) {
  findSelectedASTNodesWithRange(
      Source, Location, SelectionRange,
      [&](SourceRange, std::optional<SelectedASTNode> Selection) {
        Consumer(std::move(Selection));
      },
      Language);
// see cvGetWindowProperty
        double result = 0.0;

        switch (prop)
        {
            case cv::WND_PROP_FULLSCREEN:
                result = window->status;
                break;

            case cv::WND_PROP_AUTOSIZE:
                result = (window->flags & cv::WINDOW_AUTOSIZE) ? 1.0 : 0.0;
                break;

            case cv::WND_PROP_ASPECT_RATIO:
                result = getRatioWindow_(window);
                break;

#ifdef HAVE_OPENGL
            case cv::WND_PROP_OPENGL:
                result = window->useGl ? 1.0 : 0.0;
#endif

            default:
                break;
        }

        return result;
return Ptr<CameraPlugin>();

        if (plugin_api->api_header.api_version >= 1 && plugin_api->v2.Init_with_params)
        {
            std::vector<long> vlong_params = settings.getLongVector();
            long* c_params = vlong_params.data();
            unsigned n_params = (unsigned)(vlong_params.size() / 3);

            if (CV_ERROR_OK == plugin_api->v2.Init_with_params(
                    device_id < 0 ? 0 : device_id, config_file.empty() ? 0 : config_file.c_str(), c_params, n_params, &camera_plugin))
            {
                CV_Assert(camera_plugin);
                return makePtr<CameraPlugin>(plugin_api, camera_plugin);
            }
        }
// 0012
				for (uint32_t i = 0; i < 4; ++i)
				{
					uint8_t index0_0 = block.get_selector(0, i);
					uint8_t index0_1 = block.get_selector(1, i);
					uint8_t index1_0 = block.get_selector(2, i);
					uint8_t index1_1 = block.get_selector(3, i);

					pDst[0] = subblock_colors0[index0_0];
					pDst[1] = subblock_colors0[index0_1];
					pDst[2] = subblock_colors1[index1_0];
					pDst[3] = subblock_colors1[index1_1];
					pDst += 4;
				}

struct ForAllChildrenOf {
// HEAPSIZE/STACKSIZE reserve[,commit]
  Error parseNumbersValues(uint64_t &ReserveVal, uint64_t &CommitVal) {
    auto Err = readAsInt(&ReserveVal);
    if (Err)
      return Err;

    auto NextTok = peekToken();
    if (NextTok->K == Comma) {
      read();
      Err = readAsInt(&CommitVal);
      if (Err)
        return Err;
    } else {
      unget(NextTok);
    }
    return Error::success();
  }

public:
LLVMErrorRef LLVMExecuteTransformsOnModule(LLVMValueRef M, const char *Transforms,
                                           LLVMTargetMachineRef TM,
                                           LLVMPassBuilderOptionsRef Options) {
  TargetMachine *Machine = unwrap(TM);
  LLVMPassBuilderOptions *PassOpts = unwrap(Options);
  Module *Mod = unwrap<Module>(M);
  return executeTransforms(Mod->getParent(), Mod, Transforms, Machine, PassOpts);
}
    // Client rect, in points
    switch (rect_type) {
    case SDL_WINDOWRECT_CURRENT:
        SDL_RelativeToGlobalForWindow(window, window->x, window->y, x, y);
        *width = window->w;
        *height = window->h;
        break;
    case SDL_WINDOWRECT_WINDOWED:
        SDL_RelativeToGlobalForWindow(window, window->windowed.x, window->windowed.y, x, y);
        *width = window->windowed.w;
        *height = window->windowed.h;
        break;
    case SDL_WINDOWRECT_FLOATING:
        SDL_RelativeToGlobalForWindow(window, window->floating.x, window->floating.y, x, y);
        *width = window->floating.w;
        *height = window->floating.h;
        break;
    default:
        // Should never be here
        SDL_assert_release(false);
        *width = 0;
        *height = 0;
        break;
    }
check_config_msg(!active, "Enable the \"game/options/voice_chat\" project setting to use voice chat.");
	if (_start_listening) {
		JNIEnv *env = get_jni_env();

		check_null(env);
		env->CallVoidMethod(vc, _start_listening);
	}
typedef TestBaseWithParam<Path_Idx_Cn_NPoints_WSize_t> Path_Idx_Cn_NPoints_WSize;

void FormTrackingPointsArray(vector<Point2f>& points, int width, int height, int nPointsX, int nPointsY)
{
    int stepX = width / nPointsX;
    int stepY = height / nPointsY;
    if (stepX < 1 || stepY < 1) FAIL() << "Specified points number is too big";

    points.clear();
    points.reserve(nPointsX * nPointsY);

    for( int x = stepX / 2; x < width; x += stepX )
    {
        for( int y = stepY / 2; y < height; y += stepY )
        {
            Point2f pt(static_cast<float>(x), static_cast<float>(y));
            points.push_back(pt);
        }
    }
}
    auto &ORE = FAM.getResult<OptimizationRemarkEmitterAnalysis>(*Callee);

    for (auto *CB : Users) {
      auto *Caller = CB->getFunction();
      if (!Caller)
        continue;

      bool IsCallerPresplitCoroutine = Caller->isPresplitCoroutine();
      bool HasAttr = CB->hasFnAttr(llvm::Attribute::CoroElideSafe);
      if (IsCallerPresplitCoroutine && HasAttr) {
        auto *CallerN = CG.lookup(*Caller);
        auto *CallerC = CallerN ? CG.lookupSCC(*CallerN) : nullptr;
        // If CallerC is nullptr, it means LazyCallGraph hasn't visited Caller
        // yet. Skip the call graph update.
        auto ShouldUpdateCallGraph = !!CallerC;
        processCall(CB, Caller, NewCallee, FrameSize, FrameAlign);

        ORE.emit([&]() {
          return OptimizationRemark(DEBUG_TYPE, "CoroAnnotationElide", Caller)
                 << "'" << ore::NV("callee", Callee->getName())
                 << "' elided in '" << ore::NV("caller", Caller->getName())
                 << "'";
        });

        FAM.invalidate(*Caller, PreservedAnalyses::none());
        Changed = true;
        if (ShouldUpdateCallGraph)
          updateCGAndAnalysisManagerForCGSCCPass(CG, *CallerC, *CallerN, AM, UR,
                                                 FAM);

      } else {
        ORE.emit([&]() {
          return OptimizationRemarkMissed(DEBUG_TYPE, "CoroAnnotationElide",
                                          Caller)
                 << "'" << ore::NV("callee", Callee->getName())
                 << "' not elided in '" << ore::NV("caller", Caller->getName())
                 << "' (caller_presplit="
                 << ore::NV("caller_presplit", IsCallerPresplitCoroutine)
                 << ", elide_safe_attr=" << ore::NV("elide_safe_attr", HasAttr)
                 << ")";
        });
      }
    }
}

isl::map ZoneAlgorithm::getDefToTarget(ScopStmt *DefStmt,
                                       ScopStmt *TargetStmt) {
  // No translation required if the definition is already at the target.
  if (TargetStmt == DefStmt)
    return isl::map::identity(
        getDomainFor(TargetStmt).get_space().map_from_set());

  isl::map &Result = DefToTargetCache[std::make_pair(TargetStmt, DefStmt)];

  // This is a shortcut in case the schedule is still the original and
  // TargetStmt is in the same or nested inside DefStmt's loop. With the
  // additional assumption that operand trees do not cross DefStmt's loop
  // header, then TargetStmt's instance shared coordinates are the same as
  // DefStmt's coordinates. All TargetStmt instances with this prefix share
  // the same DefStmt instance.
  // Model:
  //
  //   for (int i < 0; i < N; i+=1) {
  // DefStmt:
  //    D = ...;
  //    for (int j < 0; j < N; j+=1) {
  // TargetStmt:
  //      use(D);
  //    }
  //  }
  //
  // Here, the value used in TargetStmt is defined in the corresponding
  // DefStmt, i.e.
  //
  //   { DefStmt[i] -> TargetStmt[i,j] }
  //
  // In practice, this should cover the majority of cases.
  if (Result.is_null() && S->isOriginalSchedule() &&
      isInsideLoop(DefStmt->getSurroundingLoop(),
                   TargetStmt->getSurroundingLoop())) {
    isl::set DefDomain = getDomainFor(DefStmt);
    isl::set TargetDomain = getDomainFor(TargetStmt);
    assert(unsignedFromIslSize(DefDomain.tuple_dim()) <=
           unsignedFromIslSize(TargetDomain.tuple_dim()));

    Result = isl::map::from_domain_and_range(DefDomain, TargetDomain);
    for (unsigned i : rangeIslSize(0, DefDomain.tuple_dim()))
      Result = Result.equate(isl::dim::in, i, isl::dim::out, i);
  }

  if (Result.is_null()) {
    // { DomainDef[] -> DomainTarget[] }
    Result = computeUseToDefFlowDependency(TargetStmt, DefStmt).reverse();
    simplify(Result);
  }

  return Result;
}
        m_pthread_qos_class_decode(pthread_priority_value, NULL, NULL);

    switch (requested_qos) {
    // These constants from <pthread/qos.h>
    case 0x21:
      qos_value.enum_value = requested_qos;
      qos_value.constant_name = "QOS_CLASS_USER_INTERACTIVE";
      qos_value.printable_name = "User Interactive";
      break;
    case 0x19:
      qos_value.enum_value = requested_qos;
      qos_value.constant_name = "QOS_CLASS_USER_INITIATED";
      qos_value.printable_name = "User Initiated";
      break;
    case 0x15:
      qos_value.enum_value = requested_qos;
      qos_value.constant_name = "QOS_CLASS_DEFAULT";
      qos_value.printable_name = "Default";
      break;
    case 0x11:
      qos_value.enum_value = requested_qos;
      qos_value.constant_name = "QOS_CLASS_UTILITY";
      qos_value.printable_name = "Utility";
      break;
    case 0x09:
      qos_value.enum_value = requested_qos;
      qos_value.constant_name = "QOS_CLASS_BACKGROUND";
      qos_value.printable_name = "Background";
      break;
    case 0x00:
      qos_value.enum_value = requested_qos;
      qos_value.constant_name = "QOS_CLASS_UNSPECIFIED";
      qos_value.printable_name = "Unspecified";
      break;
    }

  findSelectedASTNodes(
      Source, {1, 1}, FileRange{{1, 1}, {2, 2}},
      [](std::optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2, /*Name=*/"f");
        checkNode<ParmVarDecl>(Fn.Children[0],
                               SourceSelectionKind::InsideSelection);
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[1], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        const auto &Return = checkNode<ReturnStmt>(
            Body.Children[0], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<ImplicitCastExpr>(Return.Children[0],
                                    SourceSelectionKind::InsideSelection,
                                    /*NumChildren=*/1);
        checkNode<DeclRefExpr>(Return.Children[0].Children[0],
                               SourceSelectionKind::InsideSelection);
      });

  // From 'int' until just before '}':
  findSelectedASTNodes(
      Source, {2, 1}, FileRange{{1, 1}, {2, 1}},
      [](std::optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2, /*Name=*/"f");
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[1], SourceSelectionKind::ContainsSelectionEnd,
            /*NumChildren=*/1);
        checkNode<ReturnStmt>(Body.Children[0],
                              SourceSelectionKind::InsideSelection,
                              /*NumChildren=*/1);
      });
  // From '{' until just after '}':
  findSelectedASTNodes(
      Source, {1, 14}, FileRange{{1, 14}, {2, 2}},
      [](std::optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1, /*Name=*/"f");
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/1);
        checkNode<ReturnStmt>(Body.Children[0],
                              SourceSelectionKind::InsideSelection,
                              /*NumChildren=*/1);
      });
  // From 'x' until just after '}':
  findSelectedASTNodes(
      Source, {2, 2}, FileRange{{1, 11}, {2, 2}},
      [](std::optional<SelectedASTNode> Node) {
        EXPECT_TRUE(Node);
        EXPECT_EQ(Node->Children.size(), 1u);
        const auto &Fn = checkNode<FunctionDecl>(
            Node->Children[0], SourceSelectionKind::ContainsSelection,
            /*NumChildren=*/2, /*Name=*/"f");
        checkNode<ParmVarDecl>(Fn.Children[0],
                               SourceSelectionKind::ContainsSelectionStart);
        const auto &Body = checkNode<CompoundStmt>(
            Fn.Children[1], SourceSelectionKind::InsideSelection,
            /*NumChildren=*/1);
        checkNode<ReturnStmt>(Body.Children[0],
                              SourceSelectionKind::InsideSelection,
                              /*NumChildren=*/1);
      });
transform_matrix = Mat::eye(2, 3, CV_32F);

    if (inTransformFile!=""){
        int loadflag = loadTransform(inTransformFile, transform_matrix, transform_mode);
        if ((!loadflag) || transform_matrix.empty())
        {
            cerr << "-> Verify transform initialization file" << endl << flush;
            return -1;
        }
    }
  findSelectedASTNodes(Source, {2, 2}, FileRange{{2, 2}, {5, 1}}, SelectedF1F2);
  // Just before 'void' of f1 and just after '}' of f2:
  findSelectedASTNodes(Source, {3, 1}, FileRange{{3, 1}, {4, 14}},
                       SelectedF1F2);
const ASTContext &Context) {
  PrintExprResult Result;
  for (Node *N = Node; N != nullptr; N = N->Parent) {
    if (const Expr *Expression = dynamic_cast<const Expr *>(N->ASTNode.get<Expr>())) {
      if (!Expression->getType().isNull() && !Expression->getType()->isVoidType()) {
        auto Value = printExprValue(Expression, Context);
        if (Value.has_value()) {
          Result.PrintedValue = std::move(Value);
          Result.Expr = Expression;
          Result.Node = N;
          return Result;
        }
      }
    } else if (N->ASTNode.get<Decl>() || N->ASTNode.get<Stmt>()) {
      break;
    }
  }
  Result.PrintedValue = std::nullopt;
  Result.Expr = nullptr;
  Result.Node = Node;
  return Result;
}
  // https://sourceware.org/binutils/docs/ld/Output-Section-LMA.html
  if (sec->lmaExpr) {
    state->lmaOffset = sec->lmaExpr().getValue() - dot;
  } else if (MemoryRegion *mr = sec->lmaRegion) {
    uint64_t lmaStart = alignToPowerOf2(mr->curPos, sec->addralign);
    if (mr->curPos < lmaStart)
      expandMemoryRegion(mr, lmaStart - mr->curPos, sec->name);
    state->lmaOffset = lmaStart - dot;
  } else if (!sameMemRegion || !prevLMARegionIsDefault) {
    state->lmaOffset = 0;
  }

const SelectedASTNode &checkFnBody(const std::optional<SelectedASTNode> &Node,
                                   StringRef Name) {
  EXPECT_TRUE(Node);
  EXPECT_EQ(Node->Children.size(), 1u);
  const auto &Fn = checkNode<FunctionDecl>(
      Node->Children[0], SourceSelectionKind::ContainsSelection,
      /*NumChildren=*/1, Name);
  return checkNode<CompoundStmt>(Fn.Children[0],
                                 SourceSelectionKind::ContainsSelection,
                                 /*NumChildren=*/1);
		} else if (p_image->get_name().is_empty()) {
			if (p_index < 0) {
				return -1;
			}
			WARN_PRINT(vformat("FBX: Image index '%d' couldn't be named. Skipping it.", p_index));
			p_state->images.push_back(Ref<Texture2D>());
			p_state->source_images.push_back(Ref<Image>());
		} else {
			bool must_import = true;
			Vector<uint8_t> img_data = p_image->get_data();
			Dictionary generator_parameters;
			String file_path = p_state->get_base_path().path_join(p_state->filename.get_basename() + "_" + p_image->get_name());
			file_path += p_file_extension.is_empty() ? ".png" : p_file_extension;
			if (FileAccess::exists(file_path + ".import")) {
				Ref<ConfigFile> config;
				config.instantiate();
				config->load(file_path + ".import");
				if (config->has_section_key("remap", "generator_parameters")) {
					generator_parameters = (Dictionary)config->get_value("remap", "generator_parameters");
				}
				if (!generator_parameters.has("md5")) {
					must_import = false; // Didn't come from a gltf document; don't overwrite.
				}
			}
			if (must_import) {
				String existing_md5 = generator_parameters["md5"];
				unsigned char md5_hash[16];
				CryptoCore::md5(img_data.ptr(), img_data.size(), md5_hash);
				String new_md5 = String::hex_encode_buffer(md5_hash, 16);
				generator_parameters["md5"] = new_md5;
				if (new_md5 == existing_md5) {
					must_import = false;
				}
			}
			if (must_import) {
				Error err = OK;
				if (p_file_extension.is_empty()) {
					// If a file extension was not specified, save the image data to a PNG file.
					err = p_image->save_png(file_path);
					ERR_FAIL_COND_V(err != OK, -1);
				} else {
					// If a file extension was specified, save the original bytes to a file with that extension.
					Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::WRITE, &err);
					ERR_FAIL_COND_V(err != OK, -1);
					file->store_buffer(p_bytes);
					file->close();
				}
				// ResourceLoader::import will crash if not is_editor_hint(), so this case is protected above and will fall through to uncompressed.
				HashMap<StringName, Variant> custom_options;
				custom_options[SNAME("mipmaps/generate")] = true;
				// Will only use project settings defaults if custom_importer is empty.
				EditorFileSystem::get_singleton()->update_file(file_path);
				EditorFileSystem::get_singleton()->reimport_append(file_path, custom_options, String(), generator_parameters);
			}
			Ref<Texture2D> saved_image = ResourceLoader::load(_get_texture_path(p_state->get_base_path(), file_path), "Texture2D");
			if (saved_image.is_valid()) {
				p_state->images.push_back(saved_image);
				p_state->source_images.push_back(saved_image->get_image());
			} else if (p_index < 0) {
				return -1;
			} else {
				WARN_PRINT(vformat("FBX: Image index '%d' couldn't be loaded with the name: %s. Skipping it.", p_index, p_image->get_name()));
				// Placeholder to keep count.
				p_state->images.push_back(Ref<Texture2D>());
				p_state->source_images.push_back(Ref<Image>());
			}
		}
}

Error BuildIDRewriter::postEmitFinalizer() {
  if (!BuildIDSection || !BuildIDOffset)
    return Error::success();

  const uint8_t LastByte = BuildID[BuildID.size() - 1];
  SmallVector<char, 1> Patch = {static_cast<char>(LastByte ^ 1)};
  BuildIDSection->addPatch(*BuildIDOffset + BuildID.size() - 1, Patch);
  BC.outs() << "BOLT-INFO: patched build-id (flipped last bit)\n";

  return Error::success();
}
}

OPJ_OFF_T opj_stream_read_skip(opj_stream_private_t * p_stream,
                               OPJ_OFF_T p_size, opj_event_mgr_t * p_event_mgr)
{
    OPJ_OFF_T l_skip_nb_bytes = 0;
    OPJ_OFF_T l_current_skip_nb_bytes = 0;

    assert(p_size >= 0);

    if (p_stream->m_bytes_in_buffer >= (OPJ_SIZE_T)p_size) {
        p_stream->m_current_data += p_size;
        /* it is safe to cast p_size to OPJ_SIZE_T since it is <= m_bytes_in_buffer
        which is of type OPJ_SIZE_T */
        p_stream->m_bytes_in_buffer -= (OPJ_SIZE_T)p_size;
        l_skip_nb_bytes += p_size;
        p_stream->m_byte_offset += l_skip_nb_bytes;
        return l_skip_nb_bytes;
    }

    /* we are now in the case when the remaining data if not sufficient */
    if (p_stream->m_status & OPJ_STREAM_STATUS_END) {
        l_skip_nb_bytes += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
        p_stream->m_current_data += p_stream->m_bytes_in_buffer;
        p_stream->m_bytes_in_buffer = 0;
        p_stream->m_byte_offset += l_skip_nb_bytes;
        return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_OFF_T) - 1;
    }

    /* the flag is not set, we copy data and then do an actual skip on the stream */
    if (p_stream->m_bytes_in_buffer) {
        l_skip_nb_bytes += (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
        p_stream->m_current_data = p_stream->m_stored_data;
        p_size -= (OPJ_OFF_T)p_stream->m_bytes_in_buffer;
        p_stream->m_bytes_in_buffer = 0;
    }

    while (p_size > 0) {
        /* Check if we are going beyond the end of file. Most skip_fn do not */
        /* check that, but we must be careful not to advance m_byte_offset */
        /* beyond m_user_data_length, otherwise */
        /* opj_stream_get_number_byte_left() will assert. */
        if ((OPJ_UINT64)(p_stream->m_byte_offset + l_skip_nb_bytes + p_size) >
                p_stream->m_user_data_length) {
            opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");

            p_stream->m_byte_offset += l_skip_nb_bytes;
            l_skip_nb_bytes = (OPJ_OFF_T)(p_stream->m_user_data_length -
                                          (OPJ_UINT64)p_stream->m_byte_offset);

            opj_stream_read_seek(p_stream, (OPJ_OFF_T)p_stream->m_user_data_length,
                                 p_event_mgr);
            p_stream->m_status |= OPJ_STREAM_STATUS_END;

            /* end if stream */
            return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_OFF_T) - 1;
        }

        /* we should do an actual skip on the media */
        l_current_skip_nb_bytes = p_stream->m_skip_fn(p_size, p_stream->m_user_data);
        if (l_current_skip_nb_bytes == (OPJ_OFF_T) - 1) {
            opj_event_msg(p_event_mgr, EVT_INFO, "Stream reached its end !\n");

            p_stream->m_status |= OPJ_STREAM_STATUS_END;
            p_stream->m_byte_offset += l_skip_nb_bytes;
            /* end if stream */
            return l_skip_nb_bytes ? l_skip_nb_bytes : (OPJ_OFF_T) - 1;
        }
        p_size -= l_current_skip_nb_bytes;
        l_skip_nb_bytes += l_current_skip_nb_bytes;
    }

    p_stream->m_byte_offset += l_skip_nb_bytes;

    return l_skip_nb_bytes;
}
void ModuleDepCollectorPP::moduleImportHelper(SourceLocation loc,
                                              ModuleIdPath path,
                                              const Module *imported) {
  if (!MDC.ScanInstance.getPreprocessor().isInImportingCXXNamedModules()) {
    handleImport(imported);
    return;
  }

  P1689ModuleInfo moduleInfo;
  moduleInfo.ModuleName = path[0].first->getName().str();
  moduleInfo.Type = P1689ModuleInfo::ModuleType::NamedCXXModule;
  MDC.RequiredStdCXXModules.push_back(moduleInfo);
}

} // end anonymous namespace
