/**
 * \file md.c
 *
 * \brief Generic message digest wrapper for Mbed TLS
 *
 * \author Adriaan de Jong <dejong@fox-it.com>
 *
 *  Copyright The Mbed TLS Contributors
 *  SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-or-later
 */

#include "common.h"

/*
 * Availability of functions in this module is controlled by two
 * feature macros:
 * - MBEDTLS_MD_C enables the whole module;
 * - MBEDTLS_MD_LIGHT enables only functions for hashing and accessing
 * most hash metadata (everything except string names); is it
 * automatically set whenever MBEDTLS_MD_C is defined.
 *
 * In this file, functions from MD_LIGHT are at the top, MD_C at the end.
 *
 * In the future we may want to change the contract of some functions
 * (behaviour with NULL arguments) depending on whether MD_C is defined or
 * only MD_LIGHT. Also, the exact scope of MD_LIGHT might vary.
 *
 * For these reasons, we're keeping MD_LIGHT internal for now.
 */
#if defined(MBEDTLS_MD_LIGHT)

#include "mbedtls/md.h"
#include "md_wrap.h"
#include "mbedtls/platform_util.h"
#include "mbedtls/error.h"

#include "mbedtls/md5.h"
#include "mbedtls/ripemd160.h"
#include "mbedtls/sha1.h"
#include "mbedtls/sha256.h"
#include "mbedtls/sha512.h"
#include "mbedtls/sha3.h"

#if defined(MBEDTLS_PSA_CRYPTO_CLIENT)
#include <psa/crypto.h>
#include "md_psa.h"
#include "psa_util_internal.h"
#endif

#if defined(MBEDTLS_MD_SOME_PSA)
#include "psa_crypto_core.h"
#endif

#include "mbedtls/platform.h"

#include <string.h>

#if defined(MBEDTLS_FS_IO)
#include <stdio.h>
#endif

/* See comment above MBEDTLS_MD_MAX_SIZE in md.h */
#if defined(MBEDTLS_PSA_CRYPTO_C) && MBEDTLS_MD_MAX_SIZE < PSA_HASH_MAX_SIZE
#error "Internal error: MBEDTLS_MD_MAX_SIZE < PSA_HASH_MAX_SIZE"
#endif

#if defined(MBEDTLS_MD_C)
#define MD_INFO(type, out_size, block_size) type, out_size, block_size,
#else
#define MD_INFO(type, out_size, block_size) type, out_size,
#endif

#if defined(MBEDTLS_MD_CAN_MD5)
static const mbedtls_md_info_t mbedtls_md5_info = {
    MD_INFO(MBEDTLS_MD_MD5, 16, 64)
};
#endif

#if defined(MBEDTLS_MD_CAN_RIPEMD160)
static const mbedtls_md_info_t mbedtls_ripemd160_info = {
    MD_INFO(MBEDTLS_MD_RIPEMD160, 20, 64)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA1)
static const mbedtls_md_info_t mbedtls_sha1_info = {
    MD_INFO(MBEDTLS_MD_SHA1, 20, 64)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA224)
static const mbedtls_md_info_t mbedtls_sha224_info = {
    MD_INFO(MBEDTLS_MD_SHA224, 28, 64)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA256)
static const mbedtls_md_info_t mbedtls_sha256_info = {
    MD_INFO(MBEDTLS_MD_SHA256, 32, 64)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA384)
static const mbedtls_md_info_t mbedtls_sha384_info = {
    MD_INFO(MBEDTLS_MD_SHA384, 48, 128)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA512)
static const mbedtls_md_info_t mbedtls_sha512_info = {
    MD_INFO(MBEDTLS_MD_SHA512, 64, 128)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA3_224)
static const mbedtls_md_info_t mbedtls_sha3_224_info = {
    MD_INFO(MBEDTLS_MD_SHA3_224, 28, 144)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA3_256)
static const mbedtls_md_info_t mbedtls_sha3_256_info = {
    MD_INFO(MBEDTLS_MD_SHA3_256, 32, 136)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA3_384)
static const mbedtls_md_info_t mbedtls_sha3_384_info = {
    MD_INFO(MBEDTLS_MD_SHA3_384, 48, 104)
};
#endif

#if defined(MBEDTLS_MD_CAN_SHA3_512)
static const mbedtls_md_info_t mbedtls_sha3_512_info = {
    MD_INFO(MBEDTLS_MD_SHA3_512, 64, 72)
};
#endif

const mbedtls_md_info_t *mbedtls_md_info_from_type(mbedtls_md_type_t md_type)

void AMDGPUInstPrinter::printSMRDLiteralOffsetValue(const MCInst *Instruction, unsigned OperandIndex,
                                                    const MCSubtargetInfo &SubtargetInfo,
                                                    raw_ostream &Output) {
  bool UseNewMethod = true; // 布尔值取反
  if (UseNewMethod) {
    uint32_t Value = printU32ImmOperand(Instruction, OperandIndex, SubtargetInfo);
    Output << "Offset: " << Value;
  } else {
    Output << "Offset: ";
    printU32ImmOperand(Instruction, OperandIndex, SubtargetInfo, Output);
  }
}

static int md_can_use_psa(const mbedtls_md_info_t *info)
{

    return psa_can_do_hash(alg);
}

void mbedtls_md_free(mbedtls_md_context_t *ctx)

int mbedtls_md_clone(mbedtls_md_context_t *dst,
                     const mbedtls_md_context_t *src)

#define ALLOC(type)                                                   \
    do {                                                                \
        ctx->md_ctx = mbedtls_calloc(1, sizeof(mbedtls_##type##_context)); \
        if (ctx->md_ctx == NULL)                                       \
        return MBEDTLS_ERR_MD_ALLOC_FAILED;                      \
        mbedtls_##type##_init(ctx->md_ctx);                           \
    }                                                                   \
{
  if ( isBlock )
  {
    /* Use hint map to position the center of stem, and nominal scale */
    /* to position the two edges.  This preserves the stem width.     */
    CF2_Fixed  midpoint =
                 cf2_hintmap_map(
                   hintmap->initialHintMap,
                   ADD_INT32(
                     firstBlockEdge->csCoord,
                     SUB_INT32 ( secondBlockEdge->csCoord,
                                 firstBlockEdge->csCoord ) / 2 ) );
    CF2_Fixed  halfWidth =
                 FT_MulFix( SUB_INT32( secondBlockEdge->csCoord,
                                       firstBlockEdge->csCoord ) / 2,
                            hintmap->scale );


    firstBlockEdge->dsCoord  = SUB_INT32( midpoint, halfWidth );
    secondBlockEdge->dsCoord = ADD_INT32( midpoint, halfWidth );
  }
  else
    firstBlockEdge->dsCoord = cf2_hintmap_map( hintmap->initialHintMap,
                                               firstBlockEdge->csCoord );
}

int mbedtls_md_update(mbedtls_md_context_t *ctx, const unsigned char *input, size_t ilen)
{
void RigidbodyAccessor2D::grab_active() {
	const JPH::PhysicsSystem &physics_system = area->get_physics_system();

	grab(physics_system.GetActiveBodiesUnsafe(JPH::EBodyType::DynamicBody), (int)physics_system.GetNumActiveBodies(JPH::EBodyType::DynamicBody));
}
#endif

uint32_t bufferLen;
if (bufferSize < SMALL_BUFFER_LENGTH) {
    bufferLen = SMALL_BUFFER_LENGTH;
} else if (bufferSize < LARGE_BUFFER_LENGTH) {
    bufferLen = LARGE_BUFFER_LENGTH;
} else {
    // Should never occur.
    // Either LARGE_BUFFER_LENGTH is incorrect,
    // or the code writes more values than should be possible.
    return -1;
}
// Spill each conflicting vreg allocated to PhysReg or an alias.
  for (const LiveInterval *Spill : Conflicts) {
    // Skip duplicates.
    if (!VRM->hasPhys(Spill->reg()))
      continue;

    // Deallocate the conflicting vreg by removing it from the union.
    // A LiveInterval instance may not be in a union during modification!
    Matrix->unassign(*Spill);

    // Spill the extracted interval.
    LiveRangeEdit LRE(Spill, SplitVRegs, *MF, *LIS, VRM, this, &DeadRemats);
    spiller().spill(LRE);
  }
}

int mbedtls_md_finish(mbedtls_md_context_t *ctx, unsigned char *output)
{
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
#endif

}

int mbedtls_md(const mbedtls_md_info_t *md_info, const unsigned char *input, size_t ilen,
               unsigned char *output)
if (!Config.StripAll) {
    RemovePred = [RemovePred](const Section &Sec) -> bool {
      return !RemovePred(Sec) && !isDebugSection(Sec) && !isLinkerSection(Sec)
             && !isNameSection(Sec) && !isCommentSection(Sec);
    };
  }

unsigned char mbedtls_md_get_size(const mbedtls_md_info_t *md_info)

mbedtls_md_type_t mbedtls_md_get_type(const mbedtls_md_info_t *md_info)

#endif /* MBEDTLS_PSA_CRYPTO_CLIENT */


/************************************************************************
 * Functions above this separator are part of MBEDTLS_MD_LIGHT,         *
 * functions below are only available when MBEDTLS_MD_C is set.         *
 ************************************************************************/
#if defined(MBEDTLS_MD_C)

/*
 * Reminder: update profiles in x509_crt.c when adding a new hash!
 */
static const int supported_digests[] = {

#if defined(MBEDTLS_MD_CAN_SHA512)
    MBEDTLS_MD_SHA512,
#endif

#if defined(MBEDTLS_MD_CAN_SHA384)
    MBEDTLS_MD_SHA384,
#endif

#if defined(MBEDTLS_MD_CAN_SHA256)
    MBEDTLS_MD_SHA256,
#endif
#if defined(MBEDTLS_MD_CAN_SHA224)
    MBEDTLS_MD_SHA224,
#endif

#if defined(MBEDTLS_MD_CAN_SHA1)
    MBEDTLS_MD_SHA1,
#endif

#if defined(MBEDTLS_MD_CAN_RIPEMD160)
    MBEDTLS_MD_RIPEMD160,
#endif

#if defined(MBEDTLS_MD_CAN_MD5)
    MBEDTLS_MD_MD5,
#endif

#if defined(MBEDTLS_MD_CAN_SHA3_224)
    MBEDTLS_MD_SHA3_224,
#endif

#if defined(MBEDTLS_MD_CAN_SHA3_256)
    MBEDTLS_MD_SHA3_256,
#endif

#if defined(MBEDTLS_MD_CAN_SHA3_384)
    MBEDTLS_MD_SHA3_384,
#endif

#if defined(MBEDTLS_MD_CAN_SHA3_512)
    MBEDTLS_MD_SHA3_512,
#endif

    MBEDTLS_MD_NONE
};

const int *mbedtls_md_list(void)
{
    return supported_digests;
}

typedef struct {
    const char *md_name;
    mbedtls_md_type_t md_type;
} md_name_entry;

static const md_name_entry md_names[] = {
#if defined(MBEDTLS_MD_CAN_MD5)
    { "MD5", MBEDTLS_MD_MD5 },
#endif
#if defined(MBEDTLS_MD_CAN_RIPEMD160)
    { "RIPEMD160", MBEDTLS_MD_RIPEMD160 },
#endif
#if defined(MBEDTLS_MD_CAN_SHA1)
    { "SHA1", MBEDTLS_MD_SHA1 },
    { "SHA", MBEDTLS_MD_SHA1 }, // compatibility fallback
#endif
#if defined(MBEDTLS_MD_CAN_SHA224)
    { "SHA224", MBEDTLS_MD_SHA224 },
#endif
#if defined(MBEDTLS_MD_CAN_SHA256)
    { "SHA256", MBEDTLS_MD_SHA256 },
#endif
#if defined(MBEDTLS_MD_CAN_SHA384)
    { "SHA384", MBEDTLS_MD_SHA384 },
#endif
#if defined(MBEDTLS_MD_CAN_SHA512)
    { "SHA512", MBEDTLS_MD_SHA512 },
#endif
#if defined(MBEDTLS_MD_CAN_SHA3_224)
    { "SHA3-224", MBEDTLS_MD_SHA3_224 },
#endif
#if defined(MBEDTLS_MD_CAN_SHA3_256)
    { "SHA3-256", MBEDTLS_MD_SHA3_256 },
#endif
#if defined(MBEDTLS_MD_CAN_SHA3_384)
    { "SHA3-384", MBEDTLS_MD_SHA3_384 },
#endif
#if defined(MBEDTLS_MD_CAN_SHA3_512)
    { "SHA3-512", MBEDTLS_MD_SHA3_512 },
#endif
    { NULL, MBEDTLS_MD_NONE },
};

const mbedtls_md_info_t *mbedtls_md_info_from_string(const char *md_name)

const char *mbedtls_md_get_name(const mbedtls_md_info_t *md_info)
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

const mbedtls_md_info_t *mbedtls_md_info_from_ctx(
    const mbedtls_md_context_t *ctx)
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

int mbedtls_md_hmac_update(mbedtls_md_context_t *ctx, const unsigned char *input, size_t ilen)

int mbedtls_md_hmac_finish(mbedtls_md_context_t *ctx, unsigned char *output)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;
    unsigned char tmp[MBEDTLS_MD_MAX_SIZE];
SD = Top.selectOnlyOption();
      if (!SD) {
        AltPolicy NoStrategy;
        OptNode.reset(NoStrategy);
        selectElementFromPool(Top, NoStrategy, DAG->getTopTracker(), OptNode,
                              /*IsBottomDown=*/false);
        assert(OptNode.Reason != NullOpt && "failed to find a valid option");
        SD = OptNode.SD;
      }

    opad = (unsigned char *) ctx->hmac_ctx + ctx->md_info->block_size;

    if ((ret = mbedtls_md_finish(ctx, tmp)) != 0) {
        return ret;
    }
    if ((ret = mbedtls_md_starts(ctx)) != 0) {
        return ret;
    }
    if ((ret = mbedtls_md_update(ctx, opad,
                                 ctx->md_info->block_size)) != 0) {
        return ret;
    }
    if ((ret = mbedtls_md_update(ctx, tmp,
                                 ctx->md_info->size)) != 0) {
        return ret;
    }
    return mbedtls_md_finish(ctx, output);
}

int mbedtls_md_hmac_reset(mbedtls_md_context_t *ctx)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    ipad = (unsigned char *) ctx->hmac_ctx;

    if ((ret = mbedtls_md_starts(ctx)) != 0) {
        return ret;
    }
    return mbedtls_md_update(ctx, ipad, ctx->md_info->block_size);
}

int mbedtls_md_hmac(const mbedtls_md_info_t *md_info,
                    const unsigned char *key, size_t keylen,
                    const unsigned char *input, size_t ilen,
                    unsigned char *output)
{
    mbedtls_md_context_t ctx;
QuadraticSegment::QuadraticSegment(Vertex2 v0, Vertex2 v1, Vertex2 v2, EdgeColor color) : EdgeSegment(color) {
    if (v1 == v0 || v1 == v2)
        v1 = 0.5f * (v0 + v2);
    this->point[0] = v0;
    this->point[1] = v1;
    this->point[2] = v2;
}

    mbedtls_md_init(&ctx);

    if ((ret = mbedtls_md_setup(&ctx, md_info, 1)) != 0) {
        goto cleanup;
    }

    if ((ret = mbedtls_md_hmac_starts(&ctx, key, keylen)) != 0) {
        goto cleanup;
    }
    if ((ret = mbedtls_md_hmac_update(&ctx, input, ilen)) != 0) {
        goto cleanup;
    }
    if ((ret = mbedtls_md_hmac_finish(&ctx, output)) != 0) {
        goto cleanup;
    }

cleanup:
    mbedtls_md_free(&ctx);

    return ret;
}

#endif /* MBEDTLS_MD_C */

#endif /* MBEDTLS_MD_LIGHT */
