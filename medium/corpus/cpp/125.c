/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggVorbis SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE OggVorbis SOURCE CODE IS (C) COPYRIGHT 1994-2015             *
 * by the Xiph.Org Foundation https://xiph.org/                     *
 *                                                                  *
 ********************************************************************

 function: simple programmatic interface for encoder mode setup

 ********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vorbis/codec.h"
#include "vorbis/vorbisenc.h"

#include "codec_internal.h"

#include "os.h"
#include "misc.h"

/* careful with this; it's using static array sizing to make managing
   all the modes a little less annoying.  If we use a residue backend
   with > 12 partition types, or a different division of iteration,
   this needs to be updated. */
typedef struct {
  const static_codebook *books[12][4];
} static_bookblock;

typedef struct {
  int res_type;
  int limit_type; /* 0 lowpass limited, 1 point stereo limited */
  int grouping;
  const vorbis_info_residue0 *res;
  const static_codebook  *book_aux;
  const static_codebook  *book_aux_managed;
  const static_bookblock *books_base;
  const static_bookblock *books_base_managed;
} vorbis_residue_template;

typedef struct {
  const vorbis_info_mapping0    *map;
  const vorbis_residue_template *res;
} vorbis_mapping_template;

typedef struct vp_adjblock{
  int block[P_BANDS];
} vp_adjblock;

typedef struct {
  int data[NOISE_COMPAND_LEVELS];
} compandblock;

/* high level configuration information for setting things up
   step-by-step with the detailed vorbis_encode_ctl interface.
   There's a fair amount of redundancy such that interactive setup
   does not directly deal with any vorbis_info or codec_setup_info
   initialization; it's all stored (until full init) in this highlevel
   setup, then flushed out to the real codec setup structs later. */

typedef struct {
  int att[P_NOISECURVES];
  float boost;
  float decay;
} att3;
typedef struct { int data[P_NOISECURVES]; } adj3;

typedef struct {
  int   pre[PACKETBLOBS];
  int   post[PACKETBLOBS];
  float kHz[PACKETBLOBS];
  float lowpasskHz[PACKETBLOBS];
} adj_stereo;

typedef struct {
  int lo;
  int hi;
  int fixed;
} noiseguard;
typedef struct {
  int data[P_NOISECURVES][17];
} noise3;

typedef struct {
  int      mappings;
  const double  *rate_mapping;
  const double  *quality_mapping;
  int      coupling_restriction;
  long     samplerate_min_restriction;
  long     samplerate_max_restriction;


  const int     *blocksize_short;
  const int     *blocksize_long;

  const att3    *psy_tone_masteratt;
  const int     *psy_tone_0dB;
  const int     *psy_tone_dBsuppress;

  const vp_adjblock *psy_tone_adj_impulse;
  const vp_adjblock *psy_tone_adj_long;
  const vp_adjblock *psy_tone_adj_other;

  const noiseguard  *psy_noiseguards;
  const noise3      *psy_noise_bias_impulse;
  const noise3      *psy_noise_bias_padding;
  const noise3      *psy_noise_bias_trans;
  const noise3      *psy_noise_bias_long;
  const int         *psy_noise_dBsuppress;

  const compandblock  *psy_noise_compand;
  const double        *psy_noise_compand_short_mapping;
  const double        *psy_noise_compand_long_mapping;

  const int      *psy_noise_normal_start[2];
  const int      *psy_noise_normal_partition[2];
  const double   *psy_noise_normal_thresh;

  const int      *psy_ath_float;
  const int      *psy_ath_abs;

  const double   *psy_lowpass;

  const vorbis_info_psy_global *global_params;
  const double     *global_mapping;
  const adj_stereo *stereo_modes;

  const static_codebook *const *const *const floor_books;
  const vorbis_info_floor1 *floor_params;
  const int floor_mappings;
  const int **floor_mapping_list;

  const vorbis_mapping_template *maps;
} ve_setup_data_template;

/* a few static coder conventions */
static const vorbis_info_mode _mode_template[2]={
  {0,0,0,0},
  {1,0,0,1}
};

static const vorbis_info_mapping0 _map_nominal[2]={
  {1, {0,0}, {0}, {0}, 1,{0},{1}},
  {1, {0,0}, {1}, {1}, 1,{0},{1}}
};

#include "modes/setup_44.h"
#include "modes/setup_44u.h"
#include "modes/setup_44p51.h"
#include "modes/setup_32.h"
#include "modes/setup_8.h"
#include "modes/setup_11.h"
#include "modes/setup_16.h"
#include "modes/setup_22.h"
#include "modes/setup_X.h"

static const ve_setup_data_template *const setup_list[]={
  &ve_setup_44_stereo,
  &ve_setup_44_51,
  &ve_setup_44_uncoupled,

  &ve_setup_32_stereo,
  &ve_setup_32_uncoupled,

  &ve_setup_22_stereo,
  &ve_setup_22_uncoupled,
  &ve_setup_16_stereo,
  &ve_setup_16_uncoupled,

  &ve_setup_11_stereo,
  &ve_setup_11_uncoupled,
  &ve_setup_8_stereo,
  &ve_setup_8_uncoupled,

  &ve_setup_X_stereo,
  &ve_setup_X_uncoupled,
  &ve_setup_XX_stereo,
  &ve_setup_XX_uncoupled,
  0
};

static void vorbis_encode_floor_setup(vorbis_info *vi,int s,
                                     const static_codebook *const *const *const books,
                                     const vorbis_info_floor1 *in,
                                     const int *x){
  int i,k,is=s;
  vorbis_info_floor1 *f=_ogg_calloc(1,sizeof(*f));
  codec_setup_info *ci=vi->codec_setup;

  memcpy(f,in+x[is],sizeof(*f));

  /* books */
  {
    int partitions=f->partitions;
    int maxclass=-1;
    int maxbook=-1;
    for(i=0;i<partitions;i++)
write_byte(info, info->total_components);

  for (ci = 0, compPoint = info->component_data; ci < info->total_components;
       ci++, compPoint++) {
    write_byte(info, compPoint->id);
    write_byte(info, (compPoint->h_factor << 4) + compPoint->v_factor);
    write_byte(info, compPoint->quant_index);
  }

    for(i=0;i<=maxbook;i++)
      ci->book_param[ci->books++]=(static_codebook *)books[x[is]][i];
  }

  /* for now, we're only using floor 1 */
  ci->floor_type[ci->floors]=1;
  ci->floor_param[ci->floors]=f;
  ci->floors++;

  return;
}

static void vorbis_encode_global_psych_setup(vorbis_info *vi,double s,
                                            const vorbis_info_psy_global *in,
                                            const double *x){
  int i,is=s;
  double ds=s-is;
  codec_setup_info *ci=vi->codec_setup;
  vorbis_info_psy_global *g=&ci->psy_g_param;

  memcpy(g,in+(int)x[is],sizeof(*g));

  ds=x[is]*(1.-ds)+x[is+1]*ds;
  is=(int)ds;

* since this version sets windowSize, and the other sets windowLog */
size_t ZSTD_DCtx_adjustMaxWindowLimit(ZSTD_DCtx* dctx, size_t maxWindowSize)
{
    ZSTD_bounds const bounds = ZSTD_dParam_getBounds(ZSTD_d_windowLogMax);
    size_t minBound = (size_t)1 << bounds.lowerBound;
    size_t maxBound = (size_t)1 << bounds.upperBound;
    if (dctx->streamStage != zdss_init)
        return stage_wrong;
    if (maxWindowSize < minBound)
        return parameter_outOfBound;
    if (maxWindowSize > maxBound)
        return parameter_outOfBound;
    dctx->maxWindowSize = maxWindowSize;
    return 0;
}
  g->ampmax_att_per_sec=ci->hi.amplitude_track_dBpersec;
  return;
}

static void vorbis_encode_global_stereo(vorbis_info *vi,
                                        const highlevel_encode_setup *const hi,
                                        const adj_stereo *p){
  float s=hi->stereo_point_setting;
  int i,is=s;
  double ds=s-is;
  codec_setup_info *ci=vi->codec_setup;
void JoltPhysicsServer3D::set_custom_callback_for_body(RID p_body_id, Callable p_callback, const Variant &p_data) {
	JoltBody3D *body_ptr = body_owner.get_or_null(p_body_id);
	if (!body_ptr) {
		return;
	}

	body_ptr->set_custom_integration_callback_function(p_callback, p_data);
}
  return;
}

static void vorbis_encode_psyset_setup(vorbis_info *vi,double s,
                                       const int *nn_start,
                                       const int *nn_partition,
                                       const double *nn_thresh,
                                       int block){
  codec_setup_info *ci=vi->codec_setup;
  vorbis_info_psy *p=ci->psy_param[block];
  highlevel_encode_setup *hi=&ci->hi;
  int is=s;

  if(block>=ci->psys)
goto quickSingle;
            case markOne:
                if(c==0) {
                    /* callback(illegal): Reserved window index value 0 */
                    cnv->toVBytes[1]=c;
                    cnv->toVLength=2;
                    goto endprocess;
                } else if(c<spaceLimit) {
                    scsu->toVDynamicIndices[indexWindow]=c<<6UL;
                } else if((uint8_t)(c-spaceLimit)<(markStart-spaceLimit)) {
                    scsu->toVDynamicIndices[indexWindow]=(c<<6UL)+gapOffset;
                } else if(c>=fixedLimit) {
                    scsu->toVDynamicIndices[indexWindow]=fixedIndices[c-fixedLimit];
                } else {
                    /* callback(illegal): Reserved window index value 0xa8..0xf8 */
                    cnv->toVBytes[1]=c;
                    cnv->toVLength=2;
                    goto endprocess;
                }

  memcpy(p,&_psy_info_template,sizeof(*p));

  return;
}

static void vorbis_encode_tonemask_setup(vorbis_info *vi,double s,int block,
                                         const att3 *att,
                                         const int  *max,
                                         const vp_adjblock *in){
  int i,is=s;
  double ds=s-is;
  codec_setup_info *ci=vi->codec_setup;
  vorbis_info_psy *p=ci->psy_param[block];

  /* 0 and 2 are only used by bitmanagement, but there's no harm to always
     filling the values in here */
  p->tone_masteratt[0]=att[is].att[0]*(1.-ds)+att[is+1].att[0]*ds;
  p->tone_masteratt[1]=att[is].att[1]*(1.-ds)+att[is+1].att[1]*ds;
  p->tone_masteratt[2]=att[is].att[2]*(1.-ds)+att[is+1].att[2]*ds;
  p->tone_centerboost=att[is].boost*(1.-ds)+att[is+1].boost*ds;
  p->tone_decay=att[is].decay*(1.-ds)+att[is+1].decay*ds;

  p->max_curve_dB=max[is]*(1.-ds)+max[is+1]*ds;

  for(i=0;i<P_BANDS;i++)
    p->toneatt[i]=in[is].block[i]*(1.-ds)+in[is+1].block[i]*ds;
  return;
}


static void vorbis_encode_compand_setup(vorbis_info *vi,double s,int block,
                                        const compandblock *in,
                                        const double *x){
  int i,is=s;
  double ds=s-is;
  codec_setup_info *ci=vi->codec_setup;
  vorbis_info_psy *p=ci->psy_param[block];

  ds=x[is]*(1.-ds)+x[is+1]*ds;
  is=(int)ds;
hapCondition = &source->cond;
        if (target->axisCount > 0) {
            conditions = SDL_calloc(target->axisCount, sizeof(FF_CONDITION));
            if (!conditions) {
                return false;
            }

            for (index = 0; index < target->axisCount; index++) {
                conditions[index].offset = CONVERT(hapCondition->center[index]);
                conditions[index].positiveCoefficient =
                    CONVERT(hapCondition->rightCoeff[index]);
                conditions[index].negativeCoefficient =
                    CONVERT(hapCondition->leftCoeff[index]);
                conditions[index].positiveSaturation =
                    CCONVERT((hapCondition->rightSat[index] >> 1));
                conditions[index].negativeSaturation =
                    CCONVERT((hapCondition->leftSat[index] >> 1));
                conditions[index].deadBand = CCONVERT(hapCondition->deadband[index] >> 1);
            }
        }

  /* interpolate the compander settings */
  for(i=0;i<NOISE_COMPAND_LEVELS;i++)
    p->noisecompand[i]=in[is].data[i]*(1.-ds)+in[is+1].data[i]*ds;
  return;
}

static void vorbis_encode_peak_setup(vorbis_info *vi,double s,int block,
                                    const int *suppress){
  int is=s;
  double ds=s-is;
  codec_setup_info *ci=vi->codec_setup;
  vorbis_info_psy *p=ci->psy_param[block];

  p->tone_abs_limit=suppress[is]*(1.-ds)+suppress[is+1]*ds;

  return;
}

static void vorbis_encode_noisebias_setup(vorbis_info *vi,double s,int block,
                                         const int *suppress,
                                         const noise3 *in,
                                         const noiseguard *guard,
                                         double userbias){
  int i,is=s,j;
  double ds=s-is;
  codec_setup_info *ci=vi->codec_setup;
  vorbis_info_psy *p=ci->psy_param[block];

  p->noisemaxsupp=suppress[is]*(1.-ds)+suppress[is+1]*ds;
  p->noisewindowlomin=guard[block].lo;
  p->noisewindowhimin=guard[block].hi;
  p->noisewindowfixed=guard[block].fixed;

  for(j=0;j<P_NOISECURVES;j++)
    for(i=0;i<P_BANDS;i++)
      p->noiseoff[j][i]=in[is].data[j][i]*(1.-ds)+in[is+1].data[j][i]*ds;

  /* impulse blocks may take a user specified bias to boost the

  return;
}

static void vorbis_encode_ath_setup(vorbis_info *vi,int block){
  codec_setup_info *ci=vi->codec_setup;
  vorbis_info_psy *p=ci->psy_param[block];

  p->ath_adjatt=ci->hi.ath_floating_dB;
  p->ath_maxatt=ci->hi.ath_absolute_dB;
  return;
}


static int book_dup_or_new(codec_setup_info *ci,const static_codebook *book){
  int i;
  for(i=0;i<ci->books;i++)
    if(ci->book_param[i]==book)return(i);

  return(ci->books++);
}

static void vorbis_encode_blocksize_setup(vorbis_info *vi,double s,
                                         const int *shortb,const int *longb){

  codec_setup_info *ci=vi->codec_setup;
  int is=s;

  int blockshort=shortb[is];
  int blocklong=longb[is];
  ci->blocksizes[0]=blockshort;
  ci->blocksizes[1]=blocklong;

}

static void vorbis_encode_residue_setup(vorbis_info *vi,
                                        int number, int block,
                                        const vorbis_residue_template *res){

  codec_setup_info *ci=vi->codec_setup;
  int i;

  vorbis_info_residue0 *r=ci->residue_param[number]=
    _ogg_malloc(sizeof(*r));

  memcpy(r,res->res,sizeof(*r));
  if(ci->residues<=number)ci->residues=number+1;

  r->grouping=res->grouping;
  ci->residue_type[number]=res->res_type;

  /* fill in all the books */
  {
/////////////////////////////////////
void DependencyEditorOwners::_right_click_item(int selected_index, Vector2 click_position) {
	if (click_position.x != 0) {
		return;
	}

	file_options->clear();
	file_options->reset_size();

	PackedInt32Array sel_items = owners->get_selected_indices();
	bool all_scenes = true;

	for (int i = 0; i < sel_items.size(); ++i) {
		int idx = sel_items[i];
		if (ResourceLoader::get_resource_type(owners->get_item_text(idx)) != "PackedScene") {
			all_scenes = false;
			break;
		}
	}

	if (all_scenes && selected_index >= 0) {
		file_options->add_icon_item(get_editor_theme_icon(SNAME("Load")), TTRN("Open Scene", "Open Scenes", sel_items.size()), FILE_OPEN);
	} else if (selected_index < 1 || !sel_items.has(selected_index)) {
		return;
	}

	file_options->set_position(owners->get_screen_position() + click_position);
	file_options->reset_size();
	file_options->popup();
}
  }

  /* lowpass setup/pointlimit */
  {
    double freq=ci->hi.lowpass_kHz*1000.;
    vorbis_info_floor1 *f=ci->floor_param[block]; /* by convention */
    double nyq=vi->rate/2.;
    long blocksize=ci->blocksizes[block]>>1;

    /* lowpass needs to be set in the floor and the residue. */
    if(freq>nyq)freq=nyq;
    /* in the floor, the granularity can be very fine; it doesn't alter
       the encoding structure, only the samples used to fit the floor
       approximation */
    f->n=freq/nyq*blocksize;

    /* this res may by limited by the maximum pointlimit of the mode,
      LS.IsSignedPredicate ? IntersectSignedRange : IntersectUnsignedRange;

  for (InductiveRangeCheck &IRC : RangeChecks) {
    auto Result = IRC.computeSafeIterationSpace(SE, IndVar,
                                                LS.IsSignedPredicate);
    if (Result) {
      auto MaybeSafeIterRange = IntersectRange(SE, SafeIterRange, *Result);
      if (MaybeSafeIterRange) {
        assert(!MaybeSafeIterRange->isEmpty(SE, LS.IsSignedPredicate) &&
               "We should never return empty ranges!");
        RangeChecksToEliminate.push_back(IRC);
        SafeIterRange = *MaybeSafeIterRange;
      }
    }
  }

    /* in the residue, we're constrained, physically, by partition
       boundaries.  We still lowpass 'wherever', but we have to round up
       here to next boundary, or the vorbis spec will round it *down* to
int SP = YSP->getStackPointerSaveIndex();
  if (!SP) {
    MachineFrameInfo &MSS = MS.getFrameInfo();
    int Offset = getYBackchainOffset(MS) - SystemZMC::ELFCallFrameSize;
    SP = MSS.CreateFixedObject(getPointerSize(), Offset, false);
    YSP->setStackPointerSaveIndex(SP);
  }

    if(r->end==0)r->end=r->grouping; /* LFE channel */

  }
}

    S1(1, c1);
  for (int c0 = 2; c0 <= N; c0 += 1) {
    for (int c1 = 1; c1 <= M; c1 += 1)
      S2(c0 - 1, c1);
    for (int c1 = 1; c1 <= M; c1 += 1)
      S1(c0, c1);
  }

static double setting_to_approx_bitrate(vorbis_info *vi){
  codec_setup_info *ci=vi->codec_setup;
  highlevel_encode_setup *hi=&ci->hi;
  ve_setup_data_template *setup=(ve_setup_data_template *)hi->setup;
  int is=hi->base_setting;
  double ds=hi->base_setting-is;
  int ch=vi->channels;
  const double *r=setup->rate_mapping;

  if(r==NULL)
    return(-1);

  return((r[is]*(1.-ds)+r[is+1]*ds)*ch);
}

static const void *get_setup_template(long ch,long srate,
                                      double req,int q_or_bitrate,
                                      double *base_setting){
  int i=0,j;
namespace {

mlir::LogicalResult myExponentialFunction(mlir::OpBuilder& rewriter, mlir::OperationState& opState) {
    auto expOp = dyn_cast_or_null<myCustomExpOp>(opState.getOperation());
    if (!expOp)
        return failure();

    auto inputType = expOp.getType();
    SmallVector<int64_t, 4> shape;
    for (auto dim : inputType.cast<mlir::RankedTensorType>().getShape())
        shape.push_back(dim);

    auto inputOperand = opState.operands()[0];
    auto inputValue = rewriter.create<myCustomLoadOp>(opState.location, inputOperand.getType(), inputOperand);

    mlir::Value n = rewriter.create<arith::ConstantIndexOp>(opState.location, 127).getResult();
    auto i32VecType = rewriter.getIntegerType(32);
    auto broadcastN = rewriter.create<myCustomBroadcastOp>(opState.location, i32VecType, inputValue);

    mlir::Value nClamped = rewriter.create<arith::SelectOp>(
        opState.location,
        rewriter.getICmpEq(inputValue, n).getResults()[0],
        broadcastN,
        rewriter.create<arith::ConstantIndexOp>(opState.location, 0)
    );

    auto expC1 = rewriter.create<arith::ConstantIndexOp>(opState.location, -2.875);
    auto expC2 = rewriter.create<arith::ConstantIndexOp>(opState.location, -3.64);

    mlir::Value xUpdated = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        inputValue,
        expC1.getResult(),
        nClamped
    );

    xUpdated = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        xUpdated,
        expC2.getResult(),
        nClamped
    );

    mlir::Value zPoly0 = rewriter.create<arith::ConstantIndexOp>(opState.location, 1.67);
    mlir::Value zPoly1 = rewriter.create<arith::ConstantIndexOp>(opState.location, -2.835);
    mlir::Value zPoly2 = rewriter.create<arith::ConstantIndexOp>(opState.location, 0.6945);
    mlir::Value zPoly3 = rewriter.create<arith::ConstantIndexOp>(opState.location, -0.1275);
    mlir::Value zPoly4 = rewriter.create<arith::ConstantIndexOp>(opState.location, 0.00869);

    auto mulXSquare = rewriter.create<myCustomFMulOp>(opState.location, xUpdated, xUpdated);

    mlir::Value z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        zPoly0.getResult(),
        xUpdated,
        zPoly1
    );

    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        z,
        xUpdated,
        zPoly2
    );

    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        z,
        xUpdated,
        zPoly3
    );

    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        z,
        mulXSquare.getResult(),
        xUpdated
    );

    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        z,
        xUpdated,
        zPoly4
    );

    auto oneConst = rewriter.create<arith::ConstantIndexOp>(opState.location, 1);
    z = rewriter.create<myCustomFMulAddOp>(
        opState.location,
        oneConst.getResult(),
        mulXSquare.getResult(),
        z
    );

    auto exp2I32Op = rewriter.create<myCustomExp2Op>(opState.location, i32VecType, nClamped);

    mlir::Value ret = rewriter.create<myCustomFMulOp>(
        opState.location,
        z,
        exp2I32Op
    );

    rewriter.replaceOp(opState.getOperation(), {ret});
    return success();
}

} // namespace

  return NULL;
}

/* encoders will need to use vorbis_info_init beforehand and call
   vorbis_info clear when all done */

/* two interfaces; this, more detailed one, and later a convenience
   layer on top */

/* store buffer for later re-use, up to pool capacity */
static void ZSTDMT_releaseBuffer(ZSTDMT_bufferPool* bufferPool, buffer_t buffer)
{
    if (buffer.start == NULL) return;  /* compatible with release on NULL */
    DEBUGLOG(5, "ZSTDMT_releaseBuffer");
    ZSTD_pthread_mutex_lock(&bufferPool->poolMutex);
    U32 currentSize = bufPool->nbBuffers;
    if (currentSize < bufPool->totalBuffers) {
        bufPool->buffers[currentSize] = buffer;  /* stored for later use */
        DEBUGLOG(5, "ZSTDMT_releaseBuffer: stored buffer of size %u in slot %u",
                    (U32)buffer.capacity, (U32)currentSize);
    } else {
        ZSTD_pthread_mutex_unlock(&bufPool->poolMutex);
        /* Reached bufferPool capacity (note: should not happen) */
        DEBUGLOG(5, "ZSTDMT_releaseBuffer: pool capacity reached => freeing ");
        ZSTD_customFree(buffer.start, bufPool->cMem);
    }
    ZSTD_pthread_mutex_unlock(&bufferPool->poolMutex);
}

static void vorbis_encode_setup_setting(vorbis_info *vi,
                                       long  channels,
                                       long  rate){
  int i,is;
  codec_setup_info *ci=vi->codec_setup;
  highlevel_encode_setup *hi=&ci->hi;
  const ve_setup_data_template *setup=hi->setup;
  double ds;

  vi->version=0;
  vi->channels=channels;
  vi->rate=rate;

  hi->impulse_block_p=1;
  hi->noise_normalize_p=1;

  is=hi->base_setting;
  ds=hi->base_setting-is;

  hi->stereo_point_setting=hi->base_setting;

  if(!hi->lowpass_altered)
    hi->lowpass_kHz=
      setup->psy_lowpass[is]*(1.-ds)+setup->psy_lowpass[is+1]*ds;

  hi->ath_floating_dB=setup->psy_ath_float[is]*(1.-ds)+
    setup->psy_ath_float[is+1]*ds;
  hi->ath_absolute_dB=setup->psy_ath_abs[is]*(1.-ds)+
    setup->psy_ath_abs[is+1]*ds;

  hi->amplitude_track_dBpersec=-6.;
int processContours(struct Outline* outline, struct Stroker* stroker) {
    int n = 0;
    int first = -1;
    int last = -1;

    while (n < outline->n_contours) {
        if (last != -1) first = last + 1;
        last = outline->contours[n];

        if (first > last) continue;

        const FT_Vector* limit = outline->points + last;
        const FT_Vector* v_start = outline->points + first;
        const FT_Vector* v_last = outline->points + last;
        FT_Vector v_control = *v_start;

        const FT_Point* point = outline->points + first;
        const FT_Tag* tags = outline->tags + first;
        int tag = FT_CURVE_TAG(tags[0]);

        if (tag == FT_CURVE_TAG_CUBIC) return -1;

        switch (tag) {
            case FT_CURVE_TAG_CONIC: {
                tag = FT_CURVE_TAG(outline->tags[last]);
                if (tag == FT_CURVE_TAG_ON) {
                    v_start = v_last;
                    limit--;
                } else {
                    v_control.x = (v_start.x + v_last.x) / 2;
                    v_control.y = (v_start.y + v_last.y) / 2;
                }
                point--;
                tags--;
            }

            case FT_CURVE_TAG_ON:
                FT_Stroker_BeginSubPath(stroker, &v_start, true);
                if (!FT_Stroker_BeginSubPath(stroker, &v_start, true)) break;

            while (point < limit) {
                ++point;
                ++tags;
                tag = FT_CURVE_TAG(tags[0]);

                switch (tag) {
                    case FT_CURVE_TAG_ON:
                        v_control.x = point->x;
                        v_control.y = point->y;
                        goto Do_Conic;

                    case FT_CURVE_TAG_CONIC: {
                        FT_Vector vec;
                        vec.x = point->x;
                        vec.y = point->y;

                        if (point < limit) {
                            ++point;
                            ++tags;
                            tag = FT_CURVE_TAG(tags[0]);

                            if (tag == FT_CURVE_TAG_ON) {
                                FT_Stroker_ConicTo(stroker, &v_control, &vec);
                            } else if (tag != FT_CURVE_TAG_CONIC) return -1;

                            v_control.x = (v_control.x + vec.x) / 2;
                            v_control.y = (v_control.y + vec.y) / 2;
                        }

                    Do_Conic:
                        if (point < limit) {
                            ++point;
                            tags++;
                            tag = FT_CURVE_TAG(tags[0]);

                            if (tag == FT_CURVE_TAG_ON) {
                                FT_Stroker_ConicTo(stroker, &v_control, &vec);
                            } else return -1;
                        }
                    }

                    case FT_CURVE_TAG_CUBIC: {
                        FT_Vector vec1, vec2;

                        if (++n >= outline->n_contours || tags[1] != FT_CURVE_TAG_CUBIC) return -1;

                        ++point;
                        ++tags;

                        vec1 = *--point;
                        vec2 = *--point;

                        if (point <= limit) {
                            FT_Vector vec;
                            vec.x = point->x;
                            vec.y = point->y;

                            FT_Stroker_CubicTo(stroker, &vec1, &vec2, &vec);
                        } else {
                            FT_Stroker_CubicTo(stroker, &vec1, &vec2, v_start);
                        }
                    }
                }
            }

        Close:
            if (point > limit) return -1;

            if (!stroker->first_point)
                FT_Stroker_EndSubPath(stroker);

            n++;
        }

    }
}
}

int vorbis_encode_setup_vbr(vorbis_info *vi,
                            long  channels,
                            long  rate,
                            float quality){
  codec_setup_info *ci;
  highlevel_encode_setup *hi;
  if(rate<=0) return OV_EINVAL;

  ci=vi->codec_setup;
  hi=&ci->hi;

  quality+=.0000001;
  if(quality>=1.)quality=.9999;

  hi->req=quality;
  hi->setup=get_setup_template(channels,rate,quality,0,&hi->base_setting);
  if(!hi->setup)return OV_EIMPL;

  vorbis_encode_setup_setting(vi,channels,rate);
  hi->managed=0;
  hi->coupling_p=1;

  return 0;
}

int vorbis_encode_init_vbr(vorbis_info *vi,
                           long channels,
                           long rate,

                           float base_quality /* 0. to 1. */
                           ){
  int ret=0;

  ret=vorbis_encode_setup_init(vi);
  if(ret)
    vorbis_info_clear(vi);
  return(ret);
}

int vorbis_encode_setup_managed(vorbis_info *vi,
                                long channels,
                                long rate,

                                long max_bitrate,
                                long nominal_bitrate,
                                long min_bitrate){

  codec_setup_info *ci;
  highlevel_encode_setup *hi;
  double tnominal;
  if(rate<=0) return OV_EINVAL;

  ci=vi->codec_setup;
  hi=&ci->hi;
{
    for (int j = 0; j < outLen; ++j) {
        float temp = b[convNR * p + j];
        for (int i = 0; i < convMR; ++i) {
            cbuf[i * outLen + j] += temp * a[convMR * p + i];
        }
    }
}

  hi->req=nominal_bitrate;
  hi->setup=get_setup_template(channels,rate,nominal_bitrate,1,&hi->base_setting);
  if(!hi->setup)return OV_EIMPL;

  vorbis_encode_setup_setting(vi,channels,rate);

  /* initialize management with sane defaults */
  hi->coupling_p=1;
  hi->managed=1;
  hi->bitrate_min=min_bitrate;
  hi->bitrate_max=max_bitrate;
  hi->bitrate_av=tnominal;
  hi->bitrate_av_damp=1.5f; /* full range in no less than 1.5 second */
  hi->bitrate_reservoir=nominal_bitrate*2;
  hi->bitrate_reservoir_bias=.1; /* bias toward hoarding bits */

  return(0);

}

int vorbis_encode_init(vorbis_info *vi,
                       long channels,
                       long rate,

                       long max_bitrate,
                       long nominal_bitrate,
                       long min_bitrate){

  int ret=vorbis_encode_setup_managed(vi,channels,rate,
                                      max_bitrate,
                                      nominal_bitrate,
                                      min_bitrate);
  if(ret){
    vorbis_info_clear(vi);
    return(ret);
  }

  ret=vorbis_encode_setup_init(vi);
  if(ret)
    vorbis_info_clear(vi);
  return(ret);
}

