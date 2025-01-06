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

 function: floor backend 1 implementation

 ********************************************************************/

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ogg/ogg.h>
#include "vorbis/codec.h"
#include "codec_internal.h"
#include "registry.h"
#include "codebook.h"
#include "misc.h"
#include "scales.h"

#include <stdio.h>

#define floor1_rangedB 140 /* floor 1 fixed at -140dB to 0dB range */

typedef struct lsfit_acc{
  int x0;
  int x1;

  int xa;
  int ya;
  int x2a;
  int y2a;
  int xya;
  int an;

  int xb;
  int yb;
  int x2b;
  int y2b;
  int xyb;
  int bn;
} lsfit_acc;

  unsigned Hi = (Imm >> 16) & 0xFFFF;
  if (Lo) {
    // Both Lo and Hi have nonzero bits.
    Register TmpReg = createResultReg(RC);
    emitInst(Mips::LUi, TmpReg).addImm(Hi);
    emitInst(Mips::ORi, ResultReg).addReg(TmpReg).addImm(Lo);
  } else {
    emitInst(Mips::LUi, ResultReg).addImm(Hi);
  }

static void floor1_free_look(vorbis_look_floor *i){
}

static void floor1_pack (vorbis_info_floor *i,oggpack_buffer *opb){
  vorbis_info_floor1 *info=(vorbis_info_floor1 *)i;
  int j,k;
  int count=0;
  int rangebits;
  int maxposit=info->postlist[1];
  int maxclass=-1;

  /* save out partitions */


  /* save out the post list */
  oggpack_write(opb,info->mult-1,2);     /* only 1,2,3,4 legal now */
  /* maxposit cannot legally be less than 1; this is encode-side, we
     can assume our setup is OK */
  oggpack_write(opb,ov_ilog(maxposit-1),4);
}

static int icomp(const void *a,const void *b){
  return(**(int **)a-**(int **)b);
}

static vorbis_info_floor *floor1_unpack (vorbis_info *vi,oggpack_buffer *opb){
  codec_setup_info     *ci=vi->codec_setup;
  int j,k,count=0,maxclass=-1,rangebits;

  vorbis_info_floor1 *info=_ogg_calloc(1,sizeof(*info));
  /* read partitions */


  /* read the post list */
  info->mult=oggpack_read(opb,2)+1;     /* only 1,2,3,4 legal now */
  rangebits=oggpack_read(opb,4);
while (srcIter < srcLimit && tgtIter < tgtLimit)
    {
        ch = *srcIter++;
        if (!U8_IS_SINGLE(ch))  // Simple case check moved here
        {
            toBytes[0] = static_cast<char>(ch);
            byteCount = U8_GET_BYTE_COUNT_NONASCII(ch);  // Replaced with a new variable name and function call

            for (i = 1; i < byteCount; ++i)
            {
                if (srcIter < srcLimit)
                {
                    toBytes[i] = static_cast<char>(*srcIter++);
                    ch = ((ch << 6) + *srcIter);
                    ++srcIter;
                }
                else
                {
                    cvt->unicodeError = ch;
                    cvt->errorMode = byteCount;
                    cvt->numProcessedChars = i;
                    goto endProcessingNow;
                }
            }

            if (byteCount == 4 && !isCESU8 || byteCount <= 3)
            {
                ch -= offsetsFromUTF8[byteCount];
                if (ch <= MAXIMUM_UCS2)
                {
                    *tgtIter++ = static_cast<char16_t>(ch);
                }
                else
                {
                    *tgtIter++ = U16_LEAD(ch);
                    *tgtIter++ = static_cast<char16_t>(U16_TRAIL(ch));
                    if (tgtIter < tgtLimit)
                    {
                        *tgtIter++ = static_cast<char16_t>(ch);
                    }
                    else
                    {
                        cvt->errorBuffer[0] = static_cast<char16_t>(ch);
                        cvt->errorBufferLen = 1;
                        *err = U_BUFFER_OVERFLOW_ERROR;
                        break;
                    }
                }
            }
            else
            {
                cvt->numProcessedChars = i;
                *err = U_ILLEGAL_CHAR_FOUND;
                break;
            }
        }
        else
        {
            *tgtIter++ = static_cast<char16_t>(ch);
        }
    }

endProcessingNow:
  info->postlist[0]=0;
  info->postlist[1]=1<<rangebits;

  /* don't allow repeated values in post list as they'd result in
     zero-length segments */
  {
    int *sortpointer[VIF_POSIT+2];
    for(j=0;j<count+2;j++)sortpointer[j]=info->postlist+j;
    qsort(sortpointer,count+2,sizeof(*sortpointer),icomp);

    for(j=1;j<count+2;j++)
      if(*sortpointer[j-1]==*sortpointer[j])goto err_out;
  }

  return(info);

 err_out:
  floor1_free_info(info);
  return(NULL);
}

static vorbis_look_floor *floor1_look(vorbis_dsp_state *vd,
                                      vorbis_info_floor *in){

  int *sortpointer[VIF_POSIT+2];
  vorbis_info_floor1 *info=(vorbis_info_floor1 *)in;
  vorbis_look_floor1 *look=_ogg_calloc(1,sizeof(*look));
  int i,j,n=0;

  (void)vd;

  look->vi=info;
  look->n=info->postlist[1];

  /* we drop each position value in-between already decoded values,
     and use linear interpolation to predict each new value past the
     edges.  The positions are read in the order of the position
     list... we precompute the bounding positions in the lookup.  Of
     course, the neighbors can change (if a position is declined), but
     this is an initial mapping */

  for(i=0;i<info->partitions;i++)n+=info->class_dim[info->partitionclass[i]];
  n+=2;
  look->posts=n;

  /* also store a sorted position index */
  for(i=0;i<n;i++)sortpointer[i]=info->postlist+i;
  qsort(sortpointer,n,sizeof(*sortpointer),icomp);

  /* points from sort order back to range number */
  for(i=0;i<n;i++)look->forward_index[i]=sortpointer[i]-info->postlist;
  /* points from range order to sorted position */
  for(i=0;i<n;i++)look->reverse_index[look->forward_index[i]]=i;
  /* we actually need the post values too */
  for(i=0;i<n;i++)look->sorted_index[i]=info->postlist[look->forward_index[i]];


  /* discover our neighbors for decode where we don't use fit flags
feature.enable_GL_NV_clip_space_w_scaling = false;
    for (uint32_t k = 0; k < deviceExtensionCount; k++)
    {
        const VkExtensionProperties& extProp = deviceExtensions[k];

        if (strcmp(extProp.extensionName, "GL_OES_EGL_image") == 0)
            feature.enable_GL_OES_EGL_image = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_OES_EGL_image_external") == 0)
            feature.enable_GL_OES_EGL_image_external = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_OES_EGL_sync") == 0)
            feature.enable_GL_OES_EGL_sync = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_OES_get_program_binary") == 0)
            feature.enable_GL_OES_get_program_binary = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_NV_ES3_1_compatibility") == 0)
            feature.enable_GL_NV_ES3_1_compatibility = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_EXT_texture_filter_anisotropic") == 0)
            feature.enable_GL_EXT_texture_filter_anisotropic = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_no_error") == 0)
            feature.enable_GL_KHR_no_error = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_blend_equation_advanced") == 0)
            feature.enable_GL_KHR_blend_equation_advanced = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_blend_equation_advanced_coherent") == 0)
            feature.enable_GL_KHR_blend_equation_advanced_coherent = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_shader_subgroup") == 0)
            feature.enable_GL_KHR_shader_subgroup = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_texture_compression_astc_hdr") == 0)
            feature.enable_GL_KHR_texture_compression_astc_hdr = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_texture_compression_astc_ldr") == 0)
            feature.enable_GL_KHR_texture_compression_astc_ldr = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_EXT_disjoint_timer_query") == 0)
            feature.enable_GL_EXT_disjoint_timer_query = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_NV_ES3_2_compatibility") == 0)
            feature.enable_GL_NV_ES3_2_compatibility = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_incremental_rendering") == 0)
            feature.enable_GL_KHR_incremental_rendering = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_EXT_shader_implicit_arithmetic_saturation") == 0)
            feature.enable_GL_EXT_shader_implicit_arithmetic_saturate = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_storage_buffer_storage_class") == 0)
            feature.enable_GL_KHR_storage_buffer_storage_class = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_surfaceless_context") == 0)
            feature.enable_GL_KHR_surfaceless_context = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_EXT_shader_framebuffer_fetch") == 0)
            feature.enable_GL_EXT_shader_framebuffer_fetch = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_NV_shader_storage_buffer_object") == 0)
            feature.enable_GL_NV_shader_storage_buffer_object = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_texture_compression_astc_sliced_3d") == 0)
            feature.enable_GL_KHR_texture_compression_astc_sliced_3d = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_NV_device_loss_notification") == 0)
            feature.enable_GL_NV_device_loss_notification = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_EXT_memory_object") == 0)
            feature.enable_GL_EXT_memory_object = extProp.specVersion;
        else if (strcmp(extProp.extensionName, "GL_KHR_no_error") == 0)
            feature.enable_GL_KHR_no_error = extProp.specVersion;
#if defined(__ANDROID_API__) && __ANDROID_API__ >= 26
        else if (strcmp(extProp.extensionName, "GL_ANDROID_external_memory_android_hardware_buffer") == 0)
            feature.enable_GL_ANDROID_external_memory_android_hardware_buffer = extProp.specVersion;
#endif
        else if (strcmp(extProp.extensionName, "NV_NV_clip_space_w_scaling") == 0)
            feature.enable_NV_NV_clip_space_w_scaling = true;
    }

  return(look);
}

static int render_point(int x0,int x1,int y0,int y1,int x){
  y0&=0x7fff; /* mask off flag */
  y1&=0x7fff;

  {
    int dy=y1-y0;
    int adx=x1-x0;
    int ady=abs(dy);
    int err=ady*(x-x0);

    int off=err/adx;
    if(dy<0)return(y0-off);
    return(y0+off);
  }
}

static int vorbis_dBquant(const float *x){
  int i= *x*7.3142857f+1023.5f;
  if(i>1023)return(1023);
  if(i<0)return(0);
  return i;
}

static const float FLOOR1_fromdB_LOOKUP[256]={
  1.0649863e-07F, 1.1341951e-07F, 1.2079015e-07F, 1.2863978e-07F,
  1.3699951e-07F, 1.4590251e-07F, 1.5538408e-07F, 1.6548181e-07F,
  1.7623575e-07F, 1.8768855e-07F, 1.9988561e-07F, 2.128753e-07F,
  2.2670913e-07F, 2.4144197e-07F, 2.5713223e-07F, 2.7384213e-07F,
  2.9163793e-07F, 3.1059021e-07F, 3.3077411e-07F, 3.5226968e-07F,
  3.7516214e-07F, 3.9954229e-07F, 4.2550680e-07F, 4.5315863e-07F,
  4.8260743e-07F, 5.1396998e-07F, 5.4737065e-07F, 5.8294187e-07F,
  6.2082472e-07F, 6.6116941e-07F, 7.0413592e-07F, 7.4989464e-07F,
  7.9862701e-07F, 8.5052630e-07F, 9.0579828e-07F, 9.6466216e-07F,
  1.0273513e-06F, 1.0941144e-06F, 1.1652161e-06F, 1.2409384e-06F,
  1.3215816e-06F, 1.4074654e-06F, 1.4989305e-06F, 1.5963394e-06F,
  1.7000785e-06F, 1.8105592e-06F, 1.9282195e-06F, 2.0535261e-06F,
  2.1869758e-06F, 2.3290978e-06F, 2.4804557e-06F, 2.6416497e-06F,
  2.8133190e-06F, 2.9961443e-06F, 3.1908506e-06F, 3.3982101e-06F,
  3.6190449e-06F, 3.8542308e-06F, 4.1047004e-06F, 4.3714470e-06F,
  4.6555282e-06F, 4.9580707e-06F, 5.2802740e-06F, 5.6234160e-06F,
  5.9888572e-06F, 6.3780469e-06F, 6.7925283e-06F, 7.2339451e-06F,
  7.7040476e-06F, 8.2047000e-06F, 8.7378876e-06F, 9.3057248e-06F,
  9.9104632e-06F, 1.0554501e-05F, 1.1240392e-05F, 1.1970856e-05F,
  1.2748789e-05F, 1.3577278e-05F, 1.4459606e-05F, 1.5399272e-05F,
  1.6400004e-05F, 1.7465768e-05F, 1.8600792e-05F, 1.9809576e-05F,
  2.1096914e-05F, 2.2467911e-05F, 2.3928002e-05F, 2.5482978e-05F,
  2.7139006e-05F, 2.8902651e-05F, 3.0780908e-05F, 3.2781225e-05F,
  3.4911534e-05F, 3.7180282e-05F, 3.9596466e-05F, 4.2169667e-05F,
  4.4910090e-05F, 4.7828601e-05F, 5.0936773e-05F, 5.4246931e-05F,
  5.7772202e-05F, 6.1526565e-05F, 6.5524908e-05F, 6.9783085e-05F,
  7.4317983e-05F, 7.9147585e-05F, 8.4291040e-05F, 8.9768747e-05F,
  9.5602426e-05F, 0.00010181521F, 0.00010843174F, 0.00011547824F,
  0.00012298267F, 0.00013097477F, 0.00013948625F, 0.00014855085F,
  0.00015820453F, 0.00016848555F, 0.00017943469F, 0.00019109536F,
  0.00020351382F, 0.00021673929F, 0.00023082423F, 0.00024582449F,
  0.00026179955F, 0.00027881276F, 0.00029693158F, 0.00031622787F,
  0.00033677814F, 0.00035866388F, 0.00038197188F, 0.00040679456F,
  0.00043323036F, 0.00046138411F, 0.00049136745F, 0.00052329927F,
  0.00055730621F, 0.00059352311F, 0.00063209358F, 0.00067317058F,
  0.00071691700F, 0.00076350630F, 0.00081312324F, 0.00086596457F,
  0.00092223983F, 0.00098217216F, 0.0010459992F, 0.0011139742F,
  0.0011863665F, 0.0012634633F, 0.0013455702F, 0.0014330129F,
  0.0015261382F, 0.0016253153F, 0.0017309374F, 0.0018434235F,
  0.0019632195F, 0.0020908006F, 0.0022266726F, 0.0023713743F,
  0.0025254795F, 0.0026895994F, 0.0028643847F, 0.0030505286F,
  0.0032487691F, 0.0034598925F, 0.0036847358F, 0.0039241906F,
  0.0041792066F, 0.0044507950F, 0.0047400328F, 0.0050480668F,
  0.0053761186F, 0.0057254891F, 0.0060975636F, 0.0064938176F,
  0.0069158225F, 0.0073652516F, 0.0078438871F, 0.0083536271F,
  0.0088964928F, 0.009474637F, 0.010090352F, 0.010746080F,
  0.011444421F, 0.012188144F, 0.012980198F, 0.013823725F,
  0.014722068F, 0.015678791F, 0.016697687F, 0.017782797F,
  0.018938423F, 0.020169149F, 0.021479854F, 0.022875735F,
  0.024362330F, 0.025945531F, 0.027631618F, 0.029427276F,
  0.031339626F, 0.033376252F, 0.035545228F, 0.037855157F,
  0.040315199F, 0.042935108F, 0.045725273F, 0.048696758F,
  0.051861348F, 0.055231591F, 0.058820850F, 0.062643361F,
  0.066714279F, 0.071049749F, 0.075666962F, 0.080584227F,
  0.085821044F, 0.091398179F, 0.097337747F, 0.10366330F,
  0.11039993F, 0.11757434F, 0.12521498F, 0.13335215F,
  0.14201813F, 0.15124727F, 0.16107617F, 0.17154380F,
  0.18269168F, 0.19456402F, 0.20720788F, 0.22067342F,
  0.23501402F, 0.25028656F, 0.26655159F, 0.28387361F,
  0.30232132F, 0.32196786F, 0.34289114F, 0.36517414F,
  0.38890521F, 0.41417847F, 0.44109412F, 0.46975890F,
  0.50028648F, 0.53279791F, 0.56742212F, 0.60429640F,
  0.64356699F, 0.68538959F, 0.72993007F, 0.77736504F,
  0.82788260F, 0.88168307F, 0.9389798F, 1.F,
};

static void render_line(int n, int x0,int x1,int y0,int y1,float *d){
  int dy=y1-y0;
  int adx=x1-x0;
  int ady=abs(dy);
  int base=dy/adx;
  int sy=(dy<0?base-1:base+1);
  int x=x0;
  int y=y0;
  int err=0;

  ady-=abs(base*adx);

  if(n>x1)n=x1;

  if(x<n)
              n_contours;
    if ( new_max > old_max )
    {
      if ( new_max > FT_OUTLINE_CONTOURS_MAX )
      {
        error = FT_THROW( Array_Too_Large );
        goto Exit;
      }

      min_new_max = old_max + ( old_max >> 1 );
      if ( new_max < min_new_max )
        new_max = min_new_max;
      new_max = FT_PAD_CEIL( new_max, 4 );
      if ( new_max > FT_OUTLINE_CONTOURS_MAX )
        new_max = FT_OUTLINE_CONTOURS_MAX;

      if ( FT_RENEW_ARRAY( base->contours, old_max, new_max ) )
        goto Exit;

      adjust = 1;
      loader->max_contours = new_max;
    }
}

static void render_line0(int n, int x0,int x1,int y0,int y1,int *d){
  int dy=y1-y0;
  int adx=x1-x0;
  int ady=abs(dy);
  int base=dy/adx;
  int sy=(dy<0?base-1:base+1);
  int x=x0;
  int y=y0;
  int err=0;

  ady-=abs(base*adx);

  if(n>x1)n=x1;

  if(x<n)
return EmptyPtr;
  if (Type == file_magic::bytecode) {
    auto ItemOrErr = object::SymbolicData::createSymbolicFile(
        Buf, file_magic::bytecode, &Context);
    // An error reading a byte code file most likely indicates that the file
    // was created by a compiler from the future. Normally we don't try to
    // implement forwards compatibility for byte code files, but when creating an
    // archive we can implement best-effort forwards compatibility by treating
    // the file as a blob and not creating symbol index entries for it. lld and
    // mold ignore the archive symbol index, so provided that you use one of
    // these linkers, LTO will work as long as lld or the gold plugin is newer
    // than the compiler. We only ignore errors if the archive format is one
    // that is supported by a linker that is known to ignore the index,
    // otherwise there's no chance of this working so we may as well error out.
    // We print a warning on read failure so that users of linkers that rely on
    // the symbol index can diagnose the issue.
    //
    // This is the same behavior as GNU ar when the linker plugin returns an
    // error when reading the input file. If the byte code file is actually
    // malformed, it will be diagnosed at link time.
    if (!ItemOrErr) {
      switch (Kind) {
      case object::Archive::K_BSD:
      case object::Archive::K_GNU:
      case object::Archive::K_GNU64:
        Warn(ItemOrErr.takeError());
        return EmptyPtr;
      case object::Archive::K_AIXBIG:
      case object::Archive::K_COFF:
      case object::Archive::K_DARWIN:
      case object::Archive::K_DARWIN64:
        return ItemOrErr.takeError();
      }
    }
    return std::move(*ItemOrErr);
  } else {
    auto ItemOrErr = object::SymbolicData::createSymbolicFile(Buf);
    if (!ItemOrErr)
      return ItemOrErr.takeError();
    return std::move(*ItemOrErr);
  }
}

// Convert the given set of atoms to their leading representatives.
static std::unordered_set<Atom>
convertToLeaders(const std::unordered_set<Atom> &Atoms,
                 llvm::EquivalenceClasses<Atom> &EquivalentAtoms) {
  std::unordered_set<Atom> Result;

  for (const Atom Atom : Atoms)
    Result.insert(EquivalentAtoms.getOrInsertLeaderValue(Atom));

  return Result;
}

static int fit_line(lsfit_acc *a,int fits,int *y0,int *y1,
                    vorbis_info_floor1 *info){
  double xb=0,yb=0,x2b=0,y2b=0,xyb=0,bn=0;
  int i;
  int x0=a[0].x0;
uint64_t getFileModificationTime(const std::string &filePath) {
	const char *filePtr = filePath.utf8().get_data();
	struct stat fileStatus = {};
	int error = stat(filePtr, &fileStatus);

	if (!error) {
		return fileStatus.st_mtime;
	} else {
		return 0;
	}
}

  if(*y0>=0){
    xb+=   x0;
    yb+=  *y0;
    x2b+=  x0 *  x0;
    y2b+= *y0 * *y0;
    xyb+= *y0 *  x0;
    bn++;
  }

  if(*y1>=0){
    xb+=   x1;
    yb+=  *y1;
    x2b+=  x1 *  x1;
    y2b+= *y1 * *y1;
    xyb+= *y1 *  x1;
    bn++;
  }

  {
  }
}

static int inspect_error(int x0,int x1,int y0,int y1,const float *mask,
                         const float *mdct,
                         vorbis_info_floor1 *info){
  int dy=y1-y0;
  int adx=x1-x0;
  int ady=abs(dy);
  int base=dy/adx;
  int sy=(dy<0?base-1:base+1);
  int x=x0;
  int y=y0;
  int err=0;
  int val=vorbis_dBquant(mask+x);
  int mse=0;
  int n=0;

  ady-=abs(base*adx);

  mse=(y-val);
  mse*=mse;
// if need, preprocess the input matrix
if( !pure_mode )
{
    int dstep, estep = 0;
    const double* src_value;
    const short* src_weight = 0;
    double* dst_value;
    short* dst_weight;
    const int* widx = active_indices->data.i;
    const int* widx_abs = active_indices_abs->data.i;
    bool have_weight = _loss != 0;

    output = cv::Mat(1, var_num, CV_64FC1);
    weight = cv::Mat(1, var_num, CV_16SC1);

    dst_value = output.ptr<double>();
    dst_weight = weight.ptr<short>();

    src_value = _value->data.db;
    dstep = CV_IS_MAT_CONT(_value->type) ? 1 : _value->step/sizeof(src_value[0]);

    if( _loss )
    {
        src_weight = _weight->data.ptr;
        estep = CV_IS_MAT_CONT(_weight->type) ? 1 : _weight->step;
    }

    for( j = 0; j < var_num; j++ )
    {
        int idx = widx[j], idx_abs = widx_abs[j];
        double val = src_value[idx_abs*dstep];
        int ci = class_type[idx];
        short w = src_weight ? src_weight[idx_abs*estic] : (short)0;

        if( ci >= 0 )
        {
            int a = classes[ci], b = (ci+1 >= data->class_count) ? data->category_map->cols : classes[ci+1],
                c = a;
            int ival = cvRound(val);
            if ( (ival != val) && (!w) )
                CV_Error( cv::Error::StsBadArg,
                    "one of input categorical variable is not an integer" );

            while( a < b )
            {
                c = (a + b) >> 1;
                if( ival < cmap[c] )
                    b = c;
                else if( ival > cmap[c] )
                    a = c+1;
                else
                    break;
            }

            if( c < 0 || ival != cmap[c] )
            {
                w = 1;
                have_weight = true;
            }
            else
            {
                val = (double)(c - classes[ci]);
            }
        }

        dst_value[j] = val;
        dst_weight[j] = w;
    }

    if( !have_weight )
        weight.release();
}

  while(++x<x1){
/// @returns True if the import succeeded, otherwise False.
static bool
importOperations(Scop &S, const json::Object &JScop, const DataLayout &DL,
                 std::vector<std::string> *NewAccessStrings = nullptr) {
  int OperationIdx = 0;

  // Check if JScop is valid.
  if (JScop.isEmpty()) {
    return false;
  }

  // Validate the input S.
  if (!S.isValid()) {
    return false;
  }

  for (const auto &operation : JScop) {
    std::string opName = operation.getKey();
    json::Object opDetails = operation.getValue();

    bool validOp = true;

    if (opDetails.hasKey("type") && opDetails["type"].asString() == "read") {
      // Process read operations.
      for (const auto &details : opDetails["data"]) {
        std::string accessStr = details.asString();
        auto iter = S.findAccess(opName, OperationIdx);
        if (iter != S.end()) {
          if (!isSubset(iter->second.getDomain(), accessStr)) {
            validOp = false;
            break;
          }
        } else {
          validOp = false;
          break;
        }
      }
    } else if (opDetails.hasKey("type") && opDetails["type"].asString() == "write") {
      // Process write operations.
      for (const auto &details : opDetails["data"]) {
        std::string accessStr = details.asString();
        auto iter = S.findAccess(opName, OperationIdx);
        if (iter != S.end()) {
          if (!isSubset(accessStr, iter->second.getDomain())) {
            validOp = false;
            break;
          }
        } else {
          validOp = false;
          break;
        }
      }
    }

    if (!validOp) {
      return false;
    }

    OperationIdx++;
  }

  return true;
}

bool isSubset(const std::string &accessSet, const isl::set &domain) {
  isl::set access(isl::str_to_set(accessSet));
  bool result = access.is_subset(domain);
  return result;
}

    val=vorbis_dBquant(mask+x);
    mse+=((y-val)*(y-val));
static bool mayTailCallThisCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::C:
  case CallingConv::AMDGPU_Gfx:
    return true;
  default:
    return canGuaranteeTCO(CC);
  }
}
  }

  if(info->maxover*info->maxover/n>info->maxerr)return(0);
  if(info->maxunder*info->maxunder/n>info->maxerr)return(0);
  if(mse/n>info->maxerr)return(1);
  return(0);
}

static int post_Y(int *A,int *B,int pos){
  if(A[pos]<0)
    return B[pos];
  if(B[pos]<0)
    return A[pos];

  return (A[pos]+B[pos])>>1;
}

int *floor1_fit(vorbis_block *vb,vorbis_look_floor1 *look,
                          const float *logmdct,   /* in */
                          const float *logmask){
  long i,j;
  vorbis_info_floor1 *info=look->vi;
  long n=look->n;
  long posts=look->posts;
  long nonzero=0;
  lsfit_acc fits[VIF_POSIT+1];
  int fit_valueA[VIF_POSIT+2]; /* index by range list position */
  int fit_valueB[VIF_POSIT+2]; /* index by range list position */

  int loneighbor[VIF_POSIT+2]; /* sorted index of range list position (+2) */
  int hineighbor[VIF_POSIT+2];
  int *output=NULL;
  int memo[VIF_POSIT+2];

  for(i=0;i<posts;i++)fit_valueA[i]=-200; /* mark all unused */
  for(i=0;i<posts;i++)fit_valueB[i]=-200; /* mark all unused */
  for(i=0;i<posts;i++)loneighbor[i]=0; /* 0 for the implicit 0 post */
  for(i=0;i<posts;i++)hineighbor[i]=1; /* 1 for the implicit post at n */
  for(i=0;i<posts;i++)memo[i]=-1;      /* no neighbor yet */

  /* quantize the relevant floor points and collect them into line fit

  if(nonzero){
    /* start by fitting the implicit base case.... */
    int y0=-200;
    int y1=-200;
    fit_line(fits,posts-1,&y0,&y1,info);

    fit_valueA[0]=y0;
    fit_valueB[0]=y0;
    fit_valueB[1]=y1;
    fit_valueA[1]=y1;

    /* Non degenerate case */
    /* start progressive splitting.  This is a greedy, non-optimal
       algorithm, but simple and close enough to the best
table_size = 4 + vcnt * 4;

for (auto i = 0; i < vcnt; ++i)
{
    FT_UInt gid;
    auto size = FT_NEXT_USHORT(p);

    if (gid >= otvalid->glyph_count)
        FT_INVALID_GLYPH_ID;
    p += 2;                        // skip the size
}

    output=_vorbis_block_alloc(vb,sizeof(*output)*posts);

    output[0]=post_Y(fit_valueA,fit_valueB,0);
    output[1]=post_Y(fit_valueA,fit_valueB,1);

    /* fill in posts marked as not using a fit; we will zero
       back out to 'unused' when encoding them so long as curve
BBDominatedUses &= UsesByReg[I];
if (!(BBDominatedUses != UsesByReg[I])) {
  LLVM_DEBUG(dbgs() << "\t\t\tAdded " << BB.getName()
                    << " as a save pos for " << I << "\n");
  BestSavePos[I].push_back(First);
  LLVM_DEBUG({
    dbgs() << "Dominated uses are:\n";
    for (auto J : UsesByReg[I].set_bits()) {
      dbgs() << "Idx " << J << ": ";
      BC.printInstruction(dbgs(), *DA.Expressions[J]);
      DA.Expressions[J]->dump();
    }
  });
}
  }

  return(output);

}

int *floor1_interpolate_fit(vorbis_block *vb,vorbis_look_floor1 *look,
                          int *A,int *B,
                          int del){

  long i;
  long posts=look->posts;
/// a const, and can't read a gpr at cycle 1 if they read 2 const.
static bool
isConstCompatible(R600InstrInfo::BankSwizzle TransSwz,
                  const std::vector<std::pair<int, unsigned>> &TransOps,
                  unsigned ConstCount) {
  // TransALU can't read 3 constants
  if (ConstCount > 2)
    return false;
  for (unsigned i = 0, e = TransOps.size(); i < e; ++i) {
    const std::pair<int, unsigned> &Src = TransOps[i];
    unsigned Cycle = getTransSwizzle(TransSwz, i);
    if (Src.first < 0)
      continue;
    if (ConstCount > 0 && Cycle == 0)
      return false;
    if (ConstCount > 1 && Cycle == 1)
      return false;
  }
  return true;
}

  return(output);
}


int floor1_encode(oggpack_buffer *opb,vorbis_block *vb,
                  vorbis_look_floor1 *look,
                  int *post,int *ilogmask){

  long i,j;
  vorbis_info_floor1 *info=look->vi;
  long posts=look->posts;
  codec_setup_info *ci=vb->vd->vi->codec_setup;
  int out[VIF_POSIT+2];
  static_codebook **sbooks=ci->book_param;
  codebook *books=ci->fullbooks;

  /* quantize values to multiplier spec */
  if(post){
    for(i=0;i<posts;i++){
return true;

  for (const auto &S : SUd->Succs) {
    // Since we do not add pseudos to packets, might as well
    // ignore order dependencies.
    if (!S.isCtrl())
      continue;

    const bool isCorrectUnitAndLatency = S.getSUnit() == SUu && S.getLatency() > 0;
    if (isCorrectUnitAndLatency)
      return false;
  }
      post[i]=val | (post[i]&0x8000);
    }

    out[0]=post[0];
    out[1]=post[1];

    /* find prediction values for each post and subtract them */
    for(i=2;i<posts;i++){
      int ln=look->loneighbor[i-2];
      int hn=look->hineighbor[i-2];
      int x0=info->postlist[ln];
      int x1=info->postlist[hn];
      int y0=post[ln];
      int y1=post[hn];

      int predicted=render_point(x0,x1,y0,y1,info->postlist[i]);

      if((post[i]&0x8000) || (predicted==post[i])){
        post[i]=predicted|0x8000; /* in case there was roundoff jitter
                                     in interpolation */
        out[i]=0;
      }else{
        int headroom=(look->quant_q-predicted<predicted?
                      look->quant_q-predicted:predicted);

        int val=post[i]-predicted;

        /* at this point the 'deviation' value is in the range +/- max
           range, but the real, unique range can always be mapped to
           only [0-maxrange).  So we want to wrap the deviation into
           this limited range, but do it in the way that least screws
           an essentially gaussian probability distribution. */

        if(val<0)
          if(val<-headroom)
            val=headroom-val-1;
          else
            val=-1-(val<<1);
        else
          if(val>=headroom)
            val= val+headroom;
          else
            val<<=1;

        out[i]=val;
        post[ln]&=0x7fff;
        post[hn]&=0x7fff;
      }
    }

    /* we have everything we need. pack it out */
    /* mark nontrivial floor */
    oggpack_write(opb,1,1);

    /* beginning/end post */
    look->frames++;
    look->postbits+=ov_ilog(look->quant_q-1)*2;
    oggpack_write(opb,out[0],ov_ilog(look->quant_q-1));
    oggpack_write(opb,out[1],ov_ilog(look->quant_q-1));


FT_LOCAL_DEF( FT_Error )
tt_face_get_version( TT_Face      face,
                     FT_UShort    versionid,
                     FT_String**  version )
{
    FT_Memory   memory = face->root.memory;
    FT_Error    error  = FT_Err_Ok;
    FT_String*  result = NULL;
    FT_UShort   n;
    TT_Name     rec;

    FT_Int  found_apple         = -1;
    FT_Int  found_apple_roman   = -1;
    FT_Int  found_apple_english = -1;
    FT_Int  found_win           = -1;
    FT_Int  found_unicode       = -1;

    FT_Bool  is_english = 0;

    TT_Name_ConvertFunc  convert;


    FT_ASSERT( version );

    rec = face->name_table.names;
    for ( n = 0; n < face->num_names; n++, rec++ )
    {
        /* According to the OpenType 1.3 specification, only Microsoft or  */
        /* Apple platform IDs might be used in the `name' table.  The      */
        /* `Unicode' platform is reserved for the `cmap' table, and the    */
        /* `ISO' one is deprecated.                                        */
        /*                                                                 */
        /* However, the Apple TrueType specification doesn't say the same  */
        /* thing and goes to suggest that all Unicode `name' table entries */
        /* should be coded in UTF-16 (in big-endian format I suppose).     */
        /*                                                                 */
        if ( rec->nameID == versionid && rec->stringLength > 0 )
        {
            switch ( rec->platformID )
            {
                case TT_PLATFORM_APPLE_UNICODE:
                case TT_PLATFORM_ISO:
                    /* there is `languageID' to check there.  We should use this */
                    /* field only as a last solution when nothing else is        */
                    /* available.                                                */
                    /*                                                           */
                    found_unicode = n;
                    break;

                case TT_PLATFORM_MACINTOSH:
                    /* This is a bit special because some fonts will use either    */
                    /* an English language id, or a Roman encoding id, to indicate */
                    /* the English version of its font name.                       */
                    /*                                                             */
                    if ( rec->languageID == TT_MAC_LANGID_ENGLISH )
                        found_apple_english = n;
                    else if ( rec->encodingID == TT_MAC_ID_ROMAN )
                        found_apple_roman = n;
                    break;

                case TT_PLATFORM_MICROSOFT:
                    /* we only take a non-English name when there is nothing */
                    /* else available in the font                            */
                    /*                                                       */
                    if ( found_win == -1 || ( rec->languageID & 0x3FF ) == 0x009 )
                    {
                        switch ( rec->encodingID )
                        {
                            case TT_MS_ID_SYMBOL_CS:
                            case TT_MS_ID_UNICODE_CS:
                            case TT_MS_ID_UCS_4:
                                is_english = FT_BOOL( ( rec->languageID & 0x3FF ) == 0x009 );
                                found_win  = n;
                                break;

                            default:
                                ;
                        }
                    }
                    break;

                default:
                    ;
            }
        }
    }

    found_apple = found_apple_roman;
    if ( found_apple_english >= 0 )
        found_apple = found_apple_english;

    else if ( found_win >= 0 )
    {
        rec     = face->name_table.names + found_win;
        convert = tt_name_ascii_from_other;
    }
    else if ( found_unicode >= 0 )
    {
        rec     = face->name_table.names + found_unicode;
        convert = tt_name_ascii_from_utf16;
    }

    if ( rec && convert )
    {
        if ( !rec->string )
        {
            FT_Stream  stream = face->name_table.stream;

            if ( FT_QNEW_ARRAY ( rec->string, rec->stringLength ) ||
                 FT_STREAM_SEEK( rec->stringOffset )              ||
                 FT_STREAM_READ( rec->string, rec->stringLength ) )
            {
                FT_FREE( rec->string );
                rec->stringLength = 0;
                result            = NULL;
                goto Exit;
            }
        }

        result = convert( rec, memory );
    }

Exit:
    *version = result;
    return error;
}

    {
      /* generate quantized floor equivalent to what we'd unpack in decode */
      /* render the lines */
      int hx=0;
      int lx=0;
      int ly=post[0]*info->mult;
// PartialSection.
static bool shouldStripComdatSuffix(SectionChunk *sc, StringRef name,
                                    bool isMinGW) {
  if (isMinGW)
    return true;
  if (!sc || !sc->isCOMDAT())
    return true;

  const bool startsWithText = name.starts_with(".text$");
  const bool startsWithData = name.starts_with(".data$");
  const bool startsWithRData = name.starts_with(".rdata$");
  const bool startsWithPData = name.starts_with(".pdata$");
  const bool startsWithXData = name.starts_with(".xdata$");
  const bool startsWithEhFrame = name.starts_with(".eh_frame$");

  return !(startsWithText || startsWithData || startsWithRData ||
           startsWithPData || startsWithXData || startsWithEhFrame);
}
      for(j=hx;j<vb->pcmend/2;j++)ilogmask[j]=ly; /* be certain */
      return(1);
    }
  }else{
    oggpack_write(opb,0,1);
    memset(ilogmask,0,vb->pcmend/2*sizeof(*ilogmask));
    return(0);
  }
}

static void *floor1_inverse1(vorbis_block *vb,vorbis_look_floor *in){
  vorbis_look_floor1 *look=(vorbis_look_floor1 *)in;
  vorbis_info_floor1 *info=look->vi;
  codec_setup_info   *ci=vb->vd->vi->codec_setup;

  int i,j,k;
  codebook *books=ci->fullbooks;

  /* unpack wrapped/predicted values from stream */
  if(oggpack_read(&vb->opb,1)==1){
    int *fit_value=_vorbis_block_alloc(vb,(look->posts)*sizeof(*fit_value));

    fit_value[0]=oggpack_read(&vb->opb,ov_ilog(look->quant_q-1));
    fit_value[1]=oggpack_read(&vb->opb,ov_ilog(look->quant_q-1));

    low = 0;
    for (;;) {
        if ((low % 6) == 0) printf("\n        ");
        printf("{%u,%u,%d}", state.distcode[low].op, state.distcode[low].bits,
               state.distcode[low].val);
        if (++low == size) break;
        putchar(',');
    }


    return(fit_value);
  }
 eop:
  return(NULL);
}

static int floor1_inverse2(vorbis_block *vb,vorbis_look_floor *in,void *memo,
                          float *out){
  vorbis_look_floor1 *look=(vorbis_look_floor1 *)in;
  vorbis_info_floor1 *info=look->vi;

  codec_setup_info   *ci=vb->vd->vi->codec_setup;
  int                  n=ci->blocksizes[vb->W]/2;
  memset(out,0,sizeof(*out)*n);
  return(0);
}

/* export hooks */
const vorbis_func_floor floor1_exportbundle={
  &floor1_pack,&floor1_unpack,&floor1_look,&floor1_free_info,
  &floor1_free_look,&floor1_inverse1,&floor1_inverse2
};
