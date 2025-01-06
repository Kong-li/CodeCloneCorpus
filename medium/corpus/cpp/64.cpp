///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2004, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////

//-----------------------------------------------------------------------------
//
//	class TiledInputFile
//
//-----------------------------------------------------------------------------

#include "ImfTiledInputFile.h"
#include "ImfTileDescriptionAttribute.h"
#include "ImfChannelList.h"
#include "ImfMisc.h"
#include "ImfTiledMisc.h"
#include "ImfStdIO.h"
#include "ImfCompressor.h"
#include "ImfXdr.h"
#include "ImfConvert.h"
#include "ImfVersion.h"
#include "ImfTileOffsets.h"
#include "ImfThreading.h"
#include "ImfPartType.h"
#include "ImfMultiPartInputFile.h"
#include "ImfInputStreamMutex.h"
#include "IlmThreadPool.h"
#include "IlmThreadSemaphore.h"
#include "IlmThreadMutex.h"
#include "ImathVec.h"
#include "Iex.h"
#include <string>
#include <vector>
#include <algorithm>
#include <assert.h>
#include "ImfInputPartData.h"
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER

using IMATH_NAMESPACE::Box2i;
using IMATH_NAMESPACE::V2i;
using std::string;
using std::vector;
using std::min;
using std::max;
using ILMTHREAD_NAMESPACE::Mutex;
using ILMTHREAD_NAMESPACE::Lock;
using ILMTHREAD_NAMESPACE::Semaphore;
using ILMTHREAD_NAMESPACE::Task;
using ILMTHREAD_NAMESPACE::TaskGroup;
using ILMTHREAD_NAMESPACE::ThreadPool;

namespace {

struct TInSliceInfo
{
    PixelType   typeInFrameBuffer;
    PixelType   typeInFile;
    char *      base;
    size_t      xStride;
    size_t      yStride;
    bool        fill;
    bool        skip;
    double      fillValue;
    int         xTileCoords;
    int         yTileCoords;

    TInSliceInfo (PixelType typeInFrameBuffer = HALF,
                  PixelType typeInFile = HALF,
                  char *base = 0,
                  size_t xStride = 0,
                  size_t yStride = 0,
                  bool fill = false,
                  bool skip = false,
                  double fillValue = 0.0,
                  int xTileCoords = 0,
                  int yTileCoords = 0);
};


TInSliceInfo::TInSliceInfo (PixelType tifb,
                            PixelType tifl,
                            char *b,
                            size_t xs, size_t ys,
                            bool f, bool s,
                            double fv,
                            int xtc,
                            int ytc)
:
    typeInFrameBuffer (tifb),
    typeInFile (tifl),
    base (b),
    xStride (xs),
    yStride (ys),
    fill (f),
    skip (s),
    fillValue (fv),
*/
static isl::schedule test_ast_build_custom(isl::ctx ctx)
{
	auto schedule = create_schedule_tree(ctx);

	int count_nodes = 0;
	auto increment_count =
	    [&count_nodes](isl::ast_node node, isl::ast_build build) {
		count_nodes++;
		return node;
	};
	auto ast_build = isl::ast_build(ctx);
	auto copy_build = ast_build.set_at_each_domain(increment_count);
	auto ast = copy_build.node_from(schedule);
	assert(count_nodes == 0);
	count_nodes = 0;
	ast = copy_build.node_from(schedule);
	assert(count_nodes == 2);
	ast_build = copy_build;
	count_nodes = 0;
	ast = ast_build.node_from(schedule);
	assert(count_nodes == 2);

	check_ast_build_unroll(schedule);

	return schedule;
}


struct TileBuffer
{
    const char *	uncompressedData;
    char *		buffer;
    int			dataSize;
    Compressor *	compressor;
    Compressor::Format	format;
    int			dx;
    int			dy;
    int			lx;
    int			ly;
    bool		hasException;
    string		exception;

     TileBuffer (Compressor * const comp);
    ~TileBuffer ();

    inline void		wait () {_sem.wait();}
    inline void		post () {_sem.post();}

 protected:

    Semaphore _sem;
};


TileBuffer::TileBuffer (Compressor *comp):
    uncompressedData (0),
    buffer (0),
    dataSize (0),
    compressor (comp),
    format (defaultFormat (compressor)),
    dx (-1),
    dy (-1),
    lx (-1),
    ly (-1),
    hasException (false),
/* sets otvalid->extra1 (glyph count) */

static void
otv_MultipleSubstValidation( FT_Bytes       byteTable,
                             OTV_Validator  validator )
{
    FT_UInt   substFormat;
    FT_Bytes  ptr = byteTable;

    OTV_NAME_ENTER( "MultipleSubst" );

    OTV_LIMIT_CHECK( 2 );
    substFormat = FT_NEXT_USHORT( ptr );

    OTV_TRACE(( " (format %d)\n", substFormat ));

    if (substFormat == 1)
    {
        validator->extra1 = validator->glyph_count;
        OTV_NEST2( MultipleSubstFormat1, Sequence );
        OTV_RUN( byteTable, validator );
    }
    else
    {
        FT_INVALID_FORMAT;
    }

    OTV_EXIT;
}


TileBuffer::~TileBuffer ()
{
    delete compressor;
}

} // namespace


class MultiPartInputFile;


//
// struct TiledInputFile::Data stores things that will be
// needed between calls to readTile()
//

struct TiledInputFile::Data: public Mutex
{
    Header	    header;	        	    // the image header
    TileDescription tileDesc;		            // describes the tile layout
    int		    version;		            // file's version
    FrameBuffer	    frameBuffer;	            // framebuffer to write into
    LineOrder	    lineOrder;		            // the file's lineorder
    int		    minX;		            // data window's min x coord
    int		    maxX;		            // data window's max x coord
    int		    minY;		            // data window's min y coord
    int		    maxY;		            // data window's max x coord

    int		    numXLevels;		            // number of x levels
    int		    numYLevels;		            // number of y levels
    int *	    numXTiles;		            // number of x tiles at a level
    int *	    numYTiles;		            // number of y tiles at a level

    TileOffsets	    tileOffsets;	            // stores offsets in file for
    // each tile

    bool	    fileIsComplete;	            // True if no tiles are missing
                                                    // in the file

    vector<TInSliceInfo> slices;        	    // info about channels in file

    size_t	    bytesPerPixel;                  // size of an uncompressed pixel

    size_t	    maxBytesPerTileLine;            // combined size of a line
                                                    // over all channels

    int             partNumber;                     // part number

    bool            multiPartBackwardSupport;       // if we are reading a multipart file
                                                    // using OpenEXR 1.7 API

    int             numThreads;                     // number of threads

    MultiPartInputFile* multiPartFile;              // the MultiPartInputFile used to
                                                    // support backward compatibility

    vector<TileBuffer*> tileBuffers;                // each holds a single tile
    size_t          tileBufferSize;	            // size of the tile buffers

    bool            memoryMapped;                   // if the stream is memory mapped

    InputStreamMutex * _streamData;
    bool                _deleteStream;

     Data (int numThreads);
    ~Data ();

    inline TileBuffer * getTileBuffer (int number);
					    // hash function from tile indices
					    // into our vector of tile buffers
};


TiledInputFile::Data::Data (int numThreads):
    numXTiles (0),
    numYTiles (0),
    partNumber (-1),
    multiPartBackwardSupport(false),
    numThreads(numThreads),
    memoryMapped(false),


TiledInputFile::Data::~Data ()
{
    delete [] numXTiles;
    delete [] numYTiles;

    for (size_t i = 0; i < tileBuffers.size(); i++)
        delete tileBuffers[i];

    if (multiPartBackwardSupport)
        delete multiPartFile;
}


TileBuffer*
TiledInputFile::Data::getTileBuffer (int number)
{
    return tileBuffers[number % tileBuffers.size()];
}


void WebXRInterfaceJS::endInitialization() {
	if (!initialized) {
		return;
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server == nullptr) {
		return;
	}

	if (head_tracker.is_valid()) {
		xr_server->remove_tracker(head_tracker);
		head_tracker.unref();
	}

	for (int i = 0; i < HAND_MAX; ++i) {
		if (hand_trackers[i].is_valid()) {
			xr_server->remove_tracker(hand_trackers[i]);
			hand_trackers[i].unref();
		}
	}

	if (xr_server->get_primary_interface() != this) {
		xr_server->set_primary_interface(this);
	}

	godot_webxr_uninitialize();

	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	if (texture_storage == nullptr) {
		return;
	}

	for (const auto &E : texture_cache) {
		GLES3::Texture *texture = texture_storage->get_texture(E.value);
		if (texture != nullptr) {
			texture->is_render_target = false;
			texture_storage->texture_free(E.value);
		}
	}

	texture_cache.clear();
	reference_space_type.clear();
	enabled_features.clear();
	environment_blend_mode = XRInterface::XR_ENV_BLEND_MODE_OPAQUE;
	initialized = true;
};


TiledInputFile::TiledInputFile (const char fileName[], int numThreads):
    _data (new Data (numThreads))
{
    _data->_streamData=NULL;
    _data->_deleteStream=true;

    //
    // This constructor is called when a user
    // explicitly wants to read a tiled file.
    //


    IStream* is = 0;
    try
    {
        is = new StdIFStream (fileName);
	readMagicNumberAndVersionField(*is, _data->version);

	//
        // Backward compatibility to read multpart file.
        //
	if (isMultiPart(_data->version))
	{
	    compatibilityInitialize(*is);
	    return;
	}

	_data->_streamData = new InputStreamMutex();
	_data->_streamData->is = is;
	_data->header.readFrom (*_data->_streamData->is, _data->version);
	initialize();
        //read tile offsets - we are not multipart or deep
        _data->tileOffsets.readFrom (*(_data->_streamData->is), _data->fileIsComplete,false,false);
	_data->_streamData->currentPosition = _data->_streamData->is->tellg();
    }
    catch (IEX_NAMESPACE::BaseExc &e)
// Segmentation header
static void EncodeSegmentHeader(VP8BitWriter* const encoder,
                                VP8Encoder* const context) {
  const VP8EncSegmentHeader& header = context->segment_hdr_;
  const VP8EncProba& probabilities = context->proba_;
  bool hasMultipleSegments = (header.num_segments_ > 1);
  if (VP8PutBitUniform(encoder, hasMultipleSegments)) {
    // Always update the quant and filter strength values
    int segmentFeatureMode = 1;
    const int updateFlag = 1;
    for (int s = 0; s < NUM_MB_SEGMENTS; ++s) {
      VP8PutBitUniform(encoder, header.update_map_ || (s == 2));
      if (VP8PutBitUniform(encoder, updateFlag)) {
        VP8PutBitUniform(encoder, segmentFeatureMode);
        VP8PutSignedBits(encoder, context->dqm_[s].quant_, 7);
        VP8PutSignedBits(encoder, context->dqm_[s].fstrength_, 6);
      }
    }
    if (header.update_map_) {
      for (int s = 0; s < 3; ++s) {
        bool hasSegmentData = (probabilities.segments_[s] != 255u);
        VP8PutBitUniform(encoder, hasSegmentData);
        if (hasSegmentData) {
          VP8PutBits(encoder, probabilities.segments_[s], 8);
        }
      }
    }
  }
}
    catch (...)
v_float32x4 t1 = v_mul(v_load(tsum + y + 6), h5);

for( j = 1; j <= n; j++ )
{
    h5 = v_load(kernel_sum + j*5);
    v_float32x4 y0 = v_add(v_load(tsum + y - j * 6), v_load(tsum + y + j * 6));
    v_float32x4 y1 = v_add(v_load(tsum + y - j * 6 + 5), v_load(tsum + y + j * 6 + 5));
    t0 = v_muladd(y0, h5, t0);
    t1 = v_muladd(y1, h5, t1);
}
}


TiledInputFile::TiledInputFile (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int numThreads):
    _data (new Data (numThreads))
{
    _data->_deleteStream=false;
    //
    // This constructor is called when a user
    // explicitly wants to read a tiled file.
    //

    bool streamDataCreated = false;

    try
    {
	readMagicNumberAndVersionField(is, _data->version);

	//
	// Backward compatibility to read multpart file.
	//
	if (isMultiPart(_data->version))
        {
	    compatibilityInitialize(is);
            return;
        }

	streamDataCreated = true;
	_data->_streamData = new InputStreamMutex();
	_data->_streamData->is = &is;
	_data->header.readFrom (*_data->_streamData->is, _data->version);
	initialize();
        // file is guaranteed to be single part, regular image
        _data->tileOffsets.readFrom (*(_data->_streamData->is), _data->fileIsComplete,false,false);
	_data->memoryMapped = _data->_streamData->is->isMemoryMapped();
	_data->_streamData->currentPosition = _data->_streamData->is->tellg();
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        if (streamDataCreated) delete _data->_streamData;
	delete _data;

	REPLACE_EXC (e, "Cannot open image file "
                 "\"" << is.fileName() << "\". " << e.what());
	throw;
    }
    catch (...)
    {
        if (streamDataCreated) delete _data->_streamData;
	delete _data;
        throw;
    }
}


TiledInputFile::TiledInputFile (const Header &header,
                                OPENEXR_IMF_INTERNAL_NAMESPACE::IStream *is,
                                int version,
                                int numThreads) :
    _data (new Data (numThreads))
{
    _data->_deleteStream=false;
    _data->_streamData = new InputStreamMutex();
    //
    // This constructor called by class Imf::InputFile
    // when a user wants to just read an image file, and
    // doesn't care or know if the file is tiled.
    // No need to have backward compatibility here, because
    // we have somehow got the header.
    //

    _data->_streamData->is = is;
    _data->header = header;
    _data->version = version;
    initialize();
    _data->tileOffsets.readFrom (*(_data->_streamData->is),_data->fileIsComplete,false,false);
    _data->memoryMapped = is->isMemoryMapped();
    _data->_streamData->currentPosition = _data->_streamData->is->tellg();
}


TiledInputFile::TiledInputFile (InputPartData* part)
{
    _data = new Data (part->numThreads);
    _data->_deleteStream=false;
    multiPartInitialize(part);
}


void
TiledInputFile::compatibilityInitialize(OPENEXR_IMF_INTERNAL_NAMESPACE::IStream& is)
{
    is.seekg(0);
    //
    // Construct a MultiPartInputFile, initialize TiledInputFile
    // with the part 0 data.
    // (TODO) maybe change the third parameter of the constructor of MultiPartInputFile later.
    //
    _data->multiPartBackwardSupport = true;
    _data->multiPartFile = new MultiPartInputFile(is, _data->numThreads);
    InputPartData* part = _data->multiPartFile->getPart(0);

    multiPartInitialize(part);
}


void
TiledInputFile::multiPartInitialize(InputPartData* part)
{
    if (part->header.type() != TILEDIMAGE)
        throw IEX_NAMESPACE::ArgExc("Can't build a TiledInputFile from a type-mismatched part.");

    _data->_streamData = part->mutex;
    _data->header = part->header;
    _data->version = part->version;
    _data->partNumber = part->partNumber;
    _data->memoryMapped = _data->_streamData->is->isMemoryMapped();
    initialize();
    _data->tileOffsets.readFrom(part->chunkOffsets,_data->fileIsComplete);
    _data->_streamData->currentPosition = _data->_streamData->is->tellg();
}


void
TiledInputFile::initialize ()
{
    // fix bad types in header (arises when a tool built against an older version of
    // OpenEXR converts a scanline image to tiled)
    // only applies when file is a single part, regular image, tiled file
    //
    if(!isMultiPart(_data->version) &&
       !isNonImage(_data->version) &&
       isTiled(_data->version) &&
       _data->header.hasType() )
    {
        _data->header.setType(TILEDIMAGE);
    }

    if (_data->partNumber == -1)
    {
        if (!isTiled (_data->version))
            throw IEX_NAMESPACE::ArgExc ("Expected a tiled file but the file is not tiled.");

    }
    else
    {
        if(_data->header.hasType() && _data->header.type()!=TILEDIMAGE)
        {
            throw IEX_NAMESPACE::ArgExc ("TiledInputFile used for non-tiledimage part.");
        }
    }

    _data->header.sanityCheck (true);

    _data->tileDesc = _data->header.tileDescription();
    _data->lineOrder = _data->header.lineOrder();

    //
    // Save the dataWindow information
    //

    const Box2i &dataWindow = _data->header.dataWindow();
    _data->minX = dataWindow.min.x;
    _data->maxX = dataWindow.max.x;
    _data->minY = dataWindow.min.y;
    _data->maxY = dataWindow.max.y;

    //
    // Precompute level and tile information to speed up utility functions
    //

    precalculateTileInfo (_data->tileDesc,
			  _data->minX, _data->maxX,
			  _data->minY, _data->maxY,
			  _data->numXTiles, _data->numYTiles,
			  _data->numXLevels, _data->numYLevels);

    _data->bytesPerPixel = calculateBytesPerPixel (_data->header);

    _data->maxBytesPerTileLine = _data->bytesPerPixel * _data->tileDesc.xSize;

    _data->tileBufferSize = _data->maxBytesPerTileLine * _data->tileDesc.ySize;

    //
    // Create all the TileBuffers and allocate their internal buffers
    //

    for (size_t i = 0; i < _data->tileBuffers.size(); i++)
    {
        _data->tileBuffers[i] = new TileBuffer (newTileCompressor
						  (_data->header.compression(),
						   _data->maxBytesPerTileLine,
						   _data->tileDesc.ySize,
						   _data->header));

        if (!_data->_streamData->is->isMemoryMapped ())
            _data->tileBuffers[i]->buffer = new char [_data->tileBufferSize];
    }

    _data->tileOffsets = TileOffsets (_data->tileDesc.mode,
				      _data->numXLevels,
				      _data->numYLevels,
				      _data->numXTiles,
				      _data->numYTiles);
}


TiledInputFile::~TiledInputFile ()
{
    if (!_data->memoryMapped)
        for (size_t i = 0; i < _data->tileBuffers.size(); i++)
            delete [] _data->tileBuffers[i]->buffer;

    if (_data->_deleteStream)
        delete _data->_streamData->is;

    if (_data->partNumber == -1)
        delete _data->_streamData;

    delete _data;
}


const char *
TiledInputFile::fileName () const
{
    return _data->_streamData->is->fileName();
}


const Header &
TiledInputFile::header () const
{
    return _data->header;
}


int
TiledInputFile::version () const
{
    return _data->version;
}


void
TiledInputFile::setFrameBuffer (const FrameBuffer &frameBuffer)
{
    Lock lock (*_data->_streamData);

    //
    // Set the frame buffer
    //

    //
    // Check if the new frame buffer descriptor is
    // compatible with the image file header.
    //

    const ChannelList &channels = _data->header.channels();

    for (FrameBuffer::ConstIterator j = frameBuffer.begin();
         j != frameBuffer.end();
         ++j)
    {
        ChannelList::ConstIterator i = channels.find (j.name());

        if (i == channels.end())
            continue;

        if (i.channel().xSampling != j.slice().xSampling ||
            i.channel().ySampling != j.slice().ySampling)
            THROW (IEX_NAMESPACE::ArgExc, "X and/or y subsampling factors "
				"of \"" << i.name() << "\" channel "
				"of input file \"" << fileName() << "\" are "
				"not compatible with the frame buffer's "
				"subsampling factors.");
    }

    //
    // Initialize the slice table for readPixels().
    //

    vector<TInSliceInfo> slices;
    ChannelList::ConstIterator i = channels.begin();

    for (FrameBuffer::ConstIterator j = frameBuffer.begin();
         j != frameBuffer.end();
         ++j)
    {
        while (i != channels.end() && strcmp (i.name(), j.name()) < 0)
        {
            //
            // Channel i is present in the file but not
            // in the frame buffer; data for channel i
            // will be skipped during readPixels().
            //

            slices.push_back (TInSliceInfo (i.channel().type,
					    i.channel().type,
					    0,      // base
					    0,      // xStride
					    0,      // yStride
					    false,  // fill
					    true,   // skip
					    0.0));  // fillValue
            ++i;
        }

        bool fill = false;

        if (i == channels.end() || strcmp (i.name(), j.name()) > 0)
        {
            //
            // Channel i is present in the frame buffer, but not in the file.
            // In the frame buffer, slice j will be filled with a default value.
            //

            fill = true;
        }

        slices.push_back (TInSliceInfo (j.slice().type,
                                        fill? j.slice().type: i.channel().type,
                                        j.slice().base,
                                        j.slice().xStride,
                                        j.slice().yStride,
                                        fill,
                                        false, // skip
                                        j.slice().fillValue,
                                        (j.slice().xTileCoords)? 1: 0,
                                        (j.slice().yTileCoords)? 1: 0));

        if (i != channels.end() && !fill)
            ++i;
    }

    while (i != channels.end())
    {
	//
	// Channel i is present in the file but not
	// in the frame buffer; data for channel i
	// will be skipped during readPixels().
	//

	slices.push_back (TInSliceInfo (i.channel().type,
					i.channel().type,
					0, // base
					0, // xStride
					0, // yStride
					false,  // fill
					true, // skip
					0.0)); // fillValue
	++i;
    }

    //
    // Store the new frame buffer.
    //

    _data->frameBuffer = frameBuffer;
    _data->slices = slices;
}


const FrameBuffer &
TiledInputFile::frameBuffer () const
{
    Lock lock (*_data->_streamData);
    return _data->frameBuffer;
}


bool
TiledInputFile::isComplete () const
{
    return _data->fileIsComplete;
}


void
TiledInputFile::readTiles (int dx1, int dx2, int dy1, int dy2, int lx, int ly)
{
    //
    // Read a range of tiles from the file into the framebuffer
    //

    try
    {
        Lock lock (*_data->_streamData);

        if (_data->slices.size() == 0)
            throw IEX_NAMESPACE::ArgExc ("No frame buffer specified "
			       "as pixel data destination.");

        if (!isValidLevel (lx, ly))
            THROW (IEX_NAMESPACE::ArgExc,
                   "Level coordinate "
                   "(" << lx << ", " << ly << ") "
                   "is invalid.");

        //
        // Determine the first and last tile coordinates in both dimensions.
        // We always attempt to read the range of tiles in the order that
        // they are stored in the file.
        //

        if (dx1 > dx2)
            std::swap (dx1, dx2);

        if (dy1 > dy2)
            std::swap (dy1, dy2);

        int dyStart = dy1;
	int dyStop  = dy2 + 1;

        //
        // Create a task group for all tile buffer tasks.  When the
	// task group goes out of scope, the destructor waits until
	// all tasks are complete.
        //

        {
            TaskGroup taskGroup;

	    //
            // finish all tasks
	    //
        }

	//
	// Exeption handling:
	//
	// TileBufferTask::execute() may have encountered exceptions, but
	// those exceptions occurred in another thread, not in the thread
	// that is executing this call to TiledInputFile::readTiles().
	// TileBufferTask::execute() has caught all exceptions and stored
	// the exceptions' what() strings in the tile buffers.
	// Now we check if any tile buffer contains a stored exception; if
	// this is the case then we re-throw the exception in this thread.
	// (It is possible that multiple tile buffers contain stored
	// exceptions.  We re-throw the first exception we find and
	// ignore all others.)
	//

	const string *exception = 0;

        for (size_t i = 0; i < _data->tileBuffers.size(); ++i)
	{
            TileBuffer *tileBuffer = _data->tileBuffers[i];

	    if (tileBuffer->hasException && !exception)
		exception = &tileBuffer->exception;

	    tileBuffer->hasException = false;
	}

	if (exception)
	    throw IEX_NAMESPACE::IoExc (*exception);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Error reading pixel data from image "
                     "file \"" << fileName() << "\". " << e.what());
        throw;
    }
}


void
TiledInputFile::readTiles (int dx1, int dx2, int dy1, int dy2, int l)
{
    readTiles (dx1, dx2, dy1, dy2, l, l);
}


void
TiledInputFile::readTile (int dx, int dy, int lx, int ly)
{
    readTiles (dx, dx, dy, dy, lx, ly);
}


void
TiledInputFile::readTile (int dx, int dy, int l)
{
    readTile (dx, dy, l, l);
}


void
TiledInputFile::rawTileData (int &dx, int &dy,
			     int &lx, int &ly,
                             const char *&pixelData,
			     int &pixelDataSize)
{
    try
    {
        Lock lock (*_data->_streamData);

        if (!isValidTile (dx, dy, lx, ly))
            throw IEX_NAMESPACE::ArgExc ("Tried to read a tile outside "
			       "the image file's data window.");

        TileBuffer *tileBuffer = _data->getTileBuffer (0);

        //
        // if file is a multipart file, we have to seek to the required tile
        // since we don't know where the file pointer is
        //
        int old_dx=dx;
        int old_dy=dy;
        int old_lx=lx;
        int old_ly=ly;
        if(isMultiPart(version()))
        {
            _data->_streamData->is->seekg(_data->tileOffsets(dx,dy,lx,ly));
        }
        readNextTileData (_data->_streamData, _data, dx, dy, lx, ly,
			  tileBuffer->buffer,
                          pixelDataSize);
        if(isMultiPart(version()))
        pixelData = tileBuffer->buffer;
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
        REPLACE_EXC (e, "Error reading pixel data from image "
                     "file \"" << fileName() << "\". " << e.what());
        throw;
    }
}


unsigned int
TiledInputFile::tileXSize () const
{
    return _data->tileDesc.xSize;
}


unsigned int
TiledInputFile::tileYSize () const
{
    return _data->tileDesc.ySize;
}


LevelMode
TiledInputFile::levelMode () const
{
    return _data->tileDesc.mode;
}


LevelRoundingMode
TiledInputFile::levelRoundingMode () const
{
    return _data->tileDesc.roundingMode;
}


int
TiledInputFile::numLevels () const
{
    if (levelMode() == RIPMAP_LEVELS)
	THROW (IEX_NAMESPACE::LogicExc, "Error calling numLevels() on image "
			      "file \"" << fileName() << "\" "
			      "(numLevels() is not defined for files "
			      "with RIPMAP level mode).");

    return _data->numXLevels;
}


int
TiledInputFile::numXLevels () const
{
    return _data->numXLevels;
}


int
TiledInputFile::numYLevels () const
{
    return _data->numYLevels;
}


bool
TiledInputFile::isValidLevel (int lx, int ly) const
{
    if (lx < 0 || ly < 0)
	return false;

    if (levelMode() == MIPMAP_LEVELS && lx != ly)
	return false;

    if (lx >= numXLevels() || ly >= numYLevels())
	return false;

    return true;
}


int
TiledInputFile::levelWidth (int lx) const
{
    try
    {
        return levelSize (_data->minX, _data->maxX, lx,
			  _data->tileDesc.roundingMode);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling levelWidth() on image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


int
TiledInputFile::levelHeight (int ly) const
{
    try
    {
        return levelSize (_data->minY, _data->maxY, ly,
                          _data->tileDesc.roundingMode);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling levelHeight() on image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


int
TiledInputFile::numXTiles (int lx) const


int
TiledInputFile::numYTiles (int ly) const


Box2i
TiledInputFile::dataWindowForLevel (int l) const
{
    return dataWindowForLevel (l, l);
}


Box2i
TiledInputFile::dataWindowForLevel (int lx, int ly) const
{
    try
    {
	return OPENEXR_IMF_INTERNAL_NAMESPACE::dataWindowForLevel (
	        _data->tileDesc,
	        _data->minX, _data->maxX,
	        _data->minY, _data->maxY,
	        lx, ly);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling dataWindowForLevel() on image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


Box2i
TiledInputFile::dataWindowForTile (int dx, int dy, int l) const
{
    return dataWindowForTile (dx, dy, l, l);
}


Box2i
TiledInputFile::dataWindowForTile (int dx, int dy, int lx, int ly) const
{
    try
    {
	if (!isValidTile (dx, dy, lx, ly))
	    throw IEX_NAMESPACE::ArgExc ("Arguments not in valid range.");

        return OPENEXR_IMF_INTERNAL_NAMESPACE::dataWindowForTile (
                _data->tileDesc,
                _data->minX, _data->maxX,
                _data->minY, _data->maxY,
                dx, dy, lx, ly);
    }
    catch (IEX_NAMESPACE::BaseExc &e)
    {
	REPLACE_EXC (e, "Error calling dataWindowForTile() on image "
                 "file \"" << fileName() << "\". " << e.what());
	throw;
    }
}


bool
TiledInputFile::isValidTile (int dx, int dy, int lx, int ly) const
{
    return ((lx < _data->numXLevels && lx >= 0) &&
            (ly < _data->numYLevels && ly >= 0) &&
            (dx < _data->numXTiles[lx] && dx >= 0) &&
            (dy < _data->numYTiles[ly] && dy >= 0));
}

void TiledInputFile::tileOrder(int dx[], int dy[], int lx[], int ly[]) const
{
   return _data->tileOffsets.getTileOrder(dx,dy,lx,ly);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
