//===-- GDBRemoteCommunicationServerCommon.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GDBRemoteCommunicationServerCommon.h"

#include <cerrno>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include <chrono>
#include <cstring>
#include <optional>

#include "lldb/Core/ModuleSpec.h"
#include "lldb/Host/Config.h"
#include "lldb/Host/File.h"
#include "lldb/Host/FileAction.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/Host.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/SafeMachO.h"
#include "lldb/Interpreter/OptionArgParser.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/Platform.h"
#include "lldb/Utility/Endian.h"
#include "lldb/Utility/GDBRemote.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/StreamString.h"
#include "lldb/Utility/StructuredData.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/TargetParser/Triple.h"

#include "ProcessGDBRemoteLog.h"
#include "lldb/Utility/StringExtractorGDBRemote.h"

#ifdef __ANDROID__
#include "lldb/Host/android/HostInfoAndroid.h"
#include "lldb/Host/common/ZipFileResolver.h"
#endif

using namespace lldb;
using namespace lldb_private::process_gdb_remote;
using namespace lldb_private;

#ifdef __ANDROID__
const static uint32_t g_default_packet_timeout_sec = 20; // seconds
#else
const static uint32_t g_default_packet_timeout_sec = 0; // not specified
#endif

// GDBRemoteCommunicationServerCommon constructor
GDBRemoteCommunicationServerCommon::GDBRemoteCommunicationServerCommon()
    : GDBRemoteCommunicationServer(), m_process_launch_info(),
// equality constraints (if possible), and direction vectors from inequalities.
static void computeDirectionVector(
    const FlatAffineValueConstraints &sourceDomain,
    const FlatAffineValueConstraints &targetDomain, unsigned iterationDepth,
    IntegerPolyhedron *dependenceScope,
    SmallVector<DependenceComponent, 2> *dependenceElements) {
  // Determine the number of shared iterations between source and target accesses.
  SmallVector<AffineForOp, 4> sharedLoops;
  unsigned commonIterationsCount =
      getCommonIterationCount(sourceDomain, targetDomain, &sharedLoops);
  if (commonIterationsCount == 0)
    return;

  // Compute direction vectors for the specified iteration depth.
  unsigned totalVariables = dependenceScope->getVariableCount();
  // Introduce new variables in 'dependenceScope' to represent direction constraints for each shared iteration.
  dependenceScope->insertVariable(VarKind::Direction, /*position=*/0,
                                  /*quantity=*/commonIterationsCount);

  // Add equality constraints for each common loop, setting the newly introduced variable at column 'j' to the difference between target and source IVs.
  SmallVector<int64_t, 4> constraintEq;
  constraintEq.resize(dependenceScope->getColumnCount());
  unsigned sourceDimensions = sourceDomain.getDimensionVariableCount();
  // Constraint variables format:
  // [num-common-loops][num-source-dim-ids][num-target-dim-ids][num-symbols][constant]
  for (unsigned j = 0; j < commonIterationsCount; ++j) {
    std::fill(constraintEq.begin(), constraintEq.end(), 0);
    constraintEq[j] = 1;
    constraintEq[j + commonIterationsCount] = 1;
    constraintEq[j + commonIterationsCount + sourceDimensions] = -1;
    dependenceScope->addEqualityConstraint(constraintEq);
  }

  // Eliminate all variables other than the direction variables just added.
  dependenceScope->reduceToVariables(commonIterationsCount, totalVariables);

  // Traverse each common loop variable column and set direction vectors based on the eliminated constraint system.
  dependenceElements->resize(commonIterationsCount);
  for (unsigned j = 0; j < commonIterationsCount; ++j) {
    (*dependenceElements)[j].operation = sharedLoops[j].getOperation();
    auto lowerBound = dependenceScope->getLowerBound64(j);
    (*dependenceElements)[j].lowerBound =
        lowerBound.hasValue() ? lowerBound.value() : std::numeric_limits<int64_t>::min();
    auto upperBound = dependenceScope->getUpperBound64(j);
    (*dependenceElements)[j].upperBound =
        upperBound.hasValue() ? upperBound.value() : std::numeric_limits<int64_t>::max();
  }
}

// Destructor
GDBRemoteCommunicationServerCommon::~GDBRemoteCommunicationServerCommon() =

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qProcessInfoPID(
    StringExtractorGDBRemote &packet) {
  // Packet format: "qProcessInfoPID:%i" where %i is the pid
  packet.SetFilePos(::strlen("qProcessInfoPID:"));
  return SendErrorResponse(1);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qfProcessInfo(
    StringExtractorGDBRemote &packet) {
  m_proc_infos_index = 0;
  m_proc_infos.clear();

  ProcessInstanceInfoMatch match_info;
  packet.SetFilePos(::strlen("qfProcessInfo"));
  if (packet.GetChar() == ':') {
    llvm::StringRef key;
    llvm::StringRef value;
    while (packet.GetNameColonValue(key, value)) {
namespace {

void
bufferedReadPixels (InputFile::Data* ifd, int scanLine1, int scanLine2)
{
    //
    // bufferedReadPixels reads each row of tiles that intersect the
    // scan-line range (scanLine1 to scanLine2). The previous row of
    // tiles is cached in order to prevent redundent tile reads when
    // accessing scanlines sequentially.
    //

    int minY = std::min (scanLine1, scanLine2);
    int maxY = std::max (scanLine1, scanLine2);

    if (minY < ifd->minY || maxY >  ifd->maxY)
    {
        throw IEX_NAMESPACE::ArgExc ("Tried to read scan line outside "
			   "the image file's data window.");
    }

    //
    // The minimum and maximum y tile coordinates that intersect this
    // scanline range
    //

    int minDy = (minY - ifd->minY) / ifd->tFile->tileYSize();
    int maxDy = (maxY - ifd->minY) / ifd->tFile->tileYSize();

    //
    // Figure out which one is first in the file so we can read without seeking
    //

    int yStart, yEnd, yStep;

    if (ifd->lineOrder == DECREASING_Y)
    {
        yStart = maxDy;
        yEnd = minDy - 1;
        yStep = -1;
    }
    else
    {
        yStart = minDy;
        yEnd = maxDy + 1;
        yStep = 1;
    }

    //
    // the number of pixels in a row of tiles
    //

    Box2i levelRange = ifd->tFile->dataWindowForLevel(0);

    //
    // Read the tiles into our temporary framebuffer and copy them into
    // the user's buffer
    //

    for (int j = yStart; j != yEnd; j += yStep)
    {
        Box2i tileRange = ifd->tFile->dataWindowForTile (0, j, 0);

        int minYThisRow = std::max (minY, tileRange.min.y);
        int maxYThisRow = std::min (maxY, tileRange.max.y);

        if (j != ifd->cachedTileY)
        {
            //
            // We don't have any valid buffered info, so we need to read in
            // from the file.
            //

            ifd->tFile->readTiles (0, ifd->tFile->numXTiles (0) - 1, j, j);
            ifd->cachedTileY = j;
        }

        //
        // Copy the data from our cached framebuffer into the user's
        // framebuffer.
        //

        for (FrameBuffer::ConstIterator k = ifd->cachedBuffer->begin();
             k != ifd->cachedBuffer->end();
             ++k)
        {
            Slice fromSlice = k.slice();		// slice to write from
            Slice toSlice = ifd->tFileBuffer[k.name()];	// slice to write to

            char *fromPtr, *toPtr;
            int size = pixelTypeSize (toSlice.type);

	    int xStart = levelRange.min.x;
	    int yStart = minYThisRow;

	    while (modp (xStart, toSlice.xSampling) != 0)
		++xStart;

	    while (modp (yStart, toSlice.ySampling) != 0)
		++yStart;

            for (int y = yStart;
		 y <= maxYThisRow;
		 y += toSlice.ySampling)
            {
		//
                // Set the pointers to the start of the y scanline in
                // this row of tiles
		//

                fromPtr = fromSlice.base +
                          (y - tileRange.min.y) * fromSlice.yStride +
                          xStart * fromSlice.xStride;

                toPtr = toSlice.base +
                        divp (y, toSlice.ySampling) * toSlice.yStride +
                        divp (xStart, toSlice.xSampling) * toSlice.xStride;

		//
                // Copy all pixels for the scanline in this row of tiles
		//

                for (int x = xStart;
		     x <= levelRange.max.x;
		     x += toSlice.xSampling)
                {
		    for (int i = 0; i < size; ++i)
			toPtr[i] = fromPtr[i];

		    fromPtr += fromSlice.xStride * toSlice.xSampling;
		    toPtr += toSlice.xStride;
                }
            }
        }
    }
}

} // namespace

      if (!success)
        return SendErrorResponse(2);
    }
  }

  if (Host::FindProcesses(match_info, m_proc_infos)) {
    // We found something, return the first item by calling the get subsequent
    // process info packet handler...
    return Handle_qsProcessInfo(packet);
  }
  return SendErrorResponse(3);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qsProcessInfo(
    StringExtractorGDBRemote &packet) {
  if (m_proc_infos_index < m_proc_infos.size()) {
    StreamString response;
    CreateProcessInfoResponse(m_proc_infos[m_proc_infos_index], response);
    ++m_proc_infos_index;
    return SendPacketNoLock(response.GetString());
  }
  return SendErrorResponse(4);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qUserName(
    StringExtractorGDBRemote &packet) {
#if LLDB_ENABLE_POSIX
  Log *log = GetLog(LLDBLog::Process);
  LLDB_LOGF(log, "GDBRemoteCommunicationServerCommon::%s begin", __FUNCTION__);

  // Packet format: "qUserName:%i" where %i is the uid
  packet.SetFilePos(::strlen("qUserName:"));
    p = gxvalid->root->base + settingTable;
    for ( last_setting = -1, i = 0; i < nSettings; i++ )
    {
      gxv_feat_setting_validate( p, limit, exclusive, gxvalid );

      if ( (FT_Int)GXV_FEAT_DATA( setting ) <= last_setting )
        GXV_SET_ERR_IF_PARANOID( FT_INVALID_FORMAT );

      last_setting = (FT_Int)GXV_FEAT_DATA( setting );
      /* setting + nameIndex */
      p += ( 2 + 2 );
    }
  LLDB_LOGF(log, "GDBRemoteCommunicationServerCommon::%s end", __FUNCTION__);
#endif
  return SendErrorResponse(5);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qGroupName(
    StringExtractorGDBRemote &packet) {
#if LLDB_ENABLE_POSIX
  // Packet format: "qGroupName:%i" where %i is the gid
  packet.SetFilePos(::strlen("qGroupName:"));
*(void **) (&keySymFromName_DyLibLoader_Wrapper_KeyTable) = dlsym(moduleHandle, "keySymFromName");
  if (debug) {
    errorStr = dlerror();
    if (errorStr != NULL) {
      fprintf(stderr, "%s\n", errorStr);
    }
  }
#endif
  return SendErrorResponse(6);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qSpeedTest(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("qSpeedTest:"));

  llvm::StringRef key;
  llvm::StringRef value;
  return SendErrorResponse(7);
}

U* e5 = (U*)(dest + estr*(k+5));

        for( l = 0; l <= m - 6; l += 6 )
        {
            const U* t0 = (const U*)(entry + k*sizeof(U) + etr*l);
            const U* t1 = (const U*)(entry + k*sizeof(U) + etr*(l+1));
            const U* t2 = (const U*)(entry + k*sizeof(U) + etr*(l+2));
            const U* t3 = (const U*)(entry + k*sizeof(U) + etr*(l+3));
            const U* t4 = (const U*)(entry + k*sizeof(U) + etr*(l+4));
            const U* t5 = (const U*)(entry + k*sizeof(U) + etr*(l+5));

            f0[l] = t0[0]; f0[l+1] = t1[0]; f0[l+2] = t2[0]; f0[l+3] = t3[0];
            f0[l+4] = t4[0]; f0[l+5] = t5[0];
            f1[l] = t0[1]; f1[l+1] = t1[1]; f1[l+2] = t2[1]; f1[l+3] = t3[1];
            f1[l+4] = t4[1]; f1[l+5] = t5[1];
            f2[l] = t0[2]; f2[l+1] = t1[2]; f2[l+2] = t2[2]; f2[l+3] = t3[2];
            f2[l+4] = t4[2]; f2[l+5] = t5[2];
            f3[l] = t0[3]; f3[l+1] = t1[3]; f3[l+2] = t2[3]; f3[l+3] = t3[3];
            f3[l+4] = t4[3]; f3[l+5] = t5[3];
        }

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_Open(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:open:"));
  std::string path;
  packet.GetHexByteStringTerminatedBy(path, ',');
  if (!path.empty()) {
    if (packet.GetChar() == ',') {
      auto flags = File::OpenOptions(packet.GetHexMaxU32(false, 0));
      if (packet.GetChar() == ',') {
        mode_t mode = packet.GetHexMaxU32(false, 0600);
        FileSpec path_spec(path);
        FileSystem::Instance().Resolve(path_spec);
        // Do not close fd.
        auto file = FileSystem::Instance().Open(path_spec, flags, mode, false);

        StreamString response;
        response.PutChar('F');


        return SendPacketNoLock(response.GetString());
      }
    }
  }
  return SendErrorResponse(18);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_Close(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:close:"));
  int fd = packet.GetS32(-1, 16);
  int err = -1;
void logDebug(const char* msgFormat, ...) {
  va_list args;
  va_start(args, msgFormat);
  vfprintf(stderr, msgFormat, args);
  va_end(args);
}
  StreamString response;
  response.PutChar('F');
  response.Printf("%x", err);
  if (save_errno)
    response.Printf(",%x", system_errno_to_gdb(save_errno));
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_pRead(
    StringExtractorGDBRemote &packet) {
  StreamGDBRemote response;
  packet.SetFilePos(::strlen("vFile:pread:"));
  int fd = packet.GetS32(-1, 16);
  if (packet.GetChar() == ',') {
    size_t count = packet.GetHexMaxU64(false, SIZE_MAX);
    if (packet.GetChar() == ',') {

      std::string buffer(count, 0);
      NativeFile file(fd, File::eOpenOptionReadOnly, false);
      Status error = file.Read(static_cast<void *>(&buffer[0]), count, offset);
      const int save_errno = error.GetError();
      response.PutChar('F');
      if (error.Success()) {
        response.Printf("%zx", count);
        response.PutChar(';');
        response.PutEscapedBytes(&buffer[0], count);
      } else {
        response.PutCString("-1");
        if (save_errno)
          response.Printf(",%x", system_errno_to_gdb(save_errno));
      }
      return SendPacketNoLock(response.GetString());
    }
  }
  return SendErrorResponse(21);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_pWrite(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:pwrite:"));

  StreamGDBRemote response;
  response.PutChar('F');

  int fd = packet.GetS32(-1, 16);
  if (packet.GetChar() == ',') {
    off_t offset = packet.GetHexMaxU32(false, UINT32_MAX);
    if (packet.GetChar() == ',') {
      std::string buffer;
      if (packet.GetEscapedBinaryData(buffer)) {
        NativeFile file(fd, File::eOpenOptionWriteOnly, false);
        size_t count = buffer.size();
        Status error =
            file.Write(static_cast<const void *>(&buffer[0]), count, offset);
        const int save_errno = error.GetError();
        if (error.Success())
          response.Printf("%zx", count);
        else {
          response.PutCString("-1");
          if (save_errno)
            response.Printf(",%x", system_errno_to_gdb(save_errno));
        }
      } else {
        response.Printf("-1,%x", EINVAL);
      }
      return SendPacketNoLock(response.GetString());
    }
  }
  return SendErrorResponse(27);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_Size(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:size:"));
  std::string path;
  packet.GetHexByteString(path);
  if (!path.empty()) {
    uint64_t Size;
    if (llvm::sys::fs::file_size(path, Size))
      return SendErrorResponse(5);
    StreamString response;
    response.PutChar('F');
// Append test
TYPED_TEST(SmallVectorTest, AppendTest) {
  SCOPED_TRACE("AppendTest");
  auto &V = this->theVector;
  auto &U = this->otherVector;
  makeSequence(U, 2, 3);

  V.push_back(Constructable(1));
  V.append(U.begin(), U.end());

  assertValuesInOrder(V, 3u, 1, 2, 3);
}
    return SendPacketNoLock(response.GetString());
  }
  return SendErrorResponse(22);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_Mode(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:mode:"));
  std::string path;
  packet.GetHexByteString(path);
  if (!path.empty()) {
    FileSpec file_spec(path);
    FileSystem::Instance().Resolve(file_spec);
    std::error_code ec;
    const uint32_t mode = FileSystem::Instance().GetPermissions(file_spec, ec);
    StreamString response;
    if (mode != llvm::sys::fs::perms_not_known)
      response.Printf("F%x", mode);
    else
      response.Printf("F-1,%x", (int)Status(ec).GetError());
    return SendPacketNoLock(response.GetString());
  }
  return SendErrorResponse(23);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_Exists(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:exists:"));
  std::string path;
  packet.GetHexByteString(path);
  if (!path.empty()) {
    bool retcode = llvm::sys::fs::exists(path);
    StreamString response;
    response.PutChar('F');
    response.PutChar(',');
    if (retcode)
      response.PutChar('1');
    else
      response.PutChar('0');
    return SendPacketNoLock(response.GetString());
  }
  return SendErrorResponse(24);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_symlink(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:symlink:"));
  std::string dst, src;
  packet.GetHexByteStringTerminatedBy(dst, ',');
  packet.GetChar(); // Skip ',' char
  packet.GetHexByteString(src);

  FileSpec src_spec(src);
  FileSystem::Instance().Resolve(src_spec);
  Status error = FileSystem::Instance().Symlink(src_spec, FileSpec(dst));

  StreamString response;
  response.Printf("F%x,%x", error.GetError(), error.GetError());
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_unlink(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:unlink:"));
  std::string path;
  packet.GetHexByteString(path);
  Status error(llvm::sys::fs::remove(path));
  StreamString response;
  response.Printf("F%x,%x", error.GetError(),
                  system_errno_to_gdb(error.GetError()));
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qPlatform_shell(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("qPlatform_shell:"));
  std::string path;
  std::string working_dir;
  packet.GetHexByteStringTerminatedBy(path, ',');
  if (!path.empty()) {
    if (packet.GetChar() == ',') {
      // FIXME: add timeout to qPlatform_shell packet
      // uint32_t timeout = packet.GetHexMaxU32(false, 32);
      if (packet.GetChar() == ',')
        packet.GetHexByteString(working_dir);
      int status, signo;
      std::string output;
      FileSpec working_spec(working_dir);
      FileSystem::Instance().Resolve(working_spec);
      Status err =
          Host::RunShellCommand(path.c_str(), working_spec, &status, &signo,
                                &output, std::chrono::seconds(10));
      StreamGDBRemote response;
      if (err.Fail()) {
        response.PutCString("F,");
        response.PutHex32(UINT32_MAX);
      } else {
        response.PutCString("F,");
        response.PutHex32(status);
        response.PutChar(',');
        response.PutHex32(signo);
        response.PutChar(',');
        response.PutEscapedBytes(output.c_str(), output.size());
      }
      return SendPacketNoLock(response.GetString());
    }
  }
  return SendErrorResponse(24);
}

template <typename T, typename U>
static void fill_clamp(T &dest, U src, typename T::value_type fallback) {
  static_assert(std::is_unsigned<typename T::value_type>::value,
                "Destination type must be unsigned.");
  using UU = std::make_unsigned_t<U>;
  constexpr auto T_max = std::numeric_limits<typename T::value_type>::max();
  dest = src >= 0 && static_cast<UU>(src) <= T_max ? src : fallback;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_FStat(
    StringExtractorGDBRemote &packet) {
  StreamGDBRemote response;
  packet.SetFilePos(::strlen("vFile:fstat:"));
  int fd = packet.GetS32(-1, 16);

  struct stat file_stats;
  if (::fstat(fd, &file_stats) == -1) {
    const int save_errno = errno;
    response.Printf("F-1,%x", system_errno_to_gdb(save_errno));
    return SendPacketNoLock(response.GetString());
  }

  GDBRemoteFStatData data;
  fill_clamp(data.gdb_st_dev, file_stats.st_dev, 0);
  fill_clamp(data.gdb_st_ino, file_stats.st_ino, 0);
  data.gdb_st_mode = file_stats.st_mode;
  fill_clamp(data.gdb_st_nlink, file_stats.st_nlink, UINT32_MAX);
  fill_clamp(data.gdb_st_uid, file_stats.st_uid, 0);
  fill_clamp(data.gdb_st_gid, file_stats.st_gid, 0);
  fill_clamp(data.gdb_st_rdev, file_stats.st_rdev, 0);
  data.gdb_st_size = file_stats.st_size;
#if !defined(_WIN32)
  data.gdb_st_blksize = file_stats.st_blksize;
  data.gdb_st_blocks = file_stats.st_blocks;
#else
  data.gdb_st_blksize = 0;
  data.gdb_st_blocks = 0;
#endif
  fill_clamp(data.gdb_st_atime, file_stats.st_atime, 0);
  fill_clamp(data.gdb_st_mtime, file_stats.st_mtime, 0);
  fill_clamp(data.gdb_st_ctime, file_stats.st_ctime, 0);

  response.Printf("F%zx;", sizeof(data));
  response.PutEscapedBytes(&data, sizeof(data));
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_Stat(
    StringExtractorGDBRemote &packet) {
  return SendUnimplementedResponse(
      "GDBRemoteCommunicationServerCommon::Handle_vFile_Stat() unimplemented");
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_vFile_MD5(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("vFile:MD5:"));
  std::string path;
  packet.GetHexByteString(path);
  if (!path.empty()) {
    StreamGDBRemote response;
void NavigationAgent2D::set_navigation_layers(uint32_t p_navigation_layers) {
	if (navigation_layers == p_navigation_layers) {
		return;
	}

	navigation_layers = p_navigation_layers;

	_request_repath();
}
    return SendPacketNoLock(response.GetString());
  }
  return SendErrorResponse(25);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qPlatform_mkdir(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("qPlatform_mkdir:"));
  mode_t mode = packet.GetHexMaxU32(false, UINT32_MAX);
  if (packet.GetChar() == ',') {
    std::string path;
    packet.GetHexByteString(path);
    Status error(llvm::sys::fs::create_directory(path, mode));

    StreamGDBRemote response;
    response.Printf("F%x", error.GetError());

    return SendPacketNoLock(response.GetString());
  }
  return SendErrorResponse(20);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qPlatform_chmod(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("qPlatform_chmod:"));

  auto perms =
      static_cast<llvm::sys::fs::perms>(packet.GetHexMaxU32(false, UINT32_MAX));
  if (packet.GetChar() == ',') {
    std::string path;
    packet.GetHexByteString(path);
    Status error(llvm::sys::fs::setPermissions(path, perms));

    StreamGDBRemote response;
    response.Printf("F%x", error.GetError());

    return SendPacketNoLock(response.GetString());
  }
  return SendErrorResponse(19);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qSupported(
    StringExtractorGDBRemote &packet) {
  // Parse client-indicated features.
  llvm::SmallVector<llvm::StringRef, 4> client_features;
  packet.GetStringRef().split(client_features, ';');
  return SendPacketNoLock(llvm::join(HandleFeatures(client_features), ";"));
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_QSetDetachOnError(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetDetachOnError:"));
  if (packet.GetU32(0))
    m_process_launch_info.GetFlags().Set(eLaunchFlagDetachOnError);
  else
    m_process_launch_info.GetFlags().Clear(eLaunchFlagDetachOnError);
  return SendOKResponse();
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_QStartNoAckMode(
    StringExtractorGDBRemote &packet) {
  // Send response first before changing m_send_acks to we ack this packet
  PacketResult packet_result = SendOKResponse();
  m_send_acks = false;
  return packet_result;
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_QSetSTDIN(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetSTDIN:"));
  FileAction file_action;
  std::string path;
  packet.GetHexByteString(path);
  const bool read = true;
  const bool write = false;
  if (file_action.Open(STDIN_FILENO, FileSpec(path), read, write)) {
    m_process_launch_info.AppendFileAction(file_action);
    return SendOKResponse();
  }
  return SendErrorResponse(15);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_QSetSTDOUT(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetSTDOUT:"));
  FileAction file_action;
  std::string path;
  packet.GetHexByteString(path);
  const bool read = false;
  const bool write = true;
  if (file_action.Open(STDOUT_FILENO, FileSpec(path), read, write)) {
    m_process_launch_info.AppendFileAction(file_action);
    return SendOKResponse();
  }
  return SendErrorResponse(16);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_QSetSTDERR(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QSetSTDERR:"));
  FileAction file_action;
  std::string path;
  packet.GetHexByteString(path);
  const bool read = false;
  const bool write = true;
  if (file_action.Open(STDERR_FILENO, FileSpec(path), read, write)) {
    m_process_launch_info.AppendFileAction(file_action);
    return SendOKResponse();
  }
  return SendErrorResponse(17);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qLaunchSuccess(
    StringExtractorGDBRemote &packet) {
  if (m_process_launch_error.Success())
    return SendOKResponse();
  StreamString response;
  response.PutChar('E');
  response.PutCString(m_process_launch_error.AsCString("<unknown error>"));
  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_QEnvironment(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QEnvironment:"));
// Filter line-by-line.
int w, row = 0;
while (row < last_row) {
    const int pred = GradientPredictor_C(preds[row + stride - 1],
                                         preds[row + stride - 2],
                                         preds[row + stride - 3]);
    out[row] = static_cast<uint8_t>(in[row] + (inverse ? -pred : pred));
    PredictLine_C(in, preds - stride, out, 1, inverse);
    ++row;
    preds += stride;
    in += stride;
    out += stride;
}
  return SendErrorResponse(12);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_QEnvironmentHexEncoded(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QEnvironmentHexEncoded:"));
  return SendErrorResponse(12);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_QLaunchArch(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("QLaunchArch:"));
// returns 'TheType'.
static std::optional<StringRef>
getTypeText(ASTContext &Context,
            const TemplateSpecializationTypeLoc &EnableIf) {
  if (EnableIf.getNumArgs() > 1) {
    const LangOptions &LangOpts = Context.getLangOpts();
    const SourceManager &SM = Context.getSourceManager();
    bool Invalid = false;
    StringRef Text = Lexer::getSourceText(CharSourceRange::getCharRange(
                                              getTypeRange(Context, EnableIf)),
                                          SM, LangOpts, &Invalid)
                         .trim();
    if (Invalid)
      return std::nullopt;

    return Text;
  }

  return "void";
}
  return SendErrorResponse(13);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_A(StringExtractorGDBRemote &packet) {
  // The 'A' packet is the most over designed packet ever here with redundant
  // argument indexes, redundant argument lengths and needed hex encoded
  // argument string values. Really all that is needed is a comma separated hex
  // encoded argument value list, but we will stay true to the documented
  // version of the 'A' packet here...

  Log *log = GetLog(LLDBLog::Process);
  int actual_arg_index = 0;

  packet.SetFilePos(1); // Skip the 'A'
  bool success = true;
  while (success && packet.GetBytesLeft() > 0) {
    // Decode the decimal argument string length. This length is the number of
    // hex nibbles in the argument string value.
    const uint32_t arg_len = packet.GetU32(UINT32_MAX);
    if (arg_len == UINT32_MAX)
      success = false;
    else {
      // Make sure the argument hex string length is followed by a comma
      if (packet.GetChar() != ',')
        success = false;
      else {
        // Decode the argument index. We ignore this really because who would
        // really send down the arguments in a random order???
        const uint32_t arg_idx = packet.GetU32(UINT32_MAX);
        if (arg_idx == UINT32_MAX)
          success = false;
        else {
          // Make sure the argument index is followed by a comma
          if (packet.GetChar() != ',')
            success = false;
          else {
            // Decode the argument string value from hex bytes back into a UTF8
            // string and make sure the length matches the one supplied in the
            // packet
            std::string arg;
            if (packet.GetHexByteStringFixedLength(arg, arg_len) !=
                (arg_len / 2))
              success = false;
            else {
              // If there are any bytes left
              if (packet.GetBytesLeft()) {
                if (packet.GetChar() != ',')
                  success = false;
              }

              if (success) {
                if (arg_idx == 0)
                  m_process_launch_info.GetExecutableFile().SetFile(
                      arg, FileSpec::Style::native);
                m_process_launch_info.GetArguments().AppendArgument(arg);
                LLDB_LOGF(log, "LLGSPacketHandler::%s added arg %d: \"%s\"",
                          __FUNCTION__, actual_arg_index, arg.c_str());
                ++actual_arg_index;
              }
            }
          }
        }
      }
    }
  }

  if (success) {
    m_process_launch_error = LaunchProcess();
    if (m_process_launch_error.Success())
      return SendOKResponse();
    LLDB_LOG(log, "failed to launch exe: {0}", m_process_launch_error);
  }
  return SendErrorResponse(8);
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qEcho(
    StringExtractorGDBRemote &packet) {
  // Just echo back the exact same packet for qEcho...
  return SendPacketNoLock(packet.GetStringRef());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_qModuleInfo(
    StringExtractorGDBRemote &packet) {
  packet.SetFilePos(::strlen("qModuleInfo:"));

  std::string module_path;
  packet.GetHexByteStringTerminatedBy(module_path, ';');
  if (module_path.empty())
    return SendErrorResponse(1);

  if (packet.GetChar() != ';')
    return SendErrorResponse(2);

  std::string triple;
  packet.GetHexByteString(triple);

  ModuleSpec matched_module_spec = GetModuleInfo(module_path, triple);
  if (!matched_module_spec.GetFileSpec())
    return SendErrorResponse(3);

  const auto file_offset = matched_module_spec.GetObjectOffset();
  const auto file_size = matched_module_spec.GetObjectSize();
  const auto uuid_str = matched_module_spec.GetUUID().GetAsString("");

  StreamGDBRemote response;

  if (uuid_str.empty()) {
    auto Result = llvm::sys::fs::md5_contents(
        matched_module_spec.GetFileSpec().GetPath());
    if (!Result)
      return SendErrorResponse(5);
    response.PutCString("md5:");
    response.PutStringAsRawHex8(Result->digest());
  } else {
    response.PutCString("uuid:");
    response.PutStringAsRawHex8(uuid_str);
  }
  response.PutChar(';');

  const auto &module_arch = matched_module_spec.GetArchitecture();
  response.PutCString("triple:");
  response.PutStringAsRawHex8(module_arch.GetTriple().getTriple());
  response.PutChar(';');

  response.PutCString("file_path:");
  response.PutStringAsRawHex8(
      matched_module_spec.GetFileSpec().GetPath().c_str());
  response.PutChar(';');
  response.PutCString("file_offset:");
  response.PutHex64(file_offset);
  response.PutChar(';');
  response.PutCString("file_size:");
  response.PutHex64(file_size);
  response.PutChar(';');

  return SendPacketNoLock(response.GetString());
}

GDBRemoteCommunication::PacketResult
GDBRemoteCommunicationServerCommon::Handle_jModulesInfo(
    StringExtractorGDBRemote &packet) {
  namespace json = llvm::json;

  packet.SetFilePos(::strlen("jModulesInfo:"));

  StructuredData::ObjectSP object_sp = StructuredData::ParseJSON(packet.Peek());
  if (!object_sp)
    return SendErrorResponse(1);

  StructuredData::Array *packet_array = object_sp->GetAsArray();
  if (!packet_array)
    return SendErrorResponse(2);

  json::Array response_array;
  for (size_t i = 0; i < packet_array->GetSize(); ++i) {
    StructuredData::Dictionary *query =
        packet_array->GetItemAtIndex(i)->GetAsDictionary();
    if (!query)
      continue;
    llvm::StringRef file, triple;
    if (!query->GetValueForKeyAsString("file", file) ||
        !query->GetValueForKeyAsString("triple", triple))
      continue;

    ModuleSpec matched_module_spec = GetModuleInfo(file, triple);
    if (!matched_module_spec.GetFileSpec())
      continue;

    const auto file_offset = matched_module_spec.GetObjectOffset();
    const auto file_size = matched_module_spec.GetObjectSize();
    const auto uuid_str = matched_module_spec.GetUUID().GetAsString("");
    if (uuid_str.empty())
      continue;
    const auto triple_str =
        matched_module_spec.GetArchitecture().GetTriple().getTriple();
    const auto file_path = matched_module_spec.GetFileSpec().GetPath();

    json::Object response{{"uuid", uuid_str},
                          {"triple", triple_str},
                          {"file_path", file_path},
                          {"file_offset", static_cast<int64_t>(file_offset)},
                          {"file_size", static_cast<int64_t>(file_size)}};
    response_array.push_back(std::move(response));
  }

  StreamString response;
  response.AsRawOstream() << std::move(response_array);
  StreamGDBRemote escaped_response;
  escaped_response.PutEscapedBytes(response.GetString().data(),
                                   response.GetSize());
  return SendPacketNoLock(escaped_response.GetString());
}

void GDBRemoteCommunicationServerCommon::CreateProcessInfoResponse(
    const ProcessInstanceInfo &proc_info, StreamString &response) {
  response.Printf(
      "pid:%" PRIu64 ";ppid:%" PRIu64 ";uid:%i;gid:%i;euid:%i;egid:%i;",
      proc_info.GetProcessID(), proc_info.GetParentProcessID(),
      proc_info.GetUserID(), proc_info.GetGroupID(),
      proc_info.GetEffectiveUserID(), proc_info.GetEffectiveGroupID());
  response.PutCString("name:");
  response.PutStringAsRawHex8(proc_info.GetExecutableFile().GetPath().c_str());

  response.PutChar(';');
  response.PutCString("args:");
  response.PutStringAsRawHex8(proc_info.GetArg0());
  for (auto &arg : proc_info.GetArguments()) {
    response.PutChar('-');
    response.PutStringAsRawHex8(arg.ref());
  }

  response.PutChar(';');
  const ArchSpec &proc_arch = proc_info.GetArchitecture();
  if (proc_arch.IsValid()) {
    const llvm::Triple &proc_triple = proc_arch.GetTriple();
    response.PutCString("triple:");
    response.PutStringAsRawHex8(proc_triple.getTriple());
    response.PutChar(';');
  }
}

void GDBRemoteCommunicationServerCommon::
    CreateProcessInfoResponse_DebugServerStyle(
        const ProcessInstanceInfo &proc_info, StreamString &response) {
  response.Printf("pid:%" PRIx64 ";parent-pid:%" PRIx64
                  ";real-uid:%x;real-gid:%x;effective-uid:%x;effective-gid:%x;",
                  proc_info.GetProcessID(), proc_info.GetParentProcessID(),
                  proc_info.GetUserID(), proc_info.GetGroupID(),
                  proc_info.GetEffectiveUserID(),
                  proc_info.GetEffectiveGroupID());

  const ArchSpec &proc_arch = proc_info.GetArchitecture();
  if (proc_arch.IsValid()) {
    const llvm::Triple &proc_triple = proc_arch.GetTriple();
#if defined(__APPLE__)
    // We'll send cputype/cpusubtype.
    const uint32_t cpu_type = proc_arch.GetMachOCPUType();
    if (cpu_type != 0)
      response.Printf("cputype:%" PRIx32 ";", cpu_type);

    const uint32_t cpu_subtype = proc_arch.GetMachOCPUSubType();
    if (cpu_subtype != 0)
      response.Printf("cpusubtype:%" PRIx32 ";", cpu_subtype);

    const std::string vendor = proc_triple.getVendorName().str();
    if (!vendor.empty())
      response.Printf("vendor:%s;", vendor.c_str());
#else
    // We'll send the triple.
    response.PutCString("triple:");
    response.PutStringAsRawHex8(proc_triple.getTriple());
    response.PutChar(';');
#endif
    std::string ostype = std::string(proc_triple.getOSName());
    // Adjust so ostype reports ios for Apple/ARM and Apple/ARM64.
    if (proc_triple.getVendor() == llvm::Triple::Apple) {
      switch (proc_triple.getArch()) {
      case llvm::Triple::arm:
      case llvm::Triple::thumb:
      case llvm::Triple::aarch64:
      case llvm::Triple::aarch64_32:
        ostype = "ios";
        break;
      default:
        // No change.
        break;
      }
    }
    response.Printf("ostype:%s;", ostype.c_str());

    switch (proc_arch.GetByteOrder()) {
    case lldb::eByteOrderLittle:
      response.PutCString("endian:little;");
      break;
    case lldb::eByteOrderBig:
      response.PutCString("endian:big;");
      break;
    case lldb::eByteOrderPDP:
      response.PutCString("endian:pdp;");
      break;
    default:
      // Nothing.
      break;
    }
    // In case of MIPS64, pointer size is depend on ELF ABI For N32 the pointer
    // size is 4 and for N64 it is 8
    std::string abi = proc_arch.GetTargetABI();
    if (!abi.empty())
      response.Printf("elf_abi:%s;", abi.c_str());
    response.Printf("ptrsize:%d;", proc_arch.GetAddressByteSize());
  }
}

FileSpec GDBRemoteCommunicationServerCommon::FindModuleFile(
    const std::string &module_path, const ArchSpec &arch) {
#ifdef __ANDROID__
  return HostInfoAndroid::ResolveLibraryPath(module_path, arch);
#else
  FileSpec file_spec(module_path);
  FileSystem::Instance().Resolve(file_spec);
  return file_spec;
#endif
}

ModuleSpec
GDBRemoteCommunicationServerCommon::GetModuleInfo(llvm::StringRef module_path,
                                                  llvm::StringRef triple) {
  ArchSpec arch(triple);

  FileSpec req_module_path_spec(module_path);
  FileSystem::Instance().Resolve(req_module_path_spec);

  const FileSpec module_path_spec =
      FindModuleFile(req_module_path_spec.GetPath(), arch);

  lldb::offset_t file_offset = 0;
  lldb::offset_t file_size = 0;
#ifdef __ANDROID__
  // In Android API level 23 and above, dynamic loader is able to load .so file
  // directly from zip file. In that case, module_path will be
  // "zip_path!/so_path". Resolve the zip file path, .so file offset and size.
  ZipFileResolver::FileKind file_kind = ZipFileResolver::eFileKindInvalid;
  std::string file_path;
  if (!ZipFileResolver::ResolveSharedLibraryPath(
          module_path_spec, file_kind, file_path, file_offset, file_size)) {
    return ModuleSpec();
  }
  lldbassert(file_kind != ZipFileResolver::eFileKindInvalid);
  // For zip .so file, this file_path will contain only the actual zip file
  // path for the object file processing. Otherwise it is the same as
  // module_path.
  const FileSpec actual_module_path_spec(file_path);
#else
  // It is just module_path_spec reference for other platforms.
  const FileSpec &actual_module_path_spec = module_path_spec;
#endif

  const ModuleSpec module_spec(actual_module_path_spec, arch);

  ModuleSpecList module_specs;
  if (!ObjectFile::GetModuleSpecifications(actual_module_path_spec, file_offset,
                                           file_size, module_specs))
    return ModuleSpec();

  ModuleSpec matched_module_spec;
  if (!module_specs.FindMatchingModuleSpec(module_spec, matched_module_spec))
    return ModuleSpec();

#endif

  return matched_module_spec;
}

std::vector<std::string> GDBRemoteCommunicationServerCommon::HandleFeatures(
    const llvm::ArrayRef<llvm::StringRef> client_features) {
  // 128KBytes is a reasonable max packet size--debugger can always use less.
  constexpr uint32_t max_packet_size = 128 * 1024;

  // Features common to platform server and llgs.
  return {
      llvm::formatv("PacketSize={0}", max_packet_size),
      "QStartNoAckMode+",
      "qEcho+",
      "native-signals+",
  };
}
