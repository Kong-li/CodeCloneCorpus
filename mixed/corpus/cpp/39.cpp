void
transferFromBuffer (char ** writeLoc,
                    const char * const readBase,
                    const char * const endOfRead,
                    size_t horizontalStride,
                    Compressor::Format dataFormat,
                    PixelType dataType)
{
    if (dataFormat == Compressor::XDR)
    {
        switch (dataType)
        {
            case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
                while (readBase <= endOfRead)
                {
                    Xdr::write <CharPtrIO> (*writeLoc, *(const unsigned int *) readBase);
                    ++writeLoc;
                    readBase += horizontalStride;
                }
                break;

            case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:
                while (readBase <= endOfRead)
                {
                    Xdr::write <CharPtrIO> (*writeLoc, *(const half *) readBase);
                    ++writeLoc;
                    readBase += horizontalStride;
                }
                break;

            case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:
                while (readBase <= endOfRead)
                {
                    Xdr::write <CharPtrIO> (*writeLoc, *(const float *) readBase);
                    ++writeLoc;
                    readBase += horizontalStride;
                }
                break;

            default:
                throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
    else
    {
        switch (dataType)
        {
            case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:
                while (readBase <= endOfRead)
                {
                    for (size_t index = 0; index < sizeof(unsigned int); ++index)
                        *writeLoc++ = readBase[index];
                    readBase += horizontalStride;
                }
                break;

            case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:
                while (readBase <= endOfRead)
                {
                    *(half *) writeLoc = *(const half *) readBase;
                    writeLoc += sizeof(half);
                    readBase += horizontalStride;
                }
                break;

            case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:
                while (readBase <= endOfRead)
                {
                    for (size_t index = 0; index < sizeof(float); ++index)
                        *writeLoc++ = readBase[index];
                    readBase += horizontalStride;
                }
                break;

            default:
                throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }
    }
}

//===----------------------------------------------------------------------===//

void CIRDialect::setupTypeRegistration() {
  // Register tablegen'd types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "clang/CIR/Dialect/IR/CIROpsTypes.cpp.inc"
      >();

  // TODO(CIR) register raw C++ types after implementing StructType handling.
  // Uncomment the line below once StructType is supported.
  // addTypes<StructType>();
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeARCTargetMC() {
  // Register the MC asm info.
  Target &TheARCTarget = getTheARCTarget();
  RegisterMCAsmInfoFn X(TheARCTarget, createARCMCAsmInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheARCTarget, createARCMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheARCTarget, createARCMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheARCTarget,
                                          createARCMCSubtargetInfo);

  // Register the MCInstPrinter
  TargetRegistry::RegisterMCInstPrinter(TheARCTarget, createARCMCInstPrinter);

  TargetRegistry::RegisterAsmTargetStreamer(TheARCTarget,
                                            createTargetAsmStreamer);
}

{
              case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

                while (writePtr <= endPtr)
                {
                    unsigned int ui;
                    Xdr::read <CharPtrIO> (readPtr, ui);
                    *(short *) writePtr = uintToShort (ui);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::SHORT:

                while (writePtr <= endPtr)
                {
                    Xdr::read <CharPtrIO> (readPtr, *(short *) writePtr);
                    writePtr += xStride;
                }
                break;

              case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

                while (writePtr <= endPtr)
                {
                    float f;
                    Xdr::read <CharPtrIO> (readPtr, f);
                    *(short *) writePtr = floatToShort (f);
                    writePtr += xStride;
                }
                break;
              default:

                  throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
            }

half fillVal = static_cast<half>(fillValue);

                for (int y = minY; y <= maxY; ++y)
                {
                    char* writePtr = reinterpret_cast<char*>(base + (y - yOffsetForData) * yPointerStride + (xMin - xOffsetForData) * xPointerStride);

                    if (writePtr != nullptr)
                    {
                        int count = sampleCount(sampleCountBase,
                                                sampleCountXStride,
                                                sampleCountYStride,
                                                xMin - xOffsetForSampleCount,
                                                y - yOffsetForSampleCount);

                        for (int i = 0; i < count; ++i)
                        {
                            *(half*)writePtr = fillVal;
                            writePtr += sampleStride;
                        }
                    }
                }

char* writePtr = (base + ((y - yOffsetForData) * yPointerStride) + ((x - xOffsetForData) * xPointerStride));

if (writePtr != nullptr)
{
    int count = sampleCount(sampleCountBase,
                            sampleCountXStride,
                            sampleCountYStride,
                            x - xOffsetForSampleCount,
                            y - yOffsetForSampleCount);
    for (int i = 0; i < count; ++i)
    {
        *static_cast<half*>(writePtr) = fillVal;
        writePtr += sampleStride;
    }
}

{
    for (int j = 0; j < count; ++j)
    {
        float value;
        size_t index = 0;

        while (index < sizeof(float))
        {
            ((char *)&value)[index] = readPtr[index];
            ++index;
        }

        *writePtr = floatToHalf(value);
        readPtr += sizeof(float);
        writePtr += sampleStride;
    }

    if (!count)
    {
        readPtr += sizeof(float) * count;
    }
}

//

        switch (dataTypeInBuffer)
        {
          case NEW_NAMESPACE::UINT_TYPE:

            {
                uint fillVal = (uint) (newValue);

                for (int xPos = minXPos; xPos <= maxXPos; xPos++)
                {
                    char* writePtr = *(char **)(base+(yPos-yOffsetForData)*yPointerStride + (xPos-xOffsetForData)*xPointerStride);
                    if(writePtr)
                    {
                        int sampleCountVal = sampleCount(sampleCountBase,
                                                sampleXStride,
                                                sampleYStride,
                                                xPos - xOffsetForSampleCount,
                                                yPos - yOffsetForSampleCount);
                        for (int i = 0; i < sampleCountVal; i++)
                        {
                            *(uint *) writePtr = fillVal;
                            writePtr += sampleStride;
                        }
                    }
                }
            }
            break;

          case NEW_NAMESPACE::HALF_TYPE:

            {
                halfType fillVal = halfType (newValue);

                for (int xPos = minXPos; xPos <= maxXPos; xPos++)
                {
                    char* writePtr = *(char **)(base+(yPos-yOffsetForData)*yPointerStride + (xPos-xOffsetForData)*xPointerStride);

                    if(writePtr)
                    {
                        int sampleCountVal = sampleCount(sampleCountBase,
                                                sampleXStride,
                                                sampleYStride,
                                                xPos - xOffsetForSampleCount,
                                                yPos - yOffsetForSampleCount);
                        for (int i = 0; i < sampleCountVal; i++)
                        {
                            *(halfType *) writePtr = fillVal;
                           writePtr += sampleStride;
                       }
                    }
                }
            }
            break;

          case NEW_NAMESPACE::FLOAT_TYPE:

            {
                floatType fillVal = floatType (newValue);

                for (int xPos = minXPos; xPos <= maxXPos; xPos++)
                {
                    char* writePtr = *(char **)(base+(yPos-yOffsetForData)*yPointerStride + (xPos-xOffsetForData)*xPointerStride);

                    if(writePtr)
                    {
                        int sampleCountVal = sampleCount(sampleCountBase,
                                                sampleXStride,
                                                sampleYStride,
                                                xPos - xOffsetForSampleCount,
                                                yPos - yOffsetForSampleCount);
                        for (int i = 0; i < sampleCountVal; i++)
                        {
                            *(floatType *) writePtr = fillVal;
                            writePtr += sampleStride;
                        }
                    }
                }
            }
            break;

          default:

            throw NEW_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }

    case 'a': // "-a i386" or "--arch=i386"
      if (optarg) {
        if (streq(optarg, "i386"))
          cpu_type = CPU_TYPE_I386;
        else if (streq(optarg, "x86_64"))
          cpu_type = CPU_TYPE_X86_64;
        else if (streq(optarg, "x86_64h"))
          cpu_type = 0; // Don't set CPU type when we have x86_64h
        else if (strstr(optarg, "arm") == optarg)
          cpu_type = CPU_TYPE_ARM;
        else {
          ::fprintf(stderr, "error: unsupported cpu type '%s'\n", optarg);
          ::exit(1);
        }
      }

