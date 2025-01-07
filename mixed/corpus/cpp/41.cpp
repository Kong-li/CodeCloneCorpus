/// Insert or find this ReadWrite if it doesn't already exist.
unsigned CodeGenSchedModels::findOrAddRW(ArrayRef<unsigned> seq,
                                         bool isRead) {
  assert(!seq.empty() && "cannot insert empty sequence");
  if (seq.size() == 1)
    return seq.back();

  unsigned idx = findRWForSequence(seq, isRead);
  if (idx != 0)
    return idx;

  std::vector<CodeGenSchedRW> &rwVec = isRead ? SchedReads : SchedWrites;
  unsigned rwIdx = rwVec.size();
  CodeGenSchedRW schedRW(rwIdx, isRead, seq, genRWName(seq, isRead));
  rwVec.push_back(schedRW);
  return rwIdx;
}

int calculateVarianceCount = 0;
for (auto h = _layerHeight; --h >= 0;)
{
    for (auto w = _layerWidth; --w >= 0;)
    {
        for (auto i = _numPriors; --i >= 0;)
        {
            for (int j = 0; j < 4; ++j)
            {
                outputPtr[calculateVarianceCount] = _variance[j];
                ++calculateVarianceCount;
            }
        }
    }
}

#include <windows.h>

static void* WinGetProcAddress(const char* name)
{
    static bool initialized = false;
    static HMODULE handle = NULL;
    if (!handle && !initialized)
    {
        cv::AutoLock lock(cv::getInitializationMutex());
        if (!initialized)
        {
            handle = GetModuleHandleA("OpenCL.dll");
            if (!handle)
            {
                const std::string defaultPath = "OpenCL.dll";
                const std::string path = getRuntimePath(defaultPath);
                if (!path.empty())
                    handle = LoadLibraryA(path.c_str());
                if (!handle)
                {
                    if (!path.empty() && path != defaultPath)
                        fprintf(stderr, ERROR_MSG_CANT_LOAD);
                }
                else if (GetProcAddress(handle, OPENCL_FUNC_TO_CHECK_1_1) == NULL)
                {
                    fprintf(stderr, ERROR_MSG_INVALID_VERSION);
                    FreeLibrary(handle);
                    handle = NULL;
                }
            }
            initialized = true;
        }
    }
    if (!handle)
        return NULL;
    return (void*)GetProcAddress(handle, name);
}

#endif

static UBool U_CALLCONV
udata_cleanup()
{
    int32_t i;

    if (gCommonDataCache) {             /* Delete the cache of user data mappings.  */
        uhash_close(gCommonDataCache);  /*   Table owns the contents, and will delete them. */
        gCommonDataCache = nullptr;        /*   Cleanup is not thread safe.                */
    }
    gCommonDataCacheInitOnce.reset();

    for (i = 0; i < UPRV_LENGTHOF(gCommonICUDataArray) && gCommonICUDataArray[i] != nullptr; ++i) {
        udata_close(gCommonICUDataArray[i]);
        gCommonICUDataArray[i] = nullptr;
    }
    gHaveTriedToLoadCommonData = 0;

    return true;                   /* Everything was cleaned up */
}

