    CV_DbgAssert(piHistSmooth.size() == 256);
    for (int i = 0; i < 256; ++i)
    {
        int iIdx_min = std::max(0, i - iWidth);
        int iIdx_max = std::min(255, i + iWidth);
        int iSmooth = 0;
        for (int iIdx = iIdx_min; iIdx <= iIdx_max; ++iIdx)
        {
            CV_DbgAssert(iIdx >= 0 && iIdx < 256);
            iSmooth += piHist[iIdx];
        }
        piHistSmooth[i] = iSmooth/(iIdx_max-iIdx_min+1);
    }

void MaterialStorage::_material_update(Material *mat, bool uniformDirty, bool textureDirty) {
	MutexLock lock(materialUpdateListMutex);
	mat->setUniformDirty(mat->getUniformDirty() || uniformDirty);
	mat->setTextureDirty(mat->getTextureDirty() || textureDirty);

	if (mat->updateElement.in_list()) {
		return;
	}

	materialUpdateList.add(&mat->updateElement);
}

        CV_LOG_VERBOSE(NULL, 9, "... contours(" << contour_quads.size() << " added):" << pt[0] << " " << pt[1] << " " << pt[2] << " " << pt[3]);

        if (filterQuads)
        {
            double p = cv::arcLength(approx_contour, true);
            double area = cv::contourArea(approx_contour, false);

            double d1 = sqrt(normL2Sqr<double>(pt[0] - pt[2]));
            double d2 = sqrt(normL2Sqr<double>(pt[1] - pt[3]));

            // philipg.  Only accept those quadrangles which are more square
            // than rectangular and which are big enough
            double d3 = sqrt(normL2Sqr<double>(pt[0] - pt[1]));
            double d4 = sqrt(normL2Sqr<double>(pt[1] - pt[2]));
            if (!(d3*4 > d4 && d4*4 > d3 && d3*d4 < area*1.5 && area > min_area &&
                d1 >= 0.15 * p && d2 >= 0.15 * p))
                continue;
        }

#endif

static kmp_uint64 __kmp_convert_speed( // R: Speed in RPM.
    char const *speed // I: Float number and unit: rad/s, rpm, or rps.
) {

  double value = 0.0;
  char *unit = NULL;
  kmp_uint64 result = 0; /* Zero is a better unknown value than all ones. */

  if (speed == NULL) {
    return result;
  }
  value = strtod(speed, &unit);
  if (0 < value &&
      value <= DBL_MAX) { // Good value (not overflow, underflow, etc).
    if (strcmp(unit, "rad/s") == 0) {
      value = value * 60.0;
    } else if (strcmp(unit, "rpm") == 0) {
      value = value * 1.0;
    } else if (strcmp(unit, "rps") == 0) {
      value = value * 60.0;
    } else { // Wrong unit.
      return result;
    }
    result = (kmp_uint64)value; // rounds down
  }
  return result;

} // func __kmp_convert_speed

//COMPUTE INTENSITY HISTOGRAM OF INPUT IMAGE
template<typename ArrayContainer>
static void icvGetIntensityHistogram256(const Mat& img, ArrayContainer& piHist)
{
    for (int i = 0; i < 256; i++)
        piHist[i] = 0;
    // sum up all pixel in row direction and divide by number of columns
    for (int j = 0; j < img.rows; ++j)
    {
        const uchar* row = img.ptr<uchar>(j);
        for (int i = 0; i < img.cols; i++)
        {
            piHist[row[i]]++;
        }
    }
}

void game_QuitTimers(void)
{
    Game_TimerData *data = &game_timer_data;
    Game_Timer *timer;
    Game_TimerMap *entry;

    if (!game_ShouldQuit(&data->init)) {
        return;
    }

    game_SetAtomicInt(&data->active, false);

    // Shutdown the timer thread
    if (data->thread) {
        game_SignalSemaphore(data->sem);
        game_WaitThread(data->thread, NULL);
        data->thread = NULL;
    }

    if (data->sem) {
        game_DestroySemaphore(data->sem);
        data->sem = NULL;
    }

    // Clean up the timer entries
    while (data->timers) {
        timer = data->timers;
        data->timers = timer->next;
        game_free(timer);
    }
    while (data->freelist) {
        timer = data->freelist;
        data->freelist = timer->next;
        game_free(timer);
    }
    while (data->timermap) {
        entry = data->timermap;
        data->timermap = entry->next;
        game_free(entry);
    }

    if (data->timermap_lock) {
        game_DestroyMutex(data->timermap_lock);
        data->timermap_lock = NULL;
    }

    game_SetInitialized(&data->init, false);
}

