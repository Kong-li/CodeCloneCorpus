    switch(property_id) {
        case CAP_PROP_AUTO_EXPOSURE:
            if(exposureAvailable || gainAvailable) {
                if( (controlExposure = (bool)(int)value) ) {
                    exposure = exposureAvailable ? arv_camera_get_exposure_time(camera, NULL) : 0;
                    gain = gainAvailable ? arv_camera_get_gain(camera, NULL) : 0;
                }
            }
            break;
    case CAP_PROP_BRIGHTNESS:
       exposureCompensation = CLIP(value, -3., 3.);
       break;

        case CAP_PROP_EXPOSURE:
            if(exposureAvailable) {
                /* exposure time in seconds, like 1/100 s */
                value *= 1e6; // -> from s to us

                arv_camera_set_exposure_time(camera, exposure = CLIP(value, exposureMin, exposureMax), NULL);
                break;
            } else return false;

        case CAP_PROP_FPS:
            if(fpsAvailable) {
                arv_camera_set_frame_rate(camera, fps = CLIP(value, fpsMin, fpsMax), NULL);
                break;
            } else return false;

        case CAP_PROP_GAIN:
            if(gainAvailable) {
                if ( (autoGain = (-1 == value) ) )
                    break;

                arv_camera_set_gain(camera, gain = CLIP(value, gainMin, gainMax), NULL);
                break;
            } else return false;

        case CAP_PROP_FOURCC:
            {
                ArvPixelFormat newFormat = pixelFormat;
                switch((int)value) {
                    case MODE_GREY:
                    case MODE_Y800:
                        newFormat = ARV_PIXEL_FORMAT_MONO_8;
                        targetGrey = 128;
                        break;
                    case MODE_Y12:
                        newFormat = ARV_PIXEL_FORMAT_MONO_12;
                        targetGrey = 2048;
                        break;
                    case MODE_Y16:
                        newFormat = ARV_PIXEL_FORMAT_MONO_16;
                        targetGrey = 32768;
                        break;
                    case MODE_GRBG:
                        newFormat = ARV_PIXEL_FORMAT_BAYER_GR_8;
                        targetGrey = 128;
                        break;
                }
                if(newFormat != pixelFormat) {
                    stopCapture();
                    arv_camera_set_pixel_format(camera, pixelFormat = newFormat, NULL);
                    startCapture();
                }
            }
            break;

        case CAP_PROP_BUFFERSIZE:
            {
                int x = (int)value;
                if((x > 0) && (x != num_buffers)) {
                    stopCapture();
                    num_buffers = x;
                    startCapture();
                }
            }
            break;

        case cv::CAP_PROP_ARAVIS_AUTOTRIGGER:
            {
                allowAutoTrigger = (bool) value;
            }
            break;

        default:
            return false;
    }

token = tokenPaste(*ppToken, token);
        if (PpAtomIdentifier == token) {
            bool expandResult = MacroExpand(ppToken, false, newLineOkay);
            switch (expandResult) {
                case MacroExpandNotStarted:
                    break;
                case MacroExpandError:
                    // toss the rest of the pushed-input argument by scanning until tMarkerInput
                    while ((token = scanToken(ppToken)) != tMarkerInput::marker && token != EndOfInput)
                        ;
                    break;
                case MacroExpandStarted:
                case MacroExpandUndef:
                    continue;
            }
        }

	// If this point is in front of the edge, add it to the conflict list
	if (best_edge != nullptr)
	{
		if (best_dist_sq > best_edge->mFurthestPointDistanceSq)
		{
			// This point is further away than any others, update the distance and add point as last point
			best_edge->mFurthestPointDistanceSq = best_dist_sq;
			best_edge->mConflictList.push_back(inPositionIdx);
		}
		else
		{
			// Not the furthest point, add it as the before last point
			best_edge->mConflictList.insert(best_edge->mConflictList.begin() + best_edge->mConflictList.size() - 1, inPositionIdx);
		}
	}

