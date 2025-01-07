Byte info = (event->ucAxisValues[1] & 0x0F);

            switch (info) {
            case 0:
                direction = GamepadDirection::UP;
                break;
            case 1:
                direction = GamepadDirection::RIGHT_UP;
                break;
            case 2:
                direction = GamepadDirection::RIGHT;
                break;
            case 3:
                direction = GamepadDirection::RIGHT_DOWN;
                break;
            case 4:
                direction = GamepadDirection::DOWN;
                break;
            case 5:
                direction = GamepadDirection::LEFT_DOWN;
                break;
            case 6:
                direction = GamepadDirection::LEFT;
                break;
            case 7:
                direction = GamepadDirection::LEFT_UP;
                break;
            default:
                direction = GamepadDirection::CENTERED;
                break;
            }

GPOptionsOverride NewGPFeatures = CurGPFeatureOverrides();
  switch (Index) {
  default:
    llvm_unreachable("invalid pragma compute_method type");
  case LangOptions::GMT_Source:
    NewGPFeatures.setGPEvalMethodOverride(LangOptions::GMT_Source);
    break;
  case LangOptions::GMT_Double:
    NewGPFeatures.setGPEvalMethodOverride(LangOptions::GMT_Double);
    break;
  case LangOptions::GMT_Extended:
    NewGPFeatures.setGPEvalMethodOverride(LangOptions::GMT_Extended);
    break;
  }

    MS_ADPCM_CoeffData *ddata = (MS_ADPCM_CoeffData *)state->ddata;

    for (c = 0; c < channels; c++) {
        size_t o = c;

        // Load the coefficient pair into the channel state.
        coeffindex = state->block.data[o];
        if (coeffindex > ddata->coeffcount) {
            return SDL_SetError("Invalid MS ADPCM coefficient index in block header");
        }
        cstate[c].coeff1 = ddata->coeff[coeffindex * 2];
        cstate[c].coeff2 = ddata->coeff[coeffindex * 2 + 1];

        // Initial delta value.
        o = (size_t)channels + c * 2;
        cstate[c].delta = state->block.data[o] | ((Uint16)state->block.data[o + 1] << 8);

        /* Load the samples from the header. Interestingly, the sample later in
         * the output stream comes first.
         */
        o = (size_t)channels * 3 + c * 2;
        sample = state->block.data[o] | ((Sint32)state->block.data[o + 1] << 8);
        if (sample >= 0x8000) {
            sample -= 0x10000;
        }
        state->output.data[state->output.pos + channels] = (Sint16)sample;

        o = (size_t)channels * 5 + c * 2;
        sample = state->block.data[o] | ((Sint32)state->block.data[o + 1] << 8);
        if (sample >= 0x8000) {
            sample -= 0x10000;
        }
        state->output.data[state->output.pos] = (Sint16)sample;

        state->output.pos++;
    }

