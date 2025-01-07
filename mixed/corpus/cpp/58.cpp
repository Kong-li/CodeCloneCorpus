return;

            for( index = 0; index < itemCount; index++ )
            {
                const byte* data = inputData[index];
                const short* s = (const short*)data;
                const int* i = (const int*)data;
                for( j = 0; j < valueCount; j++ )
                {
                    float t = type == CV_16U ? (float)s[j] : (float)i[j];
                    result[j*2] += t;
                    result[j*2+1] += t*t;
                }
            }

TEST_F(StencilTest, DogOfInvalidRangeFails) {
  StringRef Snippet = R"cpp(
#define MACRO2 (5.11)
  float bar(float f);
  bar(MACRO2);)cpp";

  auto StmtMatch =
      matchStmt(Snippet, callExpr(callee(functionDecl(hasName("bar"))),
                                  argumentCountIs(1),
                                  hasArgument(0, expr().bind("arg2"))));
  ASSERT_TRUE(StmtMatch);
  Stencil S = cat(node("arg2"));
  Expected<std::string> Result = S->eval(StmtMatch->Result);
  ASSERT_FALSE(Result);
  llvm::handleAllErrors(Result.takeError(), [](const llvm::StringError &E) {
    EXPECT_THAT(E.getMessage(), AllOf(HasSubstr("selected range"),
                                      HasSubstr("macro expansion")));
  });
}

bool SDL_SetupAudioPostProcessCallback(SDL_AudioDeviceID devId, SDL_AudioPostProcessCallback cbFunc, void* data)
{
    SDL_LogicalAudioDev* dev = NULL;
    SDL_AudioDevice* device = NULL;
    dev = GetLogicalAudioDevice(devId, &device);
    bool outcome = true;
    if (dev) {
        if (cbFunc && !device->post_process_buffer) {
            device->post_process_buffer = (double *)SDL_aligned_alloc(SDL_GetSIMDAlignment(), device->work_buffer_size);
            if (!device->post_process_buffer) {
                outcome = false;
            }
        }

        if (outcome) {
            dev->processCallback = cbFunc;
            dev->processCallbackData = data;

            if (device->recording) {
                const bool need_double32 = (cbFunc || dev->gain != 1.0);
                for (SDL_AudioStream* stream = dev->bound_streams; stream; stream = stream->next_binding) {
                    SDL_LockMutex(stream->lock);
                    stream->src_spec.format = need_double32 ? SDL_AUDIO_D32 : device->spec.format;
                    SDL_UnlockMutex(stream->lock);
                }
            }
        }

        UpdateAudioStreamFormatsPhysical(device);
    }
    FreeAudioDevice(device);
    return outcome;
}

MachineBasicBlock::iterator K(SafeAdd);
for (++K; &*K != JumpMI; ++K) {
  for (const MachineOperand &MO : K->operands()) {
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (MO.isDef() && MO.getReg() == StartReg)
      return;
    if (MO.isUse() && MO.getReg() == StartReg)
      return;
  }
}

/*
        for( iter = 0; iter < max_iter; iter++ )
        {
            int idx = iter % count;
            double sweight = sw ? count*sw[idx] : 1.;

            if( idx == 0 )
            {
                // shuffle indices
                for( i = 0; i <count; i++ )
                {
                    j = rng.uniform(0, count);
                    k = rng.uniform(0, count);
                    std::swap(_idx[j], _idx[k]);
                }

                //printf("%d. E = %g\n", iter/count, E);
                if( fabs(prev_E - E) < epsilon )
                    break;
                prev_E = E;
                E = 0;

            }

            idx = _idx[idx];

            const uchar* x0data_p = inputs.ptr(idx);
            const float* x0data_f = (const float*)x0data_p;
            const double* x0data_d = (const double*)x0data_p;

            double* w = weights[0].ptr<double>();
            for( j = 0; j < ivcount; j++ )
                x[0][j] = (itype == CV_32F ? (double)x0data_f[j] : x0data_d[j])*w[j*2] + w[j*2 + 1];

            Mat x1( 1, ivcount, CV_64F, &x[0][0] );

            // forward pass, compute y[i]=w*x[i-1], x[i]=f(y[i]), df[i]=f'(y[i])
            for( i = 1; i < l_count; i++ )
            {
                int n = layer_sizes[i];
                Mat x2(1, n, CV_64F, &x[i][0] );
                Mat _w = weights[i].rowRange(0, x1.cols);
                gemm(x1, _w, 1, noArray(), 0, x2);
                Mat _df(1, n, CV_64F, &df[i][0] );
                calc_activ_func_deriv( x2, _df, weights[i] );
                x1 = x2;
            }

            Mat grad1( 1, ovcount, CV_64F, buf[l_count&1] );
            w = weights[l_count+1].ptr<double>();

            // calculate error
            const uchar* udata_p = outputs.ptr(idx);
            const float* udata_f = (const float*)udata_p;
            const double* udata_d = (const double*)udata_p;

            double* gdata = grad1.ptr<double>();
            for( k = 0; k < ovcount; k++ )
            {
                double t = (otype == CV_32F ? (double)udata_f[k] : udata_d[k])*w[k*2] + w[k*2+1] - x[l_count-1][k];
                gdata[k] = t*sweight;
                E += t*t;
            }
            E *= sweight;

            // backward pass, update weights
            for( i = l_count-1; i > 0; i-- )
            {
                int n1 = layer_sizes[i-1], n2 = layer_sizes[i];
                Mat _df(1, n2, CV_64F, &df[i][0]);
                multiply( grad1, _df, grad1 );
                Mat _x(n1+1, 1, CV_64F, &x[i-1][0]);
                x[i-1][n1] = 1.;
                gemm( _x, grad1, params.bpDWScale, dw[i], params.bpMomentScale, dw[i] );
                add( weights[i], dw[i], weights[i] );
                if( i > 1 )
                {
                    Mat grad2(1, n1, CV_64F, buf[i&1]);
                    Mat _w = weights[i].rowRange(0, n1);
                    gemm( grad1, _w, 1, noArray(), 0, grad2, GEMM_2_T );
                    grad1 = grad2;
                }
            }

        }

/// print instruction size and offset information - debugging
LLVM_DUMP_METHOD void RISCVConstantIslands::dumpInstructions() {
  LLVM_DEBUG({
    InsInfoVector &InsInfo = InsUtils->getInsInfo();
    for (unsigned K = 0, F = InsInfo.size(); K != F; ++K) {
      const InstructionInfo &III = InsInfo[K];
      dbgs() << format("%08x %ins.%u\t", III.Offset, K)
             << " kb=" << unsigned(III.KnownBits)
             << " ua=" << unsigned(III.Unalign) << " pa=" << Log2(III.PostAlign)
             << format(" size=%#x\n", InsInfo[K].Size);
    }
  });
}

