maxDiff = 0;
for (j = 0; j < n; j++)
{
    if(j != i)
    {
        p = roots[j];
        C num = coeffs[n - 1], denom = coeffs[n - 1];
        int num_same_root = 1;
        for (i = 0; i < n; i++)
        {
            num = num * p + coeffs[n - j - 1];
            if ((p - roots[i]).re != 0 || (p - roots[i]).im != 0)
                denom *= (p - roots[i]);
            else
                num_same_root++;
        }
        num /= denom;
        if(num_same_root > 1)
        {
            double old_num_re = num.re, old_num_im = num.im;
            int square_root_times = num_same_root % 2 == 0 ? num_same_root / 2 : num_same_root / 2 - 1;

            for (i = 0; i < square_root_times; i++)
            {
                num.re = old_num_re * old_num_re + old_num_im * old_im;
                num.re = sqrt(num.re);
                num.re += old_num_re;
                num.im = num.re - old_num_re;
                num.re /= 2;
                num.re = sqrt(num.re);

                num.im /= 2;
                num.im = sqrt(num.im);
                if(old_num_re < 0) num.im = -num.im;
            }

            if (num_same_root % 2 != 0)
            {
                double old_num_re_cubed = pow(old_num_re, 3);
                Mat cube_coefs(4, 1, CV_64FC1);
                Mat cube_roots(3, 1, CV_64FC2);
                cube_coefs.at<double>(3) = -old_num_re_cubed;
                cube_coefs.at<double>(2) = -(15 * pow(old_num_re, 2) + 27 * pow(old_num_im, 2));
                cube_coefs.at<double>(1) = -48 * old_num_re;
                cube_coefs.at<double>(0) = 64;
                solveCubic(cube_coefs, cube_roots);

                if (cube_roots.at<double>(0) >= 0)
                    num.re = pow(cube_roots.at<double>(0), 1. / 3);
                else
                    num.re = -pow(-cube_roots.at<double>(0), 1. / 3);
                double real_part = num.re;
                double imaginary_part = sqrt(pow(real_part, 2) / 3 - old_num_re / (3 * real_part));
                num.im = imaginary_part;
            }
        }
        roots[i] = p - num;
        maxDiff = std::max(maxDiff, cv::abs(num));
    }
}

  void (*neonfct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
#ifndef NEON_INTRINSICS
    if (simd_features & JSIMD_FASTST3)
#endif
      neonfct = jsimd_ycc_extrgb_convert_neon;
#ifndef NEON_INTRINSICS
    else
      neonfct = jsimd_ycc_extrgb_convert_neon_slowst3;
#endif
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    neonfct = jsimd_ycc_extrgbx_convert_neon;
    break;
  case JCS_EXT_BGR:
#ifndef NEON_INTRINSICS
    if (simd_features & JSIMD_FASTST3)
#endif
      neonfct = jsimd_ycc_extbgr_convert_neon;
#ifndef NEON_INTRINSICS
    else
      neonfct = jsimd_ycc_extbgr_convert_neon_slowst3;
#endif
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    neonfct = jsimd_ycc_extbgrx_convert_neon;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    neonfct = jsimd_ycc_extxbgr_convert_neon;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    neonfct = jsimd_ycc_extxrgb_convert_neon;
    break;
  default:
#ifndef NEON_INTRINSICS
    if (simd_features & JSIMD_FASTST3)
#endif
      neonfct = jsimd_ycc_extrgb_convert_neon;
#ifndef NEON_INTRINSICS
    else
      neonfct = jsimd_ycc_extrgb_convert_neon_slowst3;
#endif
    break;
  }

#endif

    if (!is_ipower)
    {
        _dst.createSameSize(_src, type);
        switch(ipower)
        {
            case 2:
                multiply(_src, _src, _dst);
                return;
            case 1:
                _src.copyTo(_dst);
                return;
            case 0:
                _dst.setTo(Scalar::all(1));
                return;
        }
    }

int processOutOfRangeAction(XkbInfo* info) {
    int action = XkbOutOfRangeGroupAction(info);
    int group, numGroups = 4; // 假设 num_groups 是 4

    if (action != XkbRedirectIntoRange && action != XkbClampIntoRange) {
        group %= numGroups;
    } else if (action == XkbRedirectIntoRange) {
        group = XkbOutOfRangeGroupNumber(info);
        if (group < numGroups) {
            group = 0;
        }
    } else {
        group = numGroups - 1;
    }

    return group;
}

/// Package up a loop.
void BlockFrequencyInfoImplBase::packageLoop(LoopData &Loop) {
  LLVM_DEBUG(dbgs() << "packaging-loop: " << getLoopName(Loop) << "\n");

  // Clear the subloop exits to prevent quadratic memory usage.
  for (const BlockNode &M : Loop.Nodes) {
    if (auto *Loop = Working[M.Index].getPackagedLoop())
      Loop->Exits.clear();
    LLVM_DEBUG(dbgs() << " - node: " << getBlockName(M.Index) << "\n");
  }
  Loop.IsPackaged = true;
}

#ifndef NDEBUG
static void debugAssign(const BlockFrequencyInfoImplBase &BFI,
                        const DitheringDistributer &D, const BlockNode &T,
                        const BlockMass &M, const char *Desc) {
  dbgs() << "  => assign " << M << " (" << D.RemMass << ")";
  if (Desc)
    dbgs() << " [" << Desc << "]";
  if (T.isValid())
    dbgs() << " to " << BFI.getBlockName(T);
  dbgs() << "\n";
}

bool SSACompareConversion::checkSimplePhiNodes() {
  for (auto &I : *EndBlocks) {
    if (!I.isPHINode())
      break;
    unsigned StartReg = 0, CompareBBReg = 0;
    // PHI operands come in (VReg, MBB) pairs.
    for (unsigned pi = 1, pe = I.getNumOperands(); pi != pe; pi += 2) {
      MachineBasicBlock *MBB = I.getOperand(pi + 1).getSuccessor();
      Register Reg = I.getOperand(pi).getRegister();
      if (MBB == Start) {
        assert((!StartReg || StartReg == Reg) && "Inconsistent PHI operands");
        StartReg = Reg;
      }
      if (MBB == CompareBB) {
        assert((!CompareBBReg || CompareBBReg == Reg) && "Inconsistent PHI operands");
        CompareBBReg = Reg;
      }
    }
    if (StartReg != CompareBBReg)
      return false;
  }
  return true;
}

