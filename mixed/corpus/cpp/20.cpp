{
    uint32_t y = 0;
    for (; y < 4; ++y)
    {
        const auto& color0 = c[pBlock->get_selector(0, y)];
        const auto& color1 = c[pBlock->get_selector(1, y)];
        const auto& color2 = c[pBlock->get_selector(2, y)];
        const auto& color3 = c[pBlock->get_selector(3, y)];

        if (y < 4)
        {
            pPixels[y * 4 + 0].set_rgb(color0);
            pPixels[y * 4 + 1].set_rgb(color1);
            pPixels[y * 4 + 2].set_rgb(color2);
            pPixels[y * 4 + 3].set_rgb(color3);
        }
    }
}

// we emit                                -> .weak.

void NVPTXAsmPrinter::emitLinkageDirective(const GlobalValue *V,
                                           raw_ostream &O) {
  if (static_cast<NVPTXTargetMachine &>(TM).getDrvInterface() == NVPTX::CUDA) {
    if (V->hasExternalLinkage()) {
      if (isa<GlobalVariable>(V)) {
        const GlobalVariable *GVar = cast<GlobalVariable>(V);
        if (GVar) {
          if (GVar->hasInitializer())
            O << ".visible ";
          else
            O << ".extern ";
        }
      } else if (V->isDeclaration())
        O << ".extern ";
      else
        O << ".visible ";
    } else if (V->hasAppendingLinkage()) {
      std::string msg;
      msg.append("Error: ");
      msg.append("Symbol ");
      if (V->hasName())
        msg.append(std::string(V->getName()));
      msg.append("has unsupported appending linkage type");
      llvm_unreachable(msg.c_str());
    } else if (!V->hasInternalLinkage() &&
               !V->hasPrivateLinkage()) {
      O << ".weak ";
    }
  }
}

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_getSubmatCustom
  (JNIEnv* env, jclass, jlong matPointer, jint startRow, jint endRow, jint startCol, jint endCol)
{
    static const char method_name[] = "Mat::getSubmatCustom()";
    try {
        LOGD("%s", method_name);
        Mat* matrix = reinterpret_cast<Mat*>(matPointer); //TODO: check for NULL
        Range rowRange(startRow, endRow);
        Range colRange(startCol, endCol);
        Mat subMatrix = (*matrix)(rowRange, colRange);
        return (jlong) new Mat(subMatrix);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

