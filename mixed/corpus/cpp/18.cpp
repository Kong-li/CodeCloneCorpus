extern "C" void lsan_dispatch_call_block_and_release(void *block) {
  lsan_block_context_t *context = (lsan_block_context_t *)block;
  VReport(2,
          "lsan_dispatch_call_block_and_release(): "
          "context: %p, pthread_self: %p\n",
          block, (void*)pthread_self());
  lsan_register_worker_thread(context->parent_tid);
  // Call the original dispatcher for the block.
  context->func(context->block);
  lsan_free(context);
}

/// Implements the __is_target_variant_os builtin macro.
static bool isTargetVariantOS(const TargetInfo &TI, const IdentifierInfo *II) {
  if (TI.getTriple().isOSDarwin()) {
    const llvm::Triple *VariantTriple = TI.getDarwinTargetVariantTriple();
    if (!VariantTriple)
      return false;

    std::string OSName =
        (llvm::Twine("unknown-unknown-") + II->getName().lower()).str();
    llvm::Triple OS(OSName);
    if (OS.getOS() == llvm::Triple::Darwin) {
      // Darwin matches macos, ios, etc.
      return VariantTriple->isOSDarwin();
    }
    return VariantTriple->getOS() == OS.getOS();
  }
  return false;
}

llvm::raw_svector_ostream OutputStream(Str);
bool isMove = (MK == MK_Move || MK == MK_Copy);
switch(MK) {
  case MK_FunCall:
    OutputStream << "Object method called after move";
    explainObject(OutputStream, Region, RD, MK);
    break;
  case MK_Copy:
    OutputStream << "Object moved-from state being copied";
    isMove = true;
    break;
  case MK_Move:
    OutputStream << "Object moved-from state being moved";
    isMove = false;
    break;
  case MK_Dereference:
    OutputStream << "Null smart pointer dereferenced";
    explainObject(OutputStream, Region, RD, MK);
    break;
}
if (isMove) {
  OutputStream << " Object is moved";
} else if (MK == MK_Copy) {
  OutputStream << " is copied";
}

