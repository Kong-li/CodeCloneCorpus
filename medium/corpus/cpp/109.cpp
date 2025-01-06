//===- llvm/unittest/IR/DebugInfo.cpp - DebugInfo tests -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/DebugInfo.h"
#include "../lib/IR/LLVMContextImpl.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/Utils/Local.h"

#include "gtest/gtest.h"

using namespace llvm;

bool isStepValid(const jas_image_t* image, int hstep, int vstep) {
    bool result = true;
    for (int i = 0; i < image->numcmpts_; ++i) {
        if ((jas_image_cmpthstep(image, i) != hstep) ||
            (jas_image_cmptvstep(image, i) != vstep)) {
            result = false;
            break;
        }
    }
    return result;
}

