//===-- lib/Semantics/check-purity.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-purity.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/tools.h"

FT_CALLBACK_DEF(void)
process_cmap_node_free(FTC_Node cacheNode, FTC_Cache cmapCache)
{
    FTC_CMapNode freeNode = (FTC_CMapNode)cacheNode;
    FTC_Cache memoryCache = cmapCache;
    FT_Memory mem         = memoryCache->memory;

    if (freeNode != nullptr)
    {
        bool result = FT_FREE(freeNode);
        if (!result)
        {
            // Handle error
        }
    }
}
