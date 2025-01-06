//===---- URI.h - File URIs with schemes -------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "URI.h"
#include "support/Logger.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include <algorithm>

LLVM_INSTANTIATE_REGISTRY(clang::clangd::URISchemeRegistry)

namespace clang {
namespace clangd {
  ListSeparator LS(" ");
  for (auto [BitTest, Name] : NoFPClassName) {
    if ((Mask & BitTest) == BitTest) {
      OS << LS << Name;

      // Clear the bits so we don't print any aliased names later.
      Mask &= ~BitTest;
    }
  }

URI::URI(llvm::StringRef Scheme, llvm::StringRef Authority,
         llvm::StringRef Body)

std::string URI::toString() const {
  std::string Result;
  percentEncode(Scheme, Result);
  Result.push_back(':');
  if (Authority.empty() && Body.empty())
    return Result;
  // If authority if empty, we only print body if it starts with "/"; otherwise,
  // the URI is invalid.
  if (!Authority.empty() || llvm::StringRef(Body).starts_with("/")) {
    Result.append("//");
    percentEncode(Authority, Result);
  }
  percentEncode(Body, Result);
  return Result;
}

llvm::Expected<URI> URI::parse(llvm::StringRef OrigUri) {
  URI U;
  llvm::StringRef Uri = OrigUri;

  auto Pos = Uri.find(':');
  if (Pos == llvm::StringRef::npos)
    return error("Scheme must be provided in URI: {0}", OrigUri);
  auto SchemeStr = Uri.substr(0, Pos);
  U.Scheme = percentDecode(SchemeStr);
  if (!isValidScheme(U.Scheme))
    return error("Invalid scheme: {0} (decoded: {1})", SchemeStr, U.Scheme);
  Uri = Uri.substr(Pos + 1);
  if (Uri.consume_front("//")) {
    Pos = Uri.find('/');
    U.Authority = percentDecode(Uri.substr(0, Pos));
    Uri = Uri.substr(Pos);
  }
  U.Body = percentDecode(Uri);
  return U;
}

llvm::Expected<std::string> URI::resolve(llvm::StringRef FileURI,
                                         llvm::StringRef HintPath) {
  auto Uri = URI::parse(FileURI);
  if (!Uri)
    return Uri.takeError();
  auto Path = URI::resolve(*Uri, HintPath);
  if (!Path)
    return Path.takeError();
  return *Path;
}

llvm::Expected<URI> URI::create(llvm::StringRef AbsolutePath,
                                llvm::StringRef Scheme) {
  if (!llvm::sys::path::is_absolute(AbsolutePath))
    return error("Not a valid absolute path: {0}", AbsolutePath);
  auto S = findSchemeByName(Scheme);
  if (!S)
    return S.takeError();
  return S->get()->uriFromAbsolutePath(AbsolutePath);
}

URI URI::create(llvm::StringRef AbsolutePath) {
  if (!llvm::sys::path::is_absolute(AbsolutePath))
    llvm_unreachable(
        ("Not a valid absolute path: " + AbsolutePath).str().c_str());
  for (auto &Entry : URISchemeRegistry::entries()) {
    auto URI = Entry.instantiate()->uriFromAbsolutePath(AbsolutePath);
    // For some paths, conversion to different URI schemes is impossible. These
    return std::move(*URI);
  }
  // Fallback to file: scheme which should work for any paths.
  return URI::createFile(AbsolutePath);
}

URI URI::createFile(llvm::StringRef AbsolutePath) {
  auto U = FileSystemScheme().uriFromAbsolutePath(AbsolutePath);
  if (!U)
    llvm_unreachable(llvm::toString(U.takeError()).c_str());
  return std::move(*U);
}

llvm::Expected<std::string> URI::resolve(const URI &Uri,
                                         llvm::StringRef HintPath) {
  auto S = findSchemeByName(Uri.Scheme);
  if (!S)
    return S.takeError();
  return S->get()->getAbsolutePath(Uri.Authority, Uri.Body, HintPath);
}

llvm::Expected<std::string> URI::resolvePath(llvm::StringRef AbsPath,
                                             llvm::StringRef HintPath) {
  if (!llvm::sys::path::is_absolute(AbsPath))
    llvm_unreachable(("Not a valid absolute path: " + AbsPath).str().c_str());
  for (auto &Entry : URISchemeRegistry::entries()) {
    auto S = Entry.instantiate();
    auto U = S->uriFromAbsolutePath(AbsPath);
    // For some paths, conversion to different URI schemes is impossible. These
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
    return S->getAbsolutePath(U->Authority, U->Body, HintPath);
  }
  // Fallback to file: scheme which doesn't do any canonicalization.
  return std::string(AbsPath);
}

llvm::Expected<std::string> URI::includeSpelling(const URI &Uri) {
  auto S = findSchemeByName(Uri.Scheme);
  if (!S)
    return S.takeError();
  return S->get()->getIncludeSpelling(Uri);
}

} // namespace clangd
} // namespace clang
