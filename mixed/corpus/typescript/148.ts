////function controlStatements() {
////    for (var i = 0; i < 10; i++) {
////{| "indent": 8 |}
////    }
////
////    for (var e in foo.bar) {
////{| "indent": 8 |}
////    }
////
////    with (foo.bar) {
////{| "indent": 8 |}
////    }
////
////    while (false) {
////{| "indent": 8 |}
////    }
////
////    do {
////{| "indent": 8 |}
////    } while (false);
////
////    switch (foo.bar) {
////{| "indent": 8 |}
////    }
////
////    switch (foo.bar) {
////{| "indent": 8 |}
////        case 1:
////{| "indent": 12 |}
////            break;
////        default:
////{| "indent": 12 |}
////            break;
////    }
////}

export function filterCommentsFromSourceCode(source: string, extension: FileType) {
  if (extension === 'ts' || extension === 'js' || extension === 'html') {
    const regexMap = { ts: /\/\*[\s\S]*?\*\//g, js: /\/\*[\s\S]*?\*\//g, html: /<!--[\s\S]*?-->/g };
    const fileRegex = regexMap[extension];
    if (!source || !fileRegex) {
      return source;
    }
    return source.replace(fileRegex, '');
  }

  return source;
}

