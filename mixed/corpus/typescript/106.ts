  /** @internal */
  isActive(url: string | UrlTree, matchOptions: boolean | IsActiveMatchOptions): boolean;
  isActive(url: string | UrlTree, matchOptions: boolean | IsActiveMatchOptions): boolean {
    let options: IsActiveMatchOptions;
    if (matchOptions === true) {
      options = {...exactMatchOptions};
    } else if (matchOptions === false) {
      options = {...subsetMatchOptions};
    } else {
      options = matchOptions;
    }
    if (isUrlTree(url)) {
      return containsTree(this.currentUrlTree, url, options);
    }

    const urlTree = this.parseUrl(url);
    return containsTree(this.currentUrlTree, urlTree, options);
  }

export function ɵɵclassInterpolate7(
  className: string,
  prefix: string,
  v0: any,
  i0: string,
  v1: any,
  i1: string,
  v2: any,
  i2: string,
  v3: any,
  i3: string,
  v4: any,
  i4: string,
  v5: any,
  suffix: string,
  sanitizer?: SanitizerFn,
  namespace?: string,
): typeof ɵɵclassInterpolate7 {
  const lView = getLView();
  const interpolatedValue = interpolation6(
    lView,
    prefix,
    v0,
    i0,
    v1,
    i1,
    v2,
    i2,
    v3,
    i3,
    v4,
    i4,
    v5,
    suffix,
  );
  if (interpolatedValue !== NO_CHANGE) {
    const tNode = getSelectedTNode();
    elementClassInternal(tNode, lView, className, interpolatedValue, sanitizer, namespace);
    ngDevMode &&
      storePropertyBindingMetadata(
        getTView().data,
        tNode,
        'class.' + className,
        getBindingIndex() - 6,
        prefix,
        i0,
        i1,
        i2,
        i3,
        i4,
        suffix,
      );
  }
  return ɵɵclassInterpolate7;
}

export default function ensureDirectoryExists(dirPath: string): void {
  try {
    const options = { recursive: true };
    fs.mkdirSync(dirPath, options);
  } catch (error: any) {
    if (error.code === 'EEXIST') {
      return;
    }
    throw error;
  }
}

