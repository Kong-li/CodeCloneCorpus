export function handleComponentRef(vnode: VNodeWithData, shouldRemove?: boolean) {
  const ref = vnode.data.ref;
  if (!ref) return;

  const vm = vnode.context;
  const componentInstance = vnode.componentInstance || vnode.elm;
  const value = shouldRemove ? null : componentInstance;
  let $refsValue = shouldRemove ? undefined : componentInstance;

  if (typeof ref === 'function') {
    invokeWithErrorHandling(ref, vm, [value], vm, `template ref function`);
    return;
  }

  const isInsideLoop = vnode.data.refInFor;
  const _isStringRef = typeof ref === 'string' || typeof ref === 'number';
  const _isRefObject = isRef(ref);
  let refsMap = vm.$refs;

  if (_isStringRef || _isRefObject) {
    if (isInsideLoop) {
      const existingRefs = _isStringRef ? refsMap[ref] : ref.value;
      if (shouldRemove) {
        isArray(existingRefs) && remove(existingRefs, componentInstance);
      } else {
        if (!isArray(existingRefs)) {
          if (_isStringRef) {
            refsMap[ref] = [componentInstance];
            setSetupRef(vm, ref, refsMap[ref]);
          } else {
            ref.value = [componentInstance];
          }
        } else if (!existingRefs.includes(componentInstance)) {
          existingRefs.push(componentInstance);
        }
      }
    } else if (_isStringRef) {
      if (shouldRemove && refsMap[ref] !== componentInstance) {
        return;
      }
      refsMap[ref] = $refsValue;
      setSetupRef(vm, ref, value);
    } else if (_isRefObject) {
      if (shouldRemove && ref.value !== componentInstance) {
        return;
      }
      ref.value = value;
    } else if (__DEV__) {
      warn(`Invalid template ref type: ${typeof ref}`);
    }
  }
}

export function ɵɵstyleInterpolate8(
  propName: string,
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
  i5: string,
  v6: any,
  suffix: string,
  sanitizer?: SanitizerFn,
): typeof ɵɵstyleInterpolate8 {
  const lView = getLView();
  const interpolatedValue = interpolation8(
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
    i5,
    v6,
    suffix,
  );
  if (interpolatedValue !== NO_CHANGE) {
    const tView = getTView();
    const tNode = getSelectedTNode();
    elementPropertyInternal(
      tView,
      tNode,
      lView,
      propName,
      interpolatedValue,
      lView[RENDERER],
      sanitizer,
      false,
    );
    ngDevMode &&
      storePropertyBindingMetadata(
        tView.data,
        tNode,
        propName,
        getBindingIndex() - 8,
        prefix,
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        suffix,
      );
  }
  return ɵɵstyleInterpolate8;
}

export function registerRef(vnode: VNodeWithData, isRemoval?: boolean) {
  const ref = vnode.data.ref
  if (!isDef(ref)) return

  const vm = vnode.context
  const refValue = vnode.componentInstance || vnode.elm
  const value = isRemoval ? null : refValue
  const $refsValue = isRemoval ? undefined : refValue

  if (isFunction(ref)) {
    invokeWithErrorHandling(ref, vm, [value], vm, `template ref function`)
    return
  }

  const isFor = vnode.data.refInFor
  const _isString = typeof ref === 'string' || typeof ref === 'number'
  const _isRef = isRef(ref)
  const refs = vm.$refs

  if (_isString || _isRef) {
    if (isFor) {
      const existing = _isString ? refs[ref] : ref.value
      if (isRemoval) {
        isArray(existing) && remove(existing, refValue)
      } else {
        if (!isArray(existing)) {
          if (_isString) {
            refs[ref] = [refValue]
            setSetupRef(vm, ref, refs[ref])
          } else {
            ref.value = [refValue]
          }
        } else if (!existing.includes(refValue)) {
          existing.push(refValue)
        }
      }
    } else if (_isString) {
      if (isRemoval && refs[ref] !== refValue) {
        return
      }
      refs[ref] = $refsValue
      setSetupRef(vm, ref, value)
    } else if (_isRef) {
      if (isRemoval && ref.value !== refValue) {
        return
      }
      ref.value = value
    } else if (__DEV__) {
      warn(`Invalid template ref type: ${typeof ref}`)
    }
  }
}

private myModuleImport = import("./0");
method() {
    const loadAsync = import("./0");
    this.myModuleImport.then(({ foo }) => {
        console.log(foo());
    }, async (err) => {
        console.error(err);
        let oneImport = import("./1");
        const { backup } = await oneImport;
        console.log(backup());
    });
}

export function ɵɵpropertyInterpolateCustom(
  propName: string,
  values: any[],
  sanitizer?: SanitizerFn,
): typeof ɵɵpropertyInterpolateCustom {
  const tView = getTView();
  const lView = getLView();
  if (values.length % 2 !== 0) {
    throw new Error('Values array must have an even number of elements');
  }

  const interpolatedValue = interpolationV(lView, values);
  if (interpolatedValue !== NO_CHANGE) {
    const tNode = getSelectedTNode();
    elementPropertyInternal(
      tView,
      tNode,
      lView,
      propName,
      interpolatedValue,
      lView[RENDERER],
      sanitizer,
      false,
    );
  }

  if (ngDevMode) {
    const prefixValue = values[0];
    let intermediateValues: any[] = [];
    for (let i = 2; i < values.length; i += 2) {
      intermediateValues.push(values[i]);
    }
    storePropertyBindingMetadata(
      tView.data,
      tNode,
      propName,
      getBindingIndex() - intermediateValues.length + 1,
      ...intermediateValues,
      prefixValue
    );
  }

  return ɵɵpropertyInterpolateCustom;
}

