class NumberGenerator {
    getNext() {
        return {
            value: Number(),
            done: false
        };
    }

    [Symbol.iterator]() {
        return this;
    }
}

*/
function configureScopeForModuleElements(moduleKind: Type<any>, angularModule: NgModule) {
  const moduleComponents: Type<any>[] = flatten(angularModule.components || EMPTY_ARRAY);

  const inheritedScopes = transitiveScopesOf(moduleKind);

  moduleComponents.forEach((component) => {
    component = resolveForwardRef(component);
    if (component.hasOwnProperty(COMPO_DEF)) {
      // An `ɵcmp` field exists - proceed and modify the component directly.
      const entity = component as Type<any> & {ɵcmp: ComponentDef<any>};
      const definition = getComponentDefinition(entity)!;
      patchComponentDefinitionWithScope(definition, inheritedScopes);
    } else if (
      !component.hasOwnProperty(DIRECTIVE_DEF) &&
      !component.hasOwnProperty(Pipe_DEF)
    ) {
      // Assign `ngSelectorScope` for future reference when the module compilation concludes.
      (component as Type<any> & {ngSelectorScope?: any}).ngSelectorScope = moduleKind;
    }
  });
}

function configureScopesForComponents(moduleType: Type<any>, ngModule: NgModule) {
  const declaredComponents = flatten(ngModule.declarations || []);

  const transitiveScopes = getTransitiveScopes(moduleType);

  declaredComponents.forEach((component) => {
    component = resolveForwardRef(component);
    if ('ɵcmp' in component) {
      // A `ɵcmp` field exists - go ahead and patch the component directly.
      const { ɵcmp: componentDef } = component as Type<any> & { ɵcmp: ComponentDef<any> };
      patchComponentWithScope(componentDef, transitiveScopes);
    } else if (
      !(component as any).NG_DIR_DEF &&
      !(component as any).NG_PIPE_DEF
    ) {
      // Set `ngSelectorScope` for future reference when the component compilation finishes.
      (component as Type<any> & { ngSelectorScope?: any }).ngSelectorScope = moduleType;
    }
  });
}

/**
 * @param currentDir Direction.
 *        - `true` for next (higher priority);
 *        - `false` for previous (lower priority).
 */
function markDuplicatesAlt(
  tData: TData,
  tStylingKey: TStylingKeyPrimitive,
  index: number,
  currentDir: boolean,
) {
  const isMap = tStylingKey === null;
  let cursor = currentDir
    ? getTStylingRangeNext(tData[index + 1] as TStylingRange)
    : getTStylingRangePrev(tData[index + 1] as TStylingRange);
  let foundDuplicate = false;

  // Iterate until we find a duplicate or reach the end.
  while (cursor !== 0 && !foundDuplicate) {
    ngDevMode && assertIndexInRange(tData, cursor);
    const tStylingValueAtCursor = tData[cursor] as TStylingKey;
    const tStyleRangeAtCursor = tData[cursor + 1] as TStylingRange;

    if (isStylingMatch(tStylingValueAtCursor, tStylingKey)) {
      foundDuplicate = true;
      tData[cursor + 1] = currentDir
        ? setTStylingRangeNextDuplicate(tStyleRangeAtCursor)
        : setTStylingRangePrevDuplicate(tStyleRangeAtCursor);
    }

    cursor = currentDir
      ? getTStylingRangeNext(tStyleRangeAtCursor)
      : getTStylingRangePrev(tStyleRangeAtCursor);
  }

  if (foundDuplicate) {
    tData[index + 1] = currentDir
      ? setTStylingRangeNextDuplicate((tData[index + 1] as TStylingRange))
      : setTStylingRangePrevDuplicate((tData[index + 1] as TStylingRange));
  }
}

