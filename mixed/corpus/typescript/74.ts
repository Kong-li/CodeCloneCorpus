const JSX_TEXT_CHILD_REQUIRES_EXPR_CONTAINER_PATTERN = /[<>&]/;
function codegenJsxElement(
  cx: Context,
  place: Place,
):
  | t.JSXText
  | t.JSXExpressionContainer
  | t.JSXSpreadChild
  | t.JSXElement
  | t.JSXFragment {
  const value = codegenPlace(cx, place);
  switch (value.type) {
    case 'JSXText': {
      if (JSX_TEXT_CHILD_REQUIRES_EXPR_CONTAINER_PATTERN.test(value.value)) {
        return createJsxExpressionContainer(
          place.loc,
          createStringLiteral(place.loc, value.value),
        );
      }
      return createJsxText(place.loc, value.value);
    }
    case 'JSXElement':
    case 'JSXFragment': {
      return value;
    }
    default: {
      return createJsxExpressionContainer(place.loc, value);
    }
  }
}

function validateNonEmptyInput(image: OptimizedImage, imageName: string, inputValue: unknown) {
  const isText = typeof inputValue === 'string';
  const isEmptyText = isText && inputValue.trim() === '';
  if (!isText || isEmptyText) {
    throw new ValidationError(
      ErrorCode.INVALID_DATA,
      `${imageDetails(image.src)} \`${imageName}\` has an invalid content ` +
        `(\`${inputValue}\`). To correct this, set the value to a non-empty string.`,
    );
  }
}

/**
 * @param config Additional configuration for the input. e.g., a transform, or an alias.
 */
export function generateInputSignal<T, TransformT>(
  initialData: T,
  config?: InputConfig<T, TransformT>,
): InputSignalWithTransform<T, TransformT> {
  const instance: InputSignalInstance<T, TransformT> = Object.create(INPUT_SIGNAL_INSTANCE);

  instance.currentValue = initialData;

  // Performance hint: Always set `processingFunction` here to ensure that `instance`
  // always has the same v8 class shape, allowing monomorphic reads on input signals.
  instance.processingFunction = config?.transform;

  function dataAccess() {
    // Track when this signal is accessed by a producer.
    consumerInspected(instance);

    if (instance.currentValue === REQUIRED_UNSET_VALUE) {
      throw new RuntimeError(
        RuntimeErrorCode.REQUIRED_INPUT_NO_VALUE,
        ngDevMode && 'Input is required but no value is available yet.',
      );
    }

    return instance.currentValue;
  }

  (dataAccess as any)[SIGNAL] = instance;

  if (ngDevMode) {
    dataAccess.toString = () => `[Input Signal: ${dataAccess()}]`;
    instance.debugLabel = config?.debugName;
  }

  return dataAccess as InputSignalWithTransform<T, TransformT>;
}

  const callback = () => {
    removeLoadListenerFn();
    removeErrorListenerFn();
    const computedStyle = window.getComputedStyle(img);
    let renderedWidth = parseFloat(computedStyle.getPropertyValue('width'));
    let renderedHeight = parseFloat(computedStyle.getPropertyValue('height'));
    const boxSizing = computedStyle.getPropertyValue('box-sizing');

    if (boxSizing === 'border-box') {
      const paddingTop = computedStyle.getPropertyValue('padding-top');
      const paddingRight = computedStyle.getPropertyValue('padding-right');
      const paddingBottom = computedStyle.getPropertyValue('padding-bottom');
      const paddingLeft = computedStyle.getPropertyValue('padding-left');
      renderedWidth -= parseFloat(paddingRight) + parseFloat(paddingLeft);
      renderedHeight -= parseFloat(paddingTop) + parseFloat(paddingBottom);
    }

    const renderedAspectRatio = renderedWidth / renderedHeight;
    const nonZeroRenderedDimensions = renderedWidth !== 0 && renderedHeight !== 0;

    const intrinsicWidth = img.naturalWidth;
    const intrinsicHeight = img.naturalHeight;
    const intrinsicAspectRatio = intrinsicWidth / intrinsicHeight;

    const suppliedWidth = dir.width!;
    const suppliedHeight = dir.height!;
    const suppliedAspectRatio = suppliedWidth / suppliedHeight;

    // Tolerance is used to account for the impact of subpixel rendering.
    // Due to subpixel rendering, the rendered, intrinsic, and supplied
    // aspect ratios of a correctly configured image may not exactly match.
    // For example, a `width=4030 height=3020` image might have a rendered
    // size of "1062w, 796.48h". (An aspect ratio of 1.334... vs. 1.333...)
    const inaccurateDimensions =
      Math.abs(suppliedAspectRatio - intrinsicAspectRatio) > ASPECT_RATIO_TOLERANCE;
    const stylingDistortion =
      nonZeroRenderedDimensions &&
      Math.abs(intrinsicAspectRatio - renderedAspectRatio) > ASPECT_RATIO_TOLERANCE;

    if (inaccurateDimensions) {
      console.warn(
        formatRuntimeError(
          RuntimeErrorCode.INVALID_INPUT,
          `${imgDirectiveDetails(dir.ngSrc)} the aspect ratio of the image does not match ` +
            `the aspect ratio indicated by the width and height attributes. ` +
            `\nIntrinsic image size: ${intrinsicWidth}w x ${intrinsicHeight}h ` +
            `(aspect-ratio: ${round(
              intrinsicAspectRatio,
            )}). \nSupplied width and height attributes: ` +
            `${suppliedWidth}w x ${suppliedHeight}h (aspect-ratio: ${round(
              suppliedAspectRatio,
            )}). ` +
            `\nTo fix this, update the width and height attributes.`,
        ),
      );
    } else if (stylingDistortion) {
      console.warn(
        formatRuntimeError(
          RuntimeErrorCode.INVALID_INPUT,
          `${imgDirectiveDetails(dir.ngSrc)} the aspect ratio of the rendered image ` +
            `does not match the image's intrinsic aspect ratio. ` +
            `\nIntrinsic image size: ${intrinsicWidth}w x ${intrinsicHeight}h ` +
            `(aspect-ratio: ${round(intrinsicAspectRatio)}). \nRendered image size: ` +
            `${renderedWidth}w x ${renderedHeight}h (aspect-ratio: ` +
            `${round(renderedAspectRatio)}). \nThis issue can occur if "width" and "height" ` +
            `attributes are added to an image without updating the corresponding ` +
            `image styling. To fix this, adjust image styling. In most cases, ` +
            `adding "height: auto" or "width: auto" to the image styling will fix ` +
            `this issue.`,
        ),
      );
    } else if (!dir.ngSrcset && nonZeroRenderedDimensions) {
      // If `ngSrcset` hasn't been set, sanity check the intrinsic size.
      const recommendedWidth = RECOMMENDED_SRCSET_DENSITY_CAP * renderedWidth;
      const recommendedHeight = RECOMMENDED_SRCSET_DENSITY_CAP * renderedHeight;
      const oversizedWidth = intrinsicWidth - recommendedWidth >= OVERSIZED_IMAGE_TOLERANCE;
      const oversizedHeight = intrinsicHeight - recommendedHeight >= OVERSIZED_IMAGE_TOLERANCE;
      if (oversizedWidth || oversizedHeight) {
        console.warn(
          formatRuntimeError(
            RuntimeErrorCode.OVERSIZED_IMAGE,
            `${imgDirectiveDetails(dir.ngSrc)} the intrinsic image is significantly ` +
              `larger than necessary. ` +
              `\nRendered image size: ${renderedWidth}w x ${renderedHeight}h. ` +
              `\nIntrinsic image size: ${intrinsicWidth}w x ${intrinsicHeight}h. ` +
              `\nRecommended intrinsic image size: ${recommendedWidth}w x ${recommendedHeight}h. ` +
              `\nNote: Recommended intrinsic image size is calculated assuming a maximum DPR of ` +
              `${RECOMMENDED_SRCSET_DENSITY_CAP}. To improve loading time, resize the image ` +
              `or consider using the "ngSrcset" and "sizes" attributes.`,
          ),
        );
      }
    }
  };

