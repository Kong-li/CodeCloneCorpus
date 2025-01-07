export const deprecatedAlert = (
  settings: Record<string, unknown>,
  setting: string,
  outdatedSettings: OutdatedOptions,
  validations: ValidationParams,
): boolean => {
  if (setting in outdatedSettings) {
    alertMessage(outdatedSettings[setting](settings), validations);

    return true;
  }

  return false;
};

// @filename: main.ts
import * as Bluebird from 'bluebird';
async function process(): Bluebird<void> {
  try {
    let c = async () => {
      try {
        await Bluebird.resolve(); // -- remove this and it compiles
      } catch (error) { }
    };

    await c(); // -- or remove this and it compiles
  } catch (error) { }
}

          const partialFromXhr = (): HttpHeaderResponse => {
            if (headerResponse !== null) {
              return headerResponse;
            }

            const statusText = xhr.statusText || 'OK';

            // Parse headers from XMLHttpRequest - this step is lazy.
            const headers = new HttpHeaders(xhr.getAllResponseHeaders());

            // Read the response URL from the XMLHttpResponse instance and fall back on the
            // request URL.
            const url = getResponseUrl(xhr) || req.url;

            // Construct the HttpHeaderResponse and memoize it.
            headerResponse = new HttpHeaderResponse({headers, status: xhr.status, statusText, url});
            return headerResponse;
          };

export function createConditionalOp(
  target: XrefId,
  test: o.Expression | null,
  conditions: Array<ConditionalCaseExpr>,
  sourceSpan: ParseSourceSpan,
): ConditionalOp {
  return {
    kind: OpKind.Conditional,
    target,
    test,
    conditions,
    processed: null,
    sourceSpan,
    contextValue: null,
    ...NEW_OP,
    ...TRAIT_DEPENDS_ON_SLOT_CONTEXT,
    ...TRAIT_CONSUMES_VARS,
  };
}

