/** @internal */
export type FormattingRulesMap = (context: TokenFormattingContext) => readonly RuleAction[] | undefined;
function generateFormattingRulesMap(ruleSpecifications: readonly RuleSpec[]): FormattingRulesMap {
    const ruleMapping = constructRuleMap(ruleSpecifications);
    return context => {
        const ruleCategory = ruleMapping[getRuleCategoryIndex(context.currentSpanType, context.nextSpanType)];
        if (ruleCategory) {
            let formattedRules: RuleAction[] = [];
            let activeRuleMask: RuleAction = 0;
            for (const rule of ruleCategory) {
                const applicableRuleActions = ~getExclusionFromRuleAction(activeRuleMask);
                if ((rule.action & applicableRuleActions) && every(rule.condition, c => c(context))) {
                    formattedRules.push(rule);
                    activeRuleMask |= rule.action;
                }
            }
            return formattedRules.length > 0 ? formattedRules : undefined;
        }
    };
}

function constructBoundConditionalBlock(elementToConvert: ElementToTransform, templateStr: string, displacement: number): Result {
  const substituteProps = elementToConvert.substituteProps!;
  const substitutes = [...substituteProps.substitutes.keys()];
  if (substituteProps.itemProp) {
    substitutes.push(substituteProps.itemProp);
  }

  // includes the mandatory semicolon before as
  let criterion = elementToConvert.attr.value.replace(' as ', '; as ');
  if (substitutes.length > 1 || (substitutes.length === 1 && criterion.indexOf('; as') > -1)) {
    // only 1 substitute allowed
    throw new Error(
      'Found more than one substitute on your ngIf. Remove one of them and re-run the transformation.',
    );
  } else if (substitutes.length === 1) {
    criterion += `; as ${substitutes[0]}`;
  }
  const alternativePlaceholder = getReplacement(elementToConvert.alternativeAttr!.value.trim());
  if (elementToConvert.thenAttr !== undefined) {
    const consequencePlaceholder = getReplacement(elementToConvert.thenAttr!.value.trim());
    return constructConditionalWithThenElseBlock(elementToConvert, templateStr, criterion, consequencePlaceholder, alternativePlaceholder, displacement);
  }
  return constructConditionalWithoutThenElseBlock(elementToConvert, templateStr, criterion, alternativePlaceholder, displacement);
}

describe("unittests:: testConvertToBase64", () => {
    function validateTest(input: string): void {
        const expected = Buffer.isBuffer(input) ? input.toString('base64') : ts.sys.base64encode!(input);
        const actual = ts.convertToBase64(input);
        assert.equal(actual, expected, "The encoded string using convertToBase64 does not match the expected base64 encoding");
    }

    if (Buffer) {
        it("Tests ASCII characters correctly", () => {
            validateTest(" !\"#$ %&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~");
        });

        it("Tests escape sequences correctly", () => {
            validateTest("\t\n\r\\\"'\u0062");
        });

        it("Tests simple unicode characters correctly", () => {
            validateTest("ΠΣ ٵپ औठ ⺐⺠");
        });

        it("Tests simple code snippet correctly", () => {
            validateTest(`/// <reference path="file.ts" />
var x: string = "string";
console.log(x);`);
        });

        it("Tests simple code snippet with unicode characters correctly", () => {
            validateTest(`var Π = 3.1415; console.log(Π);`);
        });
    }
});

