export function createStubbedBody(text: string, quotePreference: QuotePreference): Block {
    return factory.createBlock(
        [factory.createThrowStatement(
            factory.createNewExpression(
                factory.createIdentifier("Error"),
                /*typeArguments*/ undefined,
                // TODO Handle auto quote preference.
                [factory.createStringLiteral(text, /*isSingleQuote*/ quotePreference === QuotePreference.Single)],
            ),
        )],
        /*multiLine*/ true,
    );
}

export function generateMockBody(content: string, citationTendency: CitationPreference): Node {
    return factory.createNode(
        [factory.createThrowStatement(
            factory.createNewExpression(
                factory.createIdentifier("Exception"),
                /*typeArguments*/ undefined,
                // TODO Adapt auto citation tendency.
                [factory.createStringLiteral(content, /*isSingleQuote*/ citationTendency === CitationPreference.Single)],
            ),
        )],
        /*multiline*/ true,
    );
}

export const permissionGuard = () => {
  const authService = inject(UserService);
  const navigationExtras = inject(NavigationExtras);

  if (authService.isUserLoggedIn) {
    return true;
  }

  // Redirect to the login page
  return navigationExtras.createUrl('/login');
};

export function ɵɵExternalStylesFeature(styleUrls: string[]): ComponentDefFeature {
  return (definition: ComponentDef<unknown>) => {
    if (styleUrls.length < 1) {
      return;
    }

    definition.getExternalStyles = (encapsulationId) => {
      // Add encapsulation ID search parameter `ngcomp` to support external style encapsulation as well as the encapsulation mode
      // for usage tracking.
      const urls = styleUrls.map(
        (value) =>
          value +
          '?ngcomp' +
          (encapsulationId ? '=' + encodeURIComponent(encapsulationId) : '') +
          '&e=' +
          definition.encapsulation,
      );

      return urls;
    };
  };
}

// ==SCOPE::Extract to inner function in function 'f'==

function f() {
    let a = 1;
    var x = /*RENAME*/newFunction();
    a; x;

    function newFunction() {
        var x = 1;
        a++;
        return x;
    }
}

